import os, argparse, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F 
from transformers import AutoModel, AutoProcessor

PATCHES = 16
CKPT    = "google/siglip2-base-patch16-naflex"

# --- CHANGED 1: ADD CONSTANTS FROM YOUR TRAINING SCRIPT ---
# These MUST match the constants you used to train the model.
EMBED_DIM  = 768
HIDDEN_DIM = 512
DROP_RATE  = 0.1
# --- END CHANGED 1 ---


# --- UPDATED MODEL (matches train_fixed.py) ---
class SimpleProjector(nn.Module):
    """
    A simple MLP-based projector.
    It takes concatenated (img_patch_feat, txt_feat) and predicts a similarity logit.
    ADDED LayerNorm for training stability.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # ADDED for stability
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2), # ADDED for stability
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, 1) # Output a single logit
        )

    def forward(self, x):
        """
        Input x has shape (B, H, W, in_dim)
        Output will have shape (B, H, W)
        """
        return self.net(x).squeeze(-1)
# --- END UPDATED MODEL ---

def to_multiple(v, m=PATCHES):
    return int(round(v / m) * m)

def fit_image_to_patch_budget(img: Image.Image, max_patches: int = 4096) -> Image.Image:
    W0, H0 = img.size
    W, H = to_multiple(W0), to_multiple(H0)
    gh, gw = H // PATCHES, W // PATCHES
    patches = gh * gw
    if patches <= max_patches:
        return img.resize((W, H), Image.Resampling.BILINEAR) if (W, H) != (W0, H0) else img
    scale = math.sqrt(patches / max_patches)
    newW = max(PATCHES, to_multiple(int(W / scale)))
    newH = max(PATCHES, to_multiple(int(H / scale)))
    return img.resize((newW, newH), Image.Resampling.BILINEAR)

@torch.no_grad()
def build_siglip(ckpt=CKPT):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(ckpt, dtype=dtype).to(device).eval()
    proc = AutoProcessor.from_pretrained(ckpt)
    return model, proc, device

@torch.no_grad()
def encode_image_tokens(pil_img, siglip, processor, device):
    gh, gw = pil_img.height // PATCHES, pil_img.width // PATCHES
    batch = processor(images=pil_img, return_tensors="pt", do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(
        pixel_values=batch["pixel_values"],
        attention_mask=batch["pixel_attention_mask"],
        spatial_shapes=batch["spatial_shapes"]
    )
    feats = out.last_hidden_state[0].float()[batch["pixel_attention_mask"][0].bool()]
    grid_feats = feats.view(gh, gw, -1)
    grid_feats_normalized = torch.nn.functional.normalize(grid_feats, p=2, dim=-1)
    return grid_feats_normalized.cpu().numpy().astype(np.float32)

@torch.no_grad()
def encode_text_emb(text, siglip, processor, device):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids","attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

STOPS = [
    (0.00, (0,   0, 130)), (0.33, (0, 180, 255)),
    (0.66, (255, 255, 0)), (1.00, (255,   0,   0)),
]

def lerp_color(c0, c1, t):
    return tuple(int((1-t)*c0[i] + t*c1[i]) for i in range(3))

def map_prob_to_rgb(p):
    p = float(min(max(p, 0.0), 1.0))
    for i in range(len(STOPS)-1):
        p0, c0 = STOPS[i]; p1, c1 = STOPS[i+1]
        if p <= p1:
            t = 0.0 if p1 == p0 else (p - p0) / (p1 - p0)
            return lerp_color(c0, c1, t)
    return STOPS[-1][1]

def grid_to_blocky_rgba(probs_grid: np.ndarray, out_wh, alpha=255) -> Image.Image:
    gh, gw = probs_grid.shape
    small = np.zeros((gh, gw, 4), dtype=np.uint8)
    for y in range(gh):
        for x in range(gw):
            r, g, b = map_prob_to_rgb(probs_grid[y, x])
            small[y, x, :] = (r, g, b, alpha)
    return Image.fromarray(small, mode="RGBA").resize(out_wh, resample=Image.NEAREST)

def make_vertical_legend(height: int, width: int = 80, margin: int = 8, ticks=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Image.Image:
    bar_w = 24
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        p = 1.0 - (y / max(1, height-1))
        color = map_prob_to_rgb(p)
        draw.line([(margin, y), (margin + bar_w - 1, y)], fill=color, width=1)
    try: font = ImageFont.load_default()
    except Exception: font = None
    for t in ticks:
        y = int((1.0 - t) * (height-1))
        draw.line([(margin + bar_w, y), (margin + bar_w + 6, y)], fill=(255,255,255,255), width=1)
        label = f"{t:.2f}"
        draw.text((margin + bar_w + 8, max(0, y - 6)), label, fill=(255,255,255,255), font=font)
    draw.rectangle([margin, 0, margin + bar_w - 1, height - 1], outline=(255,255,255,200), width=1)
    return img

def compose_overlay_with_legend(base_img: Image.Image, blocky_heat_rgba: Image.Image, legend: Image.Image, gap: int = 8) -> Image.Image:
    over = Image.alpha_composite(base_img.convert("RGBA"), blocky_heat_rgba)
    W, H = over.size; Lw, Lh = legend.size
    canvas = Image.new("RGBA", (W + gap + Lw, H), (0,0,0,0))
    canvas.paste(over, (0, 0))
    y0 = (H - Lh) // 2 if H > Lh else 0
    canvas.paste(legend, (W + gap, y0), legend)
    return canvas

def sanitize(s: str):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="./imgs/bike.jpg")
    ap.add_argument("--text", default="bicycle wheel")
    # --- CHANGED 2: UPDATE THE DEFAULT WEIGHTS FILE NAME (if needed) ---
    ap.add_argument("--weights", default="miou.pt") # Was "strict500_epoch09.pt"
    # --- END CHANGED 2 ---
    
    ap.add_argument("--max_patches", type=int, default=410)
    ap.add_argument("--alpha", type=int, default=150)
    args = ap.parse_args()

    img_orig = Image.open(args.image).convert("RGB")
    img_fit  = fit_image_to_patch_budget(img_orig, max_patches=args.max_patches)

    siglip, processor, device = build_siglip()
    img_tokens = encode_image_tokens(img_fit, siglip, processor, device)
    txt_emb    = encode_text_emb(args.text, siglip, processor, device)

    gh, gw, D = img_tokens.shape
    
    # --- CHANGED 3: USE ADDITION, NOT CONCATENATION ---
    # This now matches your train.py logic
    # Numpy broadcasting handles (gh, gw, D) + (D)
    x = (img_tokens + txt_emb).astype(np.float32) 
    # --- END CHANGED 3 ---


    # --- UPDATED MODEL LOADING ---
    # We no longer read from the checkpoint, we use the hard-coded constants
    in_dim = EMBED_DIM     # This is 768
    hidden_dim = HIDDEN_DIM # This is 512
    
    # This now correctly instantiates the model
    model = SimpleProjector(in_dim, hidden_dim, drop=DROP_RATE).to(device)
    
    # Load the state_dict *directly* from the .pth file
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    print(f"Loaded SimpleProjector model with in_dim={in_dim}, hidden_dim={hidden_dim}")
    # --- END UPDATED MODEL LOADING ---

    # --- UPDATED INFERENCE ---
    with torch.no_grad():
        # Input has shape (gh, gw, D_in), add batch dim -> (1, gh, gw, D_in)
        x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)
        # Model outputs logits of shape (1, gh, gw)
        logits = model(x_tensor)
        # Convert logits to probabilities (0.0 to 1.0) for visualization
        probs_tensor = torch.sigmoid(logits)
        # Remove batch dim and move to cpu/numpy -> (gh, gw)
        probs = probs_tensor[0].cpu().numpy()
    # --- END UPDATED INFERENCE ---

    blocky = grid_to_blocky_rgba(probs, (img_orig.width, img_orig.height), alpha=args.alpha)
    legend = make_vertical_legend(height=img_orig.height)
    composite = compose_overlay_with_legend(img_orig, blocky, legend, gap=12)

    stem, _ = os.path.splitext(os.path.basename(args.image))
    base_dir = os.path.dirname(args.image)
    out_path = os.path.join(base_dir, f"{stem}__{sanitize(args.text)}_overlay_with_scale.png")
    composite.save(out_path)
    print(f"saved -> {out_path} | grid={probs.shape}")

if __name__ == "__main__":
    main()