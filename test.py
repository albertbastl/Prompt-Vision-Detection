#!/usr/bin/env python3
# visualize_siglip.py
import os
import argparse
import math
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

# --- keep these in sync with training ---
PATCHES = 16
EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1
CKPT = "google/siglip2-base-patch16-naflex"
# ----------------------------------------

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
    # grid_feats_normalized = torch.nn.functional.normalize(grid_feats, p=2, dim=-1)
    return grid_feats.cpu().numpy().astype(np.float32)

@torch.no_grad()
def encode_text_emb(text, siglip, processor, device):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids","attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

# color mapping / legend utilities (kept from your script)
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

# --- projector that matches training code ---
class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim)
        )
        # temperature and bias parameters from training
        self.logits_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.logits_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(self, x):
        # x: (..., in_dim) -> (..., out_dim)
        return F.normalize(self.net(x), p=2, dim=-1)
# ---------------------------------------------

def main():
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="./imgs/car.jpg")
    ap.add_argument("--text", default="car")
    ap.add_argument("--weights", default="epoch2_wandb.pt")
    ap.add_argument("--max_patches", type=int, default=500)
    ap.add_argument("--alpha", type=int, default=150)
    args = ap.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    # Load & resize
    img_orig = Image.open(args.image).convert("RGB")
    img_fit  = fit_image_to_patch_budget(img_orig, max_patches=args.max_patches)

    siglip, processor, device = build_siglip()
    img_tokens = encode_image_tokens(img_fit, siglip, processor, device)
    txt_emb    = encode_text_emb(args.text, siglip, processor, device)

    gh, gw, D = img_tokens.shape
    assert D == EMBED_DIM

    # Flatten
    patches = img_tokens.reshape(-1, D)

    # Load projector
    model = SimpleProjector(
        in_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        out_dim=EMBED_DIM,
        drop=DROP_RATE
    ).to(device)

    sd = torch.load(args.weights, map_location="cpu")
    if not any(k.startswith("net.") for k in sd.keys()):
        for candidate in ("model_state_dict", "state_dict", "state"):
            if candidate in sd and isinstance(sd[candidate], dict):
                sd = sd[candidate]
                break
    model.load_state_dict(sd)
    model.eval()

    dtype = next(model.parameters()).dtype
    patches_t = torch.from_numpy(patches).to(device=device, dtype=dtype)
    txt_t = torch.from_numpy(txt_emb).to(device=device, dtype=dtype)

    with torch.no_grad():
        proj_patches = model(patches_t)
        txt_norm = F.normalize(txt_t, p=2, dim=-1)

        sims   = torch.matmul(proj_patches, txt_norm)
        logits = sims * model.logits_scale.exp() + model.logits_bias
        probs  = torch.sigmoid(logits).cpu().numpy().reshape(gh, gw)

    # Build heatmap overlay
    blocky = grid_to_blocky_rgba(probs, (img_orig.width, img_orig.height), alpha=args.alpha)
    legend = make_vertical_legend(height=img_orig.height)
    composite = compose_overlay_with_legend(img_orig, blocky, legend)

    # ---- SHOW INSTEAD OF SAVE ----
    plt.figure(figsize=(10, 10))
    plt.imshow(composite)
    plt.axis("off")
    plt.title(f"Text: {args.text}")
    plt.show()

if __name__ == "__main__":
    main()
