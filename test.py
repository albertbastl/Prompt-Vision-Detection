# test_from_image_blocky_heatmap.py
import os, argparse, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from transformers import AutoModel, AutoProcessor

# --- constants (match training) ---
PATCH = 16
CKPT = "google/siglip2-base-patch16-naflex"

# --- decoder (same as training) ---
class ConvTileDecoder(nn.Module):
    def __init__(self, in_dim_2D: int, C: int = 256, groups: int = 8, drop: float = 0.1):
        super().__init__()
        self.D = in_dim_2D // 2; self.C = C
        self.img_proj = nn.Conv2d(self.D, C, 1, bias=False)
        self.film = nn.Linear(self.D, 2 * C)
        self.dw1 = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw1 = nn.Conv2d(C, C, 1, bias=False)
        self.gn1 = nn.GroupNorm(groups, C)
        self.dw2 = nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False)
        self.pw2 = nn.Conv2d(C, C, 1, bias=False)
        self.gn2 = nn.GroupNorm(groups, C)
        self.act = nn.GELU(); self.drop = nn.Dropout2d(drop)
        self.m1 = nn.Conv2d(C, C // 2, 1, bias=False)
        self.m2 = nn.Conv2d(C, C // 2, 3, padding=2, dilation=2, bias=False)
        self.m3 = nn.Conv2d(C, C // 2, 3, padding=3, dilation=3, bias=False)
        self.ms_fuse = nn.Sequential(
            nn.GroupNorm(groups, (C // 2) * 3),
            nn.GELU(),
            nn.Conv2d((C // 2) * 3, C, 1, bias=False)
        )
        self.head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, C // 2),
            nn.GELU(),
            nn.Dropout2d(drop),
            nn.Conv2d(C // 2, 1, 1),
        )
    def forward(self, x):  # [B, gh, gw, 2D]
        D, C = self.D, self.C
        img = x[..., :D].permute(0,3,1,2).contiguous()
        txt = x[:,0,0,D:]
        feat = self.img_proj(img)
        gamma, beta = self.film(txt).chunk(2, dim=-1)
        feat = feat * (1 + gamma[:,:,None,None]) + beta[:,:,None,None]
        y = self.pw1(self.dw1(feat)); y = self.act(self.gn1(y)); y = self.drop(y); feat = feat + y
        y = self.pw2(self.dw2(feat)); y = self.act(self.gn2(y)); y = self.drop(y); feat = feat + y
        mix = torch.cat([self.m1(feat), self.m2(feat), self.m3(feat)], dim=1)
        feat = self.ms_fuse(mix)
        probs = torch.sigmoid(self.head(feat)).squeeze(1)  # [B, gh, gw]
        return probs

# --- helpers: sizing & encoding ---
def to_multiple(v, m=PATCH, mode="round"):
    if mode == "ceil":  return ((v + m - 1) // m) * m
    if mode == "floor": return (v // m) * m
    return int(round(v / m) * m)

def fit_image_to_patch_budget(img: Image.Image, max_patches: int = 4096) -> Image.Image:
    """Make W,H multiples of PATCH and ensure (W/16)*(H/16) <= max_patches (downscale if needed)."""
    W0, H0 = img.size
    W, H = to_multiple(W0), to_multiple(H0)
    gh, gw = H // PATCH, W // PATCH
    patches = gh * gw
    if patches <= max_patches:
        return img.resize((W, H), Image.Resampling.BILINEAR) if (W, H) != (W0, H0) else img
    scale = math.sqrt(patches / max_patches)  # >1
    newW = max(PATCH, int(W / scale)); newH = max(PATCH, int(H / scale))
    newW = max(PATCH, to_multiple(newW))
    newH = max(PATCH, to_multiple(newH))
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
    gh, gw = pil_img.height // PATCH, pil_img.width // PATCH
    batch = processor(images=pil_img, return_tensors="pt", do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(pixel_values=batch["pixel_values"],
                              attention_mask=batch["pixel_attention_mask"],
                              spatial_shapes=batch["spatial_shapes"])
    feats = out.last_hidden_state[0].float()
    feats = feats[batch["pixel_attention_mask"][0].bool()]
    T, D = feats.shape
    assert T == gh * gw, f"Token count {T} != {gh}*{gw}"
    return feats.view(gh, gw, D).cpu().numpy().astype(np.float32)

@torch.no_grad()
def encode_text_emb(text, siglip, processor, device):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids","attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

# --- helpers: blocky heatmap (cold->warm), legend, and overlay ---
# Color stops: deep blue -> cyan -> yellow -> red
STOPS = [
    (0.00, (0,   0, 130)),
    (0.33, (0, 180, 255)),
    (0.66, (255, 255, 0)),
    (1.00, (255,   0,  0)),
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
    """Make a blocky (nearest) heatmap image from grid probs, with uniform color per tile."""
    gh, gw = probs_grid.shape
    # build small color image (gw x gh) with one pixel per tile
    small = np.zeros((gh, gw, 4), dtype=np.uint8)
    for y in range(gh):
        for x in range(gw):
            r, g, b = map_prob_to_rgb(probs_grid[y, x])
            small[y, x, :] = (r, g, b, alpha)
    # upscale with NEAREST to keep hard tile edges
    img = Image.fromarray(small, mode="RGBA")
    return img.resize(out_wh, resample=Image.NEAREST)

def make_vertical_legend(height: int, width: int = 80, margin: int = 8, ticks=(0.0, 0.25, 0.5, 0.75, 1.0)) -> Image.Image:
    """Create a vertical color legend with tick labels."""
    bar_w = 24
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # draw color bar (top=1.0 -> red, bottom=0.0 -> blue)
    for y in range(height):
        p = 1.0 - (y / max(1, height-1))  # 1 at top, 0 at bottom
        color = map_prob_to_rgb(p)
        draw.line([(margin, y), (margin + bar_w - 1, y)], fill=color, width=1)

    # ticks + labels
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for t in ticks:
        y = int((1.0 - t) * (height-1))
        draw.line([(margin + bar_w, y), (margin + bar_w + 6, y)], fill=(255,255,255,255), width=1)
        label = f"{t:.2f}"
        draw.text((margin + bar_w + 8, max(0, y - 6)), label, fill=(255,255,255,255), font=font)

    # border around bar
    draw.rectangle([margin, 0, margin + bar_w - 1, height - 1], outline=(255,255,255,200), width=1)
    return img

def compose_overlay_with_legend(base_img: Image.Image, blocky_heat_rgba: Image.Image, legend: Image.Image, gap: int = 8) -> Image.Image:
    """Overlay heatmap on base image and stick legend to the right."""
    over = Image.alpha_composite(base_img.convert("RGBA"), blocky_heat_rgba)
    W, H = over.size
    Lw, Lh = legend.size
    canvas = Image.new("RGBA", (W + gap + Lw, H), (0,0,0,0))
    canvas.paste(over, (0, 0))
    # center legend vertically
    y0 = (H - Lh) // 2 if H > Lh else 0
    canvas.paste(legend, (W + gap, y0), legend)
    return canvas

def sanitize(s: str):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="park.jpg", help="Path to input JPG/PNG/WEBP")
    ap.add_argument("--text", default="person", help="Text/category")
    ap.add_argument("--weights", default="op_weights.pt", help="Trained decoder .pt")
    ap.add_argument("--max_patches", type=int, default=4096, help="Cap on token patches (gh*gw).")
    ap.add_argument("--alpha", type=int, default=150, help="Patch opacity 0..255 (255 = no fade)")
    args = ap.parse_args()

    # 1) load original + budget/align to patch grid
    img_orig = Image.open(args.image).convert("RGB")
    img_fit  = fit_image_to_patch_budget(img_orig, max_patches=args.max_patches)

    # 2) build encoders and encode
    siglip, processor, device = build_siglip()
    img_tokens = encode_image_tokens(img_fit, siglip, processor, device)  # [gh, gw, D]
    txt_emb    = encode_text_emb(args.text, siglip, processor, device)    # [D]

    gh, gw, D = img_tokens.shape
    x = np.concatenate([img_tokens, np.broadcast_to(txt_emb, (gh, gw, D))], axis=-1).astype(np.float32)

    # 3) load decoder and run
    ckpt = torch.load(args.weights, map_location="cpu")
    model = ConvTileDecoder(ckpt["in_dim_2D"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    with torch.no_grad():
        probs = model(torch.from_numpy(x).unsqueeze(0).to(device))[0].cpu().numpy()  # [gh, gw] in [0,1]

    # 4) make blocky (per-patch) heatmap, compose overlay + legend
    blocky = grid_to_blocky_rgba(probs, (img_orig.width, img_orig.height), alpha=args.alpha)
    legend = make_vertical_legend(height=img_orig.height)
    composite = compose_overlay_with_legend(img_orig, blocky, legend, gap=12)

    # 5) save next to input
    stem, _ = os.path.splitext(os.path.basename(args.image))
    base_dir = os.path.dirname(args.image)
    out_path = os.path.join(base_dir, f"{stem}__{sanitize(args.text)}_overlay_with_scale.png")
    composite.save(out_path)
    print(f"saved -> {out_path} | grid={probs.shape}")

if __name__ == "__main__":
    main()
