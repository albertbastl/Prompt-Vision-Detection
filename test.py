# test_with_prompt.py
import argparse, os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor

# ───────────────────────────────────────────────────────────────────
# Model (same as training)
# ───────────────────────────────────────────────────────────────────
class TokenUNetTiny(nn.Module):
    def __init__(self, in_ch, mid=128):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, mid, 1)
        self.block1 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1),
                                    nn.GroupNorm(8, mid), nn.GELU())
        self.block2 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=2, dilation=2),
                                    nn.GroupNorm(8, mid), nn.GELU())
        self.block3 = nn.Sequential(nn.Conv2d(mid, mid, 3, padding=1),
                                    nn.GroupNorm(8, mid), nn.GELU())
        self.out = nn.Conv2d(mid, 1, 1)
    def forward(self, x):  # [B,D,H,W]
        x = self.reduce(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x)
        return self.out(x).squeeze(1)  # [B,H,W]

# ───────────────────────────────────────────────────────────────────
# SigLIP2 encoding (image patches + text emb)
# ───────────────────────────────────────────────────────────────────
PATCHES = 16

def stretch_to_multiple(img):
    W, H = img.size
    newW = ((W + PATCHES - 1) // PATCHES) * PATCHES
    newH = ((H + PATCHES - 1) // PATCHES) * PATCHES
    if (newW, newH) == (W, H):
        return img
    return img.resize((newW, newH), Image.Resampling.BILINEAR)

def build_siglip2(ckpt, device):
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(ckpt, dtype=dtype).to(device).eval()
    proc  = AutoProcessor.from_pretrained(ckpt)
    return model, proc, dtype

@torch.no_grad()
def encode_image_grid(pil_img, siglip, processor, device):
    pil_img = stretch_to_multiple(pil_img)
    gh, gw = pil_img.height // PATCHES, pil_img.width // PATCHES
    batch = processor(images=pil_img, return_tensors="pt",
                      do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(pixel_values=batch["pixel_values"],
                              attention_mask=batch["pixel_attention_mask"],
                              spatial_shapes=batch["spatial_shapes"])
    feats = out.last_hidden_state[0].float()            # [T, D] (on device)
    mask  = batch["pixel_attention_mask"][0].bool()     # [T]
    feats = feats[mask]                                  # [gh*gw, D]
    assert feats.shape[0] == gh * gw, f"T={feats.shape[0]} != {gh*gw}"
    feats = feats.view(gh, gw, -1)                       # [gh,gw,D]
    return feats  # float32 (on device)

@torch.no_grad()
def encode_text_vec(text, siglip, processor, device):
    toks = processor(text=[text], return_tensors="pt",
                     padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items()
            if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output  # [1, D]
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb[0]  # [D], device tensor

# ───────────────────────────────────────────────────────────────────
# Viz
# ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def overlay_and_save(image_path, heat01, out_path, title=None):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    up = F.interpolate(heat01[None,None,...], size=(H, W),
                       mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy()
    plt.figure(figsize=(8,6), dpi=150)
    plt.imshow(img)
    plt.imshow(up, cmap="jet", alpha=0.45, vmin=0.0, vmax=1.0)
    plt.axis("off")
    if title: plt.title(title)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()

# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="best_tokens_cnn.pt")
    ap.add_argument("--image", type=str, default="person.jpg")
    ap.add_argument("--prompt", type=str, default="person")
    ap.add_argument("--siglip_ckpt", type=str, default="google/siglip2-base-patch16-naflex")
    ap.add_argument("--out", type=str, default="overlay.png")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fuse",  type=str, choices=["mul","min","max","cnn_only","text_only"], default="mul",
                    help="How to combine CNN probs with text similarity")
    args = ap.parse_args()

    device = args.device

    # 1) Build encoders
    siglip, processor, _ = build_siglip2(args.siglip_ckpt, device)

    # 2) Encode image grid and prompt (both on the SAME device)
    pil = Image.open(args.image).convert("RGB")
    img_grid = encode_image_grid(pil, siglip, processor, device)   # [gh,gw,D] (device)
    gh, gw, D = img_grid.shape
    txt = encode_text_vec(args.prompt, siglip, processor, device)  # [D] (device)

    # 3) Prepare tokens for CNN (match training: only image tokens)
    toks = img_grid.permute(2,0,1).contiguous().to(torch.float32)  # [D,gh,gw] (device)
    toks_b = toks.unsqueeze(0)                                      # [1,D,H,W] (device)

    # 4) Load model + weights
    ckpt = torch.load(args.ckpt, map_location=device)
    in_ch = ckpt.get("in_ch", D)
    if in_ch != D:
        raise ValueError(f"Channel mismatch: ckpt expects in_ch={in_ch}, but SigLIP2 produced D={D}")
    model = TokenUNetTiny(in_ch=in_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 5) CNN forward → probs
    logits = model(toks_b)                 # [1,H,W]
    probs  = torch.sigmoid(logits)[0]      # [H,W], device

    # 6) Text gating (all on the SAME device)
    toks_dev = toks_b[0]                                      # [D,H,W] (device)
    img_flat = toks_dev.view(D, -1).t()                       # [H*W, D] (device)
    img_flat = F.normalize(img_flat, p=2, dim=-1)
    txt_n = F.normalize(txt.to(torch.float32), p=2, dim=-1)   # [D] (device)
    sim = torch.clamp(img_flat @ txt_n, -1.0, 1.0)            # [H*W] (device)
    sim = (sim + 1.0) * 0.5                                   # [-1,1] → [0,1]
    sim_map = sim.view(toks.shape[1], toks.shape[2])          # [H,W] (device)

    # 7) Fuse
    if args.fuse == "cnn_only":
        combined = probs
    elif args.fuse == "text_only":
        combined = sim_map
    elif args.fuse == "min":
        combined = torch.minimum(probs, sim_map)
    elif args.fuse == "max":
        combined = torch.maximum(probs, sim_map)
    else:  # "mul" (default)
        combined = probs * sim_map

    # Normalize for visualization
    m, M = combined.min(), combined.max()
    heat01 = (combined - m) / (M - m + 1e-6)

    # 8) Overlay (function moves to CPU for saving)
    overlay_and_save(args.image, heat01, args.out, title=args.prompt)
    print(f"Saved overlay to: {args.out}")

if __name__ == "__main__":
    main()
