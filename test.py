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
# Constants & simple helpers
# ───────────────────────────────────────────────────────────────────
PATCHES = 16  # SigLIP2 patch size

def stretch_to_multiple(img):
    """Resize PIL image so both sides are multiple of PATCHES (16)."""
    W, H = img.size
    newW = ((W + PATCHES - 1) // PATCHES) * PATCHES
    newH = ((H + PATCHES - 1) // PATCHES) * PATCHES
    if (newW, newH) == (W, H):
        return img
    return img.resize((newW, newH), Image.Resampling.BILINEAR)

def pad_to_multiple_hw(H, W, m=4):
    """Return (Hp, Wp) >= (H, W) and divisible by m (we downsample twice)."""
    Hp = ((H + m - 1) // m) * m
    Wp = ((W + m - 1) // m) * m
    return Hp, Wp

# ───────────────────────────────────────────────────────────────────
# Models (must match training scripts)
# ───────────────────────────────────────────────────────────────────
class TokenUNetTiny(nn.Module):
    """Your original tiny token CNN (no downs/ups, residual conv blocks)."""
    def __init__(self, in_ch, mid=128):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, mid, 1)
        self.block1 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2),
            nn.GroupNorm(8, mid), nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.GroupNorm(8, mid), nn.GELU(),
        )
        self.out = nn.Conv2d(mid, 1, 1)
    def forward(self, x):  # [B,D,H,W]
        x = self.reduce(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x)
        return self.out(x).squeeze(1)  # [B,H,W]

def conv_gn_gelu(c_in, c_out, k=3, s=1, p=1, groups=8, bias=False):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, s, p, bias=bias),
        nn.GroupNorm(groups, c_out),
        nn.GELU(),
    )

class TinyUNet2L(nn.Module):
    """2-level UNet used in the BCE+Dice training script."""
    def __init__(self, in_ch, base=128):
        super().__init__()
        b = base
        self.stem = conv_gn_gelu(in_ch, b, k=1, p=0)
        self.d1 = nn.Sequential(conv_gn_gelu(b, b), conv_gn_gelu(b, b))
        self.down1 = nn.Conv2d(b, b*2, 3, 2, 1)   # /2
        self.d2 = nn.Sequential(conv_gn_gelu(b*2, b*2), conv_gn_gelu(b*2, b*2))
        self.down2 = nn.Conv2d(b*2, b*4, 3, 2, 1) # /4
        self.bot = nn.Sequential(
            nn.Conv2d(b*4, b*4, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(16, b*4), nn.GELU(),
            nn.Conv2d(b*4, b*4, 3, padding=1, bias=False),
            nn.GroupNorm(16, b*4), nn.GELU(),
        )
        self.up1 = nn.ConvTranspose2d(b*4, b*2, 2, 2)
        self.u1  = nn.Sequential(conv_gn_gelu(b*4, b*2), conv_gn_gelu(b*2, b*2))
        self.up2 = nn.ConvTranspose2d(b*2, b, 2, 2)
        self.u2  = nn.Sequential(conv_gn_gelu(b*2, b), conv_gn_gelu(b, b))
        self.out = nn.Conv2d(b, 1, 1)
    def forward(self, x):  # [B,D,H,W]
        x = self.stem(x)
        s1 = self.d1(x)
        x  = self.down1(s1)
        s2 = self.d2(x)
        x  = self.down2(s2)
        x  = self.bot(x)
        x  = self.up1(x)
        if x.shape[-2:] != s2.shape[-2:]:
            x = F.interpolate(x, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        x  = torch.cat([x, s2], dim=1)
        x  = self.u1(x)
        x  = self.up2(x)
        if x.shape[-2:] != s1.shape[-2:]:
            x = F.interpolate(x, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        x  = torch.cat([x, s1], dim=1)
        x  = self.u2(x)
        return self.out(x).squeeze(1)  # [B,H,W]

class SimpleSegCNN(nn.Module):
    """Plain encoder→bottleneck→decoder CNN (no skips), used in simple trainer."""
    def __init__(self, in_ch, base=128):
        super().__init__()
        b = base
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, b, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),     nn.ReLU(inplace=True),
        )
        self.down1 = nn.Conv2d(b, b*2, 3, stride=2, padding=1)  # /2
        self.enc2  = nn.Sequential(nn.ReLU(inplace=True),
                                   nn.Conv2d(b*2, b*2, 3, padding=1),
                                   nn.ReLU(inplace=True))
        self.down2 = nn.Conv2d(b*2, b*4, 3, stride=2, padding=1) # /4
        self.bot   = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(b*4, b*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(b*4, b*4, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(nn.Conv2d(b*4, b*2, 3, padding=1), nn.ReLU(inplace=True))
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(b*2, b, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1),   nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(b, 1, 1)
        nn.init.constant_(self.out.bias, -2.0)  # slight background prior for sparse masks
    def forward(self, x):  # [B,D,H,W]
        x = self.enc1(x); x = self.down1(x)
        x = self.enc2(x); x = self.down2(x)
        x = self.bot(x)
        x = self.up1(x);  x = self.dec1(x)
        x = self.up2(x);  x = self.dec2(x)
        return self.out(x).squeeze(1)

# ───────────────────────────────────────────────────────────────────
# SigLIP2 encoders (image tokens + text emb)
# ───────────────────────────────────────────────────────────────────
def build_siglip2(ckpt, device):
    # Use the proper argument name: torch_dtype
    dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype).to(device).eval()
    proc  = AutoProcessor.from_pretrained(ckpt)
    return model, proc

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
    feats = out.last_hidden_state[0].float()        # [T, D]
    mask  = batch["pixel_attention_mask"][0].bool() # [T]
    feats = feats[mask]                              # [gh*gw, D]
    assert feats.shape[0] == gh * gw, f"T={feats.shape[0]} != {gh*gw}"
    feats = feats.view(gh, gw, -1)                   # [gh,gw,D]
    return feats

@torch.no_grad()
def encode_text_vec(text, siglip, processor, device):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items()
            if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output  # [1, D]
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb[0]  # [D]

# ───────────────────────────────────────────────────────────────────
# Visualization
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
# Arch auto-detection from checkpoint keys
# ───────────────────────────────────────────────────────────────────
def detect_arch_from_keys(keys):
    ks = list(keys)
    if any(k.startswith("reduce.") or k.startswith("block1.") for k in ks):
        return "tiny"
    if any(k.startswith("stem.") or k.startswith("u1.") or k.startswith("up1.") for k in ks):
        return "unet2l"
    if any(k.startswith("enc1.") or k.startswith("dec1.") for k in ks):
        return "simple"
    return None

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
                    help="Combine CNN mask with text similarity")
    ap.add_argument("--arch", type=str, choices=["simple","unet2l","tiny"], default=None,
                    help="(Optional) Force architecture; otherwise auto-detect from checkpoint")
    ap.add_argument("--base", type=int, default=128,
                    help="Base channels if you trained with a different width (e.g., 192)")
    args = ap.parse_args()

    device = args.device

    # 1) Build encoders
    siglip, processor = build_siglip2(args.siglip_ckpt, device)

    # 2) Encode image grid and prompt (device tensors)
    pil = Image.open(args.image).convert("RGB")
    img_grid = encode_image_grid(pil, siglip, processor, device)   # [gh,gw,D]
    gh, gw, D = img_grid.shape
    txt = encode_text_vec(args.prompt, siglip, processor, device)  # [D]

    # 3) Prepare tokens for CNN
    toks = img_grid.permute(2,0,1).contiguous().to(torch.float32)  # [D,gh,gw]
    Hp, Wp = pad_to_multiple_hw(gh, gw, m=4)  # safe for /2 twice nets
    if (Hp, Wp) != (gh, gw):
        pad = (0, Wp - gw, 0, Hp - gh)  # (left,right,top,bottom) in (W,H)
        toks = F.pad(toks, pad)
    toks_b = toks.unsqueeze(0).to(device)  # [1,D,Hp,Wp]

    # 4) Load model + weights
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"]
    in_ch = ckpt.get("in_ch", D)
    if in_ch != D:
        raise ValueError(f"Channel mismatch: ckpt in_ch={in_ch}, but SigLIP2 produced D={D}")

    auto_arch = detect_arch_from_keys(state.keys())
    arch = args.arch if args.arch is not None else (auto_arch or "simple")
    print(f"[info] checkpoint looks like '{auto_arch}', using arch='{arch}'")

    if arch == "simple":
        model = SimpleSegCNN(in_ch=in_ch, base=args.base).to(device)
    elif arch == "unet2l":
        model = TinyUNet2L(in_ch=in_ch, base=args.base).to(device)
    else:  # "tiny"
        model = TokenUNetTiny(in_ch=in_ch, mid=args.base).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    # 5) CNN forward → probs, crop back if padded
    logits = model(toks_b)            # [1,Hp,Wp]
    probs  = torch.sigmoid(logits)[0] # [Hp,Wp]
    probs  = probs[:gh, :gw]          # back to original grid

    # 6) Text similarity map (on device)
    flat = toks_b[0][:, :gh, :gw].view(D, -1).t()   # [H*W, D]
    flat = F.normalize(flat, p=2, dim=-1)
    txtn = F.normalize(txt.to(torch.float32), p=2, dim=-1)
    sim  = torch.clamp(flat @ txtn, -1.0, 1.0)  # cosine in [-1,1]
    sim  = (sim + 1.0) * 0.5                    # → [0,1]
    sim_map = sim.view(gh, gw)

    # 7) Fuse
    if args.fuse == "cnn_only":
        combined = probs
    elif args.fuse == "text_only":
        combined = sim_map
    elif args.fuse == "min":
        combined = torch.minimum(probs, sim_map)
    elif args.fuse == "max":
        combined = torch.maximum(probs, sim_map)
    else:  # "mul"
        combined = probs * sim_map

    # Normalize 0..1 for visualization
    m, M = combined.min(), combined.max()
    heat01 = (combined - m) / (M - m + 1e-6)

    # 8) Save overlay
    overlay_and_save(args.image, heat01, args.out, title=f"{args.prompt} [{arch}]")
    print(f"Saved overlay to: {args.out}")

if __name__ == "__main__":
    main()
