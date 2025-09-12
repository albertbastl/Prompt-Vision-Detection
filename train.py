# train_cnn_on_tokens.py
import os, glob, argparse, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────────
class NPZTokens(Dataset):
    def __init__(self, root="preprocessed_dataset"):
        self.paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz found under {root}")
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        d = np.load(self.paths[i])
        heat = torch.from_numpy(d["heatmap"].astype(np.float32))   # [H,W] {0,1}
        toks = torch.from_numpy(d["img_tokens"]).to(torch.float32) # [H,W,D]
        toks = toks.permute(2,0,1).contiguous()                    # [D,H,W]
        return toks, heat

def pad_batch(batch):
    # batch: list of (tokens[D,H,W], heat[H,W])
    D = batch[0][0].shape[0]
    Hs = [b[0].shape[1] for b in batch]; Ws = [b[0].shape[2] for b in batch]
    Hm, Wm = max(Hs), max(Ws)
    toks_pad, heat_pad, mask_pad = [], [], []
    for tok, heat in batch:
        pad = (0, Wm - tok.shape[2], 0, Hm - tok.shape[1])  # (W_left,W_right,H_top,H_bot)
        toks_pad.append(F.pad(tok, pad))                    # [D,Hm,Wm]
        heat_pad.append(F.pad(heat, pad))                   # [Hm,Wm]
        mask = torch.ones_like(heat)                        # valid pixels
        mask_pad.append(F.pad(mask, pad))
    toks = torch.stack(toks_pad)                            # [B,D,Hm,Wm]
    heat = torch.stack(heat_pad)                            # [B,Hm,Wm]
    mask = torch.stack(mask_pad)                            # [B,Hm,Wm]
    return toks, heat, mask

# ───────────────────────────────────────────────────────────────────
# Model: tiny conv net over tokens
# ───────────────────────────────────────────────────────────────────
class TokenUNetTiny(nn.Module):
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
        self.out = nn.Conv2d(mid, 1, 1)  # logits
    def forward(self, x):                # x: [B,D,H,W]
        x = self.reduce(x)
        x = self.block1(x) + x
        x = self.block2(x) + x
        x = self.block3(x)
        return self.out(x).squeeze(1)    # [B,H,W] logits

# ───────────────────────────────────────────────────────────────────
# Utils
# ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def batch_iou(logits, target, mask, thr=0.5):
    pred = (torch.sigmoid(logits) >= thr).float()
    target = target.float()
    mask = mask.float()
    inter = ((pred * target) * mask).sum(dim=(1,2))
    union = (((pred + target) > 0).float() * mask).sum(dim=(1,2))
    iou = torch.where(union>0, inter/union, torch.ones_like(union))
    return iou.mean().item()

def make_pos_weight(y, mask):
    # returns scalar pos_weight = neg/pos for BCEWithLogitsLoss
    with torch.no_grad():
        y = y.float(); mask = mask.float()
        pos = (y*mask).sum()
        neg = (mask.sum() - pos).clamp(min=1.0)
        pos = pos.clamp(min=1.0)
        return (neg/pos).detach()

# ───────────────────────────────────────────────────────────────────
# Train
# ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="preprocessed_dataset")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--save", type=str, default="best_tokens_cnn.pt")
    args = ap.parse_args()

    ds = NPZTokens(args.root)
    # Peek one sample to get D
    sample_tokens, sample_heat = ds[0]
    D = sample_tokens.shape[0]

    loader = DataLoader(ds, batch_size=args.bs, shuffle=True,
                        collate_fn=pad_batch, num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device, "| CUDA available:", torch.cuda.is_available())
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    model = TokenUNetTiny(in_ch=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed and (device=="cuda"))

    best_iou = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss, running_iou, steps = 0.0, 0.0, 0
        for toks, heat, mask in loader:
            toks = toks.to(device, non_blocking=True)   # [B,D,H,W]
            heat = heat.to(device, non_blocking=True)   # [B,H,W]
            mask = mask.to(device, non_blocking=True)   # [B,H,W]

            pos_weight = make_pos_weight(heat, mask)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.mixed and (device=="cuda")):
                logits = model(toks)                    # [B,H,W]
                # apply mask to loss by flattening and weighting
                loss = loss_fn(logits[mask.bool()], heat[mask.bool()])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()
            running_iou  += batch_iou(logits.detach(), heat, mask)
            steps += 1

        avg_loss = running_loss/max(steps,1)
        avg_iou  = running_iou/max(steps,1)
        print(f"Epoch {epoch:02d} | loss {avg_loss:.4f} | IoU {avg_iou:.4f}")

        # save best
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save({"model": model.state_dict(),
                        "in_ch": D}, args.save)
            print(f"  ↳ saved: {args.save} (best IoU {best_iou:.4f})")

if __name__ == "__main__":
    main()
