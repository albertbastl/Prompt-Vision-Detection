# train_tile_decoder_cnn_pad.py
import os, glob, argparse, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Dataset (per-tile training) with padding to global max H,W
# =========================
class NPZGrid(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz found under {root}")

        # Find global max grid size across the split
        max_gh, max_gw, feat_dim = 0, 0, None
        for p in self.paths:
            d = np.load(p)
            gh, gw, D = d["img_tokens"].shape
            max_gh = max(max_gh, gh)
            max_gw = max(max_gw, gw)
            feat_dim = D
        self.max_gh, self.max_gw, self.D = max_gh, max_gw, feat_dim  # save for padding

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        img_tokens = d["img_tokens"].astype(np.float32)   # [gh, gw, D]
        txt_emb    = d["txt_emb"].astype(np.float32)      # [D]
        heatmap    = d["heatmap"].astype(np.float32)      # [gh, gw] in {0,1}

        gh, gw, D = img_tokens.shape
        # build per-tile text (same everywhere)
        txt = np.broadcast_to(txt_emb, (gh, gw, D))
        x = np.concatenate([img_tokens, txt], axis=-1)    # [gh, gw, 2D]
        y = heatmap                                       # [gh, gw]

        # ---- pad to [max_gh, max_gw] ----
        Mgh, Mgw = self.max_gh, self.max_gw
        F = x.shape[-1]
        xp = np.zeros((Mgh, Mgw, F), dtype=np.float32)
        yp = np.zeros((Mgh, Mgw),     dtype=np.float32)
        mp = np.zeros((Mgh, Mgw),     dtype=np.bool_)     # mask of valid tiles

        xp[:gh, :gw] = x
        yp[:gh, :gw] = y
        mp[:gh, :gw] = True

        return torch.from_numpy(xp), torch.from_numpy(yp), torch.from_numpy(mp)

# =========================
# Model: Convolutional decoder (text-conditioned)
# =========================
class ConvTileDecoder(nn.Module):
    """
    Expects x: [B, gh, gw, 2D] where the last dim is [img_tokens(D) || txt_emb(D) broadcast].
    - Splits image features and text features
    - Projects text to D and adds it to image features (conditioning)
    - Runs a small CNN and outputs per-tile probabilities [B, gh, gw]
    """
    def __init__(self, in_dim_2D: int):
        super().__init__()
        self.D = in_dim_2D // 2  # embedding dim

        self.text_projection = nn.Linear(self.D, self.D)

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(self.D, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):  # x: [B, gh, gw, 2D]
        B, gh, gw, F = x.shape
        D = self.D
        img_feat = x[..., :D]                 # [B, gh, gw, D]
        txt_feat = x[..., D:]                 # [B, gh, gw, D] (broadcasted spatially)

        # single text vector per sample (identical across spatial dims)
        txt_vec = txt_feat[:, 0, 0, :]        # [B, D]
        txt_proj = self.text_projection(txt_vec)  # [B, D]

        # to NCHW and add conditioning
        img_feat = img_feat.permute(0, 3, 1, 2).contiguous()       # [B, D, gh, gw]
        txt_map  = txt_proj[:, :, None, None].expand(B, D, gh, gw) # [B, D, gh, gw]
        feat = img_feat + txt_map

        logits = self.conv_blocks(feat)       # [B, 1, gh, gw]
        probs  = torch.sigmoid(logits)        # BCELoss wants probabilities
        return probs.squeeze(1)               # [B, gh, gw]

# =========================
# Utilities (masked)
# =========================
def iou_from_logits_masked(probs, y, mask, thresh=0.5):
    preds = (probs >= thresh).float()
    preds = preds[mask]
    y = y[mask]
    inter = (preds * y).sum()
    union = ((preds + y) > 0).float().sum()
    return (inter / union).item() if union > 0 else 1.0

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def maybe_init_wandb(args, config):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[WARN] wandb not installed; run `pip install wandb` or set --wandb 0.")
        return None
    run = wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)
    return run

# =========================
# Train
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="preprocessed_dataset")  # base dir with train/ and val/
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default="tile_decoder_cnn_pad.pt")
    ap.add_argument("--thresh", type=float, default=0.5, help="Mask threshold for mIoU.")
    # wandb
    ap.add_argument("--wandb", type=int, default=1, help="Set 1 to enable Weights & Biases logging.")
    ap.add_argument("--wandb_project", type=str, default="simple-decoders-testing")
    ap.add_argument("--wandb_run", type=str, default="CNN, dropout 0.1, 3 layers 256 begining")
    args = ap.parse_args()

    seed_all(42)

    # Data
    train_ds = NPZGrid(os.path.join(args.root, "train"))
    val_ds   = NPZGrid(os.path.join(args.root, "val"))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    # Model dims (use dataset's feature dim)
    in_dim_2D = train_ds.D * 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "| Max grid:", (train_ds.max_gh, train_ds.max_gw))

    model = ConvTileDecoder(in_dim_2D).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")  # we'll mask then normalize

    # wandb (optional)
    wb = maybe_init_wandb(
        args,
        config=dict(
            epochs=args.epochs, lr=args.lr, thresh=args.thresh,
            in_dim_2D=in_dim_2D, bs=args.bs, root=args.root,
            max_grid=(train_ds.max_gh, train_ds.max_gw)
        )
    )

    for epoch in range(1, args.epochs + 1):
        # ---------- Train ----------
        model.train()
        total_loss, total_iou, n_train = 0.0, 0.0, 0
        for x, y, m in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            m = m.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            probs = model(x)                  # [B, gh_max, gw_max]
            loss_map = bce(probs, y)          # [B, gh_max, gw_max]
            valid = m.float()
            loss = (loss_map * valid).sum() / (valid.sum().clamp_min(1.0))  # mean over valid tiles
            loss.backward()
            opt.step()

            with torch.no_grad():
                # IoU over valid tiles only
                total_loss += loss.item()
                total_iou += iou_from_logits_masked(probs, y, m, thresh=args.thresh)
                n_train += 1

        train_loss = total_loss / max(n_train, 1)
        train_miou = total_iou / max(n_train, 1)

        # ---------- Validate ----------
        model.eval()
        val_loss, val_iou_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y, m in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)

                probs = model(x)
                loss_map = bce(probs, y)
                valid = m.float()
                loss = (loss_map * valid).sum() / (valid.sum().clamp_min(1.0))

                val_loss += loss.item()
                val_iou_sum += iou_from_logits_masked(probs, y, m, thresh=args.thresh)
                n_val += 1

        val_loss /= max(n_val, 1)
        val_miou = val_iou_sum / max(n_val, 1)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} mIoU {train_miou:.4f} | "
              f"val loss {val_loss:.4f} mIoU {val_miou:.4f}")

        if wb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/mIoU": train_miou,
                "val/loss": val_loss,
                "val/mIoU": val_miou,
                "lr": opt.param_groups[0]["lr"],
            })

    # Save
    torch.save({
        "model": model.state_dict(),
        "in_dim_2D": in_dim_2D,
        "thresh": args.thresh,
        "max_grid": (train_ds.max_gh, train_ds.max_gw)
    }, args.save)
    print(f"Saved: {args.save}")

    if wb:
        import wandb
        wandb.finish()

if __name__ == "__main__":
    main()
