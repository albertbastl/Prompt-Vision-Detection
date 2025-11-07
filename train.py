import os, glob, argparse, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = lambda x, **k: x

class NPZGrid(Dataset):
    def __init__(self, root):
        self.paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.paths: raise FileNotFoundError(f"No .npz found under {root}")
        self.sample_map = []
        max_gh, max_gw = 0, 0
        for path in tqdm(self.paths, desc="Scanning dataset"):
            try:
                with np.load(path) as d:
                    if "masks" not in d or "txt_embs" not in d or "img_tokens" not in d: continue
                    num_objects = len(d["masks"])
                    if num_objects == 0: continue
                    for obj_idx in range(num_objects):
                        self.sample_map.append((path, obj_idx))
                    gh, gw = d["masks"].shape[1:3]
                    max_gh, max_gw = max(max_gh, gh), max(max_gw, gw)
                    if not hasattr(self, 'D'):
                        self.D = d["img_tokens"].shape[2]
            except Exception:
                continue
        if not self.sample_map: raise ValueError(f"No valid samples found in {root}.")
        self.max_gh, self.max_gw = max_gh, max_gw
        print(f"Found {len(self.sample_map)} samples. Max grid: ({self.max_gh}, {self.max_gw})")

    def __len__(self): return len(self.sample_map)

    def __getitem__(self, i):
        path, obj_idx = self.sample_map[i]
        with np.load(path) as d:
            img_tokens = d["img_tokens"].astype(np.float32)
            txt_emb = d["txt_embs"][obj_idx].astype(np.float32)
            heatmap = d["masks"][obj_idx].astype(np.float32)
        gh, gw, D = img_tokens.shape
        txt = np.broadcast_to(txt_emb, (gh, gw, D))
        x = np.concatenate([img_tokens, txt], axis=-1)
        Mgh, Mgw, F = self.max_gh, self.max_gw, x.shape[-1]
        xp = np.zeros((Mgh, Mgw, F), dtype=np.float32)
        yp = np.zeros((Mgh, Mgw), dtype=np.float32)
        mp = np.zeros((Mgh, Mgw), dtype=np.bool_)
        xp[:gh, :gw] = x
        yp[:gh, :gw] = heatmap
        mp[:gh, :gw] = True
        return torch.from_numpy(xp), torch.from_numpy(yp), torch.from_numpy(mp)

class ConvTileDecoder(nn.Module):
    def __init__(self, in_dim_2D: int, C: int = 256, groups: int = 8, drop: float = 0.1):
        super().__init__()
        self.D = in_dim_2D // 2
        self.C = C
        self.img_proj = nn.Conv2d(self.D, C, 1, bias=False)
        self.film = nn.Linear(self.D, 2 * C)
        self.dw1 = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.pw1 = nn.Conv2d(C, C, 1, bias=False)
        self.gn1 = nn.GroupNorm(groups, C)
        self.dw2 = nn.Conv2d(C, C, 3, padding=2, dilation=2, groups=C, bias=False)
        self.pw2 = nn.Conv2d(C, C, 1, bias=False)
        self.gn2 = nn.GroupNorm(groups, C)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(drop)
        self.m1 = nn.Conv2d(C, C // 2, 1, bias=False)
        self.m2 = nn.Conv2d(C, C // 2, 3, padding=2, dilation=2, bias=False)
        self.m3 = nn.Conv2d(C, C // 2, 3, padding=3, dilation=3, bias=False)
        self.ms_fuse = nn.Sequential(
            nn.GroupNorm(groups, (C // 2) * 3), nn.GELU(), nn.Conv2d((C // 2) * 3, C, 1, bias=False)
        )
        self.head = nn.Sequential(
            nn.Conv2d(C, C // 2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, C // 2), nn.GELU(),
            nn.Dropout2d(drop), nn.Conv2d(C // 2, 1, 1),
        )

    def forward(self, x):
        D, C = self.D, self.C
        img = x[..., :D].permute(0, 3, 1, 2).contiguous()
        txt = x[:, 0, 0, D:]
        feat = self.img_proj(img)
        gamma, beta = self.film(txt).chunk(2, dim=-1)
        feat = feat * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        y = self.act(self.gn1(self.pw1(self.dw1(feat))))
        feat = feat + self.drop(y)
        y = self.act(self.gn2(self.pw2(self.dw2(feat))))
        feat = feat + self.drop(y)
        mix = torch.cat([self.m1(feat), self.m2(feat), self.m3(feat)], dim=1)
        feat = self.ms_fuse(mix)
        return torch.sigmoid(self.head(feat)).squeeze(1)

def iou_from_logits_masked(probs, y, mask, thresh=0.5):
    preds = (probs >= thresh).float()[mask]
    y = y[mask]
    inter = (preds * y).sum()
    union = ((preds + y) > 0).float().sum()
    return (inter / union).item() if union > 0 else 1.0

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def maybe_init_wandb(args, config):
    if not args.wandb: return None
    try:
        import wandb
        return wandb.init(project=args.wandb_project, name=args.wandb_run, config=config)
    except ImportError:
        print("[WARN] wandb not installed.")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="pd_410patches_openvocab")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default="weights.pt")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--load_weights", type=str, default=None)
    ap.add_argument("--wandb", type=int, default=1)
    ap.add_argument("--wandb_project", type=str, default="openvocab")
    ap.add_argument("--wandb_run", type=str, default="30k images, 64bs, 1e-3lr")
    ap.add_argument("--log_every_images", type=int, default=1000, help="Log metrics to wandb every N images")
    args = ap.parse_args()

    seed_all(42)
    train_ds = NPZGrid(os.path.join(args.root, "train"))
    val_ds = NPZGrid(os.path.join(args.root, "val"))
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    in_dim_2D = train_ds.D * 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Max grid: ({train_ds.max_gh}, {train_ds.max_gw})")
    model = ConvTileDecoder(in_dim_2D).to(device)

    if args.load_weights:
        try:
            ckpt = torch.load(args.load_weights, map_location=device)
            model.load_state_dict(ckpt.get("model", ckpt))
            print(f"[Info] Loaded weights from: {args.load_weights}")
        except Exception as e:
            print(f"[Error] Failed to load {args.load_weights}: {e}. Training from scratch.")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")
    wb = maybe_init_wandb(args, config={**vars(args), "in_dim_2D": in_dim_2D, "max_grid": (train_ds.max_gh, train_ds.max_gw)})

    # --- WANDB STEP LOGGING: Initializations ---
    global_step = 0
    images_since_last_log = 0
    log_loss = 0.0
    log_iou = 0.0
    log_n = 0
    # --- End Initializations ---

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch:02d}/{args.epochs} ===")
        epoch_t0 = time.time()
        model.train()
        total_loss, total_iou, n_train = 0.0, 0.0, 0 # Epoch totals
        train_iter = tqdm(train_loader, desc=f"train {epoch:02d}")
        
        for x, y, m in train_iter:
            x, y, m = x.to(device), y.to(device), m.to(device)
            batch_size = x.size(0)
            
            opt.zero_grad(set_to_none=True)
            probs = model(x)
            loss_map = bce(probs, y)
            valid = m.float()
            loss = (loss_map * valid).sum() / (valid.sum().clamp_min(1.0))
            
            loss.backward(); opt.step()
            
            global_step += batch_size
            images_since_last_log += batch_size

            with torch.no_grad():
                current_loss = loss.item()
                current_iou = iou_from_logits_masked(probs, y, m, thresh=args.thresh)
                
                # Accumulate for epoch averages
                total_loss += current_loss
                total_iou += current_iou
                n_train += 1
                
                # Accumulate for step logging
                log_loss += current_loss
                log_iou += current_iou
                log_n += 1

                if n_train > 0: 
                    train_iter.set_postfix(loss=total_loss/n_train, miou=total_iou/n_train)

                # --- WANDB STEP LOGGING: Check and Log ---
                if wb and images_since_last_log >= args.log_every_images:
                    import wandb  # <-- FIX: Import wandb module here
                    avg_step_loss = log_loss / log_n
                    avg_step_iou = log_iou / log_n
                    wandb.log({
                        "train/step_loss": avg_step_loss,
                        "train/step_mIoU": avg_step_iou,
                        "global_step": global_step
                    })
                    # Reset step accumulators
                    images_since_last_log = 0
                    log_loss = 0.0
                    log_iou = 0.0
                    log_n = 0
                # --- End Step Logging ---

        train_loss, train_miou = total_loss / max(n_train, 1), total_iou / max(n_train, 1)

        model.eval()
        val_loss, val_iou, n_val = 0.0, 0.0, 0
        val_iter = tqdm(val_loader, desc=f"valid {epoch:02d}")
        with torch.no_grad():
            for x, y, m in val_iter:
                x, y, m = x.to(device), y.to(device), m.to(device)
                probs = model(x)
                loss_map = bce(probs, y)
                valid = m.float()
                loss = (loss_map * valid).sum() / (valid.sum().clamp_min(1.0))
                val_loss += loss.item()
                val_iou += iou_from_logits_masked(probs, y, m, thresh=args.thresh)
                n_val += 1
                if n_val > 0: val_iter.set_postfix(loss=val_loss/n_val, miou=val_iou/n_val)
        val_loss, val_miou = val_loss / max(n_val, 1), val_iou / max(n_val, 1)
        epoch_dt = time.time() - epoch_t0

        print(f"E{epoch:02d} | train L {train_loss:.4f} mIoU {train_miou:.4f} | val L {val_loss:.4f} mIoU {val_miou:.4f} | {epoch_dt/60:.2f}m")

        if wb:
            import wandb
            # Log epoch averages
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss, # Renamed from train/loss
                "train/epoch_mIoU": train_miou, # Renamed from train/mIoU
                "val/loss": val_loss,
                "val/mIoU": val_miou,
                "time/epoch_sec": epoch_dt,
                "global_step": global_step # Also log global_step at epoch end
            })
        
        ckpt_data = {"model": model.state_dict(), "in_dim_2D": in_dim_2D,
                     "thresh": args.thresh, "max_grid": (train_ds.max_gh, train_ds.max_gw)}
        epoch_path = f"{os.path.splitext(args.save)[0]}_epoch{epoch:02d}.pt"
        torch.save({**ckpt_data, "epoch": epoch}, epoch_path)
        print(f"[ckpt] saved {epoch_path}")

    torch.save(ckpt_data, args.save)
    print(f"Saved: {args.save}")
    if wb: import wandb; wb.finish()

if __name__ == "__main__":
    main()