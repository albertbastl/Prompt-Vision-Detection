import os, glob, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

TILE = 16  # must match your preprocessing stride

# --- Dataset -----------------------------------------------------------------
class NpzTokenDataset(Dataset):
    def __init__(self, folder):
        self.files = sorted([p for p in glob.glob(os.path.join(folder, "*.npz"))])
        if not self.files:
            raise RuntimeError(f"No .npz files in {folder}")
    def __len__(self): return len(self.files)

    def _downsample_to_tokens(self, heat_pix):
        # heat_pix: [H, W] (0/1). Downsample by TILE using max-pool to get [Htok, Wtok].
        H, W = heat_pix.shape
        assert H % TILE == 0 and W % TILE == 0, "Heatmap size must be divisible by TILE"
        Ht, Wt = H // TILE, W // TILE
        t = torch.from_numpy(heat_pix).float().view(Ht, TILE, Wt, TILE)
        # max over each TILE×TILE block -> 1 if any positive pixel in the tile
        t = t.amax(dim=(1, 3))  # [Ht, Wt]
        return t

    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        if "img_tokens" not in d:
            raise ValueError(f"{self.files[idx]} has no 'img_tokens'. "
                             f"Re-run preprocessing to save per-patch tokens.")
        img_tokens = torch.from_numpy(d["img_tokens"]).float()   # [D,Ht,Wt]
        text_vec   = torch.from_numpy(d["text_vec"]).float()     # [D]
        heat_pix   = d["heatmap"].astype(np.uint8)               # [H,W] pixels
        target     = self._downsample_to_tokens(heat_pix).unsqueeze(0)  # [1,Ht,Wt]
        return {
            "img_tokens": img_tokens,  # [D,Ht,Wt]
            "text_vec":   text_vec,    # [D]
            "target":     target       # [1,Ht,Wt]
        }

def pad_collate(batch):
    # Pad variable [Ht,Wt] to the max in this batch; create valid mask.
    D = batch[0]["img_tokens"].shape[0]
    Hts = [b["img_tokens"].shape[1] for b in batch]
    Wts = [b["img_tokens"].shape[2] for b in batch]
    Ht, Wt = max(Hts), max(Wts)

    imgs, txts, tgts, masks = [], [], [], []
    for b in batch:
        it = b["img_tokens"]; tgt = b["target"]
        ht, wt = it.shape[1], it.shape[2]
        pad_h = Ht - ht; pad_w = Wt - wt
        imgs.append(F.pad(it, (0,pad_w, 0,pad_h)))            # [D,Ht,Wt]
        tgts.append(F.pad(tgt, (0,pad_w, 0,pad_h)))           # [1,Ht,Wt]
        m = torch.zeros(1, Ht, Wt, dtype=torch.float32)
        m[:, :ht, :wt] = 1.0
        masks.append(m)
        txts.append(b["text_vec"])
    return {
        "img_tokens": torch.stack(imgs, 0),   # [B,D,Ht,Wt]
        "text_vec":   torch.stack(txts, 0),   # [B,D]
        "target":     torch.stack(tgts, 0),   # [B,1,Ht,Wt]
        "valid":      torch.stack(masks, 0),  # [B,1,Ht,Wt]
    }

# --- Model: FiLM-conditioned tiny CNN head -----------------------------------
class FiLMConvLocator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Map text → (gamma, beta) to modulate image tokens
        self.film = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model*2)
        )
        # Lightweight conv head
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model//2, 1, 1)
        )

    def forward(self, img_tokens: torch.Tensor, text_vec: torch.Tensor):
        # img_tokens: [B,D,H,W], text_vec: [B,D]
        B, D, H, W = img_tokens.shape
        film_params = self.film(text_vec)               # [B, 2D]
        gamma, beta = film_params.chunk(2, dim=1)       # [B,D], [B,D]
        gamma = gamma.view(B, D, 1, 1)
        beta  = beta.view(B, D, 1, 1)

        feat = img_tokens * (1 + gamma) + beta         # FiLM
        logits = self.head(feat)                       # [B,1,H,W]
        return logits

# --- Loss --------------------------------------------------------------------
def heatmap_loss(logits, target, valid_mask, pos_weight=2.0):
    # BCE with masking to ignore padded tiles
    pw = torch.tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pw)
    loss = (bce * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    return loss

# --- Train (minimal) ---------------------------------------------------------
def train_once(
    data_dir,
    d_model=768,
    epochs=2,
    batch_size=8,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    ds = NpzTokenDataset(data_dir)
    # Infer d_model from first sample if needed
    if d_model is None:
        s = ds[0]["img_tokens"]; d_model = s.shape[0]

    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=pad_collate)
    model = FiLMConvLocator(d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl, 1):
            img = batch["img_tokens"].to(device)  # [B,D,Ht,Wt]
            txt = batch["text_vec"].to(device)    # [B,D]
            tgt = batch["target"].to(device)      # [B,1,Ht,Wt]
            msk = batch["valid"].to(device)       # [B,1,Ht,Wt]

            opt.zero_grad()
            logits = model(img, txt)
            loss = heatmap_loss(logits, tgt, msk, pos_weight=2.0)
            loss.backward()
            opt.step()

            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch+1}  step {step}  loss {running/50:.4f}")
                running = 0.0

    return model

if __name__ == "__main__":
    trained_model = train_once(data_dir="out_images", d_model=768, epochs=2, batch_size=8)
