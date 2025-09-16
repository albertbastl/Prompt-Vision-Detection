# train_linear_decoder.py
import os, glob, argparse, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Dataset: loads .npz pairs
# label = 1 if any heatmap tile == 1, else 0
# x = concat(mean_pool(img_tokens), txt_emb) -> [1536]
# -------------------------
class NPZPairs(Dataset):
    def __init__(self, root="preprocessed_dataset"):
        self.paths = sorted(glob.glob(os.path.join(root, "*.npz")))
        if not self.paths:
            raise FileNotFoundError(f"No .npz found under {root}")

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        # img_tokens: [gh, gw, D] (float16) -> mean-pool -> [D]
        img_feat = d["img_tokens"].astype(np.float32).mean(axis=(0,1))
        # txt_emb: [D] (float16) -> [D]
        txt_feat = d["txt_emb"].astype(np.float32)
        x = np.concatenate([img_feat, txt_feat], axis=0).astype(np.float32)  # [2D]
        # heatmap: [gh, gw] uint8 -> label {0,1}
        y = np.array([1.0 if d["heatmap"].any() else 0.0], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)

# -------------------------
# Model: single linear layer
# -------------------------
class LinearDecoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)  # logits

# -------------------------
# Train
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="preprocessed_dataset")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--save", type=str, default="linear_decoder.pt")
    args = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    ds = NPZPairs(args.root)
    # infer 2D from first sample
    in_dim = ds[0][0].numel()
    loader = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = LinearDecoder(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True)          # [B, 1536]
            y = y.to(device, non_blocking=True).squeeze(1)  # [B]

            opt.zero_grad(set_to_none=True)
            logits = model(x)                             # [B]
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            with torch.no_grad():
                total_loss += loss.item() * x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (preds == y).sum().item()
                total += x.size(0)

        print(f"Epoch {epoch:02d} | loss {total_loss/total:.4f} | acc {total_correct/total:.4f}")

    torch.save({"model": model.state_dict(), "in_dim": in_dim}, args.save)
    print(f"Saved: {args.save}")

if __name__ == "__main__":
    main()
