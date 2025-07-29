#!/usr/bin/env python3
import os
import pickle
import bisect
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel
from tqdm import tqdm

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATASET_DIR  = "processed_batches_empty"                    # folder containing image_batch_*.pkl
MODEL_ID     = "google/siglip2-base-patch16-224"
GRID_SIZE    = 14                                     # must match preprocessing
EMBED_DIM    = 768
NUM_HEADS    = 2
NUM_LAYERS   = 4
LEARNING_RATE= 6.9e-5
BATCH_SIZE   = 256
EPOCHS       = 10
OUTPUT_PATH  = "weights_siglip_localizer.pth"

# ─── DATASET CLASS ────────────────────────────────────────────────────────────
class PreprocessedDataset(Dataset):
    def __init__(self, pkl_dir):
        # find all .pkl files
        files = sorted(
            os.path.join(pkl_dir, fn)
            for fn in os.listdir(pkl_dir)
            if fn.endswith(".pkl")
        )
        if not files:
            raise RuntimeError(f"No .pkl files found in {pkl_dir}")
        self.batch_files = files

        # build cumulative counts
        self.cum_counts = [0]
        for path in self.batch_files:
            with open(path, "rb") as f:
                batch = pickle.load(f)
            self.cum_counts.append(self.cum_counts[-1] + len(batch))

        print(f"Found {len(self.batch_files)} batch files, total samples: {self.cum_counts[-1]}")

    def __len__(self):
        return self.cum_counts[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.__len__():
            raise IndexError(idx)
        batch_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        sample_idx = idx - self.cum_counts[batch_idx]
        with open(self.batch_files[batch_idx], "rb") as f:
            batch = pickle.load(f)
        return batch[sample_idx]

def collate_fn(batch):
    return {
        "pixel_values":    torch.stack([x["pixel_values"]    for x in batch]),
        "input_ids":       torch.stack([x["input_ids"]       for x in batch]),
        "attention_mask":  torch.stack([x["attention_mask"]  for x in batch]),
        "target_heatmap":  torch.stack([x["target_heatmap"]  for x in batch]),
    }

# ─── MODEL ─────────────────────────────────────────────────────────────────────
class LocalizationDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, grid_size):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.head    = nn.Linear(embed_dim, grid_size * grid_size)

    def forward(self, text_vec, patch_vecs):
        # text_vec: (B, D), patch_vecs: (B, N, D)
        q = self.query_token + text_vec.unsqueeze(1)    # (B, 1, D)
        out = self.decoder(tgt=q, memory=patch_vecs)     # (B, 1, D)
        return self.head(out.squeeze(1))                # (B, G²)

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # load frozen SigLIP
    siglip = AutoModel.from_pretrained(MODEL_ID).to(device)
    for p in siglip.parameters():
        p.requires_grad = False

    # decoder to train
    decoder = LocalizationDecoder(
        EMBED_DIM, NUM_HEADS, NUM_LAYERS, GRID_SIZE
    ).to(device)

    opt     = AdamW(decoder.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    # dataset & loader
    ds = PreprocessedDataset(DATASET_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )

    for epoch in range(1, EPOCHS + 1):
        decoder.train()
        total_loss = 0.0
        pbar = tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}")
        for batch in pbar:
            px     = batch["pixel_values"].to(device)      # (B, 3, H, W)
            ids    = batch["input_ids"].to(device)         # (B, T)
            mask   = batch["attention_mask"].to(device)    # (B, T)
            target = batch["target_heatmap"].to(device)    # (B, G²)

            with torch.no_grad():
                v = siglip.vision_model(pixel_values=px).last_hidden_state.float()  # (B, N, D)
                t = siglip.text_model(
                    input_ids=ids, attention_mask=mask
                ).pooler_output.float()  # (B, D)

            pred = decoder(t, v)                                  # (B, G²)
            loss = loss_fn(pred, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * px.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / len(ds)
        print(f"Epoch {epoch} complete — avg loss: {avg:.4f}")

    # save decoder weights
    torch.save(decoder.state_dict(), OUTPUT_PATH)
    print(f"Saved trained decoder to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
