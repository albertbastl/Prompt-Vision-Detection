#!/usr/bin/env python3
import os
import re
import glob
import time
import pickle
import bisect

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATASET_DIR               = "processed_batches_200"
MODEL_ID                  = "google/siglip2-base-patch16-224"
GRID_SIZE                 = 14
EMBED_DIM                 = 768
NUM_HEADS                 = 2
NUM_LAYERS                = 4
LEARNING_RATE             = 6.9e-5
BATCH_SIZE                = 256
EPOCHS                    = 10
LOG_IMAGES_EVERY_N_STEPS  = 2
KEEP_LAST_N_EPOCH_WEIGHTS = 3
CHECKPOINT_DIR            = "./checkpoints"

# ─── DATASET ──────────────────────────────────────────────────────────────────
class PreprocessedDataset(Dataset):
    def __init__(self, pkl_dir):
        files = sorted(
            os.path.join(pkl_dir, fn)
            for fn in os.listdir(pkl_dir)
            if fn.endswith(".pkl")
        )
        if not files:
            raise RuntimeError(f"No .pkl files found in {pkl_dir}")
        self.batch_files = files

        self.cum_counts = [0]
        for path in self.batch_files:
            with open(path, "rb") as f:
                batch = pickle.load(f)
            self.cum_counts.append(self.cum_counts[-1] + len(batch))

        print(f"Found {len(self.batch_files)} batch files, total samples: {self.cum_counts[-1]}")

    def __len__(self):
        return self.cum_counts[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        batch_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        sample_idx = idx - self.cum_counts[batch_idx]
        with open(self.batch_files[batch_idx], "rb") as f:
            batch = pickle.load(f)
        return batch[sample_idx]

def collate_fn(batch):
    return {
        "pixel_values":   torch.stack([x["pixel_values"]   for x in batch]),
        "input_ids":      torch.stack([x["input_ids"]      for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "target_heatmap": torch.stack([x["target_heatmap"] for x in batch]),
    }

# ─── MODEL ────────────────────────────────────────────────────────────────────
class LocalizationDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, grid_size):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.1,
            activation='gelu',
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.head    = nn.Linear(embed_dim, grid_size * grid_size)

    def forward(self, text_vec, patch_vecs):
        q   = self.query_token + text_vec.unsqueeze(1)
        out = self.decoder(tgt=q, memory=patch_vecs)
        return self.head(out.squeeze(1))

# ─── CHECKPOINT / W&B HELPERS ────────────────────────────────────────────────
def save_ckpt_to_disk(decoder, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"decoder_epoch{epoch}.pth")
    torch.save(decoder.state_dict(), ckpt_path)
    return ckpt_path

def upload_ckpt_to_wandb(ckpt_path, epoch, run):
    # Show file under Run → Files for this run (immediate visibility)
    wandb.save(ckpt_path, base_path=CHECKPOINT_DIR, policy="now")

    # Log as a versioned Artifact named "decoder"
    art = wandb.Artifact("decoder", type="model", metadata={"epoch": int(epoch)})
    art.add_file(ckpt_path, name=os.path.basename(ckpt_path))
    logged = run.log_artifact(art, aliases=[f"epoch-{epoch}", "latest"])
    logged.wait()  # ensure processed server-side

def cleanup_wandb_artifacts(entity, project, keep_last_n=3):
    """Keep only the newest N versions of the single 'decoder' artifact."""
    try:
        api = wandb.Api()
        coll = api.artifact_collection(f"{entity}/{project}/decoder", type="model")
        versions = list(coll.versions())  # newest first
        for a in versions[keep_last_n:]:
            print(f"Deleting old artifact version: {a.name} (epoch={a.metadata.get('epoch')})")
            a.delete()
    except Exception as e:
        print(f"Warning: artifact cleanup failed: {e}")

def cleanup_wandb_run_files(run, keep_last_n=3):
    """Keep only the newest N checkpoint files in Run → Files."""
    try:
        api = wandb.Api()
        api_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        ckpt_files = []
        for f in api_run.files():
            if f.name.endswith(".pth") and ("decoder_epoch" in f.name):
                m = re.search(r"decoder_epoch(\d+)\.pth$", f.name)
                epoch = int(m.group(1)) if m else -1
                ckpt_files.append((epoch, f.name, f))
        ckpt_files.sort(key=lambda x: (x[0], x[1]), reverse=True)  # newest first
        for _, name, fobj in ckpt_files[keep_last_n:]:
            print(f"Deleting old run file: {name}")
            fobj.delete()
    except Exception as e:
        print(f"Warning: run-file cleanup failed: {e}")

# ─── TRAINING LOOP ────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    run = wandb.init(
        project="prompt-guided-object-localizer",
        name=f"decoder_e{EPOCHS}_h{NUM_HEADS}_l{NUM_LAYERS}_b{BATCH_SIZE}_lr{LEARNING_RATE}",
        config={
            "model_id": MODEL_ID,
            "grid_size": GRID_SIZE,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "log_images_every": LOG_IMAGES_EVERY_N_STEPS,
        },
    )
    cfg = wandb.config

    siglip = AutoModel.from_pretrained(cfg.model_id).to(device)
    for p in siglip.parameters():
        p.requires_grad = False

    decoder = LocalizationDecoder(cfg.embed_dim, cfg.num_heads, cfg.num_layers, cfg.grid_size).to(device)
    wandb.watch(decoder, log="all", log_freq=100)
    optimizer = AdamW(decoder.parameters(), lr=cfg.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    dataset = PreprocessedDataset(DATASET_DIR)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        decoder.train()
        epoch_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            global_step += 1
            px     = batch["pixel_values"].to(device)
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            target = batch["target_heatmap"].to(device)

            with torch.no_grad():
                v = siglip.vision_model(pixel_values=px).last_hidden_state.float()
                t = siglip.text_model(input_ids=ids, attention_mask=mask).pooler_output.float()

            logits = decoder(t, v)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss += loss_val * px.size(0)
            wandb.log({"train/loss": loss_val, "step": global_step})

            if global_step % cfg.log_images_every == 0:
                try:
                    img = batch["pixel_values"][0].cpu() * 0.5 + 0.5
                    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
                    gt  = batch["target_heatmap"][0].view(cfg.grid_size, cfg.grid_size).cpu().numpy()
                    pr  = torch.sigmoid(logits[0]).view(cfg.grid_size, cfg.grid_size).detach().cpu().numpy()

                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    axes[0].imshow(img); axes[0].set_title("Input Image"); axes[0].axis('off')
                    axes[1].imshow(gt, vmin=0, vmax=1); axes[1].set_title("Ground Truth"); axes[1].axis('off')
                    im = axes[2].imshow(pr, vmin=0, vmax=1); axes[2].set_title("Prediction"); axes[2].axis('off')
                    fig.suptitle(f"Step {global_step}")
                    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)

                    wandb.log({"examples/comparison": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    print(f"Warning: could not log image at step {global_step}: {e}")

        avg_loss = epoch_loss / len(dataset)
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

        # Save → Upload → Cleanup (keep last N)
        ckpt_path = save_ckpt_to_disk(decoder, epoch)
        upload_ckpt_to_wandb(ckpt_path, epoch, run)
        try:
            os.remove(ckpt_path)  # remove local copy
        except OSError:
            pass
        cleanup_wandb_artifacts(entity=run.entity, project=run.project,
                                keep_last_n=KEEP_LAST_N_EPOCH_WEIGHTS)
        cleanup_wandb_run_files(run, keep_last_n=KEEP_LAST_N_EPOCH_WEIGHTS)

    wandb.finish()
    print("Training complete.")

if __name__ == "__main__":
    main()
