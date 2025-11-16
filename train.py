import torch
import glob
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import wandb
import random
import torch.nn.functional as F

TRAIN_DIR = "pd_10k_500patches/train"
VAL_DIR = "pd_10k_500patches/val"
SAVE_PATH = "miou.pt"

BATCH_SIZE = 64
NUM_WORKERS = 4

EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1

LEARNING_RATE = 3e-4
EPOCHS = 20

WANDB_PROJECT = "openvocab"
WANDB_RUN_NAME = "siglip loss"


def calculate_miou(logits, labels):
    preds = (logits > 0.0).float()
    labels = labels.float()
    
    intersection = (preds * labels).sum(dim=1)
    union = (preds + labels).clamp(min=0, max=1).sum(dim=1)
    
    epsilon = 1e-6
    
    iou = (intersection + epsilon) / (union + epsilon)
    
    return iou.mean()


class SimplePatchDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        
        img_patch_embs = torch.from_numpy(data['img_patch_embs']).float()
        txt_emb = torch.from_numpy(data['txt_emb']).float()
        labels = torch.from_numpy(data['labels']).float()
        grid_hw = torch.from_numpy(data['grid_hw']).int()
        
        return {
            "img_embs": img_patch_embs,
            "txt_emb": txt_emb,
            "labels": labels,
            "grid_hw": grid_hw
        }

class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, img_embs):
        projected_embs = self.net(img_embs)
        projected_embs = F.normalize(projected_embs, p=2, dim=-1)
        return projected_embs
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
        }
    )
    
    print(f"Loading train dataset from {TRAIN_DIR}...")
    train_dataset = SimplePatchDataset(data_dir=TRAIN_DIR)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"Loading validation dataset from {VAL_DIR}...")
    val_dataset = SimplePatchDataset(data_dir=VAL_DIR)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Initializing model on {device}...")
    model = SimpleProjector(
        in_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        out_dim=EMBED_DIM,
        drop=DROP_RATE
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    
    best_val_miou = -1.0
    global_step = 0

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        train_miou = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            img_embs = batch['img_embs'].to(device)
            txt_emb = batch['txt_emb'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            projected_embs = model(img_embs)
            
            positive_masks = (labels == 1)
            positive_projected_embs = projected_embs[positive_masks]

            if positive_projected_embs.nelement() == 0:
                continue

            batch_indices = torch.arange(BATCH_SIZE, device=device).unsqueeze(1)
            positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]
            target_txt_embs = txt_emb[positive_batch_indices]
            
            logits = torch.matmul(positive_projected_embs, txt_emb.T)

            contrastive_labels = torch.zeros_like(logits, device=device)
            contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0
            
            loss = nn.BCEWithLogitsLoss()(logits, contrastive_labels)
            
            loss.backward()
            
            optimizer.step()
            global_step += 1
            
            miou = calculate_miou(logits.detach(), labels)
            train_loss += loss.item()
            train_miou += miou.item()
            
            running_loss = train_loss / (batch_idx + 1)
            running_miou = train_miou / (batch_idx + 1)
            train_pbar.set_postfix(loss=f"{running_loss:.4f}", miou=f"{running_miou:.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_miou = train_miou / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_miou = 0.0
        
        log_val_batch_idx = random.randint(0, len(val_loader) - 1)
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                img_embs = batch['img_embs'].to(device)
                txt_emb = batch['txt_emb'].to(device)
                labels = batch['labels'].to(device)
                grid_hw = batch['grid_hw']
                
                logits = model(img_embs, txt_emb)
                
                labels_siglip = (labels * 2) - 1.0
                loss = -F.logsigmoid(labels_siglip * logits).mean()
                
                miou = calculate_miou(logits, labels)
                val_loss += loss.item()
                val_miou += miou.item()
                
                running_loss = val_loss / (batch_idx + 1)
                running_miou = val_miou / (batch_idx + 1)
                val_pbar.set_postfix(loss=f"{running_loss:.4f}", miou=f"{running_miou:.4f}")

                if batch_idx == log_val_batch_idx:
                    sample_logits = logits[0]
                    sample_labels = labels[0]
                    gh, gw = grid_hw[0]
                    
                    gh, gw = gh.item(), gw.item()

                    pred_mask = torch.sigmoid(sample_logits).reshape(gh, gw).cpu().numpy()
                    gt_mask = sample_labels.float().reshape(gh, gw).cpu().numpy()
                    
                    gt_mask_rgb = np.stack([gt_mask]*3, axis=-1)
                    pred_mask_rgb = np.stack([pred_mask]*3, axis=-1)

                    border_rgb = np.zeros((gh, 1, 3), dtype=np.float32)
                    border_rgb[:, :, 0] = 1.0
                    
                    combined_mask = np.concatenate((gt_mask_rgb, border_rgb, pred_mask_rgb), axis=1)

                    wandb.log({
                        "val/Mask Comparison": wandb.Image(
                            combined_mask, 
                            caption=f"Epoch {epoch+1} | GT (left) vs Pred (right)"
                        )
                    }, step=global_step)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_miou = val_miou / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train mIoU: {avg_train_miou:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_val_miou:.4f}")

        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_miou": avg_train_miou,
            "val/epoch_loss": avg_val_loss,
            "val/epoch_miou": avg_val_miou,
            "epoch": epoch + 1
        }, step=global_step)
        
        if avg_val_miou > best_val_miou:
            best_val_miou = avg_val_miou
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New best model saved to {SAVE_PATH} (mIoU: {best_val_miou:.4f})")
            wandb.save(SAVE_PATH)

    print("--- Training Complete ---")
    
    wandb.finish()
