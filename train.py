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
import itertools

TRAIN_DIR = "pd_10k_500patches/train"
VAL_DIR = "pd_10k_500patches/val"
SAVE_PATH = "miou.pt"

BATCH_SIZE = 256
ACCUMULATION_STEPS = 4
NUM_WORKERS = 4

EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1

LEARNING_RATE = 3e-4
EPOCHS = 20

WANDB_PROJECT = "openvocab"
WANDB_RUN_NAME = "davids correction with accumulation bs 256"


def calculate_accuracy(logits, contrastive_labels):
    preds = logits.argmax(dim=1)
    ground_truth = contrastive_labels.argmax(dim=1)
    acc = (preds == ground_truth).float().mean()
    return acc


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
        
        return {
            "img_embs": img_patch_embs,
            "txt_emb": txt_emb,
            "labels": labels,
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
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
            "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
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
    
    all_params = itertools.chain(model.parameters())
    
    wandb.watch(model, log="all", log_freq=100)
    
    optimizer = AdamW(all_params, lr=LEARNING_RATE)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            img_embs = batch['img_embs'].to(device)
            txt_emb = batch['txt_emb'].to(device)
            labels = batch['labels'].to(device)
            
            B = txt_emb.shape[0]
            
            projected_img_embs = model(img_embs)
            
            positive_masks = (labels == 1)
            positive_projected_embs = projected_img_embs[positive_masks]

            if positive_projected_embs.nelement() == 0:
                continue

            batch_indices = torch.arange(B, device=device).unsqueeze(1)
            positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]
           

            logits = torch.matmul(positive_projected_embs, txt_emb.T)

            temperature = model.logit_scale.exp()
            logits = logits * temperature

            contrastive_labels = torch.zeros_like(logits, device=device)
            contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0
            
            loss = loss_fn(logits, contrastive_labels)
            
            loss = loss / ACCUMULATION_STEPS
            
            loss.backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            acc = calculate_accuracy(logits.detach(), contrastive_labels)
            train_loss += loss.item() * ACCUMULATION_STEPS
            train_acc += acc.item()
            
            running_loss = train_loss / (batch_idx + 1)
            running_acc = train_acc / (batch_idx + 1)
            train_pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_pbar):
                img_embs = batch['img_embs'].to(device)
                txt_emb = batch['txt_emb'].to(device)
                labels = batch['labels'].to(device)
                
                B = txt_emb.shape[0]

                projected_img_embs = model(img_embs)
                
                positive_masks = (labels == 1)
                positive_projected_embs = projected_img_embs[positive_masks]

                if positive_projected_embs.nelement() == 0:
                    continue
                
                batch_indices = torch.arange(B, device=device).unsqueeze(1)
                positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]

                logits = torch.matmul(positive_projected_embs, txt_emb.T)
                
                contrastive_labels = torch.zeros_like(logits, device=device)
                contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0

                loss = loss_fn(logits, contrastive_labels)
                
                acc = calculate_accuracy(logits, contrastive_labels)
                val_loss += loss.item()
                val_acc += acc.item()
                
                running_loss = val_loss / (batch_idx + 1)
                running_acc = val_acc / (batch_idx + 1)
                val_pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "train/epoch_acc": avg_train_acc,
            "val/epoch_loss": avg_val_loss,
            "val/epoch_acc": avg_val_acc,
            "epoch": epoch + 1
        }, step=global_step)
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(
                {
                    "image_projector": model.state_dict(),
                }, 
                SAVE_PATH
            )
            print(f"New best model saved to {SAVE_PATH} (Acc: {best_val_acc:.4f})")
            wandb.save(SAVE_PATH)

    print("--- Training Complete ---")
    
    wandb.finish()
