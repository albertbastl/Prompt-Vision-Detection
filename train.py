import torch
import glob
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import wandb
import itertools
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

TRAIN_DIR = "pd_30k_500patches_imgnorm/train"
VAL_DIR = "pd_30k_500patches_imgnorm/val"
# Changed default save name to be generic, specific epoch names are generated in loop
BEST_SAVE_PATH = "30k_hopefullyfixed.pt" 

BATCH_SIZE = 64
NUM_WORKERS = 4
EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1

# --- Hyperparameters Updated ---
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CLIP_GRAD = 1.0
EPOCHS = 10

WANDB_PROJECT = "openvocab"
WANDB_RUN_NAME = "30k with fixes to lr temp and bias"

def calculate_accuracy(logits, contrastive_labels):
    preds = logits.argmax(dim=1)
    ground_truth = contrastive_labels.argmax(dim=1)
    acc = (preds == ground_truth).float().mean()
    return acc

class SimplePatchDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_dir}. Using dummy data mode.")
            self.dummy_mode = True
        else:
            self.dummy_mode = False
        
    def __len__(self):
        return len(self.file_paths) if not self.dummy_mode else 100
        
    def __getitem__(self, idx):
        if self.dummy_mode:
            return {
                "img_embs": torch.randn(5, 768),
                "txt_emb": torch.randn(768),
                "labels": torch.tensor([0, 1, 0, 0, 1]).float()
            }

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
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(10))
        self.logits_bias = nn.Parameter(torch.ones([]) * -5.0)

    def forward(self, img_embs):
        projected_embs = self.net(img_embs)
        projected_embs = F.normalize(projected_embs, p=2, dim=-1)
        return projected_embs

def siglip_loss(logits, labels, normalize_factor):
    log_probs = F.logsigmoid(labels * logits)
    loss = -torch.sum(log_probs) / normalize_factor
    return loss

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode="disabled" if not os.path.exists(TRAIN_DIR) else "online",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": CLIP_GRAD,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "embed_dim": EMBED_DIM,
        }
    )
    
    train_dataset = SimplePatchDataset(data_dir=TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    val_dataset = SimplePatchDataset(data_dir=VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SimpleProjector(
        in_dim=EMBED_DIM, 
        hidden_dim=HIDDEN_DIM, 
        out_dim=EMBED_DIM,
        drop=DROP_RATE
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=100)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * len(train_loader) * EPOCHS),
        num_training_steps=len(train_loader) * EPOCHS
    )
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs (SigLIP Style) ---")
    
    best_val_acc = -1.0
    
    # List to keep track of the last 3 checkpoints
    recent_checkpoints = [] 

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            img_embs = batch['img_embs'].to(device)
            txt_emb = batch['txt_emb'].to(device)
            labels = batch['labels'].to(device)
            
            B = txt_emb.shape[0]
            
            optimizer.zero_grad()
            
            projected_img_embs = model(img_embs)
            
            positive_masks = (labels == 1)
            positive_projected_embs = projected_img_embs[positive_masks]

            if positive_projected_embs.nelement() == 0:
                continue

            normalized_txt = F.normalize(txt_emb, p=2, dim=-1)

            sim_matrix = torch.matmul(positive_projected_embs, normalized_txt.T)

            temperature = model.logits_scale.exp()
            temperature = torch.clamp(temperature, max=100.0)
            bias = model.logits_bias
            logits = sim_matrix * temperature + bias

            batch_indices = torch.arange(B, device=device).unsqueeze(1)
            positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]
            
            contrastive_labels = -torch.ones_like(logits, device=device)
            contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0
            
            loss = siglip_loss(logits, contrastive_labels, normalize_factor=B)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD)
            optimizer.step()
            scheduler.step()
            
            acc = calculate_accuracy(logits.detach(), contrastive_labels)
            train_loss += loss.item()
            train_acc += acc.item()
            
            train_pbar.set_postfix(loss=f"{train_loss/(batch_idx+1):.4f}", acc=f"{train_acc/(batch_idx+1):.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                img_embs = batch['img_embs'].to(device)
                txt_emb = batch['txt_emb'].to(device)
                labels = batch['labels'].to(device)
                B = txt_emb.shape[0]

                projected_img_embs = model(img_embs)
                positive_masks = (labels == 1)
                positive_projected_embs = projected_img_embs[positive_masks]

                if positive_projected_embs.nelement() == 0: continue
                
                normalized_txt = F.normalize(txt_emb, p=2, dim=-1)
                
                sim_matrix = torch.matmul(positive_projected_embs, normalized_txt.T)
                logits = sim_matrix * model.logits_scale.exp() + model.logits_bias

                batch_indices = torch.arange(B, device=device).unsqueeze(1)
                positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]

                contrastive_labels = -torch.ones_like(logits, device=device)
                contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0

                loss = siglip_loss(logits, contrastive_labels, normalize_factor=B)
                acc = calculate_accuracy(logits, contrastive_labels)
                
                val_loss += loss.item()
                val_acc += acc.item()
        
        if len(val_loader) > 0:
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_acc / len(val_loader)
        else:
            avg_val_loss, avg_val_acc = 0, 0
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        wandb.log({
            "train/loss": avg_train_loss,
            "train/acc": avg_train_acc,
            "val/loss": avg_val_loss,
            "val/acc": avg_val_acc,
            "epoch": epoch + 1
        })
        
        # --- SAVE LOGIC: Best Model ---
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), BEST_SAVE_PATH)
            print(f"Saved new best model: {best_val_acc:.4f}")
            wandb.save(BEST_SAVE_PATH)

        # --- SAVE LOGIC: Last 3 Epochs (Rolling) ---
        epoch_ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_ckpt_name)
        
        # Upload this specific epoch checkpoint to wandb
        wandb.save(epoch_ckpt_name)
        
        # Add to list and manage local files
        recent_checkpoints.append(epoch_ckpt_name)
        if len(recent_checkpoints) > 3:
            oldest_ckpt = recent_checkpoints.pop(0)
            if os.path.exists(oldest_ckpt):
                print(f"Removing old checkpoint locally: {oldest_ckpt}")
                os.remove(oldest_ckpt)

    wandb.finish()