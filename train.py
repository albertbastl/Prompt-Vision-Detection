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

# --- Constants ---
TRAIN_DIR = "pd_10k_500patches/train"
VAL_DIR = "pd_10k_500patches/val"
SAVE_PATH = "miou_siglip.pt"

BATCH_SIZE = 16
NUM_WORKERS = 4
EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 20

WANDB_PROJECT = "openvocab"
WANDB_RUN_NAME = "bs 16 siglip exact"

def calculate_accuracy(logits, contrastive_labels):
    """
    Calculates accuracy for the contrastive task.
    In the -1/1 label setup, we still look for the max logit 
    and compare it to the index of the positive (1) label.
    """
    # logits shape: [N_patches, N_texts]
    # contrastive_labels shape: [N_patches, N_texts] (values are -1 or 1)
    
    preds = logits.argmax(dim=1)
    ground_truth = contrastive_labels.argmax(dim=1) # The index where value is 1
    acc = (preds == ground_truth).float().mean()
    return acc

class SimplePatchDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        # Mocking file paths for standalone runnability if dir missing
        if not self.file_paths:
            print(f"Warning: No .npz files found in {data_dir}. Using dummy data mode.")
            self.dummy_mode = True
        else:
            self.dummy_mode = False
        
    def __len__(self):
        return len(self.file_paths) if not self.dummy_mode else 100
        
    def __getitem__(self, idx):
        if self.dummy_mode:
            # Generate dummy data for testing
            return {
                "img_embs": torch.randn(5, 768), # 5 patches per file
                "txt_emb": torch.randn(768),
                "labels": torch.tensor([0, 1, 0, 0, 1]).float() # Mixed labels
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
        # Learnable Temperature (t_prime in snippet)
        self.logits_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # Learnable Bias (b in snippet) - Initialize to roughly -10 to ensure sigmoid starts closed
        self.logits_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(self, img_embs):
        projected_embs = self.net(img_embs)
        # Line 7: zimg = l2_normalize(img_emb)
        projected_embs = F.normalize(projected_embs, p=2, dim=-1)
        return projected_embs

def siglip_loss(logits, labels, normalize_factor):
    """
    Line 11: l = -sum(log_sigmoid(labels * logits)) / n
    """
    # labels are -1 and 1
    # logits are (dot_product * t + b)
    
    # F.logsigmoid(x) is mathematically equivalent to -softplus(-x)
    # Corresponds to log(sigmoid(labels * logits))
    log_probs = F.logsigmoid(labels * logits)
    
    # Sum over all pairs, then divide by N (batch size, not total elements)
    loss = -torch.sum(log_probs) / normalize_factor
    return loss

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize WandB (disabled for dummy run)
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        mode="disabled" if not os.path.exists(TRAIN_DIR) else "online",
        config={
            "learning_rate": LEARNING_RATE,
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

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs (SigLIP Style) ---")
    
    best_val_acc = -1.0

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        
        # Use global step for accurate logging if needed
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            img_embs = batch['img_embs'].to(device) # [B, num_patches, Dim]
            txt_emb = batch['txt_emb'].to(device)   # [B, Dim]
            labels = batch['labels'].to(device)     # [B, num_patches]
            
            B = txt_emb.shape[0] # Mini-batch size 'n'
            
            optimizer.zero_grad()
            
            # 1. Project Image Embeddings
            projected_img_embs = model(img_embs) # [B, num_patches, Dim]
            
            # 2. Flatten patches logic
            # We only want to compute loss on patches that are actually labeled as positive
            # for the specific image they belong to.
            positive_masks = (labels == 1)
            positive_projected_embs = projected_img_embs[positive_masks] # [N_total_pos_patches, Dim]

            if positive_projected_embs.nelement() == 0:
                continue

            # Line 8: ztxt = l2_normalize(txt_emb)
            normalized_txt = F.normalize(txt_emb, p=2, dim=-1)

            # 3. Compute Dot Product (Cosine Similarity)
            # Shape: [N_total_pos_patches, B]
            # This compares every valid patch against EVERY text in the batch
            sim_matrix = torch.matmul(positive_projected_embs, normalized_txt.T)

            # 4. Apply Temperature and Bias
            # Line 6 & 9: logits = dot(...) * t + b
            temperature = model.logits_scale.exp()
            bias = model.logits_bias
            logits = sim_matrix * temperature + bias

            # 5. Construct Labels (-1 / 1)
            # Determine which text index belongs to which image patch
            batch_indices = torch.arange(B, device=device).unsqueeze(1) # [B, 1]
            # Expand batch indices to match mask shape [B, num_patches]
            # Select indices corresponding to positive_masks
            positive_batch_indices = batch_indices.expand_as(labels)[positive_masks]
            
            # Initialize labels to -1 (Line 10: labels = -ones(n) + diagonal correction)
            contrastive_labels = -torch.ones_like(logits, device=device)
            
            # Set the "diagonal" (correct text matches) to +1
            # contrastive_labels[row, correct_column] = 1
            contrastive_labels[torch.arange(len(positive_projected_embs)), positive_batch_indices] = 1.0
            
            # 6. Compute Sigmoid Loss
            loss = siglip_loss(logits, contrastive_labels, normalize_factor=B)
            
            loss.backward()
            optimizer.step()
            
            # Metrics
            acc = calculate_accuracy(logits.detach(), contrastive_labels)
            train_loss += loss.item()
            train_acc += acc.item()
            
            train_pbar.set_postfix(loss=f"{train_loss/(batch_idx+1):.4f}", acc=f"{train_acc/(batch_idx+1):.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # --- Validation Loop ---
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
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc:.4f}")

        wandb.log({
            "train/loss": avg_train_loss,
            "val/acc": avg_val_acc,
            "epoch": epoch + 1
        })
        
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model: {best_val_acc:.4f}")

    wandb.finish()