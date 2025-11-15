import torch
import glob
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm


TRAIN_DIR = "pd_10k_500patches/train"
VAL_DIR = "pd_10k_500patches/val" 

BATCH_SIZE = 64
NUM_WORKERS = 4

EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.1

LEARNING_RATE = 3e-4
EPOCHS = 10


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
            "labels": labels
        }

class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, drop: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, img_embs, txt_emb):
        txt_emb_expanded = txt_emb.unsqueeze(1).expand_as(img_embs)
        combined_embs = img_embs + txt_emb_expanded
        return self.net(combined_embs).squeeze(-1)
    
if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        drop=DROP_RATE
    ).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    
    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    
    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            img_embs = batch['img_embs'].to(device)
            txt_emb = batch['txt_emb'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(img_embs, txt_emb)
            
            loss = loss_fn(logits, labels)
            
            loss.backward()
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                img_embs = batch['img_embs'].to(device)
                txt_emb = batch['txt_emb'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(img_embs, txt_emb)
                
                loss = loss_fn(logits, labels)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print("--- Training Complete ---")
    
    torch.save(model.state_dict(), "simple_projector.pth")
    print("Model saved to simple_projector.pth")