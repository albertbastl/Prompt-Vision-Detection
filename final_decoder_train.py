import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModel
import wandb
import time
import os
import glob
import json
import pickle
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from config import MODEL_ID, PATCH_GRID_SIZE, TEXT_MAX_LENGTH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 1024
NUM_EPOCHS = 20
NUM_WORKERS = min(os.cpu_count(), 4)
PIN_MEMORY = False

LOG_IMAGES_EVERY_N_STEPS = 200
WANDB_PROJECT = "siglip2-localization-decoder"
WANDB_RUN_NAME = f"train_bs{BATCH_SIZE}_lr{LEARNING_RATE}_with_visuals"

CHECKPOINT_DIR = "./checkpoints"
DECODER_SAVE_PATH = "./siglip2_localization_decoder.pth"
CHECKPOINT_EVERY_N_STEPS = 500
KEEP_LAST_N_CHECKPOINTS = 3

class LocalizationDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, PATCH_GRID_SIZE * PATCH_GRID_SIZE)

    def forward(self, text_vector, patch_vectors):
        query = self.query_token + text_vector.unsqueeze(1)
        decoder_output = self.transformer_decoder(tgt=query, memory=patch_vectors)
        return self.output_head(decoder_output.squeeze(1))

class PreprocessedDataset(Dataset):
    def __init__(self, processed_dir):
        metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("Loading preprocessed data into memory...")
        self.data = []
        for batch_file in tqdm(metadata['batch_files'], desc="Loading batch files"):
            try:
                with open(batch_file, 'rb') as f:
                    self.data.extend(pickle.load(f))
            except FileNotFoundError:
                print(f"Warning: Batch file not found, skipping: {batch_file}")
        print(f"Loaded {len(self.data)} preprocessed samples.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    target_heatmaps = torch.stack([item['target_heatmap'] for item in batch])
    phrases = [item['phrase'] for item in batch]
    return {'pixel_values': pixel_values, 'input_ids': input_ids, 'attention_mask': attention_mask, 'target_heatmaps': target_heatmaps, 'phrases': phrases}
    
def save_checkpoint(decoder, optimizer, epoch, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch, 'global_step': step, 'model_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss
    }, path)
    
def cleanup_old_checkpoints(checkpoint_dir, keep_last_n):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if len(checkpoints) > keep_last_n:
        checkpoints.sort(key=os.path.getmtime)
        for old_checkpoint in checkpoints[:-keep_last_n]:
            try:
                os.remove(old_checkpoint)
            except OSError as e:
                print(f"Warning: Could not remove old checkpoint {old_checkpoint}. Error: {e}")

def train_one_run():
    wandb.init(project=WANDB_PROJECT)

    learning_rate = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    num_layers = wandb.config.num_layers
    num_heads = wandb.config.num_heads
    num_epochs = wandb.config.num_epochs 

    print("Loading pre-trained SigLIP model...")
    siglip_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    if DEVICE == "cuda": siglip_model = siglip_model.half()
    for param in siglip_model.parameters(): param.requires_grad = False

    decoder = LocalizationDecoder(num_layers=num_layers, num_heads=num_heads).to(DEVICE)
    optimizer = AdamW(decoder.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Loading preprocessed dataset...")
    train_dataset = PreprocessedDataset(processed_dir="./processed_dataset")
    dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)

    print(f"--- Starting Sweep Run ---")
    print(f"Config: LR={learning_rate:.1e}, BS={batch_size}, Layers={num_layers}, Heads={num_heads}, NE={num_epochs}")
    global_step = 0
    
    siglip_model.eval()
    for epoch in range(num_epochs):
        decoder.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            pixel_values = batch['pixel_values'].to(DEVICE)
            input_ids, attention_mask = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE)
            target_heatmaps = batch['target_heatmaps'].to(DEVICE)

            with torch.no_grad():
                dtype = torch.float16 if DEVICE == "cuda" else torch.float32
                with torch.autocast(device_type=DEVICE, dtype=dtype):
                    vision_outputs = siglip_model.vision_model(pixel_values=pixel_values)
                    text_outputs = siglip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
                patch_vectors = vision_outputs.last_hidden_state.float()
                text_vector = text_outputs.pooler_output.float()

            predicted_heatmap_logits = decoder(text_vector, patch_vectors)
            loss = criterion(predicted_heatmap_logits, target_heatmaps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            global_step += 1
            wandb.log({"train/loss": loss.item()})
            progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})

            if global_step % LOG_IMAGES_EVERY_N_STEPS == 0:
                try:
                    prompt_text = batch['phrases'][0]
                    img_tensor = batch['pixel_values'][0].cpu() * 0.5 + 0.5
                    unnormalized_image = torch.clamp(img_tensor, 0, 1).permute(1, 2, 0).numpy()
                    gt_heatmap = batch['target_heatmaps'][0].cpu().numpy().reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)
                    pred_heatmap = torch.sigmoid(predicted_heatmap_logits[0]).detach().cpu().numpy().reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f"Prompt: \"{prompt_text}\"", fontsize=12)
                    axes[0].imshow(unnormalized_image); axes[0].set_title("Input"); axes[0].axis('off')
                    axes[1].imshow(gt_heatmap, cmap='viridis', vmin=0, vmax=1); axes[1].set_title("Ground Truth"); axes[1].axis('off')
                    im = axes[2].imshow(pred_heatmap, cmap='viridis', vmin=0, vmax=1); axes[2].set_title("Prediction"); axes[2].axis('off')
                    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
                    wandb.log({"examples/comparison": wandb.Image(fig)})
                    plt.close(fig)
                except Exception as e:
                    print(f"\nWarning: Could not log image at step {global_step}. Error: {e}")
    
    print("--- Sweep Run Finished ---")
    wandb.finish()

if __name__ == "__main__" :
    train_one_run()
