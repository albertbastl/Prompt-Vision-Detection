#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
IMAGE_PATH      = "persons.jpg"
PROMPT          = "person"
DECODER_WEIGHTS = "decoder_epoch10.pth"
MODEL_ID        = "google/siglip2-base-patch16-224"
GRID_SIZE       = 14
EMBED_DIM       = 768
NUM_HEADS       = 2
NUM_LAYERS      = 4
TEXT_MAX_LEN    = 64
OUTPUT_PATH     = None  # or set to "my_output.png"

# ─── HEATMAP FILTERING THRESHOLDS ─────────────────────────────────────────────
ABS_THRESH  = 0.00  # absolute threshold: remove values < this
REL_THRESH  = 0.00  # relative threshold: remove values < 25% of max

# ─── MODEL DEFINITION ─────────────────────────────────────────────────────────
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
        self.head = nn.Linear(embed_dim, grid_size * grid_size)

    def forward(self, text_vec, patch_vecs):
        q = self.query_token + text_vec.unsqueeze(1)    # (B,1,D)
        out = self.decoder(tgt=q, memory=patch_vecs)     # (B,1,D)
        return self.head(out.squeeze(1))                # (B, G*G)

# ─── MAIN SCRIPT ──────────────────────────────────────────────────────────────
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load processor & tokenizer
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load frozen SigLIP model
    siglip = AutoModel.from_pretrained(MODEL_ID).to(device)
    for p in siglip.parameters():
        p.requires_grad = False

    # Load trained decoder
    decoder = LocalizationDecoder(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        grid_size=GRID_SIZE
    ).to(device)
    decoder.load_state_dict(torch.load(DECODER_WEIGHTS, map_location=device))
    decoder.eval()

    # Load and prepare image
    image = Image.open(IMAGE_PATH).convert('RGB')
    image_inputs = processor(images=image, return_tensors='pt').to(device)

    # Tokenize prompt
    text_inputs = tokenizer(
        text=[PROMPT],
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_attention_mask=True,
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        v = siglip.vision_model(pixel_values=image_inputs['pixel_values']).last_hidden_state
        t = siglip.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        ).pooler_output
        logits = decoder(t, v)
        probs  = torch.sigmoid(logits)

        # ─── HEATMAP THRESHOLDING ─────────────────────────────────────────────
        abs_mask = probs >= ABS_THRESH
        rel_mask = probs >= REL_THRESH * probs.max()
        probs    = probs * (abs_mask & rel_mask)  # zero out low-activation cells

    heatmap = probs.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)

    # Optional: gamma correction to enhance mid-tone contrast
    heatmap = np.power(heatmap, 0.8)

    # Upsample to image resolution
    W, H = image.size
    cell_w, cell_h = W // GRID_SIZE, H // GRID_SIZE
    mask_up = np.kron(heatmap, np.ones((cell_h, cell_w)))

    # Plot & overlay
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    heatmap_img = ax.imshow(
        mask_up,
        cmap='plasma',        # perceptually uniform colormap
        alpha=0.6,
        extent=(0, W, H, 0),
        interpolation='bicubic',
        vmin=0.0,              # linear color scale
        vmax=1.0
    )

    ax.axis('off')
    ax.set_title(PROMPT, color='white', backgroundcolor='black')

    # Add colorbar
    cbar = plt.colorbar(heatmap_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Confidence', rotation=270, labelpad=15)

    # Save
    out_path = OUTPUT_PATH or f"overlay_{os.path.basename(IMAGE_PATH)}"
    if not out_path.lower().endswith(('.png', '.jpg')):
        out_path += '.png'
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"✅ Saved overlay to: {out_path}")

if __name__ == '__main__':
    main()
