# predict_heatmap.py (with Overlay Visualization)

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
try:
    from config import MODEL_ID, PATCH_GRID_SIZE, TEXT_MAX_LENGTH
except ImportError:
    print("Error: Could not import from config.py."); exit()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. DECODER ARCHITECTURE (Unchanged) ---
class LocalizationDecoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(embed_dim, PATCH_GRID_SIZE * PATCH_GRID_SIZE)
    def forward(self, text_vector, patch_vectors):
        query = self.query_token + text_vector.unsqueeze(1)
        decoder_output = self.transformer_decoder(tgt=query, memory=patch_vectors)
        return self.output_head(decoder_output.squeeze(1))

# --- 3. MAIN PREDICTION FUNCTION ---
def predict(image_path, prompt_text, decoder_weights_path, output_path):
    print(f"Using device: {DEVICE}")

    # --- Load Models and Processor ---
    print("Loading pre-trained SigLIP model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    siglip_model = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    
    print("Loading trained localization decoder...")
    decoder = LocalizationDecoder().to(DEVICE)
    if not os.path.exists(decoder_weights_path):
        print(f"ERROR: Decoder weights not found at '{decoder_weights_path}'."); return
    decoder.load_state_dict(torch.load(decoder_weights_path, map_location=DEVICE))

    siglip_model.eval(); decoder.eval()

    # --- Prepare Inputs ---
    print(f"Processing image: {image_path}")
    print(f"Using prompt: '{prompt_text}'")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Image file not found at '{image_path}'"); return

    text_inputs = processor(
        text=[prompt_text], padding="max_length", truncation=True,
        max_length=TEXT_MAX_LENGTH, return_tensors="pt"
    ).to(DEVICE)
    image_inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # --- Run Inference ---
    with torch.no_grad():
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        with torch.autocast(device_type=DEVICE, dtype=dtype):
            vision_outputs = siglip_model.vision_model(**image_inputs)
            text_outputs = siglip_model.text_model(**text_inputs)
        
        patch_vectors = vision_outputs.last_hidden_state.float()
        text_vector = text_outputs.pooler_output.float()
        predicted_heatmap_logits = decoder(text_vector, patch_vectors)
        predicted_heatmap = torch.sigmoid(predicted_heatmap_logits)

    # --- Post-process for Visualization ---
    heatmap_grid = predicted_heatmap.cpu().numpy().reshape(PATCH_GRID_SIZE, PATCH_GRID_SIZE)

    # --- CREATE AND SAVE OVERLAY VISUALIZATION ---
    print("Generating overlay visualization...")
    # Create a single plot. We can set the figure size based on the image aspect ratio.
    fig_w, fig_h = 10, 10 * (image.height / image.width)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # 1. Display the original image as the base layer
    ax.imshow(image)
    
    # 2. Overlay the heatmap with transparency
    #    - `extent` stretches the low-res heatmap (e.g., 14x14) over the full image dimensions.
    #    - `alpha` controls transparency (0.6 = 60% transparent).
    #    - `interpolation` makes the stretched heatmap look smooth.
    ax.imshow(
        heatmap_grid, 
        cmap='viridis', 
        alpha=0.6,
        extent=(0, image.width, image.height, 0),
        interpolation='bicubic'
    )

    ax.set_title(f"Prompt: \"{prompt_text}\"", fontsize=14, color='white', backgroundcolor='black')
    ax.axis('off') # Hide the axes and ticks for a clean look

    # Save the final image with a tight bounding box
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"\nSuccess! Output overlay saved to '{output_path}'")
    plt.close(fig)

# --- 4. COMMAND-LINE INTERFACE (Unchanged) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained localization decoder and create an overlay.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("prompt_text", type=str, help="The text prompt describing the object to localize.")
    parser.add_argument("--weights", type=str, default="./siglip2_localization_decoder.pth", help="Path to the trained decoder weights file (.pth).")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output visualization. Defaults to 'output_overlay_[image_filename]'.")
    args = parser.parse_args()
    
    if args.output is None:
        base_name = os.path.basename(args.image_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        args.output = f"./output_overlay_{file_name_no_ext}.png"
    
    predict(args.image_path, args.prompt_text, args.weights, args.output)
