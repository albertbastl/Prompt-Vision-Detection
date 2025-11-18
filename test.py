import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import sys
from PIL import Image

# --- Safety Import for Transformers ---
try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    print("\n[ERROR] 'transformers' library not found.")
    print("Please run: pip install transformers\n")
    sys.exit(1)

# --- Configuration ---
MODEL_PATH = "miou_siglip.pt"   # Path to your trained weights
IMAGE_PATH = "imgs/park.jpg"   # Put a raw image path here
TEXT_QUERY = "human head" # The text to search for

# --- Preprocessing Constants (MUST MATCH YOUR PREPROCESSING SCRIPT) ---
CKPT = "google/siglip2-base-patch16-naflex"
TARGET_PATCHES = 500
PATCH_SIZE = 16

# --- Model Architecture Constants ---
EMBED_DIM = 768
HIDDEN_DIM = 512
DROP_RATE = 0.0 

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Define Your Projector Architecture ---
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
        self.logits_bias = nn.Parameter(torch.ones([]) * -10.0)

    def forward(self, img_embs):
        projected_embs = self.net(img_embs)
        projected_embs = F.normalize(projected_embs, p=2, dim=-1)
        return projected_embs

# --- 2. Preprocessing Helpers (From your script) ---
def get_factor_pairs(n):
    pairs = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            pairs.append((i, n // i))
            if i != n // i:
                pairs.append((n // i, i))
    if not pairs:
        pairs.append((1, n))
    return pairs

def get_grid_size(W_orig, H_orig, target_patches):
    """Calculates the best (gh, gw) to match aspect ratio and target patches."""
    A_orig = W_orig / H_orig
    best_pair = (0, 0)
    min_diff = float('inf')
    
    factor_pairs = get_factor_pairs(target_patches)
    
    for gh_cand, gw_cand in factor_pairs:
        if gh_cand == 0: continue
        A_grid = gw_cand / gh_cand
        diff = abs(A_grid - A_orig)
        
        if diff < min_diff:
            min_diff = diff
            best_pair = (gh_cand, gw_cand)
            
    return best_pair

# --- 3. Backbone Feature Extractor ---
def get_backbone_features(image_path, text, device):
    print(f"Loading SigLIP Backbone: {CKPT}...")
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load HF Model
    try:
        siglip = AutoModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
        processor = AutoProcessor.from_pretrained(CKPT)
    except Exception as e:
        print(f"[ERROR] Failed to load HuggingFace model: {e}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        sys.exit(1)

    # 1. Resize Image Logic (Matching your preprocessing)
    raw_img = Image.open(image_path).convert("RGB")
    W_orig, H_orig = raw_img.size
    
    gh, gw = get_grid_size(W_orig, H_orig, TARGET_PATCHES)
    
    if gh == 0 or gw == 0:
        print("Error: Could not calculate valid grid.")
        sys.exit(1)
        
    W_new = gw * PATCH_SIZE
    H_new = gh * PATCH_SIZE
    
    # Resize image for the model
    img_resized = raw_img.resize((W_new, H_new), Image.Resampling.BILINEAR)
    
    print(f"Original: {W_orig}x{H_orig} -> Grid: {gh}x{gw} (Patches: {gh*gw}) -> Resized: {W_new}x{H_new}")

    # 2. Encode Image
    with torch.no_grad():
        batch = processor(images=img_resized, return_tensors="pt", do_resize=False, max_num_patches=4096)
        batch = {k: v.to(device) for k, v in batch.items()}
        
        out = siglip.vision_model(**{
            "pixel_values": batch["pixel_values"],
            "attention_mask": batch["pixel_attention_mask"],
            "spatial_shapes": batch["spatial_shapes"],
        })
        
        # Extract features and cast to float32 for the projector
        feats = out.last_hidden_state[0].float()[batch["pixel_attention_mask"][0].bool()]
        img_embs = F.normalize(feats, p=2, dim=-1)
        
        # Unsqueeze to add batch dimension [1, N_patches, Dim]
        img_embs = img_embs.unsqueeze(0)

    # 3. Encode Text
    with torch.no_grad():
        toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
        
        txt_out = siglip.text_model(**toks).pooler_output
        txt_emb = F.normalize(txt_out, p=2, dim=-1)
        # txt_emb is already [1, Dim]

    # Return raw_img for plotting, embeddings for model, and grid dims for reshaping
    return raw_img, img_embs.float(), txt_emb.float(), (gh, gw)

# --- 4. Visualization Logic ---
def visualize_heatmap():
    # A. Load Trained Model
    print(f"Loading trained projector from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] {MODEL_PATH} not found. Using random weights.")
    
    projector = SimpleProjector(EMBED_DIM, HIDDEN_DIM, EMBED_DIM, DROP_RATE).to(device)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if "image_projector" in checkpoint:
            projector.load_state_dict(checkpoint["image_projector"])
        else:
            projector.load_state_dict(checkpoint)
        print("Weights loaded.")
        
    projector.eval()

    # B. Get Backbone Features
    orig_img, img_patches, txt_emb, (gh, gw) = get_backbone_features(IMAGE_PATH, TEXT_QUERY, device)

    # C. Run Projector
    with torch.no_grad():
        # [1, N, 768]
        projected_patches = projector(img_patches)
        
        # [1, 768]
        norm_txt = F.normalize(txt_emb, p=2, dim=-1)
        
        # Dot product: [1, N]
        sim_scores = torch.matmul(projected_patches, norm_txt.T).squeeze()
        
        # SigLIP Activation
        t = projector.logits_scale.exp()
        b = projector.logits_bias
        probs = torch.sigmoid(sim_scores * t + b)
        probs = probs.cpu().numpy()

    # D. Visualize
    print(f"Visualizing heatmap with grid {gh}x{gw}...")
    print(f"Max Prob: {probs.max():.4f}, Min Prob: {probs.min():.4f}")

    # Reshape 1D patches -> 2D Grid
    try:
        heatmap = probs.reshape(gh, gw)
    except ValueError:
        print(f"Error: Expected {gh*gw} patches, got {probs.shape[0]}. Grid mismatch.")
        return

    # Resize heatmap to original image size
    width, height = orig_img.size
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)

    # Normalize to 0-255
    heatmap_norm = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Overlay
    orig_cv = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
    alpha = 0.5
    overlay = cv2.addWeighted(orig_cv, 1 - alpha, heatmap_color, alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title(f"Query: '{TEXT_QUERY}'")
    plt.imshow(orig_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Heatmap ({gh}x{gw} grid)")
    plt.imshow(heatmap_resized, cmap='jet')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("SigLIP Attention")
    plt.imshow(overlay_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_heatmap()