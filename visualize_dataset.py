import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

# ─── ADJUST THIS ─────────────────────────────────────────────────────────────
BATCH_FILE = "processed_batches/image_batch_009012.pkl"  # path to your .pkl
NUM_DISPLAY = 5                                        # how many samples to show
GRID_SIZE = 14                                         # must match your preprocess

# ─── VISUALIZATION FUNCTION ─────────────────────────────────────────────────
def visualize_batch(batch_path, num_display=5):
    """Load a .pkl and display the first `num_display` image/mask/prompt triples."""
    with open(batch_path, "rb") as f:
        samples = pickle.load(f)
    
    for sample in samples[:num_display]:
        # ---- image ----
        img_t = sample["pixel_values"]                  # torch.Tensor (3,H,W)
        img_np = img_t.cpu().numpy().transpose(1,2,0)    # (H,W,3)

        # ---- mask ----
        hm_flat = sample["target_heatmap"]               # torch.Tensor (G*G,)
        hm = hm_flat.cpu().numpy().reshape(GRID_SIZE, GRID_SIZE)
        H, W, _ = img_np.shape
        cell_h, cell_w = H//GRID_SIZE, W//GRID_SIZE
        mask_up = np.kron(hm, np.ones((cell_h,cell_w)))  # upsample

        # ---- prompt ----
        prompt = sample["phrase"]

        # ---- plot ----
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
        ax1.imshow(img_np)
        ax1.set_title(prompt)
        ax1.axis("off")

        ax2.imshow(mask_up, cmap="gray", interpolation="nearest")
        ax2.set_title("Target Heatmap")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_batch(BATCH_FILE, NUM_DISPLAY)
