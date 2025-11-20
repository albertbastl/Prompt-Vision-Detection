import os, math, itertools
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from datasets import load_dataset
from transformers import AutoModel, AutoProcessor
from tqdm.auto import tqdm


CKPT = "google/siglip2-base-patch16-naflex"
OUT_DIR = "pd_30k_500patches_imgnorm"
TARGET_PATCHES = 500
PATCH_SIZE = 16

TRAIN_COUNT = 30000
VAL_COUNT = 3000


print("Loading SigLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
siglip = AutoModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(CKPT)
print(f"Models loaded on {device}")

@torch.no_grad()
def encode_image(pil_img):
    gh, gw = pil_img.height // PATCH_SIZE, pil_img.width // PATCH_SIZE
    
    batch = processor(images=pil_img, return_tensors="pt", do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(**{
        "pixel_values": batch["pixel_values"],
        "attention_mask": batch["pixel_attention_mask"],
        "spatial_shapes": batch["spatial_shapes"],
    })
    
    feats = out.last_hidden_state[0].float()[batch["pixel_attention_mask"][0].bool()]
    # grid_feats_normalized = F.normalize(feats, p=2, dim=-1)
    
    return feats.cpu().numpy(), (gh, gw)

@torch.no_grad()
def encode_text(text: str):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().numpy()

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

factor_pairs = get_factor_pairs(TARGET_PATCHES)

for split_name, split_count in [("train", TRAIN_COUNT), ("validation", VAL_COUNT)]:
    
    print(f"\n--- Processing {split_name} split ---")
    out_path_dir = os.path.join(OUT_DIR, split_name)
    os.makedirs(out_path_dir, exist_ok=True)
    
    ds = load_dataset("vikhyatk/openimages-bbox", split=split_name, streaming=True)
    ds_iter = itertools.islice(ds, split_count)
    
    sample_idx = 0

    for sample in tqdm(ds_iter, total=split_count, desc=f"Processing {split_name}"):
        try:
            raw_img = sample["image"].convert("RGB")
            W_orig, H_orig = raw_img.size
                
            A_orig = W_orig / H_orig
            
            best_pair = (0, 0)
            min_diff = float('inf')
            
            for gh_cand, gw_cand in factor_pairs:
                if gh_cand == 0: continue
                A_grid = gw_cand / gh_cand
                diff = abs(A_grid - A_orig)
                
                if diff < min_diff:
                    min_diff = diff
                    best_pair = (gh_cand, gw_cand)
            
            gh, gw = best_pair
            
            if gh == 0 or gw == 0:
                print(f"Could not find valid grid for {W_orig}x{H_orig}. Skipping.")
                continue
            
            W_new = gw * PATCH_SIZE
            H_new = gh * PATCH_SIZE
            
            img_resized = raw_img.resize((W_new, H_new), Image.Resampling.BILINEAR)

            img_patch_embs, (gh_out, gw_out) = encode_image(img_resized)
            
            gh, gw = gh_out, gw_out
            
            objects_in_image = {}
            for obj in sample.get("objects", []):
                label = obj.get('label')

                if label not in objects_in_image:
                    objects_in_image[label] = []
                
                nx0, ny0, nx1, ny1 = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
                bbox_resized = [nx0 * W_new, ny0 * H_new, (nx1 - nx0) * W_new, (ny1 - ny0) * H_new]
                objects_in_image[label].append(bbox_resized)
                
            for label, bboxes in objects_in_image.items():
                
                txt_emb = encode_text(label)
                
                grid_heatmap = np.zeros((gh, gw), dtype=np.uint8)
                for bbox in bboxes:
                    x, y, w, h = bbox
                    tx0 = max(0, min(gw - 1, int(x / PATCH_SIZE)))
                    ty0 = max(0, min(gh - 1, int(y / PATCH_SIZE)))
                    tx1 = max(0, min(gw - 1, int((x + w) / PATCH_SIZE)))
                    ty1 = max(0, min(gh - 1, int((y + h) / PATCH_SIZE)))
                    
                    grid_heatmap[ty0:ty1+1, tx0:tx1+1] = 1
                
                labels_flat = grid_heatmap.flatten()
                
                out_path = os.path.join(out_path_dir, f"{sample_idx:08d}.npz")
                np.savez_compressed(
                    out_path,
                    txt_emb=txt_emb.astype(np.float16),
                    img_patch_embs=img_patch_embs.astype(np.float16),
                    labels=labels_flat.astype(np.uint8),
                    grid_hw=np.array([gh, gw], dtype=np.int16)
                )
                sample_idx += 1
                
        except Exception as e:
            print(f"Error on sample, skipping: {e}")
            continue
            
    print(f"--- Finished {split_name} split, {sample_idx} files saved. ---")

print("\n--- Dataset Generation Complete ---")