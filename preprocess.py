from datasets import load_dataset
from PIL import Image
import os, itertools, math
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

PATCHES      = 16
MAX_PATCHES  = 410
OUT_DIR      = "pd_410patches_openvocab"
CKPT         = "google/siglip2-base-patch16-naflex"
VAL_N        = 3000
TRAIN_MAX    = 30000

def to_multiple(v, m=PATCHES):
    return int(round(v / m) * m)

def fit_image_to_patch_budget(img: Image.Image, max_patches: int) -> Image.Image:
    W0, H0 = img.size
    W, H = to_multiple(W0), to_multiple(H0)
    gh, gw = H // PATCHES, W // PATCHES
    patches = gh * gw
    if patches <= max_patches:
        return img.resize((W, H), Image.Resampling.BILINEAR) if (W, H) != (W0, H0) else img
    scale = math.sqrt(patches / max_patches)
    newW = max(PATCHES, to_multiple(int(W / scale)))
    newH = max(PATCHES, to_multiple(int(H / scale)))
    return img.resize((newW, newH), Image.Resampling.BILINEAR)

def snap_bbox_to_tiles(bbox, tile, W, H):
    x, y, w, h = map(float, bbox)
    x0 = int(math.floor(x / tile) * tile)
    y0 = int(math.floor(y / tile) * tile)
    x1 = int(math.ceil((x + w) / tile) * tile)
    y1 = int(math.ceil((y + h) / tile) * tile)
    x0 = max(0, min(x0, W)); y0 = max(0, min(y0, H))
    x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
    return x0, y0, x1, y1

def make_tile_binary_heatmap(W, H, tile, bboxes):
    gh, gw = H // tile, W // tile
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for bbox in bboxes:
        x0, y0, x1, y1 = snap_bbox_to_tiles(bbox, tile, W, H)
        if x1 <= x0 or y1 <= y0: continue
        tx0, ty0 = x0 // tile, y0 // tile
        tx1 = (x1 - 1) // tile
        grid[ty0:ty0+1, tx0:tx1+1] = 1 
    return grid

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
siglip = AutoModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(CKPT)

@torch.no_grad()
def encode_image(pil_img):
    gh, gw = pil_img.height // PATCHES, pil_img.width // PATCHES
    batch = processor(images=pil_img, return_tensors="pt", do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(**{
        "pixel_values": batch["pixel_values"],
        "attention_mask": batch["pixel_attention_mask"],
        "spatial_shapes": batch["spatial_shapes"],
    })
    feats = out.last_hidden_state[0].float()[batch["pixel_attention_mask"][0].bool()]
    grid_feats = feats.view(gh, gw, -1)
    grid_feats_normalized = torch.nn.functional.normalize(grid_feats, p=2, dim=-1)
    return grid_feats_normalized.cpu().half().numpy()

@torch.no_grad()
def encode_text(text: str):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().half().numpy()

def process_split(split_name: str, max_items, subdir: str):
    out_dir = os.path.join(OUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    
    ds = load_dataset("vikhyatk/openimages-bbox", split=split_name, streaming=True)
    it = itertools.islice(ds, max_items) if max_items is not None else ds

    for j, sample in enumerate(it):
        raw_img = sample["image"].convert("RGB")
        img = fit_image_to_patch_budget(raw_img, MAX_PATCHES)
        W, H = img.size 
        img_tokens = encode_image(img)
        
        objects, bboxes = [], []
        for obj in sample.get("objects", []):
            if obj['label']:
                objects.append(obj['label'])
                nx0, ny0, nx1, ny1 = obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']
                bboxes.append([nx0 * W, ny0 * H, (nx1 - nx0) * W, (ny1 - ny0) * H])
        
        unique_objects = set(objects)
        if not unique_objects: 
            print(f"[{subdir}] SKIPPED image #{j} (no valid objects found)")
            continue
        
        all_masks, all_txt_embs = [], []
        for cat in unique_objects:
            cat_bboxes = [bb for obj, bb in zip(objects, bboxes) if obj == cat]
            all_masks.append(make_tile_binary_heatmap(W, H, PATCHES, cat_bboxes))
            all_txt_embs.append(encode_text(cat))

        if all_masks:
            base = f"{j:06d}"
            np.savez_compressed(
                os.path.join(out_dir, f"{base}.npz"),
                img_tokens=img_tokens.astype(np.float16, copy=False),
                masks=np.stack(all_masks, axis=0),
                txt_embs=np.stack(all_txt_embs, axis=0),
            )
            print(f"[{subdir}] saved image #{j} with {len(all_masks)} objects")

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "val"), exist_ok=True)
    process_split("train", TRAIN_MAX, "train")
    process_split("validation", VAL_N, "val")