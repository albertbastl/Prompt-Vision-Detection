from datasets import load_dataset
from PIL import Image, ImageOps
import os, itertools, math, random, json
import numpy as np

# NEW: embeddings
import torch
from transformers import AutoModel, AutoProcessor

# ─── SETTINGS ──────────────────────────────────────────────────────
N = 1000
PATCHES = 16
MIN_CROP = 0.8
NSD_REF = 425
OUT_DIR = "test"
CKPT = "google/siglip2-base-patch16-naflex"

os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD PMI (least-similar negatives) ────────────────────────────
with open("./words/pmi.json", "r") as f:
    PMI = json.load(f)

# ─── HELPERS ──────────────────────────────────────────────────────
def stretch_to_multiple(img):
    """Resize image so that width and height are multiples of PATCHES."""
    W, H = img.size
    newW = ((W + PATCHES - 1) // PATCHES) * PATCHES
    newH = ((H + PATCHES - 1) // PATCHES) * PATCHES
    if (newW, newH) == (W, H):
        return img
    return img.resize((newW, newH), Image.Resampling.BILINEAR)

def snap_bbox_to_tiles(bbox, tile, W, H, img_w, img_h, crop_x0=0, crop_y0=0):
    """COCO bbox [x,y,w,h] → (x0,y0,x1,y1) snapped to tile grid and clipped."""
    x, y, w, h = map(float, bbox)
    sx = img_w / NSD_REF
    sy = img_h / NSD_REF
    x = x * sx - crop_x0
    y = y * sy - crop_y0
    w = w * sx
    h = h * sy
    x0 = int(math.floor(x / tile) * tile)
    y0 = int(math.floor(y / tile) * tile)
    x1 = int(math.ceil((x + w) / tile) * tile)
    y1 = int(math.ceil((y + h) / tile) * tile)
    x0 = max(0, min(x0, W)); y0 = max(0, min(y0, H))
    x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
    return x0, y0, x1, y1

def make_tile_binary_heatmap(W, H, tile, bboxes, original_w, original_h, crop_x0=0, crop_y0=0):
    """
    Returns:
      mask : (H, W) uint8 binary mask {0,255}, with tiles containing a bbox marked.
      grid : (H//tile, W//tile) uint8 binary array {0,1}.
    """
    gh, gw = H // tile, W // tile
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for bbox in bboxes:
        x0, y0, x1, y1 = snap_bbox_to_tiles(bbox, tile, W, H, original_w, original_h, crop_x0, crop_y0)
        if x1 <= x0 or y1 <= y0:
            continue
        tx0, ty0 = x0 // tile, y0 // tile
        tx1, ty1 = (x1 - 1) // tile, (y1 - 1) // tile
        grid[ty0:ty1+1, tx0:tx1+1] = 1
    mask = (np.kron(grid, np.ones((tile, tile), dtype=np.uint8)) * 255).astype(np.uint8)
    mask = mask[:H, :W]
    return mask, grid

def crop_image(img):
    """Random crop with size aligned to PATCHES and *origin snapped to PATCHES*. Returns cropped image AND (x0,y0)."""
    W, H = img.size
    scale_w = random.uniform(MIN_CROP, 1.0)
    scale_h = random.uniform(MIN_CROP, 1.0)
    newW = int(W * scale_w); newH = int(H * scale_h)
    newW = (newW // PATCHES) * PATCHES; newH = (newH // PATCHES) * PATCHES
    newW = max(PATCHES, min(newW, W)); newH = max(PATCHES, min(newH, H))
    x0 = (random.randint(0, W - newW) // PATCHES) * PATCHES
    y0 = (random.randint(0, H - newH) // PATCHES) * PATCHES
    return img.crop((x0, y0, x0 + newW, y0 + newH)), x0, y0

# ─── SIGLIP2 ENCODERS ──────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
siglip = AutoModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(CKPT)

@torch.no_grad()
def encode_image(pil_img):
    gh, gw = pil_img.height // PATCHES, pil_img.width // PATCHES  # grid size
    batch = processor(
        images=pil_img,
        return_tensors="pt",
        do_resize=False,
        max_num_patches=4096,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    out = siglip.vision_model(
        pixel_values=batch["pixel_values"],
        attention_mask=batch["pixel_attention_mask"],
        spatial_shapes=batch["spatial_shapes"],
    )

    feats = out.last_hidden_state[0].float()        # [T, D]
    mask  = batch["pixel_attention_mask"][0].bool() # [T]
    feats = feats[mask]                              # [T_real, D]

    # expect one token per 16×16 patch
    T, D = feats.shape
    print(D)
    assert T == gh * gw, f"Token count {T} != {gh}*{gw} ({gh*gw})"

    # reshape to a spatial grid so a CNN can read it directly
    feats = feats.view(gh, gw, D)

    return feats.cpu().half().numpy()  # [gh, gw, D], float16


@torch.no_grad()
def encode_text(text: str):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output           # [1, D]
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().half().numpy()              # [D]


# ─── MAIN LOOP ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ds = load_dataset("clane9/NSD-Flat", split="train", streaming=True)

    for i, sample in enumerate(itertools.islice(ds, N)):
        img = stretch_to_multiple(sample["image"])
        original_w, original_h = img.size  # stretched dims

        labels  = sample.get("objects", {})
        objects = labels.get("category", [])
        bboxes  = labels.get("bbox", [])
        unique_objects = set(objects)

        for cat in unique_objects:
            # crop once per (image, category)
            cropped_img, cx0, cy0 = crop_image(img)
            W, H = cropped_img.size

            # all boxes of this category (in original coords)
            cat_bboxes = [bb for obj, bb in zip(objects, bboxes) if obj == cat]

            # heatmap for this category in this crop
            mask255, grid01 = make_tile_binary_heatmap(W, H, PATCHES, cat_bboxes, original_w, original_h, cx0, cy0)
            heat01 = grid01.astype(np.uint8)  # <- shape (H//16, W//16)
        
            pos_txt_emb = encode_text(cat)                      # [D]
            img_tokens = encode_image(cropped_img)  # [gh, gw, D] float16

            gh, gw, D = img_tokens.shape
            assert heat01.shape == (gh, gw), f"heatmap {heat01.shape} != tokens {(gh, gw)}"

            # ----- save POSITIVE -----
            out_base = f"{i:05d}_{cat.replace(' ', '_')}"
            np.savez_compressed(
                os.path.join(OUT_DIR, f"{out_base}.npz"),
                heatmap=heat01.astype(np.uint8),  # HxW {0,1}
                img_tokens=img_tokens,            # [gh, gw, D] float16
                txt_emb=pos_txt_emb,              # [D] float16
            )

            # ----- save NEGATIVE -----
            d = PMI.get(cat, {})
            neg_word = (min(d, key=d.get) if d else "none")
            neg_txt_emb = encode_text(neg_word)
            np.savez_compressed(
                os.path.join(OUT_DIR, f"{i:05d}_{neg_word.replace(' ', '_')}__neg.npz"),
                heatmap=np.zeros_like(heat01, dtype=np.uint8),  # HxW zeros
                img_tokens=img_tokens,                           # same tokens
                txt_emb=neg_txt_emb,                             # negative text emb
            )


            print(f"Saved {i} image pair.")
