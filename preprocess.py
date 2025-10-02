from datasets import load_dataset
from PIL import Image
import os, itertools, math, random, json, re, glob
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

# ── SETTINGS ───────────────────────────────────────────────────────
PATCHES      = 16
MIN_CROP     = 0.8
NSD_REF      = 425
OUT_DIR      = "preprocessed_dataset_full"     # will create train/ and val/
CKPT         = "google/siglip2-base-patch16-naflex"

VAL_N        = 1000                            # exactly 1000 for validation
TRAIN_MAX    = None                            # None → whole split
START_FROM   = None                            # e.g. 12345 to resume from this image index
AUTO_RESUME  = True                            # if True and START_FROM is None, infer from files

os.makedirs(os.path.join(OUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "val"), exist_ok=True)

with open("./words/pmi.json", "r") as f:
    PMI = json.load(f)

# ── HELPERS ────────────────────────────────────────────────────────
def _infer_start_idx(path):
    files = glob.glob(os.path.join(path, "*.npz"))
    if not files: return 0
    # filenames start with zero-padded index: "00012_cat.npz" or "00012_dog__neg.npz"
    pat = re.compile(r"^(\d+)_")
    idxs = []
    for f in files:
        m = pat.search(os.path.basename(f))
        if m: idxs.append(int(m.group(1)))
    return (max(idxs) + 1) if idxs else 0

def stretch_to_multiple(img):
    W, H = img.size
    newW = ((W + PATCHES - 1) // PATCHES) * PATCHES
    newH = ((H + PATCHES - 1) // PATCHES) * PATCHES
    return img if (newW, newH)==(W,H) else img.resize((newW, newH), Image.Resampling.BILINEAR)

def snap_bbox_to_tiles(bbox, tile, W, H, img_w, img_h, crop_x0=0, crop_y0=0):
    x, y, w, h = map(float, bbox)
    sx = img_w / NSD_REF; sy = img_h / NSD_REF
    x = x * sx - crop_x0; y = y * sy - crop_y0
    w = w * sx; h = h * sy
    x0 = int(math.floor(x / tile) * tile); y0 = int(math.floor(y / tile) * tile)
    x1 = int(math.ceil((x + w) / tile) * tile); y1 = int(math.ceil((y + h) / tile) * tile)
    x0 = max(0, min(x0, W)); y0 = max(0, min(y0, H))
    x1 = max(0, min(x1, W)); y1 = max(0, min(y1, H))
    return x0, y0, x1, y1

def make_tile_binary_heatmap(W, H, tile, bboxes, original_w, original_h, crop_x0=0, crop_y0=0):
    gh, gw = H // tile, W // tile
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for bbox in bboxes:
        x0, y0, x1, y1 = snap_bbox_to_tiles(bbox, tile, W, H, original_w, original_h, crop_x0, crop_y0)
        if x1 <= x0 or y1 <= y0: continue
        tx0, ty0 = x0 // tile, y0 // tile
        tx1, ty1 = (x1 - 1) // tile, (y1 - 1) // tile
        grid[ty0:ty1+1, tx0:tx1+1] = 1
    mask = (np.kron(grid, np.ones((tile, tile), dtype=np.uint8)) * 255).astype(np.uint8)
    return mask[:H, :W], grid

def crop_image(img):
    W, H = img.size
    sw = random.uniform(MIN_CROP, 1.0); sh = random.uniform(MIN_CROP, 1.0)
    newW = max(PATCHES, min((int(W * sw) // PATCHES) * PATCHES, W))
    newH = max(PATCHES, min((int(H * sh) // PATCHES) * PATCHES, H))
    x0 = (random.randint(0, W - newW) // PATCHES) * PATCHES
    y0 = (random.randint(0, H - newH) // PATCHES) * PATCHES
    return img.crop((x0, y0, x0 + newW, y0 + newH)), x0, y0

# ── ENCODERS ───────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32
siglip = AutoModel.from_pretrained(CKPT, torch_dtype=dtype).to(device).eval()
processor = AutoProcessor.from_pretrained(CKPT)

@torch.no_grad()
def encode_image(pil_img):
    gh, gw = pil_img.height // PATCHES, pil_img.width // PATCHES
    batch = processor(images=pil_img, return_tensors="pt", do_resize=False, max_num_patches=4096)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = siglip.vision_model(
        pixel_values=batch["pixel_values"],
        attention_mask=batch["pixel_attention_mask"],
        spatial_shapes=batch["spatial_shapes"],
    )
    feats = out.last_hidden_state[0].float()
    mask  = batch["pixel_attention_mask"][0].bool()
    feats = feats[mask]
    T, D = feats.shape
    assert T == gh * gw, f"Token count {T} != {gh}*{gw} ({gh*gw})"
    return feats.view(gh, gw, D).cpu().half().numpy()

@torch.no_grad()
def encode_text(text: str):
    toks = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    toks = {k: v.to(device) for k, v in toks.items() if k in ("input_ids", "attention_mask")}
    emb = siglip.text_model(**toks).pooler_output
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.squeeze(0).cpu().half().numpy()

# ── CORE ───────────────────────────────────────────────────────────
def process_split(split_name: str, max_items, subdir: str, start_from=None, auto_resume=True):
    out_dir = os.path.join(OUT_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)

    # figure starting index
    if start_from is None and auto_resume:
        start_from = _infer_start_idx(out_dir)
    if start_from is None:
        start_from = 0

    ds = load_dataset("clane9/NSD-Flat", split=split_name, streaming=True)

    # streaming skip
    it = itertools.islice(ds, start_from, None if max_items is None else start_from + max_items)

    try:
        for j, sample in enumerate(it, start=start_from):
            img = stretch_to_multiple(sample["image"])
            original_w, original_h = img.size

            labels  = sample.get("objects", {})
            objects = labels.get("category", [])
            bboxes  = labels.get("bbox", [])
            unique_objects = set(objects)

            for cat in unique_objects:
                cropped_img, cx0, cy0 = crop_image(img)
                W, H = cropped_img.size
                cat_bboxes = [bb for obj, bb in zip(objects, bboxes) if obj == cat]
                _, grid01 = make_tile_binary_heatmap(W, H, PATCHES, cat_bboxes, original_w, original_h, cx0, cy0)
                heat01 = grid01.astype(np.uint8)

                pos_txt_emb = encode_text(cat)
                img_tokens  = encode_image(cropped_img)
                gh, gw, _ = img_tokens.shape
                assert heat01.shape == (gh, gw)

                base = f"{j:06d}_{cat.replace(' ', '_')}"
                np.savez_compressed(
                    os.path.join(out_dir, f"{base}.npz"),
                    heatmap=heat01,
                    img_tokens=float16_safe(img_tokens:=img_tokens),
                    txt_emb=pos_txt_emb,
                )

                d = PMI.get(cat, {})
                neg_word = (min(d, key=d.get) if d else "none")
                neg_txt_emb = encode_text(neg_word)
                np.savez_compressed(
                    os.path.join(out_dir, f"{j:06d}_{neg_word.replace(' ', '_')}__neg.npz"),
                    heatmap=np.zeros_like(heat01, dtype=np.uint8),
                    img_tokens=img_tokens,
                    txt_emb=neg_txt_emb,
                )

            print(f"[{subdir}] saved image #{j}")
    except KeyboardInterrupt:
        print(f"\n[{subdir}] interrupted at image #{j}. You can resume from {j}.")

# small helper to keep dtype consistent in npz
def float16_safe(x):
    return x.astype(np.float16, copy=False)

# ── MAIN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # TRAIN: full dataset
    process_split("train", TRAIN_MAX, "train", start_from=START_FROM, auto_resume=AUTO_RESUME)

    # VAL: exactly 1000
    process_split("test", VAL_N, "val", start_from=START_FROM, auto_resume=AUTO_RESUME)
