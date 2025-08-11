#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np
from datasets import load_dataset
from PIL import Image
from transformers import AutoTokenizer
import torch

# ─── CONSTANT PARAMETERS ──────────────────────────────────────────────────────
NUM_SAMPLES   = 3000     # how many images to process
SQUARE_SIZE   =  224     # final square resize (like model input)
GRID_SIZE     =   14     # number of grid cells per side (14×14 = 196)
TEXT_MAX_LEN  =   64     # max token length for phrases
MODEL_ID      = "google/siglip2-base-patch16-224"
DATASET       = "clane9/NSD-Flat"
SAVE_DIR      = "dataset_1_2_5_3000"

# crop policy (simple + stable)
BIG_MIN_FRAC  = 0.60     # of min(H, W)
BIG_MAX_FRAC  = 0.90
SM_MIN_FRAC   = 0.20
SM_MAX_FRAC   = 0.40
BIG_CROPS     = 2
SMALL_CROPS   = 5        # will create 1 small positive + (SMALL_CROPS-1) small negatives
NEG_MAX_TRIES = 50       # attempts to find a clean negative crop (no overlap with GT)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_square_size(min_frac, max_frac, W, H):
    side = int(round(random.uniform(min_frac, max_frac) * min(W, H)))
    side = max(1, min(side, min(W, H)))
    return side

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = max(0, min(x0, W-1))
    y0 = max(0, min(y0, H-1))
    x1 = max(x0+1, min(x1, W))
    y1 = max(y0+1, min(y1, H))
    return x0, y0, x1, y1

def crop_from_center(cx, cy, side, W, H):
    x0 = int(round(cx - side/2))
    y0 = int(round(cy - side/2))
    x1 = x0 + side
    y1 = y0 + side
    return clamp_box(x0, y0, x1, y1, W, H)

def rect_intersection_area(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    return iw * ih

def overlaps_any(crop, boxes):
    # boxes are (x,y,w,h) floats
    cx0, cy0, cx1, cy1 = crop
    for (bx, by, bw, bh) in boxes:
        if rect_intersection_area((cx0, cy0, cx1, cy1), (int(bx), int(by), int(bx+bw), int(by+bh))) > 0:
            return True
    return False

def sample_big_positive_crop(cls_boxes, W, H):
    """Square big crop likely containing the class; if ≥2 boxes, try to cover a pair."""
    side = get_square_size(BIG_MIN_FRAC, BIG_MAX_FRAC, W, H)
    if len(cls_boxes) >= 2:
        (bx1,by1,bw1,bh1), (bx2,by2,bw2,bh2) = random.sample(cls_boxes, 2)
        x0 = min(bx1, bx2); y0 = min(by1, by2)
        x1 = max(bx1+bw1, bx2+bw2); y1 = max(by1+bh1, by2+bh2)
        # center on union center; ensure desired side
        cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
        return crop_from_center(cx, cy, side, W, H)
    else:
        (bx, by, bw, bh) = random.choice(cls_boxes)
        cx = bx + bw/2.0; cy = by + bh/2.0
        # small jitter so it isn't always the exact same window
        jitter = 0.1 * side
        cx += random.uniform(-jitter, jitter)
        cy += random.uniform(-jitter, jitter)
        return crop_from_center(cx, cy, side, W, H)

def sample_small_positive_crop(cls_boxes, W, H):
    """Small crop centered on a random GT box with slight jitter."""
    side = get_square_size(SM_MIN_FRAC, SM_MAX_FRAC, W, H)
    (bx, by, bw, bh) = random.choice(cls_boxes)
    cx = bx + bw/2.0; cy = by + bh/2.0
    jitter = 0.15 * side
    cx += random.uniform(-jitter, jitter)
    cy += random.uniform(-jitter, jitter)
    return crop_from_center(cx, cy, side, W, H)

def sample_small_negative_crop(cls_boxes, W, H):
    """Small crop with no overlap with any GT box for this class."""
    side = get_square_size(SM_MIN_FRAC, SM_MAX_FRAC, W, H)
    for _ in range(NEG_MAX_TRIES):
        x0 = random.randint(0, max(0, W - side))
        y0 = random.randint(0, max(0, H - side))
        crop = (x0, y0, x0 + side, y0 + side)
        if not overlaps_any(crop, cls_boxes):
            return crop
    # fallback (rare): allow overlap if we failed to find a clean negative
    x0 = random.randint(0, max(0, W - side))
    y0 = random.randint(0, max(0, H - side))
    return (x0, y0, x0 + side, y0 + side)

def gridify_fixed(mask, grid_size):
    H, W = mask.shape
    assert H == W, "Mask must be square"
    assert H % grid_size == 0, "Size must be divisible by grid_size"
    cell = H // grid_size
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for i in range(grid_size):
        for j in range(grid_size):
            if mask[i*cell:(i+1)*cell, j*cell:(j+1)*cell].any():
                grid[i, j] = 1
    return grid

def encode_phrase(phrase, tokenizer):
    enc = tokenizer(
        phrase,
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
        return_attention_mask=True
    )
    return enc["input_ids"][0], enc["attention_mask"][0]

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    stream = load_dataset(
        DATASET,
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=1000).take(NUM_SAMPLES)

    batch_counter = 0
    for sample_idx, sample in enumerate(stream):
        pil_img = sample["image"]
        W, H    = pil_img.size
        img_arr = np.array(pil_img)

        raw_bboxes = sample["objects"]["bbox"]
        labels     = sample["objects"]["category"]

        # scale from dataset's 425 coordinate frame to real image size
        sx, sy     = W / 425, H / 425
        scaled = [(x*sx, y*sy, bw*sx, bh*sy) for x,y,bw,bh in raw_bboxes]

        # group boxes by class
        by_class = {}
        for (bx,by,bw,bh), lab in zip(scaled, labels):
            by_class.setdefault(lab, []).append((bx,by,bw,bh))

        image_samples = []

        for prompt in sorted(set(labels)):
            cls_boxes = by_class[prompt]

            # build full-image mask for this prompt
            full_mask = np.zeros((H, W), dtype=np.uint8)
            for (bx,by,bw,bh) in cls_boxes:
                x0, y0 = int(bx), int(by)
                x1, y1 = int(bx + bw), int(by + bh)
                full_mask[y0:y1, x0:x1] = 1

            # resize to square & gridify once per prompt
            sq = SQUARE_SIZE
            full_img_sq  = np.array(Image.fromarray(img_arr).resize((sq, sq), Image.BICUBIC))
            full_mask_sq = np.array(Image.fromarray(full_mask).resize((sq, sq), Image.NEAREST))
            full_grid    = gridify_fixed(full_mask_sq, GRID_SIZE)
            input_ids, attention_mask = encode_phrase(prompt, tokenizer)

            # 0) full-image sample (always)
            image_samples.append({
                'pixel_values': torch.tensor(full_img_sq).permute(2,0,1).float() / 255,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_heatmap': torch.tensor(full_grid.flatten(), dtype=torch.float32),
                'phrase': prompt
            })

            # 1) two BIG positive crops
            for _ in range(BIG_CROPS):
                x0, y0, x1, y1 = sample_big_positive_crop(cls_boxes, W, H)
                crop_img  = img_arr[y0:y1, x0:x1]
                crop_mask = full_mask[y0:y1, x0:x1]

                crop_img_sq  = np.array(Image.fromarray(crop_img).resize((sq, sq), Image.BICUBIC))
                crop_mask_sq = np.array(Image.fromarray(crop_mask).resize((sq, sq), Image.NEAREST))
                crop_grid    = gridify_fixed(crop_mask_sq, GRID_SIZE)

                image_samples.append({
                    'pixel_values': torch.tensor(crop_img_sq).permute(2,0,1).float() / 255,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'target_heatmap': torch.tensor(crop_grid.flatten(), dtype=torch.float32),
                    'phrase': prompt
                })

            # 2) five SMALL crops: 1 positive + 4 negatives (simple & effective)
            # 2a) one small positive
            x0, y0, x1, y1 = sample_small_positive_crop(cls_boxes, W, H)
            crop_img  = img_arr[y0:y1, x0:x1]
            crop_mask = full_mask[y0:y1, x0:x1]
            crop_img_sq  = np.array(Image.fromarray(crop_img).resize((sq, sq), Image.BICUBIC))
            crop_mask_sq = np.array(Image.fromarray(crop_mask).resize((sq, sq), Image.NEAREST))
            crop_grid    = gridify_fixed(crop_mask_sq, GRID_SIZE)
            image_samples.append({
                'pixel_values': torch.tensor(crop_img_sq).permute(2,0,1).float() / 255,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_heatmap': torch.tensor(crop_grid.flatten(), dtype=torch.float32),
                'phrase': prompt
            })

            # 2b) four small negatives (no overlap with any GT of this prompt)
            for _ in range(SMALL_CROPS - 1):
                x0, y0, x1, y1 = sample_small_negative_crop(cls_boxes, W, H)
                crop_img  = img_arr[y0:y1, x0:x1]
                # for negatives, mask will be zeros → grid all zeros
                crop_mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)

                crop_img_sq  = np.array(Image.fromarray(crop_img).resize((sq, sq), Image.BICUBIC))
                crop_mask_sq = np.array(Image.fromarray(crop_mask).resize((sq, sq), Image.NEAREST))
                crop_grid    = gridify_fixed(crop_mask_sq, GRID_SIZE)

                image_samples.append({
                    'pixel_values': torch.tensor(crop_img_sq).permute(2,0,1).float() / 255,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'target_heatmap': torch.tensor(crop_grid.flatten(), dtype=torch.float32),
                    'phrase': prompt
                })

        # save this image’s samples
        fn = os.path.join(SAVE_DIR, f"image_batch_{batch_counter:06d}.pkl")
        with open(fn, "wb") as f:
            pickle.dump(image_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {len(image_samples)} samples for image {sample_idx} → {fn}")
        batch_counter += 1

if __name__ == "__main__":
    main()
