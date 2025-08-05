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
NUM_SAMPLES   = 3000    # how many images to process
NUM_CROPS     =    3    # crops per prompt
MIN_CROP      =  50    # min crop dimension (px)
MAX_CROP      =  100    # max crop dimension (px)
SQUARE_SIZE   =  224    # final square resize (like model input)
GRID_SIZE     =   14    # number of grid cells per side (14×14 = 196)
TEXT_MAX_LEN  =   64    # max token length for phrases
MODEL_ID      = "google/siglip2-base-patch16-224"
DATASET       = "clane9/NSD-Flat"
SAVE_DIR      = "processed_batches_3000_3crops"

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def get_random_crop_coords(img_w, img_h, crop_w, crop_h):
    x0 = random.randint(0, img_w - crop_w)
    y0 = random.randint(0, img_h - crop_h)
    return x0, y0, x0 + crop_w, y0 + crop_h

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
    # avoid hanging threads at exit
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # streaming load + shuffle + take first NUM_SAMPLES
    stream = load_dataset(
        DATASET,
        split="train",
        streaming=True
    ).shuffle(seed=42, buffer_size=1000).take(NUM_SAMPLES)

    batch_counter = 0
    for sample_idx, sample in enumerate(stream):
        # sample_idx runs 0..NUM_SAMPLES-1
        pil_img = sample["image"]
        W, H    = pil_img.size
        img_arr = np.array(pil_img)

        raw_bboxes = sample["objects"]["bbox"]
        labels     = sample["objects"]["category"]
        sx, sy     = W / 425, H / 425
        scaled = [(x*sx, y*sy, bw*sx, bh*sy) for x,y,bw,bh in raw_bboxes]

        image_samples = []
        for prompt in sorted(set(labels)):
            # build full-image mask
            full_mask = np.zeros((H, W), dtype=np.uint8)
            for (bx,by,bw,bh), lab in zip(scaled, labels):
                if lab == prompt:
                    x0, y0 = map(int, (bx, by))
                    x1, y1 = map(int, (bx + bw, by + bh))
                    full_mask[y0:y1, x0:x1] = 1

            # resize to square & gridify
            sq = SQUARE_SIZE
            full_img_sq  = np.array(
                Image.fromarray(img_arr).resize((sq, sq), Image.BICUBIC)
            )
            full_mask_sq = np.array(
                Image.fromarray(full_mask).resize((sq, sq), Image.NEAREST)
            )
            full_grid    = gridify_fixed(full_mask_sq, GRID_SIZE)
            input_ids, attention_mask = encode_phrase(prompt, tokenizer)

            # full-image sample
            image_samples.append({
                'pixel_values': torch.tensor(full_img_sq).permute(2,0,1).float() / 255,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_heatmap': torch.tensor(full_grid.flatten(), dtype=torch.float32),
                'phrase': prompt
            })

            # random-crop samples
            for _ in range(NUM_CROPS):
                cw = random.randint(MIN_CROP, MAX_CROP)
                ch = random.randint(MIN_CROP, MAX_CROP)
                x0, y0, x1, y1 = get_random_crop_coords(W, H, cw, ch)

                crop_img  = img_arr[y0:y1, x0:x1]
                crop_mask = full_mask[y0:y1, x0:x1]

                crop_img_sq  = np.array(
                    Image.fromarray(crop_img).resize((sq, sq), Image.BICUBIC)
                )
                crop_mask_sq = np.array(
                    Image.fromarray(crop_mask).resize((sq, sq), Image.NEAREST)
                )
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
