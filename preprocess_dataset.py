import torch
import pickle
import os
import numpy as np
from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from datasets import load_dataset
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse
import json
import gc
# import torchvision for data augmentation
import torchvision.transforms.functional as F
from torchvision.transforms import RandomResizedCrop, InterpolationMode
from config import MODEL_ID, PATCH_GRID_SIZE, TEXT_MAX_LENGTH

DATASET_ID = "ranjaykrishna/visual_genome"
DATASET_CONFIG = "region_descriptions_v1.2.0"

OUTPUT_DIR = "./processed_dataset"
METADATA_PATH = os.path.join(OUTPUT_DIR, "dataset_metadata.json")

try:
    model_config = AutoConfig.from_pretrained(MODEL_ID)
    VISION_IMAGE_SIZE = model_config.vision_config.image_size
except Exception as e:
    print(f"Could not load model config for {MODEL_ID}. Defaulting image size to 224. Error: {e}")
    VISION_IMAGE_SIZE = 224


def generate_heatmap_from_box(box, crop_params, final_size, grid_size=PATCH_GRID_SIZE):
    """
    Creates a binary heatmap from a bounding box after calculating its new
    position within a random crop. This is the corrected, robust version.
    """
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h

    crop_top, crop_left, crop_h, crop_w = crop_params
    
    new_x1 = x1 - crop_left
    new_y1 = y1 - crop_top
    new_x2 = x2 - crop_left
    new_y2 = y2 - crop_top

    scale_w = final_size / crop_w
    scale_h = final_size / crop_h
    
    final_x1 = new_x1 * scale_w
    final_y1 = new_y1 * scale_h
    final_x2 = new_x2 * scale_w
    final_y2 = new_y2 * scale_h
    
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(final_size, final_x2)
    final_y2 = min(final_size, final_y2)

    if final_x1 >= final_x2 or final_y1 >= final_y2:
        return torch.zeros(grid_size * grid_size, dtype=torch.float32)

    patch_size = final_size / grid_size
    x_start_idx = int(final_x1 / patch_size)
    x_end_idx = int(np.ceil(final_x2 / patch_size))
    y_start_idx = int(final_y1 / patch_size)
    y_end_idx = int(np.ceil(final_y2 / patch_size))

    heatmap_np = np.zeros((grid_size, grid_size), dtype=np.float32)
    heatmap_np[y_start_idx:y_end_idx, x_start_idx:x_end_idx] = 1.0
    
    return torch.tensor(heatmap_np, dtype=torch.float32).flatten()


def process_single_example(
    example, 
    image_processor, 
    tokenizer, 
    max_regions_per_image=5,
    augmentations_per_image=3,
    debug=False  
):
    """
    Process a single Visual Genome example, creating multiple augmented versions.
    Includes detailed debugging prints if debug=True.
    """
    if debug: 
        print(f"\n--- [DEBUG] Processing New Example (Image ID: {example.get('image_id', 'N/A')}) ---")
    processed_samples = []
    
    img = example.get('image')
    if img is None or img.mode != 'RGB': 
        if debug: 
            print("[DEBUG] EXIT: Image is invalid or not RGB.")
        return []
    
    regions = example.get('regions', [])
    if not regions:
        if debug: 
            print("[DEBUG] EXIT: No 'regions' field found in example.")
        return []
    if debug: 
        print(f"[DEBUG] Found {len(regions)} initial regions.")

    valid_regions = []
    for i, region in enumerate(regions):
        phrase = region.get('phrase', '').strip()
        box = (region.get('x', 0), region.get('y', 0), region.get('width', 0), region.get('height', 0))
        is_valid = len(phrase) >= 3 and box[2] > 10 and box[3] > 10
        if debug: 
            print(f"[DEBUG] Region {i+1}: phrase='{phrase[:30]}...', box=({box[2]}x{box[3]}) -> {'VALID' if is_valid else 'INVALID'}")
        if is_valid:
            valid_regions.append(region)
    
    if not valid_regions: 
        if debug: print("[DEBUG] EXIT: No valid regions found after filtering.")
        return []
    if debug: print(f"[DEBUG] Found {len(valid_regions)} valid regions to process.")

    for i in range(augmentations_per_image):
        if debug: 
            print(f"\n  [DEBUG] --- Augmentation Pass {i+1}/{augmentations_per_image} ---")
        try:
            crop_params = RandomResizedCrop.get_params(img, scale=(0.5, 1.0), ratio=(0.75, 1.33))
            if debug: 
                print(f"  [DEBUG] Crop Params (top, left, h, w): {crop_params}")
            img_aug = F.resized_crop(img, *crop_params, size=(VISION_IMAGE_SIZE, VISION_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC)
            pixel_values = image_processor(images=[img_aug], return_tensors="pt")['pixel_values'][0]
            for region in valid_regions[:max_regions_per_image]:
                phrase = region['phrase'].strip()
                box = (region['x'], region['y'], region['width'], region['height'])
                
                target_heatmap = generate_heatmap_from_box(box, crop_params, VISION_IMAGE_SIZE)
                heatmap_sum = torch.sum(target_heatmap).item()

                if debug: 
                    print(f"    [DEBUG] Phrase '{phrase[:30]}...' -> Heatmap Sum: {heatmap_sum}")

                if heatmap_sum == 0:
                    if debug: 
                        print("      [DEBUG] SKIPPING: Bounding box was cropped out or is invalid.")
                    continue
                
                text_data = tokenizer(
                    text=phrase, padding="max_length", truncation=True,
                    max_length=TEXT_MAX_LENGTH, return_tensors="pt",return_attention_mask=True
                )
                
                processed_samples.append({
                    'pixel_values': pixel_values,
                    'input_ids': text_data['input_ids'][0],
                    'attention_mask': text_data['attention_mask'][0],
                    'target_heatmap': target_heatmap,
                    'phrase': phrase,
                })
                if debug: 
                    print("      [DEBUG] SUCCESS: Sample created and appended.")
        except Exception as e:
            if debug: 
                print(f"  [DEBUG] ERROR in augmentation loop: {e}")
            continue
            
    if debug: 
        print(f"---> [DEBUG] FINISHED EXAMPLE: Returning {len(processed_samples)} samples in total.")
    return processed_samples

def save_batch(batch_data, batch_idx, output_dir):
    """Save a batch of processed data."""
    batch_path = os.path.join(output_dir, f"batch_{batch_idx:06d}.pkl")
    with open(batch_path, 'wb') as f:
        pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    return batch_path

def preprocess_dataset(max_images=10000, batch_size=200, max_regions_per_image=5, augmentations_per_image=3, debug=False):
    """Main preprocessing function."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading processors and tokenizer...")
    image_processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"Loading dataset '{DATASET_ID}' with config '{DATASET_CONFIG}' (streaming)...")
    raw_dataset = load_dataset(DATASET_ID, name=DATASET_CONFIG, split='train', streaming=True)
    raw_dataset = raw_dataset.shuffle(seed=42, buffer_size=1000)
    
    iterable_dataset = raw_dataset.take(max_images) if max_images > 0 else raw_dataset
    
    total_images_processed, total_regions_processed = 0, 0
    batch_data, batch_files, batch_idx = [], [], 0
    
    progress_bar = tqdm(iterable_dataset, desc="Processing images")
    
    for example in progress_bar:
        processed_samples = process_single_example(
            example, image_processor, tokenizer, max_regions_per_image, augmentations_per_image, debug=debug
        )
        if processed_samples:
            batch_data.extend(processed_samples)
            total_regions_processed += len(processed_samples)
        total_images_processed += 1
        
        progress_bar.set_postfix({'Regions': total_regions_processed, 'Batch': len(batch_data)})
        
        if len(batch_data) >= batch_size and not debug:
            batch_path = save_batch(batch_data, batch_idx, OUTPUT_DIR)
            batch_files.append(str(batch_path))
            batch_data, batch_idx = [], batch_idx + 1
            gc.collect()
        
        if debug and total_images_processed >= 5:
            print("\n[DEBUG] Processed 5 images. Halting debug run.")
            break
    
    if batch_data and not debug:
        batch_path = save_batch(batch_data, batch_idx, OUTPUT_DIR)
        batch_files.append(str(batch_path))
    
    if not debug:
        metadata = { 'total_regions_processed': total_regions_processed, 'batch_files': batch_files }
        with open(METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"\nPreprocessing complete! {total_regions_processed} regions saved in {len(batch_files)} batches.")
    else:
        print("\n[DEBUG] Run finished.")

def main():
    parser = argparse.ArgumentParser(description='Preprocess Visual Genome with augmentations and debugging.')
    parser.add_argument('--max_images', type=int, default=100000, help='Maximum original images to process from the dataset.')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of processed regions to save in each batch file.')
    parser.add_argument('--max_regions', type=int, default=10, help='Maximum number of regions to use from a single image.')
    parser.add_argument('--augmentations', type=int, default=5, help='Number of random augmented views to create per region.')
    parser.add_argument('--debug', action='store_true', help="Run in debug mode with verbose prints for a few samples.")
    args = parser.parse_args()
    
    max_images_to_run = 10 if args.debug else args.max_images

    preprocess_dataset(
        max_images=max_images_to_run, 
        batch_size=args.batch_size,
        max_regions_per_image=args.max_regions, 
        augmentations_per_image=args.augmentations,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
