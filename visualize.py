import os, re, json, textwrap
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches

# --- Config ---
ROOT = "dataset16_words"         # contains images/, tiles16/, meta/, manifest.jsonl
MANIFEST = os.path.join(ROOT, "manifest.jsonl")
OUT_VIS = os.path.join(ROOT, "vis")
SAMPLE_LIMIT = None              # e.g. 50, or None for all
ALPHA = 0.35                     # heatmap transparency
CMAP = "Reds"                    # overlay colormap
DRAW_GRID = False                # draw tile grid lines
DRAW_BOX = True                  # draw GT boxes for this row

os.makedirs(OUT_VIS, exist_ok=True)

# Matches IDs like "...#obj07" to detect full-image object rows
_obj_idx_re = re.compile(r"#obj(\d+)$")

def parse_obj_index(sample_id: str):
    """
    If the row id ends with '#objNN', return NN as int; else None.
    Full-image rows use '#objNN'. Crop rows use '#<prompt>'.
    """
    m = _obj_idx_re.search(sample_id)
    return int(m.group(1)) if m else None

def load_paths(row):
    img_path  = os.path.join(ROOT, row["image"])
    heat_path = os.path.join(ROOT, row["heatmap"])
    meta_path = os.path.join(ROOT, row.get("objects_json", "")) if row.get("objects_json") else None
    return img_path, heat_path, meta_path

def load_box_for_this_object(meta_path, obj_idx):
    """
    For full-image rows: meta contains 'snapped_xyxy' per object.
    Returns (x0,y0,x1,y1) or None.
    """
    if not (meta_path and os.path.exists(meta_path) and obj_idx is not None):
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    objs = meta.get("objects", [])
    if 0 <= obj_idx < len(objs):
        x0, y0, x1, y1 = objs[obj_idx].get("snapped_xyxy", [None, None, None, None])
        if None not in (x0, y0, x1, y1):
            return (x0, y0, x1, y1)
    return None

def collect_boxes_for_prompt(meta_path, prompt):
    """
    For crop rows: meta contains objects with 'bbox_px' in crop coordinates.
    Also works for full-image meta if present (prefers 'snapped_xyxy').
    Returns a list of (x0,y0,x1,y1) for all objects matching the row's prompt.
    """
    if not (meta_path and os.path.exists(meta_path)):
        return []
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    boxes = []
    for o in meta.get("objects", []):
        if o.get("prompt_one_word") != prompt:
            continue
        if "snapped_xyxy" in o and o["snapped_xyxy"] is not None:
            x0, y0, x1, y1 = o["snapped_xyxy"]
            boxes.append((x0, y0, x1, y1))
        elif "bbox_px" in o and o["bbox_px"] is not None:
            x, y, w, h = o["bbox_px"]
            boxes.append((x, y, x + w, y + h))
    return boxes

def draw_sample(img_path, heat_path, prompt, tile, boxes_xyxy=None, out_path=None,
                alpha=ALPHA, cmap=CMAP, draw_grid=DRAW_GRID):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    heat = np.load(heat_path)  # (rows, cols) in {0,1}

    # Keep figure aspect proportional to image
    fig_w = max(4.0, W / 200.0)
    fig_h = max(4.0, H / 200.0)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = plt.gca()

    ax.imshow(img)
    # Stretch the tile heatmap over the image using pixel extent
    ax.imshow(heat, origin="upper", extent=(0, W, H, 0),
              interpolation="nearest", alpha=alpha, cmap=cmap)

    if draw_grid:
        for x in range(0, W + 1, tile):
            ax.axvline(x, linewidth=0.5, color="white", alpha=0.3)
        for y in range(0, H + 1, tile):
            ax.axhline(y, linewidth=0.5, color="white", alpha=0.3)

    if boxes_xyxy:
        for (x0, y0, x1, y1) in boxes_xyxy:
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)

    ax.set_axis_off()
    fig.suptitle(textwrap.fill(str(prompt), width=60), fontsize=12, y=0.98)
    plt.tight_layout(pad=0)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show()

# --- Run ---
with open(MANIFEST, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if SAMPLE_LIMIT is not None and idx >= SAMPLE_LIMIT:
            break
        row = json.loads(line)

        img_path, heat_path, meta_path = load_paths(row)
        tile = int(row["tile"])
        prompt = row["prompt"]
        obj_idx = parse_obj_index(row["id"])

        # Figure out which boxes to draw (if any)
        if DRAW_BOX:
            if obj_idx is not None:
                # Full-image row: draw just this object's snapped box
                box_xyxy = load_box_for_this_object(meta_path, obj_idx)
                boxes_xyxy = [box_xyxy] if box_xyxy else []
            else:
                # Crop row: draw all boxes for this prompt within the crop
                boxes_xyxy = collect_boxes_for_prompt(meta_path, prompt)
        else:
            boxes_xyxy = []

        # Make a filesystem-safe filename
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", row["id"])
        safe_prompt = re.sub(r"[^A-Za-z0-9_-]+", "_", prompt)
        out_name = f"{safe_id}_{safe_prompt}.png"
        out_path = os.path.join(OUT_VIS, out_name)

        draw_sample(img_path, heat_path, prompt, tile, boxes_xyxy=boxes_xyxy, out_path=out_path)
        print("Saved:", out_path)
