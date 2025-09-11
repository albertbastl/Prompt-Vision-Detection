# stream_cooccurrence_from_nsd_flat.py
# Purely algorithmic co-occurrence, conditional probabilities, lift, PMI.
# Uses Hugging Face Datasets *streaming* mode + progress prints.

from datasets import load_dataset
from collections import defaultdict
import numpy as np
import json
import math
import sys
from time import time

try:
    from tqdm import tqdm  # only used for pretty counters on prints, no total
except ImportError:
    def tqdm(x, **k): return x

# -----------------------------
# Config
# -----------------------------
SPLIT = "train"
PROGRESS_EVERY = 1000  # print a progress line after this many images
MAX_ROWS = None        # set to an int to stop early for debugging, else None

# -----------------------------
# 1) Load dataset in STREAMING mode
# -----------------------------
# This returns an *iterable* dataset. Length is unknown; we just stream rows.
print("Loading dataset in streaming mode…", flush=True)
ds_iter = load_dataset("clane9/NSD-Flat", split=SPLIT, streaming=True)

# -----------------------------
# 2) Build vocabulary & counts
# -----------------------------
cat2id = {}
id2cat = []

def get_id(c):
    if c not in cat2id:
        cat2id[c] = len(id2cat)
        id2cat.append(c)
    return cat2id[c]

N_images = 0
n_i = defaultdict(int)          # image count per category
co_ij = defaultdict(int)        # symmetric co-occurrence counts (i,j) with i!=j

start_time = time()
last_report_time = start_time

print("Streaming and counting…", flush=True)
try:
    for idx, row in enumerate(ds_iter, start=1):
        labels = row.get("objects", {})
        cats   = labels.get("category", [])
        if not isinstance(cats, list):
            cats = []

        if cats:
            N_images += 1
            uniq = sorted(set(cats))
            ids = [get_id(c) for c in uniq]

            # singletons
            for i in ids:
                n_i[i] += 1

            # pairwise co-occurrence (undirected)
            L = len(ids)
            for a in range(L):
                for b in range(a + 1, L):
                    i, j = ids[a], ids[b]
                    co_ij[(i, j)] += 1
                    co_ij[(j, i)] += 1

        # Progress prints
        if idx % PROGRESS_EVERY == 0:
            elapsed = time() - start_time
            uniq_cats = len(id2cat)
            print(
                f"[{idx} rows streamed] "
                f"labeled_images={N_images}  unique_categories={uniq_cats}  "
                f"elapsed={elapsed:.1f}s",
                flush=True
            )

        if MAX_ROWS is not None and idx >= MAX_ROWS:
            print(f"Reached MAX_ROWS={MAX_ROWS}, stopping early.", flush=True)
            break

except KeyboardInterrupt:
    print("\nInterrupted by user. Proceeding to compute probabilities with partial counts…", flush=True)

print(f"Streaming done. Images with ≥1 label: {N_images}, unique categories: {len(id2cat)}", flush=True)

# -----------------------------
# 3) Probabilities
# -----------------------------
# P(i) = n_i / N_images
num_cats = len(id2cat)
P = np.zeros(num_cats, dtype=float)
den_N = max(1, N_images)
for i in range(num_cats):
    P[i] = n_i.get(i, 0) / den_N

# P(j|i), Lift(A→B), PMI(A,B)
cond = {id2cat[i]: {} for i in range(num_cats)}
lift = {id2cat[i]: {} for i in range(num_cats)}
pmi  = {id2cat[i]: {} for i in range(num_cats)}

print("Computing conditional probabilities, lift, and PMI…", flush=True)
for i in range(num_cats):
    denom = n_i.get(i, 0)
    if denom == 0:
        continue
    for j in range(num_cats):
        if i == j:
            continue
        cij = co_ij.get((i, j), 0)
        if cij == 0:
            cond[id2cat[i]][id2cat[j]] = 0.0
            lift[id2cat[i]][id2cat[j]] = 0.0 if P[j] > 0 else 0.0
            continue

        p_j_given_i = cij / denom
        cond[id2cat[i]][id2cat[j]] = p_j_given_i

        # Lift(A->B) = P(B|A) / P(B)
        lift_val = p_j_given_i / (P[j] + 1e-12)
        lift[id2cat[i]][id2cat[j]] = float(lift_val)

        # PMI(A,B) = log2( P(A,B) / (P(A)P(B)) )
        p_ab = cij / den_N
        denom_pmi = (P[i] * P[j]) + 1e-12
        pmi_val = math.log2(p_ab / denom_pmi)
        pmi[id2cat[i]][id2cat[j]] = float(pmi_val)

# -----------------------------
# 4) Sort maps (for readability)
# -----------------------------
def sort_map_of_maps(d, reverse=True):
    out = {}
    for k, m in d.items():
        out[k] = dict(sorted(m.items(), key=lambda kv: kv[1], reverse=reverse))
    return out

cond_sorted = sort_map_of_maps(cond, reverse=True)
lift_sorted = sort_map_of_maps(lift, reverse=True)
pmi_sorted  = sort_map_of_maps(pmi,  reverse=True)

# -----------------------------
# 5) Save artifacts
# -----------------------------
print("Saving outputs…", flush=True)
with open("category_list.json", "w") as f:
    json.dump(id2cat, f, indent=2)

with open("prior_probs.json", "w") as f:
    json.dump({id2cat[i]: float(P[i]) for i in range(num_cats)}, f, indent=2)

with open("cond_probs.json", "w") as f:
    json.dump(cond_sorted, f, indent=2)

with open("lift.json", "w") as f:
    json.dump(lift_sorted, f, indent=2)

with open("pmi.json", "w") as f:
    json.dump(pmi_sorted, f, indent=2)

print("\nSaved files:")
print("  category_list.json   # index → category")
print("  prior_probs.json     # P(cat)")
print("  cond_probs.json      # P(B|A)")
print("  lift.json            # Lift(A→B) = P(B|A)/P(B)")
print("  pmi.json             # PMI(A,B)")

# -----------------------------
# 6) Progress-friendly quick checks
# -----------------------------
def top_k_given(cat, k=10):
    row = cond_sorted.get(cat, {})
    items = list(row.items())[:k]
    print(f"\nTop {k} by P(B|{cat}):")
    for b, v in items:
        print(f"  {b:20s}  P({b}|{cat}) = {v:.3f}")

def bottom_k_by_lift(cat, k=10):
    row = lift_sorted.get(cat, {})
    items = sorted(row.items(), key=lambda kv: kv[1])[:k]
    print(f"\nBottom {k} by Lift(B|{cat}) (negative association):")
    for b, v in items:
        if v < 1.0:
            pb_given_a = cond_sorted.get(cat, {}).get(b, 0.0)
            pb = float(P[cat2id[b]]) if b in cat2id else 0.0
            print(f"  {b:20s}  Lift={v:.3f}  P({b}|{cat})={pb_given_a:.3f}  P({b})={pb:.3f}")

# Example sanity prints (only if those categories exist)
if "person" in cat2id:
    top_k_given("person", k=10)
    bottom_k_by_lift("person", k=10)

elapsed_total = time() - start_time
print(f"\nDone. Total elapsed: {elapsed_total:.1f}s", flush=True)
