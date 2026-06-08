"""
Quick data preparation script for OpenWebText 1% subset
--------------------------------------------------------
Downloads only ~200 MB via streaming instead of the full 17 GB,
then writes train.bin and val.bin.

Changes from original prepare.py:
  - Full 8M documents -> stream only N_DOCS (80,000) documents
  - No multiprocessing (simple sequential processing)
  - Val split: use last VAL_RATIO fraction instead of train_test_split

Compatible with train_gpt2.py without any code changes (same uint16 format).

Usage:
    cd experiment_1
    pip install datasets tiktoken tqdm
    python data/openwebtext/prepare_mini.py

Output files (same directory as this script):
    data/openwebtext/train.bin  (~140-160 MB, ~72M tokens)
    data/openwebtext/val.bin    (~15-18 MB,  ~8M tokens)
"""

import os
import itertools
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ── settings ────────────────────────────────────────────────────────────────
N_DOCS    = 80000  # number of documents to stream (~1% of full 8M)
VAL_RATIO = 0.1    # last 10% of tokens reserved for val.bin

# ── output paths ────────────────────────────────────────────────────────────
# saves to the same directory as this script (data/openwebtext/)
out_dir    = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(out_dir, "train.bin")
val_path   = os.path.join(out_dir, "val.bin")

# ── tokenizer (GPT-2 BPE, same as original) ─────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256 — end-of-document token

# ── streaming download & tokenization ───────────────────────────────────────
print(f"Streaming {N_DOCS:,} documents from openwebtext (HuggingFace)...")
print("(downloading ~200 MB instead of the full 17 GB)\n")

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True,
    trust_remote_code=True,
)

chunks = []
for doc in tqdm(itertools.islice(dataset, N_DOCS), total=N_DOCS, unit="docs"):
    ids = enc.encode_ordinary(doc["text"])  # encode text without special tokens
    ids.append(EOT)                          # append end-of-document token (same as original)
    chunks.append(np.array(ids, dtype=np.uint16))

print("\nConcatenating token arrays...")
all_tokens = np.concatenate(chunks)
n = len(all_tokens)
print(f"Total tokens: {n:,}  ({n * 2 / 1e6:.1f} MB)")

# ── train / val split ────────────────────────────────────────────────────────
split_idx    = int(n * (1 - VAL_RATIO))
train_tokens = all_tokens[:split_idx]
val_tokens   = all_tokens[split_idx:]

# ── save .bin files (uint16 format, same as original) ───────────────────────
train_tokens.tofile(train_path)
val_tokens.tofile(val_path)

train_mb = os.path.getsize(train_path) / 1e6
val_mb   = os.path.getsize(val_path)   / 1e6

print(f"\nDone!")
print(f"  train.bin : {len(train_tokens):,} tokens  ({train_mb:.1f} MB)")
print(f"  val.bin   : {len(val_tokens):,} tokens   ({val_mb:.1f} MB)")
print(f"\nHow to train:")
print(f"  cd experiment_1")
print(f"  python train_gpt2.py config/train_gpt2_small_1gpu.py")
