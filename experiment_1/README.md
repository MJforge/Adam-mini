# GPT-2 Pre-training Reproduction: Adam-mini vs AdamW

A scaled-down reproduction of GPT-2 125M pre-training experiments from the paper
[Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793),
conducted on a single GPU environment.

---

## Attribution

This experiment is based on the original Adam-mini repository:
- Paper: [Adam-mini: Use Fewer Learning Rates To Gain More (arXiv 2406.16793)](https://arxiv.org/abs/2406.16793)
- Original code: [zyushun/Adam-mini](https://github.com/zyushun/Adam-mini)
- Original copyright: Copyright (c) 2024 Yushun Zhang, et al. (Adam-mini Authors)
- Modifications: Copyright (c) 2026 Min-jeong Park
- License: Apache-2.0

---

## Experiment Overview

### Objectives

- Compare validation loss trajectories of Adam-mini and AdamW during GPT-2 125M pre-training
- Critically evaluate Adam-mini's sensitivity to partitioning granularity (attention head count)
- Observe relative optimizer behavior under constrained hardware conditions

### W&B Report
Comprehensive training metrics, validation loss comparisons, and memory efficiency analysis for Experiments 1-1 and 1-2 conducted in this folder are available in the official report below.

[![W&B Report](https://img.shields.io/badge/Weights_&_Biases-Report-yellow?style=for-the-badge&logo=WeightsAndBiases)](https://wandb.ai/clairdemin-seoul/adam-mini-reproduction/reports/Adam-mini-vs-AdamW-on-GPT-2-125M---VmlldzoxNzA3NjgwNw)

### Environment Comparison

| Item | Original Paper | This Experiment |
|------|---------------|-----------------|
| GPU | A100 SXM 80GB × 8 | A100 PCIe 80GB × 1 |
| `max_iters` | 100,000 | 1,500 |
| `batch_size` | 60 | 24 |
| `gradient_accumulation_steps` | 8 | 20 |
| `tokens/iter` | 491,520 | 491,520 (preserved) |
| Total training tokens | ~49B | ~0.74B |
| Dataset | OpenWebText (full) | OpenWebText 1% subset (~80M tokens) |

`batch_size` and `gradient_accumulation_steps` were adjusted to preserve the original
`tokens/iter = 491,520`, ensuring a fair optimizer comparison despite single-GPU constraints.

---

## Repository Structure

```
experiment_1/                          # this folder
├── train_gpt2.py                      # main training script (modified)
├── model.py                           # GPT-2 model definition (copied from original_code/;
│                                      #   only change: removed dead `import algorithms`)
├── config/
│   └── train_gpt2_small_1gpu.py       # hyperparameter config for single-GPU run
├── data/
│   └── openwebtext/
│       └── prepare_mini.py            # data preparation script (1% OWT subset)
└── README.md
```

`logger.py` and `configurator.py` are **not copied**. `train_gpt2.py` references them
directly from `../original_code/examples/gpt2/`:

- `logger` — loaded via `sys.path` insert (no custom dependencies)
- `configurator.py` — loaded via `exec(open(path).read())`, not through the import system

---

## Quick Start

### Google Colab

```python
# Cell 1: clone & install
!git clone https://github.com/MJforge/Adam-mini.git
!pip install -e /content/Adam-mini/original_code   # setup.py lives here
!pip install datasets tiktoken tqdm wandb

# Cell 2: prepare data
%cd /content/Adam-mini/experiment_1
!python data/openwebtext/prepare_mini.py

# Cell 3: run training
!python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --comment=gpt2_small_adam_mini_nhead12
```

### RunPod / Single GPU

```bash
# 1. Clone the repository
git clone https://github.com/MJforge/Adam-mini.git
cd Adam-mini

# 2. Install the adam_mini package (setup.py is inside original_code/)
pip install -e original_code

# 3. Install additional dependencies
pip install datasets tiktoken tqdm wandb

# 4. Prepare the data (1% OWT subset, ~80M tokens)
cd experiment_1
python data/openwebtext/prepare_mini.py
#   → creates data/openwebtext/train.bin and val.bin

# 5. Run training (from experiment_1/)
# Adam-mini
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --comment=gpt2_small_adam_mini_nhead12

# AdamW
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw \
    --comment=gpt2_small_adamw_nhead12
```

> All commands must be run from inside `experiment_1/`. The `config/` path in the
> run command is relative to the current working directory.

---

## Modifications to train_gpt2.py

Changes made to the original `examples/gpt2/train_gpt2.py`:

| Change | Description |
|--------|-------------|
| Optimizer selection | Added `adam_mini` branch; moved `raise ValueError` to `else` clause (bug fix) |
| W&B integration | Added `wandb.init()`, `wandb.define_metric()`, `wandb.log()`, `wandb.finish()` |
| Custom x-axis | All W&B metrics use `tokens_B` (Tokens in Billions) as step metric |
| Gradient norm logging | Captured `grad_norm` from `clip_grad_norm_` for W&B logging |
| Removed unused imports | Removed `argparse`, `json`, `io_utils`, `torch_optimizer`, `SummaryWriter` |
| Dependency resolution | `sys.path` insertion → `../original_code/examples/gpt2/` for `logger`; `configurator.py` loaded via `exec(open(path).read())` |

---

## Data Preparation

### Design Rationale

Instead of downloading the full 17 GB OpenWebText dataset, `prepare_mini.py` streams
only `N_DOCS=80,000` documents (~1% of total) using HuggingFace `datasets` streaming.
The tokenized result is saved as `train.bin` and `val.bin` in `uint16` format.

During training, `numpy.memmap` loads data directly from disk with random access,
keeping the GPU fully utilized without network wait time.

```
[One-time preparation]
HuggingFace streaming → tokenize (tiktoken GPT-2 BPE) → save train.bin / val.bin

[Training loop]
numpy.memmap random sampling → GPU transfer → forward / backward
```

### Run Data Preparation

```bash
# from repository root
cd experiment_1
pip install datasets tiktoken tqdm wandb
python data/openwebtext/prepare_mini.py
```

Output files:
- `data/openwebtext/train.bin` : ~140–160 MB, ~72M tokens
- `data/openwebtext/val.bin`   : ~15–18 MB,  ~8M tokens

---

## Experiment 1: Adam-mini vs AdamW (n_head=12)

### Hyperparameters

```python
batch_size                  = 24
gradient_accumulation_steps = 20
block_size                  = 1024    # tokens/iter = 491,520
n_layer, n_head, n_embd     = 12, 12, 768   # GPT-2 Small 125M
max_iters                   = 1500
lr_decay_iters              = 1500
warmup_iters                = 30      # 2% of max_iters
learning_rate               = 6e-4
min_lr                      = 3e-5
weight_decay                = 0
beta1, beta2                = 0.9, 0.95
grad_clip                   = 1.0
seed                        = 1337
```

### Run Commands

```bash
# from repository root
cd experiment_1
pip install -e ..   # installs adam_mini package

# Adam-mini
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --comment=gpt2_small_adam_mini_nhead12

# AdamW
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw \
    --comment=gpt2_small_adamw_nhead12
```

### Results

| Optimizer | val/loss | train/loss |
|-----------|---------|-----------|
| Adam-mini | 4.393   | 4.091     |
| AdamW     | 4.238   | 3.923     |
| **gap**   | **+0.155** | — |

Over 1,500 steps (~0.74B tokens), Adam-mini showed a similar loss reduction trend
to AdamW but lagged slightly in absolute val/loss values.

---

## Experiment 2: Partitioning Sensitivity Analysis (n_head=6)

### Motivation

Adam-mini partitions parameters at the attention head level, assigning one learning
rate per partition. Reducing `n_head` decreases the number of partitions. This
experiment checks whether the relative performance gap between Adam-mini and AdamW
changes when partitioning granularity is reduced.

**Core question:**

```
gap at n_head=12 = A
gap at n_head=6  = B

If A ≠ B → Adam-mini performance depends on head count (partitioning granularity)
```

### Run Commands

```bash
# from repository root
cd experiment_1

# Adam-mini, n_head=6
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --n_head=6 \
    --comment=gpt2_small_adam_mini_nhead6
```

`n_embd=768` is divisible by 6 (head_size=128), so no other changes are needed.

### Results

| Condition | Adam-mini val/loss | AdamW val/loss | gap |
|-----------|-------------------|----------------|-----|
| n_head=12 | 4.393 | 4.238 | +0.155 |
| n_head=6  | 4.507 | 4.264 | **+0.243** |

Reducing n_head from 12 to 6 expanded the Adam-mini vs AdamW val/loss gap from
0.155 to 0.243 — a **57% increase**. This supports the paper's acknowledged
limitation that Adam-mini's performance is sensitive to the partitioning scheme.

---

### Actual Training Throughput (from per-step time series)

| Condition | Adam-mini | AdamW | Difference |
|-----------|-----------|-------|------------|
| n_head=12 | ~141,100 tok/s | ~141,300 tok/s | negligible |
| n_head=6  | ~143,500 tok/s | ~143,700 tok/s | negligible |

At GPT-2 125M scale, Adam-mini and AdamW have essentially identical training
throughput. The ~1.5% advantage of n_head=6 over n_head=12 is attributable to
Flash Attention efficiency with larger head sizes (head_size 128 vs 64).

---

## W&B Logging

The following metrics are logged to Weights & Biases during training.
All metrics use `tokens_B` (cumulative tokens in billions) as the x-axis.

| Metric | Description |
|--------|-------------|
| `val/loss` | Validation loss |
| `train/loss` | Training loss |
| `tokens_B` | Cumulative tokens processed (Billions) — x-axis |
| `throughput/tok_per_sec` | Tokens processed per second |
| `gpu/max_vram_gb` | Peak VRAM usage (GB) |
| `opt/grad_norm` | Gradient norm |
| `opt/lr` | Current learning rate |

Set `wandb_project` in `config/train_gpt2_small_1gpu.py` to your W&B project name.

---

## Setup

```bash
# from repository root
pip install -e .
pip install wandb datasets tiktoken tqdm

# login to W&B
wandb login
```

---

## Limitations

- Hardware constraints prevented full-scale reproduction (100K steps, 49B tokens)
- 1,500 steps (0.74B tokens) covers only the early training phase
- OpenWebText 1% subset limits data diversity
- Single seed run reduces statistical reliability
- Conclusions are limited to: **"relative validation loss trends in the early training
  phase under constrained conditions"**, not final convergence performance
