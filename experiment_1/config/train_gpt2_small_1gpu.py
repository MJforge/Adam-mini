# GPT-2 Small (125M) — Single GPU config
# Hardware: A100 PCIe 80GB
#
# tokens/iter = batch_size x gradient_accumulation_steps x block_size
#             = 24 x 20 x 1024 = 491,520  (same as original paper)
#
# Changes from original:
#   batch_size: 60 -> 24 (single GPU)
#   gradient_accumulation_steps: 8 -> 20 (preserving tokens/iter)
#   max_iters: 100,000 -> 1,500 (scaled down for feasibility)
#   lr_decay_iters: 100,000 -> 1,500 (preserving LR schedule shape)
#   warmup_iters: 2,000 -> 30 (2% of max_iters)

batch_size = 24
block_size = 1024
gradient_accumulation_steps = 20

n_layer = 12
n_head  = 12
n_embd  = 768
dropout = 0.0
bias    = False

max_iters      = 1500
lr_decay_iters = 1500
warmup_iters   = 30        # 2% of max_iters

eval_interval  = 50        # 30 validation checkpoints total
eval_iters     = 50
log_interval   = 10
ckpt_interval  = 500

# ── optimizer — change to 'adamw' for comparison experiment ────────────────
algorithm     = 'adam_mini'
learning_rate = 6e-4
weight_decay  = 0
beta1         = 0.9
beta2         = 0.95
epsilon       = 1e-8
grad_clip     = 1.0

decay_lr     = True
min_lr       = 3e-5

# ── W&B ────────────────────────────────────────────────────────────────────
use_wandb     = True
wandb_project = 'adam-mini-reproduction'

comment  = 'gpt2_small_adam_mini_1gpu'
save_dir = 'log_gpt2/' + comment
out_dir  = 'out-gpt2/' + comment
