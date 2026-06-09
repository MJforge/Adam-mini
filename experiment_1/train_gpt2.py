# Project: Reproduction of GPT-2 Pretraining with the Adam-mini Optimizer
# Paper: Adam-mini: Use Fewer Learning Rates to Gain More
# Based on:
#   https://github.com/zyushun/Adam-mini/tree/main/examples/gpt2
#
# Original copyright:
#   Copyright (c) 2024 Yushun Zhang, et al. (Adam-mini Authors)
#
# Modifications:
#   Copyright (c) 2026 Min-jeong Park
#
# Licensed under the same Apache-2.0 license.

"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train_gpt2.py config/train_gpt2_small_1gpu.py --algorithm=adam_mini

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train_gpt2.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train_gpt2.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train_gpt2.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

"""
Changes from the original nanoGPT train.py:

Dynamic Optimizer Selection: Integrated conditional branching to select the optimizer based on
the algorithm variable in the configuration file (adamw or adam_mini).

Custom Initialization for Adam-mini: Configured model_sharding=False tailored for a single GPU
environment and implemented parameter mapping for dim and n_heads.

W&B Integration and Granular Logging: Extended logging to include not only loss but also tokens
per second (tok_per_sec), maximum VRAM allocation (max_vram_gb), gradient norm (grad_norm),
and current learning rate (lr). All metrics use tokens_B as the custom x-axis step metric.

Execution Time Profiling and Text Logging: Added fine-grained latency measurement for
Forward/Backward passes, Gradient Clipping, and Optimizer steps, with results saved to a
local log file (e.g., logger_loss_time.txt).

Progress-based Checkpoint Strategy: Modified the checkpointing logic to trigger snapshots at
predefined intervals (ckpt_interval) as well as specific progress milestones (1%, 25%, 50%,
75%, and 100% of max_iters).

Bug Fix: Moved raise ValueError for unsupported algorithm out of the adam_mini branch into
a proper else clause to prevent incorrect errors when adam_mini is selected.
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

_script_dir = os.path.dirname(os.path.abspath(__file__))
_orig_gpt2  = os.path.normpath(os.path.join(_script_dir, '..', 'original_code', 'examples', 'gpt2'))
sys.path.insert(0, _orig_gpt2)   # for logger.py
sys.path.insert(0, _script_dir)  # for model.py — must come first to shadow original_code version

from model import GPTConfig, GPT
from adam_mini import Adam_mini
import logger
import wandb


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
resume_dir = None
eval_interval = 1000
ckpt_interval = 1000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
init_from = 'scratch'
load_iter = 0
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
epsilon = 1e-8
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
seed = 1337
comment = 'none'
algorithm = 'adam_mini'
flash_attn = True
# W&B logging
use_wandb = False
wandb_project = 'adam-mini-reproduction'
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'

print('current dtype', dtype)

save_dir = 'log_gpt2/' + comment

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join(_orig_gpt2, 'configurator.py')).read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

os.makedirs(save_dir, exist_ok=True)

logger_loss_train = logger.Logger('{}/logger_loss_train.txt'.format(save_dir), title='logger_loss_iter')
logger_loss_train.set_names(['iteration', 'trainloss'])
logger_loss_val = logger.Logger('{}/logger_loss_val.txt'.format(save_dir), title='logger_loss_iter')
logger_loss_val.set_names(['iteration', 'valloss'])
logger_loss_time = logger.Logger('{}/logger_loss_time.txt'.format(save_dir), title='logger_time_iter')
logger_loss_time.set_names(['iteration', 'forward backward time', 'clipping time', 'optimizer step time'])

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# W&B initialization — define tokens_B as the custom x-axis for all metrics
if use_wandb and master_process:
    wandb.init(
        project=wandb_project,
        name=comment,
        config=config,
    )
    wandb.define_metric("tokens_B")
    wandb.define_metric("train/*",       step_metric="tokens_B")
    wandb.define_metric("val/*",         step_metric="tokens_B")
    wandb.define_metric("throughput/*",  step_metric="tokens_B")
    wandb.define_metric("gpu/*",         step_metric="tokens_B")
    wandb.define_metric("opt/*",         step_metric="tokens_B")

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float64': torch.float64}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loader
data_dir = os.path.join('data', dataset)

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0

print('load_iter = ', load_iter, 'loading ..', load_iter)

if load_iter == 0:
    init_from = 'scratch'
else:
    init_from = 'resume'

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, flash_attn=flash_attn, device=device)
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    if resume_dir is None:
        resume_dir = out_dir
    print(f"Resuming training from {resume_dir}")
    ckpt_path = os.path.join(resume_dir, 'ckpt' + str(load_iter) + '.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    print('loading complete')

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer selection
if algorithm == 'adamw':
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
elif algorithm == 'adam_mini':
    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
        model_sharding=False,  # single GPU setting
        dim=n_embd,
        n_heads=n_head
    )
else:
    raise ValueError(f"algorithm '{algorithm}' not supported. Choose 'adamw' or 'adam_mini'.")

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None  # free up memory

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    t_eval = time.time()
    for split in ['val']:  # skip train loss estimation to save time
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    print('validation done. time used =', time.time() - t_eval)
    model.train()
    return out


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


X, Y = get_batch('train')  # fetch the very first batch


def train():
    global iter_num, X, Y
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate and checkpoint
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            logger_loss_val.append([iter_num, losses['val']])
            print(f"step {iter_num}:  val loss {losses['val']:.4f}")
            if use_wandb:
                wandb.log({
                    "val/loss":  losses['val'].item(),
                    "tokens_B":  iter_num * tokens_per_iter / 1e9,
                }, step=iter_num)

        if master_process and (iter_num > 0 and iter_num % ckpt_interval == 0 or iter_num in [round(max_iters * 0.01), round(max_iters * 0.25), round(max_iters * 0.5), round(max_iters * 0.75), round(max_iters * 1 - 1)]):
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt' + str(iter_num) + '.pt'))

        if iter_num == 0 and eval_only:
            break

        t_f_b = time.time()
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch('train')
            scaler.scale(loss).backward()

        t_f_b_e = time.time() - t_f_b

        t_clip = time.time()
        grad_norm = torch.tensor(0.0)
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        t_clip_e = time.time() - t_clip

        t_step = time.time()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        t_step_e = time.time() - t_step

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, "
                  f"forward backward time {t_f_b_e}s, clipping time {t_clip_e}s, optimizer step time {t_step_e}s")

            logger_loss_train.append([iter_num, lossf])
            logger_loss_time.append([iter_num, t_f_b_e, t_clip_e, t_step_e])

            if use_wandb and master_process:
                wandb.log({
                    "tokens_B":               iter_num * tokens_per_iter / 1e9,
                    "train/loss":             lossf,
                    "throughput/tok_per_sec": tokens_per_iter / dt,
                    "gpu/max_vram_gb":        torch.cuda.max_memory_allocated() / 1e9,
                    "opt/grad_norm":          grad_norm.item(),
                    "opt/lr":                 lr,
                }, step=iter_num)

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    if use_wandb and master_process:
        wandb.finish()
    if ddp:
        destroy_process_group()


train()
