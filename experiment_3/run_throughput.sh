#!/bin/bash
# ============================================================
# Throughput Measurement: AdamW vs Adam-mini
# Reproducing Table 2 of Adam-mini (ICLR 2025)
#
# Environment: 2× A100-80GB (RunPod)
# Model: Llama 2-7B
# Codebase: https://github.com/zyushun/Adam-mini (examples/llama)
# ============================================================

# ----- 1. Setup -----
git clone https://github.com/zyushun/Adam-mini.git
cd Adam-mini/examples/llama

pip install adam-mini
pip install -r requirements.txt
pip install torchdata torch-optimizer wandb tensorboard

# ----- 2. HuggingFace Login -----
# Requires: Meta Llama 2 access approved on huggingface.co
hf auth login

# ----- 3. Download Tokenizer -----
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Llama-2-7b-hf

# ----- 4. Download Data -----
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1B16KpuhUyz4p7mwc9xmRHuyCY37dAw-2 -O ./torchtitan/datasets/c4_mini/

# ----- 5. Fix Compatibility (PyTorch 2.8) -----
sed -i 's/^from adafactor import/# from adafactor import/' train.py

# ----- 6. Run Experiments -----
export USE_LIBUV=1
CONFIG="./train_configs/llama2_7b_throughput.toml"
RUN="torchrun --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint=localhost:0 train.py --job.config_file $CONFIG"

echo "===== AdamW batch=1 ====="
$RUN --optimizer.name AdamW --training.batch_size 1 2>&1 | tee logs/adamw_batch1.log

echo "===== AdamW batch=2 ====="
$RUN --optimizer.name AdamW --training.batch_size 2 2>&1 | tee logs/adamw_batch2.log

echo "===== AdamW batch=3 ====="
$RUN --optimizer.name AdamW --training.batch_size 3 2>&1 | tee logs/adamw_batch3.log

echo "===== Adam-mini batch=1 ====="
$RUN --optimizer.name adam_mini --training.batch_size 1 2>&1 | tee logs/adammini_batch1.log

echo "===== Adam-mini batch=2 ====="
$RUN --optimizer.name adam_mini --training.batch_size 2 2>&1 | tee logs/adammini_batch2.log

echo "===== Adam-mini batch=3 ====="
$RUN --optimizer.name adam_mini --training.batch_size 3 2>&1 | tee logs/adammini_batch3.log

echo "===== Adam-mini batch=4 ====="
$RUN --optimizer.name adam_mini --training.batch_size 4 2>&1 | tee logs/adammini_batch4.log

echo "All experiments completed!"
