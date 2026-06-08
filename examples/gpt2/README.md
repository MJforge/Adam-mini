# GPT-2 Pre-training Reproduction: Adam-mini vs AdamW

원 논문 [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793)의 GPT-2 125M pre-training 실험을 단일 GPU 환경에서 축소 재현한 실험입니다.

---

## 실험 개요

### 목적

- Adam-mini와 AdamW의 validation loss 감소 경향 비교
- Adam-mini의 파티셔닝 granularity(head 수)에 따른 성능 민감도 분석 (비판적 검토)
- 논문 환경과 다른 제약 조건에서 두 optimizer의 상대적 동작 특성 확인

### 환경 제약 및 설계 결정

| 항목 | 논문 원본 | 본 실험 |
|------|----------|---------|
| GPU | A100 SXM 80GB × 8 | A100 PCIe 80GB × 1 |
| `max_iters` | 100,000 | 1,500 |
| `batch_size` | 60 | 24 |
| `gradient_accumulation_steps` | 8 | 20 |
| `tokens/iter` | 491,520 | 491,520 (동일 유지) |
| 총 학습 토큰 | ~49B | ~0.74B |
| 데이터셋 | OpenWebText 전체 | OpenWebText 1% subset (~80M tokens) |

`tokens/iter`를 논문과 동일하게 유지하기 위해 `batch_size`와 `gradient_accumulation_steps`를 조정했습니다.

---

## 파일 구조

```
examples/gpt2/
├── train_gpt2.py                      # 메인 학습 스크립트 (W&B 통합, grad_norm 캡처 포함)
├── model.py                           # GPT-2 모델 정의
├── configurator.py                    # 커맨드라인 인수 파싱
├── logger.py                          # 로컬 loss 로그 저장
├── io_utils.py                        # 파일 입출력 유틸리티
├── plot_val_loss.py                    # 로컬 로그 기반 시각화 스크립트
├── config/
│   └── train_gpt2_small_1gpu.py       # 단일 GPU 실험용 하이퍼파라미터 설정
└── data/
    └── openwebtext/
        └── prepare_mini.py            # OpenWebText subset 준비 스크립트
```

---

## 데이터 준비 방식

### 설계 배경

HuggingFace `datasets`의 스트리밍(`streaming=True`)을 활용해 전체 17GB를 다운로드하지 않고 필요한 문서 수(`N_DOCS`)만 순차적으로 받아 토크나이징합니다. 이후 결과를 `train.bin`, `val.bin`으로 저장하면 학습 중에는 `numpy.memmap`으로 디스크에서 직접 랜덤 접근합니다.

```
[사전 준비 - 1회]
HuggingFace 스트리밍 → 토크나이징(tiktoken GPT-2 BPE) → train.bin / val.bin 저장

[학습 loop]
numpy.memmap으로 랜덤 위치 샘플링 → GPU 전송 → forward / backward
```

일반 스트리밍 학습은 매 step마다 네트워크에서 데이터를 받아오느라 GPU가 대기해야 하는 반면, 이 방식은 학습 중 GPU가 쉬지 않으며 완전한 랜덤 샘플링도 가능합니다.

### 데이터 준비 실행

```bash
cd examples/gpt2
pip install datasets tiktoken tqdm
python data/openwebtext/prepare_mini.py
```

생성 파일:
- `data/openwebtext/train.bin` : 약 140~160MB, ~72M tokens
- `data/openwebtext/val.bin`   : 약 15~18MB,  ~8M tokens

---

## 실험 1: Adam-mini vs AdamW (n_head=12)

### 하이퍼파라미터

```python
# config/train_gpt2_small_1gpu.py
batch_size                 = 24
gradient_accumulation_steps = 20
block_size                 = 1024   # tokens/iter = 491,520
n_layer, n_head, n_embd    = 12, 12, 768   # GPT-2 Small 125M
max_iters                  = 1500
lr_decay_iters             = 1500
warmup_iters               = 30           # max_iters의 2%
learning_rate              = 6e-4
min_lr                     = 3e-5
weight_decay               = 0
beta1, beta2               = 0.9, 0.95
grad_clip                  = 1.0
seed                       = 1337
eval_interval              = 50
eval_iters                 = 50
```

### 실행 명령어

```bash
cd examples/gpt2

# Adam-mini
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --comment=gpt2_small_adam_mini_nhead12

# AdamW
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw \
    --comment=gpt2_small_adamw_nhead12
```

### 결과

| Optimizer | val/loss | train/loss |
|-----------|---------|-----------|
| Adam-mini | 4.393   | 4.091     |
| AdamW     | 4.238   | 3.923     |
| **gap**   | **+0.155** | — |

1,500 steps(약 0.74B tokens) 구간에서 Adam-mini가 AdamW 대비 val/loss 기준 0.155 높음. 초기 학습 구간에서 Adam-mini는 AdamW와 유사한 감소 경향을 보이나 절대값에서 약간 뒤처짐.

---

## 실험 2: 파티셔닝 민감도 분석 (n_head=6)

### 실험 목적

Adam-mini는 attention head 단위로 파라미터를 파티셔닝하여 파티션당 하나의 학습률을 사용합니다. n_head를 줄이면 파티션 수가 감소하는데, 이때 Adam-mini와 AdamW의 상대적 성능 격차가 변화하는지 확인합니다.

### 핵심 질문

```
n_head=12에서 Adam-mini와 AdamW의 gap = A
n_head=6에서  Adam-mini와 AdamW의 gap = B

A ≠ B → Adam-mini의 성능이 head 수(파티셔닝 granularity)에 민감함
```

### 실행 명령어

```bash
# Adam-mini, n_head=6
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini \
    --n_head=6 \
    --comment=gpt2_small_adam_mini_nhead6

# AdamW, n_head=6
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw \
    --n_head=6 \
    --comment=gpt2_small_adamw_nhead6
```

`--n_head=6`을 커맨드라인에서 직접 오버라이드할 수 있습니다. `n_embd=768`은 6으로 나눠지므로(head_size=128) 별도 수정 불필요합니다.

### 결과

| 조건 | Adam-mini val/loss | AdamW val/loss | gap |
|------|-------------------|----------------|-----|
| n_head=12 | 4.393 | 4.238 | +0.155 |
| n_head=6  | 4.507 | 4.264 | **+0.243** |

n_head를 12에서 6으로 줄이자 Adam-mini와 AdamW의 val/loss 격차가 0.155에서 0.243으로 **57% 확대**되었습니다.

### 해석

파티션 수가 절반으로 줄었을 때 Adam-mini가 AdamW 대비 상대적으로 더 불리해졌습니다. 이는 Adam-mini의 성능이 파티셔닝 granularity(head 수)에 의존한다는 논문의 한계를 실험적으로 지지하는 결과입니다.

---

## Throughput 분석 및 주의사항

### 요약 값의 함정

W&B summary의 throughput 값은 마지막 step(eval + checkpoint 저장 포함)의 순간값으로, 실제 training 속도를 대표하지 않습니다.

```
step별 throughput 패턴:
  일반 training step:        ~141,000–143,900 tok/s
  eval/checkpoint step:      ~37,000–75,000  tok/s  ← 마지막 step이 여기에 해당
```

### 실제 training throughput (step별 시계열 기준)

| 조건 | Adam-mini | AdamW | 차이 |
|------|-----------|-------|------|
| n_head=12 | ~141,100 tok/s | ~141,300 tok/s | 거의 없음 |
| n_head=6  | ~143,500 tok/s | ~143,700 tok/s | 거의 없음 |

순수 training step 기준으로 Adam-mini와 AdamW의 throughput 차이는 GPT-2 125M 규모에서 유의미하지 않습니다. n_head=6이 n_head=12보다 약 1.5% 빠른 것은 Flash Attention의 head_size 효율 차이(128 vs 64)에 기인합니다.

---

## W&B 로깅

학습 중 다음 지표가 Weights & Biases에 기록됩니다.

| 지표 | 설명 |
|------|------|
| `val/loss` | validation loss |
| `train/loss` | training loss |
| `tokens_B` | 누적 학습 토큰 수 (단위: Billion) — x축 기준 |
| `throughput/tok_per_sec` | 초당 처리 토큰 수 |
| `gpu/max_vram_gb` | 최대 VRAM 사용량 |
| `opt/grad_norm` | gradient norm |
| `opt/lr` | 현재 learning rate |

x축은 step이 아닌 `tokens_B`로 설정되어 있어 논문과 동일한 "Tokens(B) vs Validation Loss" 그래프를 W&B에서 바로 확인할 수 있습니다.

W&B 프로젝트명은 `config/train_gpt2_small_1gpu.py`의 `wandb_project` 값으로 설정합니다.

---

## 환경 설정

```bash
# 저장소 클론 후
pip install -e .
pip install wandb torch-optimizer datasets tiktoken tqdm

# W&B 로그인
wandb login
```

---

## 실험의 한계

- GPU 자원 제약으로 원 논문의 전체 pre-training 규모(100K steps, 49B tokens)를 재현하지 못함
- 1,500 steps(0.74B tokens)는 초기 학습 구간만 관찰 가능
- OpenWebText 1% subset 사용으로 데이터 다양성이 제한됨
- seed 단일 실행으로 결과의 통계적 신뢰도가 낮음
- 따라서 본 실험의 결론은 "두 optimizer의 최종 성능 우위 판별"이 아닌 **"제한된 초기 학습 구간에서의 상대적 동작 경향 비교"** 로 해석해야 함

---

## 원 논문 및 원본 코드

- 논문: [Adam-mini: Use Fewer Learning Rates To Gain More (arXiv 2406.16793)](https://arxiv.org/abs/2406.16793)
- 원본 저장소: [zyushun/Adam-mini](https://github.com/zyushun/Adam-mini)
