# Adam-mini 재현 실험

> Zhang et al., *Adam-mini: Use Fewer Learning Rates To Gain More* (ICLR 2025, [arXiv:2406.16793](https://doi.org/10.48550/arXiv.2406.16793)) 의 핵심 주장을 제한된 자원 환경에서 재현하고 비판적으로 검증한 프로젝트입니다.

인공신경망과 딥러닝 기말 프로젝트 / 6팀 (권준호, 다리아, 박민정)

- 본 프로젝트 저장소: https://github.com/MJforge/Adam-mini
- 원 논문 저자 코드: https://github.com/zyushun/Adam-mini

---

## 저장소 구성 및 실행 방법

### 디렉토리 구조

```
Adam-mini-main/
├── original_code/                        # 원 논문 저자 공개 코드 (변경 없음)
│   ├── adam_mini/
│   │   └── adam_mini.py                  # 핵심 옵티마이저 구현체
│   ├── examples/
│   │   ├── gpt2/                         # NanoGPT 기반 GPT-2 사전학습
│   │   ├── llama/                        # TorchTitan 기반 Llama 사전학습
│   │   └── RLHF/                         # DeepSpeed + ReMax RLHF 파인튜닝
│   └── setup.py
├── experiment_1/                         # Effectiveness 재현
│   ├── train_gpt2.py                     # W&B 로깅 추가, optimizer 선택 분기 수정
│   ├── config/
│   │   └── train_gpt2_small_1gpu.py      # 단일 GPU 하이퍼파라미터 설정
│   ├── data/openwebtext/
│   │   └── prepare_mini.py               # OpenWebText 1% 서브셋 전처리
│   └── README.md
├── experiment_2/                         # Lightweightness 재현
│   ├── experiment_meta_memory.py         # Meta Device 이론값 시뮬레이션
│   └── experiment_real_gpu_memory.py     # 실제 GPU VRAM 측정
├── experiment_3/                         # Efficiency 재현 + Case Study
│   ├── case_study_1_hessian.ipynb        # Hessian 블록 대각 구조 시각화
│   ├── case_study_2_quadratic.ipynb      # Random Quadratic 수렴 비교
│   ├── run_throughput.sh                 # Llama 2-7B Throughput 측정 스크립트
│   ├── throughput_config.toml            # Llama 2-7B 학습 설정
│   └── README.md
└── README.md
```

`experiment_1/train_gpt2.py`는 `original_code/examples/gpt2/`의 `model.py`, `configurator.py`, `logger.py`를 `sys.path`로 직접 참조하므로 `original_code`를 삭제하면 안 된다.

세 실험 폴더는 논문의 핵심 주장 하나씩을 맡는다.

| 폴더 | 논문 주장 | 내용 |
|------|----------|------|
| `experiment_1/` | **Effectiveness** | 메모리를 줄여도 AdamW와 유사하게 수렴하는가 |
| `experiment_2/` | **Lightweightness** | 옵티마이저 상태 메모리가 모델 크기·정밀도와 무관하게 50% 절감되는가 |
| `experiment_3/` | **Efficiency** + Case Study | 메모리 절감이 더 큰 배치와 높은 throughput으로 이어지는가; 설계 동기 검증 포함 |

---

### 환경 요구사항

| 실험 | 환경 | 추가 패키지 |
|------|------|------------|
| experiment_1 | 로컬 (GPU 권장) | `datasets`, `tiktoken`, `tqdm`, `wandb` |
| experiment_2 (Meta Device) | 로컬 (CPU 가능) | `transformers` |
| experiment_2 (실제 GPU) | 로컬 (GPU 필요) | `transformers` |
| experiment_3 Case Study 1, 2 | Google Colab | — |
| experiment_3 Throughput | RunPod 2×A100-80GB | TorchTitan 의존성 포함 |

공통 사전 설치:

```bash
pip install -e ./original_code    # adam_mini 패키지 설치
```

---

### 실험별 실행 방법

#### experiment_1 — Effectiveness (GPT-2 125M 사전학습)

```bash
cd experiment_1
pip install datasets tiktoken tqdm wandb
wandb login

# 데이터 준비 (OpenWebText 1% 서브셋, 약 140~160 MB)
python data/openwebtext/prepare_mini.py

# Adam-mini vs AdamW (n_head=12, 기본 설정)
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini --comment=gpt2_small_adam_mini_nhead12
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw --comment=gpt2_small_adamw_nhead12

# 파티셔닝 민감도 검증 (n_head=6)
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adam_mini --n_head=6 --comment=gpt2_small_adam_mini_nhead6
python train_gpt2.py config/train_gpt2_small_1gpu.py \
    --algorithm=adamw --n_head=6 --comment=gpt2_small_adamw_nhead6
```

W&B 프로젝트 이름은 `config/train_gpt2_small_1gpu.py`의 `wandb_project` 항목에서 변경한다.

#### experiment_2 — Lightweightness (메모리 분석)

```bash
pip install transformers

# Meta Device 이론값 (GPU 불필요, GPT-2-1.5B ~ Llama 2-13B)
python experiment_2/experiment_meta_memory.py

# 실제 GPU VRAM 측정 (float32 / float16)
python experiment_2/experiment_real_gpu_memory.py
```

#### experiment_3 — Efficiency + Case Study

**Case Study 1, 2** (Google Colab에서 실행):

- [`experiment_3/case_study_1_hessian.ipynb`](experiment_3/case_study_1_hessian.ipynb) — Colab에서 열고 셀 순서대로 실행
- [`experiment_3/case_study_2_quadratic.ipynb`](experiment_3/case_study_2_quadratic.ipynb) — 동일

**Throughput** (RunPod 2×A100-80GB 환경):

```bash
# throughput_config.toml을 TorchTitan 설정 디렉토리에 복사 후 실행
cp experiment_3/throughput_config.toml original_code/examples/llama/train_configs/
bash experiment_3/run_throughput.sh
```

Llama 2-7B 접근을 위해 HuggingFace 토큰이 사전에 준비되어 있어야 한다.

---

## 1. Background

옵티마이저는 손실이 줄어드는 방향과 폭, 즉 어느 방향으로 얼마나 크게 파라미터를 바꿀지 결정하는 학습 알고리즘이다. Adam(과 그 변형인 AdamW)은 대규모 언어 모델 학습의 사실상 표준이지만, 파라미터마다 1차 모멘텀 `m`과 2차 모멘텀 `v`를 따로 저장한다. 파라미터가 70억 개면 `m`도 `v`도 각각 70억 개를 메모리에 올려야 하고, 분산 학습을 써도 GPU 간 통신 한계 때문에 속도가 떨어진다.

Adam-mini는 여기서 출발한다. 신경망의 Hessian을 들여다보면 완전히 뒤섞인 형태가 아니라 **블록 대각(block-diagonal)** 구조에 가깝다. 몇 개의 그룹 안에서는 파라미터들이 강하게 상호작용하지만 그룹 사이의 상호작용은 약하다. Adam-mini는 이 점을 이용해 비슷한 파라미터들을 하나의 블록으로 묶고 블록마다 학습률(2차 모멘텀 `v`)을 하나만 둔다. 이렇게 하면 `v`의 99.9% 이상을 제거할 수 있고, 옵티마이저 상태 메모리가 약 절반으로 줄어든다.

논문의 핵심 주장은 세 가지다.

| 주장 | 내용 |
|---|---|
| 경량성 (Lightweightness) | Adam의 `v` 중 99.9% 이상을 제거해 옵티마이저 메모리를 약 50% 절감 |
| 효과성 (Effectiveness) | 메모리를 줄였음에도 AdamW와 비슷하거나 더 나은 성능 |
| 효율성 (Efficiency) | Llama 2-7B 사전학습 시 AdamW 대비 49.6% 높은 처리량(throughput) |

본 프로젝트는 위 세 주장을 재현하는 실험 3개와, 주장을 뒷받침하는 동기 실험(case study) 2개, 논문의 한계를 직접 건드려 보는 비판적 검증 실험 2개로 구성된다.

---

## 2. Motivation Experiments (Case Study)

논문이 Adam-mini의 설계 근거로 제시하는 두 가지 관찰을 작은 규모에서 직접 재현했다.

### 2-1. Block-Diagonal Structure of the Hessian

논문에 코드가 공개되어 있지 않아 비슷한 구성의 신경망을 직접 만들어 확인했다. MNIST를 사용하는 은닉층 1개짜리 MLP(은닉 뉴런 8개)를 학습하면서 Hessian을 시각화했다. 그룹 내부의 상호작용이 강하고 그룹 사이는 약한 블록 대각 패턴이 학습 전 구간에서 유지됐다. 신경망 파라미터가 모두 뒤섞이는 게 아니라 그룹 단위로 동작한다는 의미로, 블록 단위로 학습률을 묶어도 되는 근거가 된다.

### 2-2. Random Quadratic Problem

무작위 2차 함수 최적화 문제에서 단일 학습률 / 블록별 학습률 / 파라미터별 학습률(Adam) 방식을 비교했다. 전체 문제에서는 파라미터마다 다른 학습률이 도움이 됐지만, 하나의 밀집(dense) 블록 안에서만 보면 잘 고른 학습률 하나로도 충분했다. "블록 하나당 학습률 하나"라는 Adam-mini의 압축 전략이 성립하는 조건을 보여주는 결과다.

---

## 3. Reproduction Experiments

각 실험은 독립적으로 읽을 수 있도록 목적 → 조건 → 결과 → 분석 순으로 정리했다.

### 3-1. Lightweightness

모델 크기와 무관하게 Adam-mini의 옵티마이저 상태 메모리가 AdamW 대비 정확히 50% 절감되는지 검증했다.

논문은 80GB VRAM 서버 클러스터에서 직접 측정했는데, 일반 Colab에서 Llama 2-13B나 Llama 3-8B를 올리면 바로 OOM이 터진다. 이를 피하려고 PyTorch의 **Meta Device**를 활용했다. 파라미터의 실제 값을 메모리에 올리지 않고 텐서 형태만 만들어 두는 가상 장치로, CUDA 캐싱 노이즈 없이 옵티마이저 상태 메모리만 분리해 계산할 수 있다. 측정 대상은 GPT-2-1.5B, Llama 2-1B, Llama 2-7B, Llama 3-8B, Llama 2-13B다.

| 모델 | AdamW | Adam-mini | 절감률 |
|---|---|---|---|
| GPT-2-1.5B | 11.61 GB | 5.80 GB | 50.0% |
| Llama 2-1B | 9.40 GB | 4.70 GB | 50.0% |
| Llama 2-7B | 50.21 GB | 25.10 GB | 50.0% |
| Llama 3-8B | 59.83 GB | 29.92 GB | 50.0% |
| Llama 2-13B | 96.98 GB | 48.49 GB | 50.0% |

캐싱 노이즈를 배제한 이론값 기준으로는 모든 스케일에서 50.0%로 일치했다. 실제 GPU에서의 결과는 [Section 4-2](#4-2-physical-gpu-memory-reduction)에서 따로 다룬다.

### 3-2. Effectiveness

GPT-2 125M 사전학습에서 Adam-mini의 validation loss가 AdamW와 초기 구간에서 비슷하게 수렴하는지 확인했다.

논문과 같은 8×A100 80GB 환경은 현실적으로 어려워 단일 GPU로 줄였다. 옵티마이저 간 비교의 공정성을 유지하기 위해 step당 토큰 수(`tokens/iter`)는 논문과 동일하게 맞췄다.

| 항목 | 논문 | 본 재현 |
|---|---|---|
| GPU | A100 SXM 80GB × 8 | A100 PCIe 80GB × 1 |
| max_iters | 100,000 | 1,500 |
| batch_size | 60 | 24 |
| gradient_accumulation_steps | 8 | 20 |
| tokens/iter | 491,520 | 491,520 (동일) |
| 전체 학습 토큰 | ~49B | ~0.74B |
| 데이터셋 | OpenWebText (전체) | OpenWebText 1% subset (~80M tokens) |

데이터는 17GB짜리 OpenWebText 전체를 내려받는 대신 Hugging Face 스트리밍으로 1%(80,000개 문서)만 가져왔다. 원본과 같은 포맷(`uint16`)의 `train.bin`/`val.bin`을 만들었기 때문에 학습 코드는 건드리지 않았고, 두 옵티마이저가 완전히 같은 데이터를 같은 순서로 보게 된다. 절대 성능은 낮아질 수 있어도 상대 비교에는 논리적 문제가 없다.

| 지표 | AdamW | Adam-mini | 해석 |
|---|---|---|---|
| val/loss | 4.238 | 4.393 | AdamW가 근소하게 우수하나 비슷한 수렴 |
| train/loss | 3.923 | 4.090 | AdamW가 학습 데이터에 약간 더 적합 |
| max VRAM | 26.30 GB | 25.81 GB | Adam-mini가 메모리를 덜 사용 |
| 학습 시간 | 5,377 s | 5,362 s | Adam-mini가 약간 더 짧음 |

1,500 step(~0.7B 토큰)이어서 loss 곡선만으로 "완전히 재현했다"고 하기는 어렵다. 다만 초기 수렴 양상은 원 논문 그래프와 비슷했고, Adam-mini가 더 적은 메모리로 비슷한 수렴과 근소하게 짧은 학습 시간을 보였다. 49B 토큰 전체를 돌린다면 논문과 같은 곡선에 수렴할 것으로 본다.

### 3-3. Efficiency

Adam-mini의 메모리 절감이 "더 큰 batch size → GPU 간 통신 감소 → 처리량 향상"으로 실제로 이어지는지 확인했다.

Llama 2-7B를 2×A100-80GB에서 AdamW와 Adam-mini 각각으로 30 step 학습하며 batch size를 1부터 올려가면서 최대 batch size와 peak throughput을 비교했다.

| 항목 | 논문 | 본 재현 | 변경 이유 |
|---|---|---|---|
| GPU | A800-80GB | A100 SXM-80GB | A800은 중국 내수용 GPU라 클라우드에서 제공되지 않음. A100은 메모리·연산 성능이 동일하고 NVLink 대역폭만 다름 (A800 400GB/s, A100 600GB/s) |
| 데이터셋 | C4 전체 | C4-mini | throughput은 초당 처리 토큰 수라 데이터셋 크기가 측정값에 영향을 주지 않음. 전체 C4는 수백 GB라 다운로드가 비효율적 |
| 학습 step | 전체 사전학습 | 30 step | throughput은 step당 처리 속도이므로 30 step으로도 동일한 비교가 가능 |
| PyTorch | 2023~2024년 버전 | 2.8 (2026년) | RunPod 환경의 기본 제공 버전 사용 |

| 옵티마이저 | Batch/GPU | Throughput (wps) | 메모리 사용 | Memory Retry | 상태 |
|---|---|---|---|---|---|
| AdamW | 1 | 5,165 | 79.2% | 0 | 정상 |
| AdamW | 2 | 7,104 | 79.2% | 0 | 실질적 최대 |
| AdamW | 3 | 5,630 | 86.6% | 87 | 성능 저하 |
| Adam-mini | 1 | 4,446 | 49.8% | 0 | 정상 |
| Adam-mini | 2 | 6,371 | 58.3% | 0 | 정상 |
| Adam-mini | 3 | 7,371 | 70.7% | 0 | 실질적 최대 |
| Adam-mini | 4 | 6,335 | 83.4% | 58 | 성능 저하 |

같은 batch=1에서 Adam-mini는 AdamW보다 GPU 메모리를 29.4%p 덜 썼다(49.8% vs 79.2%). 이 여유 덕분에 50% 더 큰 배치(3 vs 2)를 안정적으로 처리했고, 결과적으로 3.8% 높은 peak throughput을 달성했다(7,371 vs 7,104 wps). "메모리 절약 → 배치 증가 → 처리량 향상"이라는 논문의 핵심 메커니즘은 그대로 재현됐다.

**논문과의 차이.**

| 항목 | 논문 | 본 재현 |
|---|---|---|
| AdamW 최대 batch | 1 | 2 |
| Adam-mini 최대 batch | 4 | 3 |
| Throughput 향상 (실질적 최대 기준) | +49.6% | +3.8% |
| Throughput 향상 (batch=1 기준) | +49.6% | +42.7% |

향상률 수치 자체는 논문(+49.6%)과 차이가 크지만 출발점이 달라진 결과다.

1. **AdamW batch=2가 가능해진 이유.** PyTorch 2.8의 FSDP 구현이 구버전보다 내부 오버헤드가 작아 같은 80GB에서 activation 공간을 더 확보한다. 이 개선은 옵티마이저 종류와 무관하게 양쪽 모두에 적용되어, 논문에서 OOM이던 AdamW batch=2가 본 재현에서는 가능해졌다.
2. **Adam-mini batch=4가 실패한 이유.** Adam-mini의 옵티마이저 메모리 절약량(GPU당 약 13GB)은 PyTorch 버전과 무관하게 일정하다. 다만 논문이 공개하지 않은 세부 설정(activation checkpointing 여부, FSDP sharding 전략 등)에 따라 배치당 activation 비용이 달라질 수 있고, 본 환경에서는 batch=4의 activation이 여유 공간을 넘어섰다.
3. **향상률이 줄어든 이유.** AdamW의 출발점이 batch=1에서 batch=2로 올라갔기 때문이다. Adam-mini의 고정 절약량이 만드는 추가 배치 수는 같지만 비교 출발점이 높아져 상대적 증가폭이 줄었다(논문 1→4 = 4배 vs 재현 2→3 = 1.5배). batch=1을 같은 기준으로 두고 비교하면 +42.7%로 논문의 +49.6%에 근접한다.
4. **NVLink의 영향.** A100의 NVLink(600GB/s)가 A800(400GB/s)보다 빨라 FSDP 통신 오버헤드가 작다. Adam-mini의 장점 중 하나인 통신량 감소 효과가 상대적으로 덜 부각되어 향상률 축소에 일부 기여했다. 단 메모리 용량은 동일하므로 배치 크기에는 영향이 없다.

향상률 수치는 환경 차이로 논문에 미치지 못했으나, batch=1 기준 +42.7%로 근접했고 핵심 메커니즘은 동일하게 재현됐다.

---

## 4. Critical Validation

### 4-1. Sensitivity to Partitioning Granularity

Adam-mini의 핵심 가정은 같은 블록 안의 파라미터들은 최적 학습률이 비슷하다는 것이다. GPT-2 기준으로 Attention Q/K/V/O는 head 단위, MLP는 뉴런(output row) 단위, Embedding은 전체를 하나의 파티션으로 묶는다. 파티셔닝 자체가 Transformer 구조를 전제로 한다는 뜻이다.

이 가정이 얼마나 민감한지 보기 위해, 효과성 실험과 동일한 조건에서 `n_head`만 12에서 6으로 줄여 GPT-2 125M을 학습했다. 블록이 더 크고 거칠어지는 셈이다.

| 조건 | Adam-mini val/loss | AdamW val/loss | 격차 |
|---|---|---|---|
| n_head=12 | 4.393 | 4.238 | +0.155 |
| n_head=6 | 4.507 | 4.264 | +0.243 |

블록을 절반으로 거칠게 나누자 AdamW와의 격차가 0.155에서 0.243으로 약 57% 벌어졌다. 파티션이 거칠수록 Adam-mini가 불리해진다는 뜻으로, MoE처럼 구조가 다르거나 파티션 단위가 큰 모델에 적용할 때는 이 민감도를 고려해야 한다.

### 4-2. Physical GPU Memory Reduction

경량성 실험의 50% 절감이 실제 GPU에서도 같은 수준으로 나타나는지 확인했다.

Google Colab의 T4 GPU에서 GPT-2 모델의 `optimizer.step()` 전후 VRAM 변화를 측정했다. 현업에서 많이 쓰는 half precision도 포함하기 위해 float32와 float16 두 조건에서 측정했다.

| 포맷 | AdamW | Adam-mini | 절감률 |
|---|---|---|---|
| float32 | 966.65 MB | 483.73 MB | 50.0% |
| float16 | 488.95 MB | 242.25 MB | 50.5% |

연산 정밀도와 무관하게 실제 학습 파이프라인에서도 약 50%의 절감이 유지됐다. 이론값과 실측값이 일치한다.

---

## 5. Conclusion

**재현된 핵심 주장.**

- **경량성**: Meta Device 역산과 실제 T4 GPU 측정 양쪽에서, 모델 크기·연산 정밀도와 무관하게 옵티마이저 상태 메모리가 약 50% 절감됨을 확인했다.
- **효과성**: 제한된 환경(1% 서브셋, 1,500 iter)에서 Adam-mini는 AdamW와 유사한 초기 수렴 양상을 보였고, 더 적은 메모리로 근소하게 짧은 학습 시간을 기록했다.
- **효율성**: Adam-mini의 메모리 절약(29.4%p)이 더 큰 배치(2→3)로 이어져 3.8% 높은 peak throughput을 달성했다. batch=1 기준으로는 +42.7%로 논문에 근접했고, 핵심 메커니즘이 동일하게 작동함을 확인했다.
- **비판적 검증**: `n_head`를 줄이자 AdamW와의 격차가 57% 벌어져, Adam-mini의 성능이 파티셔닝 설계에 민감하다는 점을 실험으로 확인했다.

**한계.**

- 효과성 실험은 단일 GPU, 1% 서브셋(~0.7B tokens), 1,500 iter로 축소했기 때문에 초기 수렴 구간만 비교할 수 있었다.
- 효율성 실험은 논문 시점 이후 PyTorch의 GPU 메모리 관리 효율이 개선되어 완전히 동일한 환경·결과를 재현하지는 못했다.
- 보다 일반적으로, Adam-mini는 블록을 어떻게 나누느냐에 성능이 크게 좌우되고 하이퍼파라미터 재튜닝이 필요하다. 또한 표준적인 dense 트랜스포머 구조를 전제로 설계되어 GNN 등 비정형 구조에서는 같은 절감을 보장하기 어렵고, 속도 이점을 온전히 누리려면 Triton 기반 GPU 커널 환경이 필요하다.

---

## 6. Contributions

| 이름 | 담당 |
|---|---|
| 권준호 (26510118) | 효율성(Efficiency) 실험, Block Diagonal Structure test, 발표 |
| 박민정 (25512085) | 효과성(Effectiveness) 실험, 비판적 검증 1(파티셔닝), GitHub 관리 |
| 다리아 (25512084) | 경량성(Lightweightness) 실험, 비판적 검증 2(Physical VRAM) |
| 공동 | 발표자료(PPT) 제작 |

---

## 7. AI Tool Usage

본 프로젝트에서는 Claude와 Gemini를 다음 용도로 사용했다.

- **논문 번역 및 이해 검증**: 팀이 핵심이라고 판단한 실험과 contribution이 적절한지 검증
- **실험 환경 설정**: 논문의 실험 환경을 팀 자원에 맞게 이론적·비용 효율적으로 scale down
- **실험 코드 작성**: 공개된 코드의 패키지 관리·디버깅, 비공개 코드의 경우 유사 결과를 내기 위한 구조와 초안 작성
- **원인 분석 일부**: 결과가 논문과 다를 때 팀의 분석·예상 원인이 타당한지 검증·토의
- **문서 작성 일부**: README 세부 내용의 정리와 전체 구성 검증

---

## 8. References

- **Adam-mini** — Zhang, Y., Chen, C., Li, Z., Ding, T., Wu, C., Kingma, D. P., Ye, Y., Luo, Z.-Q., & Sun, R. (2024). *Adam-mini: Use fewer learning rates to gain more.* arXiv. https://doi.org/10.48550/arXiv.2406.16793
- **Adam** — Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization.* arXiv. https://arxiv.org/abs/1412.6980
- **AdamW** — Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* arXiv. https://arxiv.org/abs/1711.05101
- **PyTorch Meta Device** — https://docs.pytorch.org/docs/2.12/meta.html
