# 논문의 동기 실험 두 개(Case Study 1, 2)와 Throughput 실험에 대하여 작업한 폴더

ICLR 2025 논문 [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793) 재현 실험.

Adam-mini는 Adam의 2차 모멘텀(v)을 블록 단위로 축약해서 옵티마이저 메모리를 절반으로 줄이는 방법임. 메모리만 아끼는 게 아니라, 그 덕분에 더 큰 배치를 쓸 수 있어 학습 속도(throughput)도 빨라진다고 주장함.

본 레포에는 논문의 동기 실험 두 개(Case Study 1, 2)와 성능 검증 실험 하나(Throughput)를 재현한 결과가 들어있음.

## 파일 구성

```
.
├── case_study_1_hessian.ipynb       # Hessian 시각화
├── case_study_2_quadratic.ipynb     # Random Quadratic 실험
├── throughput/
│   ├── run_throughput.sh            # RunPod 실험 명령어
│   ├── throughput_config.toml       # Llama 2-7B 학습 설정
│   └── visualize_throughput.ipynb   # 결과 그래프 코드
└── figures/                         # 보고서용 그림들
```

## Case Study 1 — Hessian이 정말 블록 대각인가?

**학습 전 구간에서 신경망 Hessian의 블록 대각 구조를 재현함.** 세 시점(0%, 50%, 100%) 모두에서 대각 블록의 평균 절댓값이 비대각 블록의 3.4~5.4배로 나타남.

논문은 신경망 Hessian이 블록 대각 구조를 가진다는 관찰에서 출발함. 같은 뉴런에 속한 파라미터끼리는 상호작용이 강하고, 다른 뉴런과는 약하다는 얘기임. 이를 확인하기 위해 8개 뉴런짜리 MLP를 MNIST로 학습시키면서 학습 진행도 0% / 50% / 100% 시점에 Hessian을 직접 계산하여 시각화함.

논문은 CIFAR-100을 사용했으나, Hessian이 너무 커서(~2.5GB) Colab에서 다루기 어려워 MNIST로 변경함. 블록 구조는 데이터셋이 아니라 신경망 구조에서 비롯되므로 결론에는 영향 없음. 다만 클래스 수 차이(10 vs 100)로 인해 논문보다 대비가 다소 약하게 나타남.

실행: `case_study_1_hessian.ipynb`을 Colab에서 열고 위에서부터 셀 실행.

## Case Study 2 — 블록당 학습률 하나면 충분한가?

**블록 대각 이차함수에서 블록별 학습률 하나가 Adam의 파라미터별 학습률보다 빠르게 수렴함을 확인함.** Dense 서브블록 내부에서는 단일 학습률 GD가 Adam을 능가하여, 파라미터별 학습률이 dense 블록 안에서 불필요함을 보임.

논문이 주장하는 핵심은 "블록 대각 Hessian이면 블록마다 학습률 하나씩만 줘도 Adam보다 빠르게 수렴한다"는 것임. 이를 검증하기 위해 3-블록 대각 Hessian(각 30차원, 고유값 {1,2,3} / {99,100,101} / {4998,4999,5000})을 만들고 다음 세 가지를 비교함.

- 단일 학습률 GD
- 블록별 학습률 GD
- Adam (β₁=0)

전체 문제에서는 **블록별 GD > Adam > 단일 GD** 순으로 수렴함. Dense한 한 블록만 떼어놓고 보면 **단일 GD > Adam**으로 역전됨. 두 결과 모두 논문과 일치함. 후자가 Adam-mini의 설계 동기에 해당함.

실행: `case_study_2_quadratic.ipynb`을 Colab에서 실행.

## Throughput — 실제로 빨라지는가?

**Adam-mini가 AdamW보다 더 큰 배치(3 vs 2)를 안정적으로 처리하며 3.8% 높은 throughput을 달성함.** 다만 논문의 +49.6% 향상에는 미치지 못함. 그 원인은 PyTorch 버전 차이로 추정됨.

Llama 2-7B 학습에서 배치 크기를 1부터 올려가며 AdamW와 Adam-mini의 throughput을 비교함. 논문은 A800을 사용했으나 클라우드에서 A800은 제공되지 않아 A100으로 대체함. 메모리(80GB)는 동일하며 NVLink 속도만 다름. 30 step만 측정해도 throughput은 안정적으로 도출 가능함.

| Optimizer | Batch | wps   | Memory | Retry |
|-----------|-------|-------|--------|-------|
| AdamW     | 1     | 5,165 | 79.2%  | 0     |
| AdamW     | 2     | 7,104 | 79.2%  | 0     |
| AdamW     | 3     | 5,630 | 86.6%  | 87    |
| Adam-mini | 1     | 4,446 | 49.8%  | 0     |
| Adam-mini | 2     | 6,371 | 58.3%  | 0     |
| Adam-mini | 3     | 7,371 | 70.7%  | 0     |
| Adam-mini | 4     | 6,335 | 83.4%  | 58    |

Retry가 0이 아닌 행은 PyTorch가 메모리 할당을 재시도하면서 throughput이 떨어진 경우임. 형식적으로는 OOM이 아니지만 실질적으로는 실패한 것으로 판단함.

깨끗하게 돌아간 최대 배치를 비교하면 **AdamW batch=2 (7,104 wps) vs Adam-mini batch=3 (7,371 wps)**. Adam-mini가 한 단계 더 큰 배치를 처리하고 throughput도 3.8% 높음. 배치=1 기준 메모리 사용량은 79.2% → 49.8%로 약 30%p 감소함.

### 논문과의 차이

논문은 AdamW가 batch=1만 가능했고 Adam-mini는 batch=4까지 가능하여 +49.6% 향상을 보고함. 본 재현에서는 AdamW batch=2, Adam-mini batch=3으로 향상폭이 +3.8%에 그침.

원인은 PyTorch 버전 차이로 추정됨. 논문은 2024년 상반기에 작성되었고 당시 PyTorch는 2.2~2.3 버전임. 본 재현은 RunPod 기본 환경인 PyTorch 2.8 사용. 2년 사이 FSDP의 메모리 관리가 크게 변경되어 AdamW도 batch=2까지 처리 가능해진 것으로 보임. Adam-mini가 절약하는 메모리(GPU당 ~13GB)는 PyTorch 버전과 무관하게 일정하나, 비교 출발점이 높아져 상대적 이점이 축소됨.

논문이 PyTorch 정확한 버전을 명시하지 않아 이 부분은 확정할 수 없음. 다만 batch=1 기준 비교 시 +42.7%로 논문의 +49.6%에 근접하여, "메모리 절약 → 배치 증가 → throughput 향상"이라는 핵심 메커니즘은 동일하게 재현됨.

### 실행 방법

1. RunPod에서 2×A100-80GB Pod 생성 (PyTorch 템플릿)
2. `throughput_config.toml`을 `examples/llama/train_configs/`에 복사
3. `bash run_throughput.sh` 실행

스크립트가 의존성 설치부터 모든 배치 실험을 차례로 실행함. HuggingFace 토큰 사전 준비 필요 (Llama 2-7B 접근 권한 필요).

## 환경

- Case Study 1, 2: Google Colab Pro
- Throughput: RunPod, 2× A100-SXM4-80GB, PyTorch 2.8, CUDA 12.8
