# Smart Equipment Fault Diagnosis (DACON)

장비 센서 데이터(X_01~X_52) 기반 21-클래스 고장 진단 파이프라인입니다.
FT-Transformer / TabMixer / GLU-MLP 딥러닝 3종과 LightGBM 하드클러스터 전문가 모델을
5-Fold CV로 학습하고, OOF 기반 로짓 블렌드와 게이트형 후처리로 Macro-F1을 최적화합니다.

## 핵심 아이디어

### 1. 점수 병목 진단
클래스별 F1을 분석하면 대부분의 클래스는 F1 0.9~1.0으로 쉽게 분리되지만,
**{0, 3, 9, 15, 19} 클러스터가 F1 0.2~0.47로 상호 혼동**되며 Macro-F1을 지배합니다.
이 클러스터의 클래스 평균 차이는 표준편차의 5% 수준으로 거의 겹쳐 있어,
전역 모델 개선보다 클러스터 특화 전략이 효율적입니다.

### 2. FDI 잔차 피처 (analytical redundancy)
고장 진단(Fault Detection & Isolation) 문헌의 고전적 접근을 적용했습니다.
데이터에 물리적으로 중복 계측된 센서쌍이 다수 존재합니다:

- 완전 중복 (corr = 1.0): `X_06=X_45`, `X_10=X_17`
- 준중복 (|corr| > 0.95): `X_04–X_39`, `X_05–X_25`, `X_26–X_30`, `X_38–X_47` 등 12쌍

정상 상태에선 중복 센서쌍의 잔차 ≈ 0, 센서 드리프트/편향 고장에선 잔차가 벌어지는
시그니처를 이용해 **표준화 잔차 피처**를 추가합니다. 홀드아웃 검증에서
Macro-F1 **0.7942 → 0.8017 (+0.0075)** 개선을 확인했습니다.

### 3. 하드클러스터 2단계 리랭킹
전역 모델이 {0,3,9,15,19} 중 하나로 예측한 샘플에 대해, 해당 5개 클래스만으로
학습한 LightGBM 전문가 모델의 로짓을 혼합해 클러스터 내부 확률을 재분배합니다.
혼합 가중치 w는 OOF Macro-F1로 선택하며, 개선이 없으면 자동으로 비활성화됩니다.

### 4. 게이트형 후처리
train/test가 클래스 균형에 가깝다는 점을 이용해 Sinkhorn 균형 배정(예측 분포를
균등하게 유도하는 로짓 보정)을 시험하되, **OOF에서 argmax보다 좋을 때만** 적용합니다.
온도 스케일링·로짓 바이어스 튜닝도 동일하게 OOF 게이트로 보호됩니다.

### 5. 학습 안정화
- AMP(FP16) + SAM(AMP-safe) 옵티마이저
- MixUp + R-Drop + Label Smoothing
- MC-Dropout 추론 평균, 시드 3개 평균

## 저장소 구조

```
smart-fault-diagnosis/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py            # 모든 하이퍼파라미터 / 경로 설정
├── train.py             # 원클릭 실행 진입점
├── src/
│   ├── utils.py         # 시드, 메트릭, 확률 유틸
│   ├── features.py      # FE + FDI 잔차 피처 + X_11 클리핑
│   ├── models.py        # FT-Transformer / TabMixer / GLU-MLP, SAM, 손실
│   ├── expert.py        # 하드클러스터 LightGBM 전문가 + 리랭킹
│   ├── calibration.py   # 온도/바이어스/EM/Sinkhorn 균형 배정
│   └── blend.py         # OOF 그리디 가중치 탐색 (prob/logit/geomean)
├── data/                # train.csv, test.csv, sample_submission.csv (git 미포함)
└── outputs/             # 제출 파일 저장 (git 미포함)
```

## 설치

```bash
git clone <repo-url>
cd smart-fault-diagnosis
pip install -r requirements.txt
```

GPU(A100 권장) + CUDA 환경을 가정합니다. CPU에서도 동작하지만 학습이 느립니다.
Colab에서 실행하면 Google Drive가 자동 마운트됩니다(`config.py`의 경로 수정).

## 데이터 준비

DACON 대회 페이지에서 데이터를 받아 `data/`에 배치합니다:

```
data/
├── train.csv              # ID, X_01~X_52, target
├── test.csv               # ID, X_01~X_52
└── sample_submission.csv
```

로컬 경로가 다르면 `config.py`의 `DATA_DIR` / `OUT_DIR`을 수정하세요.

## 실행

```bash
python train.py                  # 전체 파이프라인 (시드 3개 × 5-Fold)
python train.py --fast           # 스모크 테스트 (시드 1개, epoch 축소)
python train.py --seeds 42       # 시드 지정
```

완료 시 `outputs/submission_<VERSION>_<timestamp>.csv`가 생성됩니다.

## 주요 설정 (config.py)

| 키 | 기본값 | 설명 |
|---|---|---|
| `CV_FOLDS` | 5 | StratifiedKFold 수 |
| `SEEDS` | [42, 52, 62] | 시드 평균 |
| `USE_FDI_FEATURES` | True | 잔차 + 편차 시그니처 피처 |
| `CLIP_X11` | True | X_11 train 극단값을 test 분위수로 클리핑 |
| `USE_EXPERT` | True | 하드클러스터 LightGBM 전문가 리랭킹 |
| `EXPERT_W_GRID` | [0.0, 0.25, 0.5] | 리랭킹 혼합 가중치 후보 (OOF 선택) |
| `USE_BALANCED_ASSIGN` | True | Sinkhorn 균형 배정 (OOF 게이트) |
| `USE_SAM` / `MIXUP_ALPHA` / `RDROP_ALPHA` | True / 0.2 / 0.5 | 학습 정규화 |
| `BLEND_MODE` | logit | 블렌드 방식 (prob/logit/geomean) |

## 검증 결과 (홀드아웃 20%, LightGBM 150 trees 기준)

| 설정 | Macro-F1 |
|---|---|
| 원본 52피처 | 0.7942 |
| + FDI 잔차/편차 피처 | 0.8017 |

DL 3종 + 전문가 블렌드의 전체 CV 점수는 실행 로그의 `[Final OOF (seed-avg)]`에서 확인합니다.

## 재현성 주의

- 모든 후처리(온도, 바이어스, 균형 배정, 전문가 가중치)는 OOF 개선 시에만 적용되는
  게이트 구조라, 데이터가 바뀌어도 성능 퇴행 위험이 낮습니다.
- 잔차 피처의 표준화 통계는 train에서만 산출해 리키지를 방지합니다.
- 이상치 스코어(IsolationForest)는 폴드별로 학습합니다.

## 라이선스

MIT
