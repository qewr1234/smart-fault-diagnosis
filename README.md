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
├── config.py            # 모든 하이퍼파라미터 / 경로 설정
├── train.py             # 학습 전용 진입점 (실행 디렉터리를 남긴다)
├── predict.py           # 추론 전용 진입점 (실행 디렉터리만 읽는다)
├── src/
│   ├── utils.py         # 시드, 메트릭, 확률 유틸
│   ├── features.py      # FE 변환의 compute_* / apply_* 쌍
│   ├── pipeline.py      # FeaturePipeline — 학습된 전처리 통계 저장/재사용
│   ├── models.py        # 아키텍처 3종, SAM, 손실, TorchPredictor(체크포인트)
│   ├── expert.py        # 하드클러스터 LightGBM 전문가 + 리랭킹
│   ├── calibration.py   # 온도/바이어스/EM/Sinkhorn 균형 배정
│   ├── blend.py         # OOF 그리디 가중치 탐색 (prob/logit/geomean)
│   ├── postprocess.py   # 후처리 파라미터 선택 + 중첩(정직한) 추정
│   └── artifacts.py     # 실행 디렉터리 규약
├── tests/               # pytest (합성 데이터, CPU)
├── data/                # train.csv, test.csv, sample_submission.csv (git 미포함)
└── outputs/             # 실행 디렉터리 (git 미포함)
```

### 실행 디렉터리 레이아웃

```
outputs/<VERSION>_<timestamp>/
├── config.json              실행에 쓰인 설정 전체
├── label_classes.json       예측 라벨 복원용 클래스
├── feature_pipeline.joblib  학습된 전처리 통계
├── postprocess.json         블렌드/전문가/바이어스/균형배정 파라미터
├── metrics.json             OOF 점수 + 정직한 중첩 추정치
├── oof_probs.npy            후처리된 OOF 확률
└── seeds/seed_<s>/fold_<i>/{isoforest.joblib, model_<arch>.pt, expert.joblib}
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

학습과 추론이 분리되어 있습니다. `train.py`는 실행 디렉터리를 남기고,
`predict.py`는 그 디렉터리만 읽어 제출 파일을 만듭니다.

```bash
python train.py                              # 학습 + 제출 파일 (시드 3개 × 5-Fold)
python train.py --fast                       # 스모크 테스트 (시드 1개, epoch 축소)
python train.py --no-predict                 # 학습만 (제출은 나중에)
python predict.py outputs/<VERSION>_<ts>     # 저장된 모델로 추론만
python predict.py outputs/<VERSION>_<ts> --test-csv new_batch.csv
```

`train.py`는 마지막에 `predict.py`를 호출하므로, 매 실행이 저장된 산출물의
복원 가능성을 함께 검증합니다. 추론에는 `test.csv`만 있으면 되고 학습 데이터는
필요하지 않습니다.

## 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # 전체 (엔드투엔드 스모크 포함, CPU 약 2분)
python -m pytest -q -m "not slow"   # 단위 테스트만
```

`tests/`는 실제 데이터 없이 합성 센서 데이터로 동작합니다. 검증 범위:

- 학습 루프의 정규화 조합(SAM / R-Drop / MixUp)과 세 아키텍처의 완주
- 체크포인트 왕복이 확률을 그대로 재현하는지
- `FeaturePipeline`이 저장/복원 후 같은 행렬을 내는지, 참조 프레임에서 X_11 분위수만 쓰는지
- 전문가 리랭킹의 확률 질량 보존, 후처리 게이트의 비퇴행성
- 중첩 추정이 선택 편향을 실제로 드러내는지 (신호 없는 데이터에서 `selection_f1 > nested_f1`)
- 학습 후 추론이 `train.csv` 없이 동작하고, 두 번 실행해도 결과가 같은지

GitHub Actions(`.github/workflows/tests.yml`)에서 push마다 실행됩니다.

## 주요 설정 (config.py)

| 키 | 기본값 | 설명 |
|---|---|---|
| `CV_FOLDS` | 5 | StratifiedKFold 수 |
| `NESTED_CALIBRATION` | True | early stopping/온도를 fold 내부 홀드아웃에서 결정 |
| `INNER_VAL_FRAC` | 0.15 | 내부 홀드아웃 비율 |
| `NESTED_POSTPROCESS_FOLDS` | 5 | 후처리 기여도의 정직한 추정에 쓸 fold 수 (0=생략) |
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

## 검증 정직성 (OOF 편향 방지)

OOF 점수를 테스트 성능의 추정치로 쓰려면, 그 점수를 만든 데이터로 아무것도 고르지
않아야 합니다. 두 지점에서 이를 강제합니다.

**1. fold 내부 홀드아웃 (`NESTED_CALIBRATION`)**
각 fold의 train 부분에서 다시 `INNER_VAL_FRAC`(기본 15%)를 떼어, early stopping의
best epoch와 온도 스케일링 계수를 그 안에서만 고릅니다. 바깥 fold는 오직 채점에만
쓰이므로 OOF가 낙관 편향되지 않습니다. 클래스당 표본이 부족해 층화 분할이 불가능하면
자동으로 폴백하며, 그 경우 `metrics.json`의 `nested_calibration`이 `false`가 되고
실행 로그에 경고가 남습니다.

**2. 후처리의 중첩 추정 (`NESTED_POSTPROCESS_FOLDS`)**
블렌드 가중치, 전문가 가중치, 로짓 바이어스, 균형 배정은 모두 OOF에서 고릅니다.
따라서 그 OOF로 잰 점수는 자기가 고른 파라미터로 자기를 채점한 값입니다.
`metrics.json`은 두 숫자를 분리해 보고합니다.

| 키 | 의미 |
|---|---|
| `raw_blend_f1` | 후처리 전 시드 평균 점수 |
| `selection_f1` | 후처리를 고른 행에서 잰 점수 (낙관적, 참고용) |
| `nested_f1` | 선택에 쓰지 않은 행에서만 잰 점수 (정직한 추정) |

후처리의 실제 기여는 `nested_f1 - raw_blend_f1`으로 읽어야 합니다.
`selection_f1`과 `nested_f1`의 격차가 크면 후처리가 OOF에 과적합된 것입니다.

## 그 밖의 재현성 주의

- 전처리 통계(윈저 경계, FDI 표준화 평균/표준편차, 중앙값, 임퓨터)는 train에서만
  산출해 `feature_pipeline.joblib`에 저장하고, 추론은 그 값을 그대로 재사용합니다.
- `X_11` 클리핑만 test 피처 분포(분위수)를 참조하는 transductive 변환입니다.
  라벨은 보지 않으며, 참조 프레임의 다른 컬럼은 전혀 쓰이지 않습니다(테스트로 검증).
- 이상치 스코어(IsolationForest)는 폴드별로 학습하고 폴드별로 저장합니다.
- 후처리는 여전히 OOF 개선 시에만 적용되는 게이트 구조입니다.

## 라이선스

MIT
