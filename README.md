# 장비 센서 고장 진단 및 센서 고장 격리

Equipment Sensor Fault Diagnosis & Sensor Fault Isolation

장비 센서 52채널(X_01~X_52)의 스냅샷 한 행을 받아 21가지 고장 유형 중 어느 것인지 판정하는 도구입니다.
판정을 그대로 믿기 어려운 행에는 검토 플래그를 붙입니다. 같은 물리량을 중복 계측하는 센서들이 서로
어긋나면 어느 센서가 틀렸는지 지목하고, 둘 중 하나까지만 좁힐 수 있을 때는 그렇게 말합니다.
중복 쌍은 익명 채널의 상관으로 추정한 것이므로, 잔차 발화는 센서 고장일 수도 있고 두 센서의 관계가
바뀐 공정 변화일 수도 있습니다. 그래서 자동 판정이 아니라 검토 플래그로 냅니다.
안에는 딥러닝 분류기 앙상블이 들어 있지만, 출력은 모델 점수가 아니라 엔지니어가 바로 읽는 행 단위
진단 리포트(CSV)와 센서 격리 평가 리포트입니다. 데이터는 DACON 대회의 익명화된 장비 센서
데이터셋이며, 반도체 장비 데이터는 아닙니다.

**English summary**
- Input: one snapshot row of 52 anonymized sensor channels (X_01..X_52) from a DACON dataset with 21 fault classes. Not semiconductor equipment data.
- Output per row: predicted fault class, confidence, runner-up classes, review flags, and sensor-health columns (abnormal residuals, suspect sensor, candidate set, ambiguity flag).
- Sensor fault isolation: sensor pairs that track each other on train data are monitored through their residual. A residual beyond 3 sigma points at the sensor whose residuals all fired, or at a candidate set when the pattern cannot tell the sensors apart.
- Caveat: pairs are inferred from correlation, so a fired residual can also mean the process changed. Isolation is a review flag, not a verdict.
- Classifier inside: an ensemble of tabular models; a second-stage model is used only for the small group of fault classes the first stage confuses with each other.
- Validation: every tuning choice is scored on rows it was not chosen on; sensor isolation is validated by injecting known sensor biases (evaluate_isolation.py).
- Only real-data number here: LightGBM 20% holdout Macro-F1 (class-averaged F1) 0.7942 -> 0.8017 with residual features. Every other number comes from a synthetic smoke run and is labelled as such.
- Inference needs only test.csv. 137 CPU tests.

## 진단 흐름

| 단계 | 하는 일 | 파일 | 엔지니어가 보는 것 |
|---|---|---|---|
| 센서 스냅샷 | `test.csv` 의 `ID` + `X_01`~`X_52` 한 행 | `data/test.csv` | - |
| 전처리 / 잔차 생성 | train 에서 학습한 통계로 윈저링, 준중복 센서쌍 잔차, 행 통계 | `src/pipeline.py`, `src/features.py`, `src/discovery.py` | - |
| 분류기 앙상블 | 여러 모델의 판정을 합산한다 (구성은 4절) | `src/models.py`, `src/expert.py`, `src/blend.py` | - |
| 후처리 | 클래스 편향 보정. 검증 데이터에서 개선될 때만 적용 (3절) | `src/postprocess.py`, `src/calibration.py` | - |
| 진단 리포트 | 행별 확신도, 검토 플래그, 센서 건강 열 | `src/report.py` | `diagnosis_*.csv` |
| 센서 격리 | 잔차 발화 패턴으로 의심 센서 지목 | `src/isolation.py` | `suspect_sensor` 열 |
| 격리 성능 평가 | 고장 주입으로 탐지율 / 격리 정확도 / 오경보율 측정 | `evaluate_isolation.py` | `isolation_report.md` |

**한계 요약** (자세한 내용은 [한계와 실제 장비 데이터로의 확장](#한계와-실제-장비-데이터로의-확장))

- 행마다 독립 스냅샷입니다. trace data 가 없어 시간축 분석(CUSUM / EWMA)은 하지 않습니다.
- 센서 고장 정답이 데이터에 없습니다. 격리 성능은 고장 주입으로만 검증했습니다.
- 중복 센서쌍은 익명 채널의 상관으로 추정했습니다. 잔차는 공정 변화에도 발화할 수 있습니다.
- 실데이터 숫자는 LightGBM 홀드아웃 한 건입니다. 나머지 숫자는 모두 합성 데이터 예시입니다.

## 1. 센서 잔차로 고장을 탐지하고 센서를 격리한다 (FDI)

FDI 는 Fault Detection and Isolation 의 약자로 제어공학 용어입니다. FDC 의 탐지(D)에 "어느 센서인가" 를
짚는 격리(I)를 더한 것입니다.

**원리 (analytical redundancy).** 같은 물리량을 재는 센서 a, b 는 정상 상태에서 함께 움직입니다.
표준화 잔차 `r_a_b = (x_a - mu_a)/sd_a - (x_b - mu_b)/sd_b` 는 정상이면 0 근처에 머물고, 한쪽 센서가
드리프트하거나 편향되면 그 센서가 들어간 잔차만 벌어집니다. FDC 에서 서로 일치해야 하는 두 계측값의
차이를 별도 모니터 항목으로 등록하고 한계선을 두는 것과 같은 원리입니다. 여기서는 그 한계선이 train
잔차 분포의 3 sigma 입니다.

- 전제가 하나 있습니다. 채널이 익명이라 중복 쌍은 train 상관(`|corr| >= 0.95`)으로 추정한 것입니다.
  두 센서가 같은 물리량을 재서 상관이 높은지, 공정 때문에 함께 움직여서 높은지 데이터만으로는 구분할 수
  없습니다. 그래서 잔차 발화는 "센서 고장" 일 수도 있고 "두 센서의 관계가 바뀐 공정 변화 또는 고장 유형"
  일 수도 있습니다. 이 도구는 그 발화를 자동 판정이 아니라 검토 사유(`sensor_residual`)와 후보 센서로 냅니다.

**격리 논리 (signature matching, `src/isolation.py`).** 센서 s 의 시그니처는 s 가 들어간 잔차의 집합입니다.
s 가 고장이면 그 잔차가 모두 발화하고 나머지는 조용해야 합니다. 발화한 잔차만 세는 것이 아니라 조용한
잔차를 알리바이로 씁니다.

| 상황 | 판정 |
|---|---|
| 쌍 2개를 가진 센서가 고장: 자기 잔차 2개 발화, 파트너들은 각각 1개씩만 | 그 센서를 `suspect_sensor` 로 단독 지목. 파트너들은 발화 잔차를 하나씩만 설명 |
| 허브 센서의 잔차 하나만 발화하고 그 파트너의 잔차는 모두 발화 | 허브의 조용한 잔차가 허브의 알리바이. 파트너를 단독 지목 |
| 1위 센서가 발화 잔차를 모두 설명하지만 자기 잔차 일부는 조용 (일치율 < 1) | 부분 드리프트 또는 파트너 동시 고장 가능. 단독 지목하지 않고 관련 센서 전체를 후보로 남김 |
| 쌍을 하나만 가진 두 센서 사이의 잔차만 발화 | 시그니처가 같아 구분 불가. `isolation_ambiguous=True`, 후보 둘을 모두 보고 |
| 1위 센서로 설명되지 않는 잔차가 더 발화 | 다중 고장 가능. 발화 잔차에 등장한 센서 전체를 후보로 남김 |
| 어느 쌍에도 없는 센서 | 잔차로는 격리 불가. `coverage()["uncovered"]` 에 보고 |

순위는 설명되는 발화 잔차 수, 시그니처 일치율, `|z|` 합의 순서로 매기고, 1위가 하나뿐이면서 시그니처가
발화 패턴과 정확히 일치할 때만 단독 지목합니다. 추측하지 않고 "둘 중 하나" 라고 말하는 것이 현장에서
추가 계측 한 번으로 끝나는 길입니다. 판정은 `isolation_note` 열에 한 줄 한국어로도 적힙니다.

**고장 주입으로 검증합니다 (`evaluate_isolation.py`).** 데이터에 센서 고장 정답이 없으므로, 커버된 센서마다
파이프라인이 저장한 train 표준편차(`fdi_stats["sd"]`, 윈저링 후 기준)의 1, 2, 3, 5배 편향을 더해 넣고
탐지율(잔차가 하나라도 발화), top-1 격리(주입 센서를 단독 지목), 후보집합 포함(주입 센서가 후보에 있음),
모호 비율, 주입 전 데이터의 탐지율(오경보율)을 잽니다. 모델 체크포인트는 쓰지 않고 실행 디렉터리의
`config.json` 과 `feature_pipeline.joblib` 만 읽습니다.

합성 데이터 예시 (168행, 주입 크기 5 x sd, `|z| >= 3`; `isolation_report.md` 4절에서 복사). `X_01`~`X_04` 의
top-1 0.0% 와 모호 비율 100% 는 설계된 결과입니다. 각각 쌍 하나를 파트너와만 공유해 시그니처가 같으므로,
도구는 둘 중 하나를 찍는 대신 두 후보를 보고합니다.

| 센서 | 쌍 수 | 단독 격리 | 탐지율 | top-1 격리 | 후보집합 포함 | 모호 비율 |
|---|---|---|---|---|---|---|
| X_01 | 1 | X | 100.0% | 0.0% | 100.0% | 100.0% |
| X_02 | 1 | X | 100.0% | 0.0% | 100.0% | 100.0% |
| X_03 | 1 | X | 100.0% | 0.0% | 100.0% | 100.0% |
| X_04 | 1 | X | 100.0% | 0.0% | 100.0% | 100.0% |
| X_05 | 2 | O | 100.0% | 99.4% | 100.0% | 0.6% |
| X_06 | 2 | O | 100.0% | 100.0% | 100.0% | 0.0% |
| X_07 | 2 | O | 100.0% | 99.4% | 100.0% | 0.6% |

이 합성 실행에서 도출된 쌍은 5개, 커버 센서 7/52, 주입 크기 1~5 sd 모두 탐지율과 후보집합 포함률 100%
였습니다. 평균 top-1 42.7% 는 단독 격리 가능한 3/7 센서라는 구조적 상한에 거의 닿은 값이고, 잔차별
최소 탐지 가능 편향은 센서 sd 의 0.20~0.29배였습니다(`ResidualIsolator.detectability`).

커버리지는 데이터에 중복 계측이 얼마나 있는가로 결정되며, 도구는 격리 불가 센서를 `isolation_report.md`
1절에 전부 적습니다. 커버되지 않은 센서의 고장은 잔차로는 잡을 수 없고 분류기와 `anomaly_score` 에만
의존합니다. 주입 전 오경보는 2/168 = 1.2% 였습니다. 다만 실데이터처럼 고장 행이 섞인 데이터에서는 이
값이 순수 오경보율이 아니라 주입 전 탐지율(기준선)입니다. `isolation_report.md` 2절에 같은 주의가 적힙니다.

**쌍은 어떻게 찾나.** 센서쌍은 하드코딩하지 않습니다. `src/discovery.py` 의 `discover_sensor_redundancy` 가
train 상관행렬에서 매 실행 도출합니다.

| 기준 | 설정 키 (기본값) | 처리 |
|---|---|---|
| 완전 중복 | `EXACT_DUP_THRESHOLD` (0.9999) | union-find 로 묶어 대표 하나만 남김. 잔차가 항상 0 이므로 쌍 후보에서 제외 |
| 준중복 | `CORR_THRESHOLD` (0.95) | 상관이 강한 순으로 `MAX_REDUNDANT_PAIRS` (12) 개까지 |

원본 데이터에서는 완전 중복(`X_06`=`X_45`, `X_10`=`X_17`)과 준중복 12쌍이 관측됐습니다. 이 값은
`src/features.py` 의 `REDUNDANT_PAIRS`(12쌍)와 `EXACT_DUP_DROP`(제거 열 `X_45`, `X_17`)에 폴백 상수로만
남아 있고 기본 경로에서는 쓰이지 않습니다. z-score 기준은 윈저링 전 원본 train 잔차의 평균 / 표준편차이며
(`FeaturePipeline.fit` 이 `residual_stats` 에 저장), `FeaturePipeline.residual_zscores(df)` 가 그 기준으로
z-score 를 내므로 `|z| >= 3` 은 sigma 단위입니다. 센서 a 에 `k x sd_a` 의 편향이 생기면 잔차는 정확히 k 만큼
움직이므로, 상관이 강한 쌍일수록 작은 편향까지 잡습니다.

**잔차 피처는 분류에도 도움이 됩니다.** 실데이터로 측정한 유일한 숫자입니다.

| 설정 (홀드아웃 20%, LightGBM 150 trees) | Macro-F1 |
|---|---|
| 원본 52피처 | 0.7942 |
| + FDI 잔차 / 편차 피처 | 0.8017 |

이 측정은 저장소에 포함되지 않은 일회성 실험 스크립트(원본 `train.csv`, 20% 홀드아웃, LightGBM 150 trees)로
얻은 값이며, 이 저장소의 `train.py` 는 이 실험을 재현하지 않습니다. 단일 홀드아웃 한 번의 측정이라 잔차
피처의 보조 효과를 확인한 것 이상으로 읽으면 안 됩니다. 최종 앙상블의 CV 점수는 실행 로그와
`metrics.json` 에서 읽어야 하며 이 README 에는 적지 않습니다.

## 2. 진단 리포트: 엔지니어가 바로 쓰는 출력

`predict.py` 는 제출 파일 `submission_<VERSION>_<ts>.csv`(ID, target) 옆에 같은 타임스탬프로
`diagnosis_<VERSION>_<ts>.csv` 를 씁니다. 행 순서는 제출 파일과 같습니다 (`sample_submission.csv` 가 있으면
그 ID 순). 두 파일의 ID 가 일치하지 않으면 리포트는 `test.csv` 순으로 두고 경고를 남깁니다. 결합 키는 어느
경우든 `ID` 입니다.

| 그룹 | 열 | 뜻 |
|---|---|---|
| 예측 | `pred`, `confidence` | 최종 판정(원본 라벨, 제출 파일의 `target` 과 같다)과 1위 확률 |
| 예측 | `top2`, `top2_prob`, `top3`, `top3_prob` | 차순위 후보와 확률 |
| 예측 | `margin` | 1위와 2위의 확률 차. 균형 배정이 argmax 와 다른 클래스를 골랐으면 음수 |
| 검토 플래그 | `in_hard_cluster` | 예측 클래스가 혼동 클러스터에 속하는가 |
| 검토 플래그 | `review_reasons` | `low_confidence`(1위 확률 < 0.5), `ambiguous_top2`(margin < 0.10), `confusable_class`(클러스터 예측), `sensor_residual`(발화 잔차 1개 이상). `;` 로 이어 씀 |
| 검토 플래그 | `needs_review` | 위 사유 중 하나라도 있으면 True |
| 센서 건강 | `anomaly_score` | IsolationForest `score_samples` 의 fold 평균. 낮을수록 학습 분포에서 멂 |
| 센서 건강 | `n_abnormal_residuals`, `abnormal_residuals` | `\|z\| >= 3` 인 잔차 수와 `잔차:z` 목록 (`\|z\|` 내림차순) |
| 센서 건강 | `involved_sensors` | 발화 잔차에 등장한 센서, 빈도순 |
| 센서 건강 | `suspect_sensor`, `suspect_candidates` | 격리기가 단독 지목한 센서(없으면 빈 칸)와 후보 센서 전체(`;` 구분) |
| 센서 건강 | `isolation_ambiguous`, `isolation_note` | 단독 지목 실패 여부와 판정을 풀어 쓴 한 줄 한국어 메모 |

혼동 클러스터는 검증 예측의 혼동행렬에서 자동 도출됩니다 (4절). `anomaly_score` 에 절대 임계는 없습니다.
같은 실행 디렉터리로 낸 리포트 사이에서 상대 비교하거나, 검토 큐 임계를 운용 중에 정합니다 (한계 절의
open-set 행 참조).

임계값은 `predict.py` 의 `--z-threshold`(기본 3.0), `--low-confidence`(0.5), `--low-margin`(0.10) 으로 바꿀 수
있습니다. `abnormal_residuals` 와 격리 판정은 같은 `--z-threshold` 를 씁니다.

합성 데이터 예시 (센서 열만). 첫 행은 주입 없는 합성 test 에서, 둘째와 셋째 행은 같은 test 에 드리프트를
주입한 실행에서 가져왔습니다. 쌍 2개를 가진 `X_05` 에 +5 sd 를 0~19행에, 쌍 1개를 가진 `X_01` 에 +5 sd 를
20~29행에 더했습니다. 3 epoch 스모크 모델이라 확신도가 낮아 모든 행에 `low_confidence` 가 붙습니다.
여기서 볼 것은 센서 열입니다.

| ID | n_abnormal_residuals | abnormal_residuals | involved_sensors | suspect_sensor | suspect_candidates | isolation_ambiguous | review_reasons |
|---|---|---|---|---|---|---|---|
| TEST_0145 | 0 | | | | | False | low_confidence |
| TEST_0000 | 2 | r_X_05_X_07:+73.7;r_X_05_X_06:+73.3 | X_05;X_07;X_06 | X_05 | X_05 | False | low_confidence;ambiguous_top2;sensor_residual |
| TEST_0020 | 1 | r_X_01_X_02:+70.6 | X_01;X_02 | | X_01;X_02 | True | low_confidence;ambiguous_top2;sensor_residual |

`involved_sensors` 는 발화한 잔차에 등장한 센서를 빈도순으로 센 것이고, `suspect_sensor` 는 발화하지 않은
잔차까지 알리바이로 써서 시그니처가 맞는 센서만 지목한 것입니다. 둘째 행에서 `X_06`, `X_07` 은 고장난
`X_05` 의 파트너라서 끌려 들어왔을 뿐이고, 조용한 `r_X_06_X_07` 이 두 센서의 알리바이입니다.

`predict.py` 는 끝에 요약을 출력합니다. 아래는 위 드리프트 주입 실행의 출력 그대로입니다 (합성 데이터 + 주입 예시):

```
[진단 요약]
  전체 168행 중 검토 필요 168행 (100.0%)
  검토 사유: 확신도 낮음(low_confidence)=168, 1·2위 근접(ambiguous_top2)=128, 혼동 클러스터(confusable_class)=53, 센서 잔차 이상(sensor_residual)=32
  평균 확신도 0.172 / 혼동 클러스터 예측 53행
  센서 잔차 이상 32행 (의심 센서: X_06x22, X_05x21, X_07x21, X_01x10, X_02x10)
  격리 단독 지목 20행 (상위 3: X_05x20) / 격리 모호 12행
  이상치 점수 범위 -0.500 ~ -0.395 (IsolationForest, 낮을수록 학습 분포에서 멂)
  예측 클래스 분포: 0:4, 1:3, 2:7, 3:12, 4:6, 5:9, 6:8, 7:5, 8:23, 9:5, 10:7, 11:15, 12:8, 13:7, 14:5, 15:4, 16:4, 17:6, 18:14, 19:4, 20:12
```

단독 지목 20행은 `X_05` 를 주입한 20행 전부입니다. 모호 12행은 `X_01` 주입 10행에, 주입 전부터 `|z|` 3.0~3.3
으로 발화해 있던 2행(TEST_0044, TEST_0072)을 더한 것입니다. 빈도순에서는 `X_06` 이 22행으로 1위지만
격리기는 `X_06` 을 한 번도 지목하지 않았습니다.

## 3. 검증을 스스로 속이지 않는다

오진의 비용을 아는 사람의 검증입니다. OOF 점수를 성능 추정치로 쓰려면, 그 점수를 만든 행으로 아무것도 고르지
않아야 합니다. 두 지점에서 강제합니다.

**fold 내부 홀드아웃 (`NESTED_CALIBRATION`).** 각 fold 의 train 부분에서 `INNER_VAL_FRAC`(0.15)를 다시 떼어
early stopping 의 best epoch 와 온도 스케일링 계수를 그 안에서만 고릅니다. 바깥 fold 는 채점에만 쓰입니다.
층화 분할이 불가능하면 폴백하고, `metrics.json` 의 `nested_calibration` 이 `false` 가 되며 로그에 경고가 남습니다.

**후처리의 중첩 추정 (`NESTED_POSTPROCESS_FOLDS`).** 블렌드 가중치, 전문가 가중치, 로짓 바이어스, 균형 배정은
모두 OOF 에서 고릅니다. 그 OOF 로 잰 점수는 자기가 고른 파라미터로 자기를 채점한 값입니다. 그래서 선택
절차를 다시 K-fold 로 감싸, 선택에 쓰지 않은 행에서만 잰 점수를 따로 보고합니다.

| 키 | 의미 | 합성 스모크 실행 값 |
|---|---|---|
| `raw_blend_f1` | 후처리 전 시드 평균 점수 | 0.6028 |
| `selection_f1` | 후처리를 고른 행에서 잰 점수 (낙관적, 참고용) | 0.6766 |
| `nested_f1` | 선택에 쓰지 않은 행에서만 잰 점수 (정직한 추정) | 0.5960 |

합성 열은 630행, 2-fold, 3 epoch 스모크 실행의 `metrics.json` 값이라 분류 성능 자체는 의미가 없고, 읽는
법만 보면 됩니다. `selection_f1` 이 `nested_f1` 보다 0.08 높고 `nested_f1` 이 `raw_blend_f1` 보다도 낮으므로,
이 실행에서 후처리는 기여가 없었고 OOF 에 과적합했다고 읽어야 합니다. 후처리의 실제 기여는 항상
`nested_f1 - raw_blend_f1` 입니다. 실행 로그에서는 `[OOF] selection` 과 `[OOF] nested` 두 줄로 출력됩니다.

## 4. 분류기 구성 (요약)

이 절은 ML 독자를 위한 것입니다. 지원 엔지니어는 건너뛰어도 됩니다.

| 항목 | 구성 |
|---|---|
| 아키텍처 | `ft` (FTTransformerTiny), `mixer` (TabMixer), `glu` (GLUMLP). `USE_FT` / `USE_MIXER` / `USE_GLU` 로 선택 |
| 학습 정규화 | AMP(FP16) + SAM(`USE_SAM`, `SAM_RHO`), MixUp(`MIXUP_ALPHA`), R-Drop(`RDROP_ALPHA`), Label Smoothing(`LS`) |
| 추론 | MC-Dropout 평균(`MC_DROPOUT`, `MC_PASSES`), 시드 3개(`SEEDS`) 평균 |
| 보조 입력 | fold 별 IsolationForest 이상치 점수를 마지막 피처로 추가(`USE_ISOFOREST`) |
| 블렌드 | 시드별 OOF 그리디 가중치 탐색, `BLEND_MODE` = prob / logit / geomean |
| 전문가 | 혼동 클러스터 전용 LightGBM, 가중치는 `EXPERT_W_GRID` 에서 OOF 선택 |
| 보정 / 게이트 | 온도(`TEMP_MODE`)는 fold 내부 홀드아웃에서 결정. 로짓 바이어스(`USE_BIAS_TUNE`, `BIAS_LIM`)와 Sinkhorn 균형 배정(`USE_BALANCED_ASSIGN`)은 OOF 에서 개선될 때만 적용 |

**혼동 클러스터와 전문가 모델.** 클래스별 F1 을 보면 대부분의 고장 유형은 쉽게 분리되고, 소수의 집합이
서로 섞이며 Macro-F1 을 끌어내립니다. 원본 데이터에서는 {0, 3, 9, 15, 19} 가 그런 집합으로 관측됐습니다.
이 집합은 `src/postprocess.py` 의 `PostProcessSpec` 폴백 기본값으로만 남아 있고 기본 경로에서는 쓰이지
않습니다. `train.py` 가 매 실행 `discover_hard_cluster` 로 도출한 클러스터를 명시적으로 넘기기 때문입니다.
도출 방법은 이렇습니다. 각 fold 의 내부 홀드아웃 예측(바깥 fold OOF 가 아님)에서 혼동행렬을 만들고, F1
하위 `CLUSTER_F1_QUANTILE` 분위의 클래스 중 상호 오분류율이 `CLUSTER_MIN_CONFUSION` 이상인 쌍을 간선으로
이어 가장 큰 연결요소를 클러스터로 삼습니다(`CLUSTER_MAX_SIZE` 상한). 합성 스모크 실행에서는
{4, 10, 13, 14, 15, 17, 18, 19} 가 나왔습니다. 데이터가 다르면 클러스터도 달라진다는 뜻입니다. 내부
홀드아웃을 만들지 못해 폴백한 실행은 `metrics.json` 의 `hard_cluster_source` 가 `outer_fold` 가 되며, 그만큼
OOF 가 낙관적입니다.

클러스터가 잡히면 그 클래스들만으로 학습한 LightGBM 전문가(`src/expert.py`)가 전역 모델이 클러스터로 예측한
행에 한해 클러스터 내부 확률을 재분배합니다. 클러스터 밖 확률은 건드리지 않습니다. 혼합 가중치는
`EXPERT_W_GRID` 에서 OOF Macro-F1 로 고르고, 0 이 선택되면 미적용입니다. 서로 혼동되는 집합이 없으면 빈
결과가 돌아오고 전문가 모델은 자동으로 꺼집니다.

## 저장소 구조

```
smart-fault-diagnosis/
├── config.py                # 모든 하이퍼파라미터 / 경로 설정
├── train.py                 # 학습 진입점 (실행 디렉터리를 남긴다)
├── predict.py               # 추론 진입점 (제출 파일 + 진단 리포트)
├── evaluate_isolation.py    # 고장 주입으로 센서 격리 성능 평가
├── src/
│   ├── discovery.py         # 센서쌍 / shift 컬럼 / 시그니처 / 하드클러스터 자동 도출
│   ├── features.py          # FE 변환의 compute_* / apply_* 쌍, 잔차 계산
│   ├── pipeline.py          # FeaturePipeline: 학습된 전처리 통계 저장 / 재사용, residual_zscores
│   ├── isolation.py         # ResidualIsolator: 잔차 발화 패턴으로 센서 격리
│   ├── report.py            # 행 단위 진단 리포트 (diagnosis CSV, 요약 출력)
│   ├── models.py            # 아키텍처 3종, SAM, 손실, TorchPredictor(체크포인트)
│   ├── expert.py            # 하드클러스터 LightGBM 전문가 + 리랭킹
│   ├── calibration.py       # 온도 / 바이어스 / Sinkhorn 균형 배정
│   ├── blend.py             # OOF 그리디 가중치 탐색
│   ├── postprocess.py       # 후처리 파라미터 선택 + 중첩(정직한) 추정
│   ├── artifacts.py         # 실행 디렉터리 규약
│   └── utils.py             # 시드, 메트릭, 확률 유틸
├── tests/                   # pytest (합성 데이터, CPU)
├── data/                    # train.csv, test.csv, sample_submission.csv (git 미포함)
└── outputs/                 # 실행 디렉터리 (git 미포함)
```

### 실행 디렉터리 레이아웃

```
outputs/<VERSION>_<timestamp>/
├── config.json                        실행에 쓰인 설정 전체          (train.py)
├── label_classes.json                 예측 라벨 복원용 클래스
├── feature_pipeline.joblib            학습된 전처리 통계 + 센서쌍 + 잔차 통계
├── postprocess.json                   블렌드 / 전문가 / 바이어스 / 균형 배정 파라미터
├── metrics.json                       OOF 점수, 중첩 추정치, discovery 결과, 하드클러스터
├── oof_probs.npy, oof_target.npy      후처리된 OOF 확률과 정답
├── seeds/seed_<s>/fold_<i>/           isoforest.joblib, model_<arch>.pt, expert.joblib
├── submission_<VERSION>_<ts>.csv      제출 파일 (ID, target)         (predict.py)
├── test_probs_<ts>.npy                후처리된 test 확률 (test.csv 순)
├── diagnosis_<VERSION>_<ts>.csv       행 단위 진단 리포트 + 센서 격리 판정
├── isolation_report.json              고장 주입 평가 결과            (evaluate_isolation.py)
└── isolation_report.md                같은 내용의 한국어 리포트
```

## 설치 / 데이터 준비 / 실행

```bash
git clone <repo-url>
cd smart-fault-diagnosis
pip install -r requirements.txt
```

학습은 GPU 를 권장하지만 CPU 에서도 동작합니다. 추론과 격리 평가는 CPU 로 충분합니다.
DACON 대회 페이지에서 데이터를 받아 `data/` 에 둡니다. `train.csv` 는 `ID, X_01~X_52, target`, `test.csv` 는
`ID, X_01~X_52`, 그리고 `sample_submission.csv` 입니다. 경로가 다르면 `config.py` 의 `DATA_DIR` / `OUT_DIR`
또는 각 명령의 `--data-dir` 를 쓰세요.

```bash
# 학습 (시드 3개 x 5-fold). 끝에 predict.py 를 호출해 제출 파일과 진단 리포트까지 만든다
python train.py
python train.py --fast                       # 스모크 테스트 (시드 1개, 3-fold, epoch 축소)
python train.py --no-predict                 # 학습만

# 추론: 실행 디렉터리만 읽는다. test.csv 만 있으면 되고 train.csv 는 필요 없다
python predict.py outputs/<VERSION>_<ts>
python predict.py outputs/<VERSION>_<ts> --data-dir other_dir
python predict.py outputs/<VERSION>_<ts> --test-csv new_batch.csv --out-dir reports/
python predict.py outputs/<VERSION>_<ts> --no-report                 # 제출 파일만
python predict.py outputs/<VERSION>_<ts> --z-threshold 2.5 --low-confidence 0.6 --low-margin 0.15

# 센서 격리 평가: 모델 없이 config.json 과 feature_pipeline.joblib 만으로 고장을 주입한다
python evaluate_isolation.py outputs/<VERSION>_<ts>
python evaluate_isolation.py outputs/<VERSION>_<ts> --test-csv new_batch.csv --magnitudes 1 2 3 5
python evaluate_isolation.py outputs/<VERSION>_<ts> --threshold 2.5 --max-rows 2000 --out-dir reports/
```

## 테스트

```bash
pip install -r requirements-dev.txt
python -m pytest -q                 # 전체. 이 환경(CPU)에서 137개 수집, 136 통과, 1 건너뜀
python -m pytest -q -m "not slow"   # 엔드투엔드 8개를 제외한 단위 테스트
```

건너뛴 1개는 train 기준 잔차 표준화(평균 0 / 표준편차 1)를 직접 확인하는 테스트로, 합성 fixture 에 준중복
쌍이 생길 때만 실행됩니다. `tests/` 는 실제 데이터 없이 합성 센서 데이터로 동작하며 다음을 고정합니다.

| 영역 | 보장하는 것 | 테스트 파일 |
|---|---|---|
| 전처리 | `FeaturePipeline` 저장 / 복원 동일성. 주입한 센서 편향이 그 센서가 든 잔차 z-score 에만 나타나는 것 | `test_feature_pipeline.py` |
| 전처리 | 자동 도출이 심어 둔 구조(중복, shift, 시그니처, 혼동 클러스터)만 찾고 없는 구조는 지어내지 않는 것 | `test_discovery.py` |
| 격리 | 단독 지목 / 동점 / 부분 발화 / 다중 고장 판정, 허브 센서 부분 발화에서 파트너를 지목하지 않는 것, 오경보, 최소 탐지 가능 편향이 임계 x 잔차 sd 인 것, 고장 주입 평가가 실행 디렉터리만 읽는 것 | `test_isolation.py` |
| 리포트 | 열 계약과 검토 사유, 격리 열의 행 위치 결합, 요약의 CSV 왕복 | `test_report.py` |
| 검증 | 중첩 추정이 선택 점수보다 높지 않고 순수 노이즈에서 과적합을 드러내는 것, 후처리 게이트의 비퇴행성, 전문가 리랭킹의 확률 질량 보존 | `test_postprocess.py`, `test_calibration.py`, `test_expert.py` |
| 추론 | 세 아키텍처와 SAM / R-Drop / MixUp 조합의 완주, 체크포인트 왕복, 학습 후 추론이 `train.csv` 없이 같은 결과를 두 번 내는 것, 제출 파일 행 순서 준수, 진단 리포트가 제출 파일 옆에 생기는 것 | `test_models.py`, `test_pipeline.py` |

GitHub Actions(`.github/workflows/tests.yml`)에서 push 마다 실행됩니다.

## 한계와 실제 장비 데이터로의 확장

이 저장소는 대회 데이터로 만들어졌습니다. 채널은 익명화되어 어떤 물리량인지 모르고, 각 행은 시간축이 없는
독립 스냅샷입니다. 그래서 trace data 를 전제로 하는 CUSUM / EWMA 나 윈도우 기반 autoencoder 는 적용할 수
없었고, 잔차 판정도 행 단위 임계 하나로 끝납니다. 센서 고장 정답이 데이터에 없어 격리 성능은 고장 주입으로만
검증했습니다. 중복 쌍은 상관으로 추정한 것이라 잔차는 공정 변화나 고장 유형 변화에도 발화할 수 있습니다.
실제 장비에서는 쌍을 상관이 아니라 알려진 중복 계측(같은 물리량의 이중 센서, setpoint 와 readback)에서
가져와야 하며, 그때 잔차 발화가 센서 고장을 뜻하게 됩니다. 정상 상태 클래스가 있다는 보장도 없어 "정상" 을
따로 판정하지 않습니다. 클래스 분포가 바뀌면 OOF 게이트로 고른 바이어스와 균형 배정은 다시 골라야 합니다.

같은 구조를 실제 장비 데이터에 옮기면 다음이 자연스러운 다음 단계입니다. 결과를 약속하는 것은 아니고, 코드
구조가 그대로 받아 줄 수 있는 범위입니다.

| 확장 | 이 저장소에서 이어지는 지점 |
|---|---|
| trace data 의 잔차 CUSUM / EWMA | `FeaturePipeline.residual_zscores` 가 내는 잔차 z-score 를 시간순으로 누적 |
| 알람 로그와의 상관 (인터락 발생 시각 포함) | `diagnosis_*.csv` 의 `needs_review`, `suspect_sensor` 를 알람 타임스탬프와 결합 |
| PM(preventive maintenance) 주기에 따른 센서 드리프트 추적 | 같은 `residual_stats` 기준으로 배치별 `\|z\|` 추세 기록 |
| 미지의 고장 유형 탐지 (open-set) | `anomaly_score`, `confidence`, `margin` 열에 임계를 두어 검토 큐로 보냄 |
| 센서 이름이 있는 쌍 설계 | `AUTO_DISCOVER=False` 로 두고 `src/features.py` 의 `REDUNDANT_PAIRS` 에 알려진 중복 계측 쌍을 적음. 상관 추정 쌍이 아니므로 잔차 발화를 센서 고장으로 읽을 수 있음 |

## 부록

<details>
<summary>자동으로 도출되는 값들</summary>

데이터 의존적 상수는 모두 `src/discovery.py` 가 매 실행 도출하고, 결과는 `metrics.json` 의 `discovery` 블록과
`hard_cluster*` 키에 기록됩니다. `AUTO_DISCOVER=False` 면 폴백 상수를 씁니다. 센서쌍 / 중복 제거 열 / 시그니처 컬럼(`X_48`)은
`src/features.py` 에, shift 컬럼(`X_11`)은 `src/pipeline.py` 에 있습니다. 혼동 클러스터의 도출은 4절에
있습니다.

| 값 | 도출 방법 | 라벨 사용 |
|---|---|---|
| 완전 중복 센서 그룹 | train 상관행렬에서 `\|corr\| >= EXACT_DUP_THRESHOLD` union-find 그룹화, 대표 하나만 유지 | 없음 |
| 준중복 센서쌍 | `\|corr\| >= CORR_THRESHOLD` 인 쌍을 강한 순으로 `MAX_REDUNDANT_PAIRS` 개까지 | 없음 |
| 분포 shift 컬럼 | 참조(test) 분위수 구간에서 그 폭의 `SHIFT_EXCESS_RATIO` 배 이상 벗어난 값이 있고, 그런 행이 `SHIFT_OUTSIDE_FRAC` 이하인 컬럼 (`MAX_SHIFT_COLS` 개까지) | 없음 |
| 분산 시그니처 컬럼 | 클래스별 std 의 변동계수 상위 `N_SIGNATURE_COLS` 개 | train 라벨 |

shift 탐지는 경계를 살짝 넘는 값을 세지 않습니다. 참조 표본이 train 보다 작으면 정상적인 표본 변동만으로도
범위 밖 행이 생기기 때문입니다. 참조 폭의 배수만큼 멀리 떨어진 값만 꼬리로 보고, 선택되지 않은 컬럼 중 최고
비율(`shift_runner_up_ratio`)을 함께 기록해 우연히 임계를 넘은 것인지 확인할 수 있게 했습니다.

</details>

<details>
<summary>주요 설정 (config.py)</summary>

| 키 | 기본값 | 설명 |
|---|---|---|
| `CV_FOLDS` / `SEEDS` | 5 / [42, 52, 62] | StratifiedKFold 수, 시드 평균 |
| `NESTED_CALIBRATION` / `INNER_VAL_FRAC` | True / 0.15 | early stopping / 온도를 fold 내부 홀드아웃(15%)에서 결정 |
| `NESTED_POSTPROCESS_FOLDS` | 5 | 후처리 기여의 정직한 추정에 쓸 fold 수 (0 = 생략) |
| `USE_FDI_FEATURES` / `CLIP_SHIFT_COLS` | True / True | 잔차 + 편차 시그니처 피처, shift 컬럼 클리핑 |
| `AUTO_DISCOVER` | True | 센서쌍 / shift / 시그니처 자동 도출 (False = 폴백 상수) |
| `CORR_THRESHOLD` / `EXACT_DUP_THRESHOLD` | 0.95 / 0.9999 | 준중복 쌍 기준, 완전 중복 기준 `\|corr\|` |
| `MAX_REDUNDANT_PAIRS` | 12 | 잔차 피처를 만들 쌍 수 상한 |
| `SHIFT_EXCESS_RATIO` / `SHIFT_OUTSIDE_FRAC` / `MAX_SHIFT_COLS` | 1.0 / 0.02 / 3 | shift 컬럼 판정 기준 |
| `N_SIGNATURE_COLS` | 1 | 분산 시그니처 컬럼 수 (0 = 비활성) |
| `HARD_CLUSTER` | "auto" | 혼동행렬에서 도출. 리스트를 주면 고정 |
| `CLUSTER_F1_QUANTILE` / `CLUSTER_MIN_CONFUSION` / `CLUSTER_MAX_SIZE` | 0.35 / 0.05 / 8 | 클러스터 후보 분위, 간선 하한, 크기 상한 |
| `USE_EXPERT` / `EXPERT_W_GRID` / `EXPERT_N_ESTIMATORS` | True / [0.0, 0.25, 0.5] / 600 | LightGBM 전문가 리랭킹 |
| `USE_BALANCED_ASSIGN` / `USE_BIAS_TUNE` / `BIAS_LIM` | True / True / 0.30 | OOF 게이트 후처리 |
| `TEMP_MODE` / `BLEND_MODE` | soft / logit | 온도 모드 (soft / grid / off), 블렌드 방식 (prob / logit / geomean) |
| `USE_SAM` / `MIXUP_ALPHA` / `RDROP_ALPHA` | True / 0.2 / 0.5 | 학습 정규화 |
| `MC_DROPOUT` / `MC_PASSES` | True / 2 | MC-Dropout 추론 |

</details>

## 라이선스

MIT
