# ============================================
# Smart Fault Diagnosis — 추론 전용 진입점
#
# train.py 가 남긴 실행 디렉터리만 읽어 제출 파일을 만든다.
# 학습 데이터도, 학습 코드 경로도 다시 타지 않는다.
#
# 제출 파일(ID, target) 옆에 행 단위 진단 리포트(diagnosis_*.csv)를 함께 남긴다.
# 리포트는 판정의 확신도·후보 클래스·센서 잔차 이상·이상치 점수를 담아, 지원 엔지니어가
# 어느 행을 다시 봐야 하는지(needs_review)와 어느 센서를 의심해야 하는지(involved_sensors)
# 바로 읽을 수 있게 한다 (src/report.py). 여기에 잔차 발화 패턴으로 고장 센서를 격리한
# 판정(suspect_sensor / suspect_candidates / isolation_ambiguous / isolation_note,
# src/isolation.py)을 덧붙인다. 두 파일은 같은 행 순서(sample_submission.csv 순)로 쓰인다.
# ============================================
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import artifacts
from src.models import TorchPredictor, resolve_device
from src.isolation import ResidualIsolator
from src.pipeline import FeaturePipeline
from src.postprocess import PostProcessFit
from src.report import build_diagnosis_report, format_summary, summarize_report
from src.utils import ensure_prob_finite

log = logging.getLogger("predict")

# 잔차 |z| 발화 임계. 리포트의 abnormal_residuals 와 센서 격리 판정이 같은 기준을 써야
# "잔차가 이상하다" 와 "어느 센서를 지목한다" 가 서로 어긋나지 않는다.
Z_THRESHOLD = 3.0        # 잔차 발화 임계 (|z|, sigma 단위)
LOW_CONFIDENCE = 0.5     # 검토 사유 low_confidence 기준
LOW_MARGIN = 0.10        # 검토 사유 ambiguous_top2 기준


def _load_fold_expert(fold_path):
    p = fold_path / "expert.joblib"
    if not p.exists():
        return None
    import joblib
    return joblib.load(p)


def _load_fold_iso(fold_path):
    p = fold_path / "isoforest.joblib"
    if not p.exists():
        return None
    import joblib
    return joblib.load(p)


def predict_probs(run_dir, X, live=True, device=None):
    """실행 디렉터리의 모든 fold/seed 모델로 확률을 만든다 (전역 바이어스 적용 전).

    반환: (pp_fit, probs, anomaly_scores)
      anomaly_scores 는 isoforest.joblib 이 있는 모든 fold/seed 의 IsolationForest
      score_samples(X) 평균 (낮을수록 학습 분포에서 먼 샘플). 저장된 것이 없으면 None.
      X 는 증강 전 변환 행렬이어야 한다 — 학습 때 IsolationForest 도 그 행렬로 학습됐다.
    """
    run_dir = Path(run_dir)
    pp_fit = PostProcessFit.from_dict(artifacts.read_json(run_dir / artifacts.POSTPROCESS_FILE))
    cfg = artifacts.read_json(run_dir / artifacts.CONFIG_FILE)
    device = device or resolve_device()
    mc_kw = dict(mc_passes=max(1, int(cfg.get("MC_PASSES", 1))),
                 enable_dropout=bool(cfg.get("MC_DROPOUT", False)),
                 tta_noise_std=float(cfg.get("TTA_NOISE_STD", 0.0)))

    per_seed_probs = []
    anomaly_acc = []
    for seed in artifacts.discover_seeds(run_dir):
        if seed not in pp_fit.seeds:
            log.warning("[Predict] seed %s 는 후처리 설정에 없어 건너뜁니다.", seed)
            continue
        needed = pp_fit.seeds[seed].keep
        folds = artifacts.discover_folds(run_dir, seed)
        if not folds:
            continue

        acc = {n: [] for n in needed}
        expert_acc = []
        for fold_path in folds:
            iso = _load_fold_iso(fold_path)
            if iso is None:
                Xa = X
            else:
                # 이상치 점수를 한 번만 계산해 모델 입력 증강과 진단 리포트 양쪽에 쓴다.
                score = iso.score_samples(X)
                anomaly_acc.append(score)
                Xa = np.hstack([X, score.reshape(-1, 1)])
            for name in needed:
                mp = fold_path / f"model_{name}.pt"
                if not mp.exists():
                    raise FileNotFoundError(f"모델 체크포인트가 없습니다: {mp}")
                predictor = TorchPredictor.load(mp, device=device)
                acc[name].append(predictor(Xa, **mc_kw))
            expert = _load_fold_expert(fold_path)
            if expert is not None:
                expert_acc.append(expert.predict_proba(Xa))
            if live:
                log.info("[Predict] seed=%s %s 완료", seed, fold_path.name)

        probs_by_name = {n: ensure_prob_finite(np.mean(np.stack(v, axis=0), axis=0))
                         for n, v in acc.items()}
        expert_probs = np.mean(np.stack(expert_acc, axis=0), axis=0) if expert_acc else None
        per_seed_probs.append(pp_fit.blend_seed(seed, probs_by_name, expert_probs))

    if not per_seed_probs:
        raise RuntimeError(f"사용할 수 있는 저장된 모델이 없습니다: {run_dir}")
    anomaly_scores = np.mean(np.stack(anomaly_acc, axis=0), axis=0) if anomaly_acc else None
    return pp_fit, pp_fit.average_seeds(per_seed_probs), anomaly_scores


def _residual_zscores_or_none(pipe, test):
    """파이프라인이 준중복 센서쌍을 갖고 있을 때만 잔차 z-score 프레임을 돌려준다."""
    if not getattr(pipe, "fdi_stats", {}).get("pairs"):
        return None
    Z = pipe.residual_zscores(test)
    return Z if Z.shape[1] > 0 else None


def predict_run(run_dir, data_dir=None, test_csv=None, out_dir=None, live=True, device=None,
                report=True, z_threshold=Z_THRESHOLD, low_confidence=LOW_CONFIDENCE,
                low_margin=LOW_MARGIN):
    """제출 파일을 만들고 그 경로를 돌려준다.

    z_threshold:    잔차 |z| 가 이 값 이상이면 "발화" (abnormal_residuals 와 센서 격리가 같은 기준을 쓴다)
    low_confidence: 1위 확률이 이보다 낮으면 검토 사유 low_confidence
    low_margin:     1위-2위 확률 차가 이보다 작으면 검토 사유 ambiguous_top2

    report=True 이면 같은 타임스탬프로 diagnosis_<VERSION>_<ts>.csv 진단 리포트를 함께 남기고,
    live 모드에서는 요약을 출력한다. 리포트에는 ResidualIsolator 의 센서 격리 판정이 포함되며,
    파이프라인에 준중복 쌍이 없으면 그 열은 비어 있다. 리포트의 행 순서는 제출 파일과 같게 맞추므로
    (sample_submission.csv 의 ID 순) 두 파일을 나란히 놓고 읽을 수 있다 — 확실한 결합 키는 ID 다.
    제출 파일은 항상 ID, target 두 열만 갖는다.
    """
    logging.basicConfig(level=logging.INFO if live else logging.WARNING,
                        format="%(message)s", force=True)
    run_dir = Path(run_dir)
    cfg = artifacts.read_json(run_dir / artifacts.CONFIG_FILE)
    data_dir = Path(data_dir or cfg["DATA_DIR"])
    test_path = Path(test_csv) if test_csv else data_dir / "test.csv"
    test = pd.read_csv(test_path)

    pipe = FeaturePipeline.load(run_dir / artifacts.PIPELINE_FILE)
    X = pipe.transform(test)
    log.info("[Predict] %s -> %s", test_path.name, X.shape)

    pp_fit, probs, anomaly_scores = predict_probs(run_dir, X, live=live, device=device)
    pred = pp_fit.predict(probs)
    probs_final = pp_fit.apply_bias(probs)

    classes = np.array(artifacts.read_json(run_dir / artifacts.LABELS_FILE))
    labels = classes[pred]

    id_col = "ID" if "ID" in test.columns else test.columns[0]
    sub = pd.DataFrame({"ID": test[id_col].values, "target": labels})

    sample_path = data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if list(sample.columns) == ["ID", "target"]:
            sub = sample[["ID"]].merge(sub, on="ID", how="left")

    out_dir = Path(out_dir) if out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    version = cfg.get("VERSION", "run")
    sub_path = out_dir / f"submission_{version}_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    np.save(out_dir / f"test_probs_{ts}.npy", probs_final)

    if report:
        residual_z = _residual_zscores_or_none(pipe, test)
        # 센서 격리: 발화한 잔차의 공통 센서(involved_sensors)에 더해, 조용한 잔차를 알리바이로
        # 쓰는 signature matching 으로 단독 지목/모호 여부를 판정한다. 쌍이 없으면 모두 비어 있다.
        # z-score 는 한 번만 계산해 리포트(abnormal_residuals)와 격리기 양쪽에 같은 값을 준다.
        isolator = ResidualIsolator.from_pipeline(pipe)
        isolation = isolator.diagnose(test, threshold=z_threshold, zscores=residual_z)
        diagnosis = build_diagnosis_report(
            test[id_col].values, probs_final, classes, list(pp_fit.spec.hard_cluster),
            residual_z=residual_z, anomaly_score=anomaly_scores, z_threshold=z_threshold,
            low_confidence=low_confidence, low_margin=low_margin,
            pred_idx=pred, isolation=isolation, pairs=isolator.pairs)
        diagnosis = _align_to_submission_order(diagnosis, sub)
        report_path = out_dir / f"diagnosis_{version}_{ts}.csv"
        diagnosis.to_csv(report_path, index=False)
        summary = summarize_report(diagnosis)
        coverage = isolator.coverage()
        unstandardized = isolator.unstandardized_residuals
        log.info("[Report] 진단 리포트 저장: %s (검토 필요 %d/%d행, 잔차 %d쌍, 이상치 점수 %s)",
                 report_path, summary["n_needs_review"], summary["n_rows"],
                 0 if residual_z is None else residual_z.shape[1],
                 "있음" if anomaly_scores is not None else "없음")
        log.info("[Isolation] 커버 센서 %d/%d (단독 격리 가능 %d), 단독 지목 %d행, 격리 모호 %d행%s",
                 len(coverage["covered"]), coverage["n_sensors"], len(coverage["uniquely_isolable"]),
                 summary["n_with_suspect_sensor"], summary["n_isolation_ambiguous"],
                 (f" — 경고: 잔차 {len(unstandardized)}개 z-score 비표준화(구버전 실행 디렉터리), "
                  f"abnormal_residuals / 격리 임계가 sigma 단위가 아님") if unstandardized else "")
        if live:
            print(format_summary(summary))
    return str(sub_path)


def _align_to_submission_order(diagnosis, sub):
    """진단 리포트를 제출 파일과 같은 ID 순서로 맞춘다.

    제출 파일은 sample_submission.csv 순, 리포트는 test.csv 순으로 만들어지므로 두 순서가 다르면
    나란히 읽을 때 행이 어긋난다. ID 집합이 같을 때만 재배열하고 (dtype 유지), 다르면 test.csv
    순서를 그대로 두고 경고한다 — 그 경우 ID 로 결합해야 한다.
    """
    order = pd.Index(diagnosis["ID"]).get_indexer(sub["ID"])
    if len(order) == len(diagnosis) and (order >= 0).all() and len(set(order)) == len(order):
        return diagnosis.iloc[order].reset_index(drop=True)
    log.warning("[Report] 제출 파일과 test.csv 의 ID 가 일치하지 않아 진단 리포트는 test.csv 순서로 둡니다 "
                "— 두 파일은 ID 로 결합하세요.")
    return diagnosis


def parse_args():
    ap = argparse.ArgumentParser(description="저장된 실행 디렉터리로 제출 파일을 만든다.")
    ap.add_argument("run_dir", help="train.py 가 남긴 outputs/<VERSION>_<timestamp> 경로")
    ap.add_argument("--data-dir", default=None, help="test.csv / sample_submission.csv 위치")
    ap.add_argument("--test-csv", default=None, help="test.csv 경로를 직접 지정")
    ap.add_argument("--out-dir", default=None, help="제출 파일을 저장할 위치 (기본: 실행 디렉터리)")
    ap.add_argument("--no-report", action="store_true",
                    help="행 단위 진단 리포트(diagnosis_*.csv, 센서 격리 판정 포함)를 만들지 않는다")
    ap.add_argument("--z-threshold", type=float, default=Z_THRESHOLD,
                    help=f"잔차 발화 임계 |z| (기본 {Z_THRESHOLD}, sigma 단위)")
    ap.add_argument("--low-confidence", type=float, default=LOW_CONFIDENCE,
                    help=f"1위 확률이 이보다 낮으면 검토 필요 (기본 {LOW_CONFIDENCE})")
    ap.add_argument("--low-margin", type=float, default=LOW_MARGIN,
                    help=f"1위-2위 확률 차가 이보다 작으면 검토 필요 (기본 {LOW_MARGIN})")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    print("SAVED submission:", predict_run(a.run_dir, data_dir=a.data_dir, test_csv=a.test_csv,
                                           out_dir=a.out_dir, live=not a.quiet,
                                           report=not a.no_report, z_threshold=a.z_threshold,
                                           low_confidence=a.low_confidence, low_margin=a.low_margin))
