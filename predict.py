# ============================================
# Smart Fault Diagnosis — 추론 전용 진입점
#
# train.py 가 남긴 실행 디렉터리만 읽어 제출 파일을 만든다.
# 학습 데이터도, 학습 코드 경로도 다시 타지 않는다.
# ============================================
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import artifacts
from src.models import TorchPredictor, resolve_device
from src.pipeline import FeaturePipeline, augment_with_anomaly_score
from src.postprocess import PostProcessFit
from src.utils import ensure_prob_finite

log = logging.getLogger("predict")


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
    """실행 디렉터리의 모든 fold/seed 모델로 확률을 만든다 (전역 바이어스 적용 전)."""
    run_dir = Path(run_dir)
    pp_fit = PostProcessFit.from_dict(artifacts.read_json(run_dir / artifacts.POSTPROCESS_FILE))
    cfg = artifacts.read_json(run_dir / artifacts.CONFIG_FILE)
    device = device or resolve_device()
    mc_kw = dict(mc_passes=max(1, int(cfg.get("MC_PASSES", 1))),
                 enable_dropout=bool(cfg.get("MC_DROPOUT", False)),
                 tta_noise_std=float(cfg.get("TTA_NOISE_STD", 0.0)))

    per_seed_probs = []
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
            Xa = augment_with_anomaly_score(X, iso)
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
    return pp_fit, pp_fit.average_seeds(per_seed_probs)


def predict_run(run_dir, data_dir=None, test_csv=None, out_dir=None, live=True, device=None):
    """제출 파일을 만들고 그 경로를 돌려준다."""
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

    pp_fit, probs = predict_probs(run_dir, X, live=live, device=device)
    pred = pp_fit.predict(probs)

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
    sub_path = out_dir / f"submission_{cfg.get('VERSION', 'run')}_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    np.save(out_dir / f"test_probs_{ts}.npy", pp_fit.apply_bias(probs))
    return str(sub_path)


def parse_args():
    ap = argparse.ArgumentParser(description="저장된 실행 디렉터리로 제출 파일을 만든다.")
    ap.add_argument("run_dir", help="train.py 가 남긴 outputs/<VERSION>_<timestamp> 경로")
    ap.add_argument("--data-dir", default=None, help="test.csv / sample_submission.csv 위치")
    ap.add_argument("--test-csv", default=None, help="test.csv 경로를 직접 지정")
    ap.add_argument("--out-dir", default=None, help="제출 파일을 저장할 위치 (기본: 실행 디렉터리)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    print("SAVED submission:", predict_run(a.run_dir, data_dir=a.data_dir, test_csv=a.test_csv,
                                           out_dir=a.out_dir, live=not a.quiet))
