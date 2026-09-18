"""실행 산출물(run directory) 규약.

outputs/<VERSION>_<timestamp>/
├── config.json            실행에 쓰인 설정 전체
├── label_classes.json     LabelEncoder 클래스 (예측 라벨 복원용)
├── feature_pipeline.joblib학습된 전처리 통계
├── postprocess.json       블렌드/전문가/바이어스/균형배정 파라미터
├── metrics.json           OOF 점수 + 정직한 중첩 추정치
├── oof_probs.npy          후처리된 OOF 확률 (seed 평균)
├── oof_target.npy         정답 라벨 (인코딩된 정수)
└── seeds/seed_<s>/fold_<i>/
    ├── isoforest.joblib
    ├── model_<arch>.pt
    └── expert.joblib
"""
import json
from pathlib import Path

CONFIG_FILE = "config.json"
LABELS_FILE = "label_classes.json"
PIPELINE_FILE = "feature_pipeline.joblib"
POSTPROCESS_FILE = "postprocess.json"
METRICS_FILE = "metrics.json"
OOF_PROBS_FILE = "oof_probs.npy"
OOF_TARGET_FILE = "oof_target.npy"


def fold_dir(run_dir, seed, fold, create=False):
    d = Path(run_dir) / "seeds" / f"seed_{seed}" / f"fold_{fold}"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def model_path(run_dir, seed, fold, arch):
    return fold_dir(run_dir, seed, fold) / f"model_{arch}.pt"


def isoforest_path(run_dir, seed, fold):
    return fold_dir(run_dir, seed, fold) / "isoforest.joblib"


def expert_path(run_dir, seed, fold):
    return fold_dir(run_dir, seed, fold) / "expert.joblib"


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def discover_folds(run_dir, seed):
    """저장된 fold 디렉터리를 번호 순으로 돌려준다."""
    base = Path(run_dir) / "seeds" / f"seed_{seed}"
    if not base.is_dir():
        return []
    folds = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("fold_")]
    return sorted(folds, key=lambda d: int(d.name.split("_")[1]))


def discover_seeds(run_dir):
    base = Path(run_dir) / "seeds"
    if not base.is_dir():
        return []
    seeds = [d.name.split("_", 1)[1] for d in base.iterdir()
             if d.is_dir() and d.name.startswith("seed_")]
    return sorted(seeds, key=lambda s: int(s) if s.isdigit() else s)
