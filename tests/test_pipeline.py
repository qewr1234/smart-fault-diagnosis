"""학습/추론 분리 엔드투엔드 테스트.

train.run() 은 실행 디렉터리만 남기고, predict.predict_run() 은 그 디렉터리만 읽어
제출 파일을 만든다. 두 경로가 실제로 분리되어 있는지(추론이 train.csv 없이 되는지)와
저장된 산출물이 완전한지를 검증한다.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from config import CONFIG  # noqa: E402
from predict import predict_run  # noqa: E402
from src import artifacts  # noqa: E402
from train import run  # noqa: E402

N_CLASSES = 21


def small_cfg(data_dir, out_dir):
    cfg = dict(CONFIG)
    cfg.update({
        "DATA_DIR": str(data_dir), "OUT_DIR": str(out_dir),
        "SEEDS": [42], "CV_FOLDS": 2, "EPOCHS": 2, "WARM": 1, "PATIENCE": 5,
        "BS": 128, "MC_PASSES": 1, "EXPERT_N_ESTIMATORS": 10,
        "NESTED_POSTPROCESS_FOLDS": 2, "LIVE": False, "AMP_ENABLED": False,
    })
    return cfg


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """모듈 전체가 공유하는 1회 학습 결과 (학습이 느리므로 재사용)."""
    from tests.conftest import FEAT_COLS, make_synthetic
    base = tmp_path_factory.mktemp("run")
    data_dir = base / "data"
    data_dir.mkdir()
    Xtr, ytr = make_synthetic(21 * 20, seed=11)
    Xte, _ = make_synthetic(21 * 6, seed=12)
    train = pd.DataFrame(Xtr, columns=FEAT_COLS)
    train.insert(0, "ID", [f"TRAIN_{i:04d}" for i in range(len(train))])
    train["target"] = ytr
    test = pd.DataFrame(Xte, columns=FEAT_COLS)
    test.insert(0, "ID", [f"TEST_{i:04d}" for i in range(len(test))])
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"ID": test["ID"], "target": 0}).to_csv(data_dir / "sample_submission.csv", index=False)

    cfg = small_cfg(data_dir, base / "out")
    assert cfg["USE_SAM"] and cfg["RDROP_ALPHA"] > 0 and cfg["MIXUP_ALPHA"] > 0
    run_dir = Path(run(cfg))
    return run_dir, data_dir


@pytest.mark.slow
def test_train_writes_complete_run_directory(trained_run):
    run_dir, _ = trained_run
    for f in (artifacts.CONFIG_FILE, artifacts.LABELS_FILE, artifacts.PIPELINE_FILE,
              artifacts.POSTPROCESS_FILE, artifacts.METRICS_FILE,
              artifacts.OOF_PROBS_FILE, artifacts.OOF_TARGET_FILE):
        assert (run_dir / f).exists(), f

    seeds = artifacts.discover_seeds(run_dir)
    assert seeds == ["42"]
    folds = artifacts.discover_folds(run_dir, "42")
    assert len(folds) == 2
    metrics = json.loads((run_dir / artifacts.METRICS_FILE).read_text())
    for fold in folds:
        assert (fold / "isoforest.joblib").exists()
        for arch in ("ft", "mixer", "glu"):
            assert (fold / f"model_{arch}.pt").exists()
        # 전문가는 하드클러스터가 도출됐을 때만 존재한다
        assert (fold / "expert.joblib").exists() == metrics["expert_enabled"]

    oof = np.load(run_dir / artifacts.OOF_PROBS_FILE)
    assert oof.shape[1] == N_CLASSES
    np.testing.assert_allclose(oof.sum(1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_metrics_report_nested_and_selection_scores(trained_run):
    run_dir, _ = trained_run
    m = json.loads((run_dir / artifacts.METRICS_FILE).read_text())
    assert m["nested_calibration"] is True          # 내부 홀드아웃이 실제로 쓰였다
    assert 0.0 <= m["selection_f1"] <= 1.0
    assert m["nested_f1"] is not None and 0.0 <= m["nested_f1"] <= 1.0
    assert m["raw_blend_f1"] is not None
    assert m["nested_folds"] == 2
    assert set(m["per_model_oof"]["42"]) == {"ft", "mixer", "glu"}


@pytest.mark.slow
def test_metrics_record_what_was_discovered(trained_run):
    """하드코딩 상수 대신 도출 결과가 기록되어야 한다."""
    run_dir, _ = trained_run
    m = json.loads((run_dir / artifacts.METRICS_FILE).read_text())
    d = m["discovery"]
    assert d["auto_discover"] is True
    # 합성 데이터에 심어 둔 완전 중복(X_45=X_06, X_17=X_10)을 찾아내야 한다
    assert d["dropped_duplicate_cols"] == ["X_17", "X_45"]
    assert ["X_06", "X_45"] in d["exact_duplicate_groups"]
    assert isinstance(d["redundant_pairs"], list)
    assert d["n_features_out"] == m["n_features"]

    # 하드클러스터는 config 상수가 아니라 검증 예측에서 나온다
    assert m["hard_cluster_source"] in {"inner_holdout", "outer_fold"}
    assert len(m["hard_cluster_class_f1"]) == N_CLASSES
    assert m["expert_enabled"] == bool(m["hard_cluster"])


@pytest.mark.slow
def test_postprocess_cluster_matches_discovered_cluster(trained_run):
    run_dir, _ = trained_run
    m = json.loads((run_dir / artifacts.METRICS_FILE).read_text())
    pp = json.loads((run_dir / artifacts.POSTPROCESS_FILE).read_text())
    assert pp["spec"]["hard_cluster"] == m["hard_cluster"]


@pytest.mark.slow
def test_predict_runs_without_training_data(trained_run, tmp_path):
    """추론은 test.csv 하나만 있으면 된다 (train.csv 가 없어도 동작)."""
    run_dir, data_dir = trained_run
    lonely = tmp_path / "inference_only"
    lonely.mkdir()
    shutil.copy(data_dir / "test.csv", lonely / "test.csv")
    assert not (lonely / "train.csv").exists()

    sub_path = predict_run(run_dir, data_dir=lonely, out_dir=tmp_path / "sub", live=False)
    sub = pd.read_csv(sub_path)
    test = pd.read_csv(lonely / "test.csv")
    assert list(sub.columns) == ["ID", "target"]
    assert len(sub) == len(test)
    assert (sub["ID"].values == test["ID"].values).all()
    assert sub["target"].between(0, N_CLASSES - 1).all()


@pytest.mark.slow
def test_predict_is_deterministic_across_calls(trained_run, tmp_path):
    run_dir, data_dir = trained_run
    a = pd.read_csv(predict_run(run_dir, data_dir=data_dir, out_dir=tmp_path / "a", live=False))
    b = pd.read_csv(predict_run(run_dir, data_dir=data_dir, out_dir=tmp_path / "b", live=False))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.slow
def test_predict_respects_sample_submission_row_order(trained_run, tmp_path):
    run_dir, data_dir = trained_run
    shuffled_dir = tmp_path / "shuffled"
    shuffled_dir.mkdir()
    shutil.copy(data_dir / "test.csv", shuffled_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv").iloc[::-1].reset_index(drop=True)
    sample.to_csv(shuffled_dir / "sample_submission.csv", index=False)

    sub = pd.read_csv(predict_run(run_dir, data_dir=shuffled_dir, out_dir=tmp_path / "s", live=False))
    assert (sub["ID"].values == sample["ID"].values).all()
    assert sub["target"].notna().all()
