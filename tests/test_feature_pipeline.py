"""FeaturePipeline: 학습 통계 저장 -> 추론 재사용이 정확히 일치하는지."""
import numpy as np
import pandas as pd
import pytest

from src.pipeline import FeaturePipeline, augment_with_anomaly_score
from tests.conftest import FEAT_COLS, make_synthetic


@pytest.fixture
def frames():
    Xtr, ytr = make_synthetic(21 * 10, seed=21)
    Xte, _ = make_synthetic(21 * 4, seed=22)
    tr = pd.DataFrame(Xtr, columns=FEAT_COLS)
    tr["target"] = ytr
    return tr, pd.DataFrame(Xte, columns=FEAT_COLS)


def _pipe():
    return FeaturePipeline(feat_cols=FEAT_COLS)


def test_transform_requires_fit(frames):
    _, te = frames
    with pytest.raises(RuntimeError):
        _pipe().transform(te)


def test_fit_and_transform_agree_on_training_rows(frames):
    tr, te = frames
    pipe = _pipe()
    M_fit = pipe.fit(tr, te)
    # 클리핑은 train 전용 변환이므로, 클리핑을 끄면 fit 결과와 transform 결과가 같아야 한다
    pipe2 = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    M2 = pipe2.fit(tr, None)
    np.testing.assert_allclose(M2, pipe2.transform(tr), rtol=1e-9, atol=1e-9)
    assert M_fit.shape[0] == len(tr)


def test_transform_is_row_independent(frames):
    """행 부분집합의 변환 결과는 전체 변환 결과의 같은 행과 일치해야 한다."""
    tr, te = frames
    pipe = _pipe()
    pipe.fit(tr, te)
    full = pipe.transform(te)
    subset = pipe.transform(te.iloc[5:15])
    np.testing.assert_allclose(subset, full[5:15], rtol=1e-9, atol=1e-9)


def test_save_load_roundtrip_reproduces_matrix(frames, tmp_path):
    tr, te = frames
    pipe = _pipe()
    pipe.fit(tr, te)
    before = pipe.transform(te)
    path = tmp_path / "pipe.joblib"
    pipe.save(path)
    after = FeaturePipeline.load(path).transform(te)
    np.testing.assert_allclose(before, after, rtol=1e-12, atol=1e-12)


def test_shift_clipping_bounds_come_from_reference(frames):
    """탐지된 shift 컬럼의 클리핑 경계는 참조(test) 분위수에서 나와야 한다."""
    tr, te = frames
    tr = tr.copy()
    tr.loc[tr.index[0], "X_11"] = 1e6      # train 에만 있는 먼 꼬리를 심는다
    pipe = _pipe()
    pipe.fit(tr, te)
    assert "X_11" in pipe.shift_report["columns"]
    lo, hi = pipe.clip_bounds["X_11"]
    assert lo == pytest.approx(te["X_11"].quantile(0.001))
    assert hi == pytest.approx(te["X_11"].quantile(0.999))


def test_no_clipping_when_no_shift_column_is_found(frames):
    """심어 둔 꼬리가 없으면 아무 컬럼도 클리핑하지 않는다."""
    tr, te = frames
    pipe = _pipe()
    pipe.fit(tr, te)
    assert pipe.shift_report["columns"] == []
    assert pipe.clip_bounds == {}


def test_pipeline_state_is_independent_of_reference_when_clipping_off(frames):
    """클리핑을 끄면 참조 프레임은 파이프라인에 전혀 영향을 주지 않는다."""
    tr, te = frames
    polluted = te.copy()
    for c in FEAT_COLS:
        polluted[c] = polluted[c] + 1000.0
    p1 = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    p2 = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    np.testing.assert_allclose(p1.fit(tr, te), p2.fit(tr, polluted))
    assert p1.winsor_bounds == p2.winsor_bounds
    assert p1.fdi_stats["mu"] == p2.fdi_stats["mu"]
    assert p1.output_columns == p2.output_columns


def test_fitted_state_is_independent_of_reference_row_order(frames):
    """참조 프레임의 행 순서는 학습된 통계에 영향을 주지 않는다."""
    tr, te = frames
    p1, p2 = _pipe(), _pipe()
    p1.fit(tr, te)
    p2.fit(tr, te.sample(frac=1.0, random_state=0))
    assert p1.clip_bounds == p2.clip_bounds
    assert p1.fdi_stats["mu"] == p2.fdi_stats["mu"]


def test_discovered_duplicates_are_dropped_from_output(frames):
    """자동 탐색이 찾은 완전 중복 컬럼은 모델 입력에서 빠진다."""
    tr, te = frames
    pipe = _pipe()
    pipe.fit(tr, te)
    assert set(pipe.redundancy["drop_cols"]) == {"X_45", "X_17"}
    assert "X_45" not in pipe.output_columns and "X_17" not in pipe.output_columns
    assert "X_06" in pipe.output_columns and "X_10" in pipe.output_columns


def test_discovered_pairs_become_residual_features(frames):
    tr, te = frames
    pipe = _pipe()
    pipe.fit(tr, te)
    for a, b in pipe.fdi_stats["pairs"]:
        assert f"r_{a}_{b}" in pipe.output_columns


def test_signature_column_requires_labels(frames):
    """라벨을 주지 않으면 분산 시그니처 피처를 만들지 않는다."""
    tr, te = frames
    y = tr.pop("target").values
    with_labels, without = _pipe(), _pipe()
    with_labels.fit(tr, te, y=y)
    without.fit(tr, te)
    assert with_labels.signature_report["columns"]
    assert without.fdi_stats["dev_cols"] == []
    dev = [c for c in with_labels.output_columns if c.startswith("dev_X_")]
    assert len(dev) == 1


def test_auto_discover_off_uses_fallback_constants(frames):
    """자동 탐색을 끄면 features.py 의 상수를 그대로 쓴다."""
    from src.features import EXACT_DUP_DROP
    tr, te = frames
    pipe = FeaturePipeline(feat_cols=FEAT_COLS, auto_discover=False)
    pipe.fit(tr, te)
    assert pipe.redundancy == {}
    for c in EXACT_DUP_DROP:
        assert c not in pipe.output_columns


def test_discovery_report_is_json_serializable(frames):
    import json
    tr, te = frames
    y = tr.pop("target").values
    pipe = _pipe()
    pipe.fit(tr, te, y=y)
    report = json.loads(json.dumps(pipe.discovery_report()))
    assert report["dropped_duplicate_cols"] == ["X_17", "X_45"]
    assert report["n_features_out"] == pipe.n_features_out


def test_augment_with_anomaly_score_adds_one_column(frames):
    from sklearn.ensemble import IsolationForest
    tr, te = frames
    pipe = _pipe()
    M = pipe.fit(tr, te)
    assert augment_with_anomaly_score(M, None).shape == M.shape
    iso = IsolationForest(n_estimators=10, random_state=0).fit(M)
    assert augment_with_anomaly_score(M, iso).shape == (M.shape[0], M.shape[1] + 1)
