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
    pipe2 = FeaturePipeline(feat_cols=FEAT_COLS, clip_x11=False)
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


def test_clip_bounds_come_from_reference_not_train(frames):
    """X_11 클리핑 경계는 참조(test) 분포에서 나와야 한다."""
    tr, te = frames
    tr = tr.copy()
    tr.loc[tr.index[0], "X_11"] = 1e6
    pipe = _pipe()
    pipe.fit(tr, te)
    lo, hi = pipe.clip_bounds["X_11"]
    assert lo == pytest.approx(te["X_11"].quantile(0.001))
    assert hi == pytest.approx(te["X_11"].quantile(0.999))


def test_reference_frame_only_leaks_x11_quantiles(frames):
    """참조 프레임에서 쓰는 정보는 X_11 분위수뿐이다.

    test의 다른 컬럼을 크게 오염시켜도 학습된 통계가 전혀 달라지지 않아야 한다.
    """
    tr, te = frames
    polluted = te.copy()
    for c in FEAT_COLS:
        if c != "X_11":
            polluted[c] = polluted[c] + 1000.0
    p1, p2 = _pipe(), _pipe()
    p1.fit(tr, te)
    p2.fit(tr, polluted)
    assert p1.clip_bounds == p2.clip_bounds
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


def test_augment_with_anomaly_score_adds_one_column(frames):
    from sklearn.ensemble import IsolationForest
    tr, te = frames
    pipe = _pipe()
    M = pipe.fit(tr, te)
    assert augment_with_anomaly_score(M, None).shape == M.shape
    iso = IsolationForest(n_estimators=10, random_state=0).fit(M)
    assert augment_with_anomaly_score(M, iso).shape == (M.shape[0], M.shape[1] + 1)
