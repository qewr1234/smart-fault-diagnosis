import numpy as np
import pandas as pd
import pytest

from src.features import (
    EXACT_DUP_DROP, REDUNDANT_PAIRS, add_fdi_features, add_pairwise_features,
    add_row_stats, clip_train_to_test_range, winsorize_selected,
)
from src.utils import infer_feature_cols
from tests.conftest import FEAT_COLS, make_synthetic


@pytest.fixture
def frames():
    Xtr, _ = make_synthetic(21 * 10, seed=7)
    Xte, _ = make_synthetic(21 * 4, seed=8)
    return pd.DataFrame(Xtr, columns=FEAT_COLS), pd.DataFrame(Xte, columns=FEAT_COLS)


def test_infer_feature_cols_ignores_id_and_target(frames):
    tr, _ = frames
    df = tr.copy()
    df.insert(0, "ID", range(len(df)))
    df["target"] = 0
    assert infer_feature_cols(df) == FEAT_COLS


def test_add_fdi_features_adds_residuals_and_drops_exact_dups(frames):
    tr, te = frames
    out_tr, out_te = add_fdi_features(tr, te, FEAT_COLS)
    residual_cols = [f"r_{a}_{b}" for a, b in REDUNDANT_PAIRS]
    for c in residual_cols + ["dev_mean", "dev_max", "dev_X_48"]:
        assert c in out_tr.columns and c in out_te.columns
    for c in EXACT_DUP_DROP:
        assert c not in out_tr.columns and c not in out_te.columns
    assert list(out_tr.columns) == list(out_te.columns)
    # 입력은 변경하지 않는다
    assert "X_45" in tr.columns and "r_X_04_X_39" not in tr.columns


def test_fdi_residual_is_zero_for_perfectly_redundant_pair():
    """센서 b = a 인 경우 표준화 잔차는 정확히 0 (analytical redundancy 의도 확인)."""
    rng = np.random.default_rng(0)
    tr = pd.DataFrame(rng.normal(size=(100, 52)), columns=FEAT_COLS)
    tr["X_39"] = tr["X_04"]
    te = pd.DataFrame(rng.normal(size=(20, 52)), columns=FEAT_COLS)
    te["X_39"] = te["X_04"]
    out_tr, out_te = add_fdi_features(tr, te, FEAT_COLS)
    np.testing.assert_allclose(out_tr["r_X_04_X_39"], 0.0, atol=1e-12)
    np.testing.assert_allclose(out_te["r_X_04_X_39"], 0.0, atol=1e-12)


def test_fdi_stats_come_from_train_only(frames):
    """test 분포를 바꿔도 잔차 계산에 쓰이는 표준화 통계는 train 기준이어야 한다."""
    tr, te = frames
    _, out_te_a = add_fdi_features(tr, te, FEAT_COLS)
    te_shift = te.copy()
    te_shift.loc[te_shift.index[:5], "X_04"] += 100.0  # 일부 행만 오염
    _, out_te_b = add_fdi_features(tr, te_shift, FEAT_COLS)
    untouched = te.index[5:]
    np.testing.assert_allclose(out_te_a.loc[untouched, "r_X_04_X_39"],
                               out_te_b.loc[untouched, "r_X_04_X_39"])


def test_clip_train_to_test_range_clips_only_train_extremes(frames):
    tr, te = frames
    tr = tr.copy()
    tr.loc[tr.index[0], "X_11"] = 1e6
    out = clip_train_to_test_range(tr, te, cols=("X_11",))
    lo, hi = te["X_11"].quantile(0.001), te["X_11"].quantile(0.999)
    assert out["X_11"].max() <= hi + 1e-12
    assert out["X_11"].min() >= lo - 1e-12
    # 다른 컬럼은 손대지 않는다
    pd.testing.assert_series_equal(out["X_01"], tr["X_01"])
    # 없는 컬럼은 조용히 건너뛴다
    clip_train_to_test_range(tr, te, cols=("X_99",))


def test_winsorize_uses_train_quantiles(frames):
    tr, te = frames
    tr, te = tr.copy(), te.copy()
    te.loc[te.index[0], "X_01"] = 1e9
    lo, hi = tr["X_01"].quantile(0.005), tr["X_01"].quantile(0.995)
    tr2, _, te2 = winsorize_selected(tr, tr, te, ["X_01"])
    assert te2["X_01"].max() <= hi + 1e-12
    assert tr2["X_01"].min() >= lo - 1e-12


def test_pairwise_and_row_stats_shapes(frames):
    tr, te = frames
    top = FEAT_COLS[:4]
    tr, _, te = add_pairwise_features(tr.copy(), tr.copy(), te.copy(), top, limit_pairs=3)
    pair_cols = [c for c in tr.columns if "_minus_" in c or "_over_" in c]
    assert len(pair_cols) == 6  # 3 pairs x (minus, over)
    assert all(c in te.columns for c in pair_cols)
    tr, _, te = add_row_stats(tr, tr, te, FEAT_COLS)
    for c in ["row_mean", "row_std", "row_min", "row_max", "row_iqr", "row_energy", "row_skew", "row_kurt"]:
        assert c in tr.columns and c in te.columns
    assert np.isfinite(tr[["row_mean", "row_std"]].values).all()
