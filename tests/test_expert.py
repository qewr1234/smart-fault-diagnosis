import numpy as np
import pytest

from src.expert import rerank_hard_cluster, select_expert_weight

HARD = [0, 3, 9, 15, 19]


def _expert_probs(n, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.random((n, len(HARD)))
    return e / e.sum(1, keepdims=True)


def test_rerank_w_zero_is_identity(probs_and_labels):
    P, _ = probs_and_labels
    out = rerank_hard_cluster(P, _expert_probs(len(P)), HARD, w=0.0)
    np.testing.assert_array_equal(out, P)


def test_rerank_preserves_out_of_cluster_probs_and_cluster_mass(probs_and_labels):
    P, _ = probs_and_labels
    out = rerank_hard_cluster(P, _expert_probs(len(P)), HARD, w=0.5)
    others = [k for k in range(P.shape[1]) if k not in HARD]

    # 클러스터 밖 확률은 그대로
    np.testing.assert_allclose(out[:, others], P[:, others])
    # 클러스터 내부 확률 질량 합은 보존
    np.testing.assert_allclose(out[:, HARD].sum(1), P[:, HARD].sum(1), atol=1e-9)
    # 여전히 유효한 확률
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-9)
    assert (out >= 0).all()


def test_rerank_only_touches_rows_predicted_in_cluster(probs_and_labels):
    P, _ = probs_and_labels
    out = rerank_hard_cluster(P, _expert_probs(len(P)), HARD, w=0.5)
    not_in_cluster = ~np.isin(P.argmax(1), HARD)
    np.testing.assert_array_equal(out[not_in_cluster], P[not_in_cluster])


def test_rerank_no_cluster_rows_returns_unchanged():
    P = np.full((5, 21), 0.01)
    P[:, 7] = 0.8
    P /= P.sum(1, keepdims=True)
    out = rerank_hard_cluster(P, _expert_probs(5), HARD, w=0.5)
    np.testing.assert_array_equal(out, P)


def test_select_expert_weight_returns_grid_member_and_never_worse(probs_and_labels):
    from sklearn.metrics import f1_score
    P, y = probs_and_labels
    E = _expert_probs(len(P))
    grid = (0.0, 0.25, 0.5)
    w = select_expert_weight(P, E, y, HARD, w_grid=grid, live=False)
    assert w in grid
    base = f1_score(y, P.argmax(1), average="macro")
    after = f1_score(y, rerank_hard_cluster(P, E, HARD, w=w).argmax(1), average="macro")
    assert after >= base - 1e-12


def test_fit_expert_predicts_cluster_size_columns(small_xy):
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from src.expert import fit_expert
    X, y = small_xy
    model = fit_expert(X, y, HARD, n_estimators=10, seed=0)
    proba = model.predict_proba(X[:8])
    assert proba.shape == (8, len(HARD))
    np.testing.assert_allclose(proba.sum(1), 1.0, atol=1e-6)
