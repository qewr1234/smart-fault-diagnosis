import numpy as np
from sklearn.metrics import f1_score

from src.calibration import (
    apply_logit_bias, apply_temperature, em_prior_shift, gated_balanced_assign,
    gated_bias_tune, logit_bias_tune, sinkhorn_balanced_assign, temperature_scale_grid,
)


def test_temperature_off_returns_one(probs_and_labels):
    P, y = probs_and_labels
    assert temperature_scale_grid(P, y, mode="off") == 1.0


def test_temperature_grid_in_range_and_apply_keeps_probs(probs_and_labels):
    P, y = probs_and_labels
    for mode, lo, hi in (("soft", 0.9, 1.1), ("grid", 0.7, 1.3)):
        T = temperature_scale_grid(P, y, mode=mode)
        assert lo - 1e-9 <= T <= hi + 1e-9
        out = apply_temperature(P, T)
        np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-9)
    # T=1 은 항등
    np.testing.assert_allclose(apply_temperature(P, 1.0), P, atol=1e-9)


def test_sinkhorn_balances_predicted_class_counts():
    rng = np.random.default_rng(0)
    n, k = 210, 21
    logits = rng.normal(size=(n, k))
    logits[:, 0] += 3.0  # 클래스 0으로 심하게 쏠린 예측
    P = np.exp(logits - logits.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    assert np.bincount(P.argmax(1), minlength=k)[0] > n * 0.5

    counts = np.bincount(sinkhorn_balanced_assign(P), minlength=k)
    assert counts.max() <= 2 * (n // k)
    assert counts.min() >= 1


def test_gated_balanced_assign_falls_back_to_argmax_when_not_better():
    # OOF 는 argmax 가 완벽 -> 균형 배정은 채택될 수 없다
    y = np.repeat(np.arange(21), 5)
    P = np.full((len(y), 21), 1e-3)
    P[np.arange(len(y)), y] = 1.0
    P /= P.sum(1, keepdims=True)
    rng = np.random.default_rng(1)
    test_P = rng.dirichlet(np.ones(21), size=40)
    pred = gated_balanced_assign(P, y, test_P, live=False)
    np.testing.assert_array_equal(pred, test_P.argmax(1))


def test_gated_bias_tune_never_degrades_oof(probs_and_labels):
    P, y = probs_and_labels
    rng = np.random.default_rng(3)
    test_P = rng.dirichlet(np.ones(21), size=30)
    oof_out, test_out = gated_bias_tune(P, y, test_P, lim=0.3, live=False)
    base = f1_score(y, P.argmax(1), average="macro")
    after = f1_score(y, oof_out.argmax(1), average="macro")
    assert after >= base - 1e-12
    assert test_out.shape == test_P.shape
    np.testing.assert_allclose(test_out.sum(1), 1.0, atol=1e-9)


def test_logit_bias_tune_within_limits(probs_and_labels):
    P, y = probs_and_labels
    b = logit_bias_tune(P, y, step=0.1, lim=0.3)
    assert b.shape == (P.shape[1],)
    assert (np.abs(b) <= 0.3 + 1e-9).all()
    np.testing.assert_allclose(apply_logit_bias(P, np.zeros_like(b)), P, atol=1e-9)


def test_em_prior_shift_returns_valid_posterior_and_prior(probs_and_labels):
    P, _ = probs_and_labels
    Q, pi = em_prior_shift(P, np.ones(P.shape[1]))
    np.testing.assert_allclose(Q.sum(1), 1.0, atol=1e-9)
    np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-9)
    assert (pi >= 0).all()
