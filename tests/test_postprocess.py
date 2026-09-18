"""후처리 선택의 저장/재현성과, 중첩 추정이 선택 편향을 실제로 드러내는지 검증."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.postprocess import (
    PostProcessFit, PostProcessSpec, fit_postprocess, fit_seed_blend, nested_estimate,
)

HARD = (0, 3, 9, 15, 19)
K = 21


def _probs(n, y=None, signal=0.0, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, K))
    if y is not None and signal:
        logits[np.arange(n), y] += signal
    p = np.exp(logits - logits.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def _expert(n, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.random((n, len(HARD)))
    return e / e.sum(1, keepdims=True)


@pytest.fixture
def oof_setup():
    n = K * 40
    y = np.repeat(np.arange(K), n // K)
    oof_by_seed = {
        "42": {"ft": _probs(n, y, 1.5, 1), "mixer": _probs(n, y, 1.7, 2), "glu": _probs(n, y, 1.3, 3)},
    }
    expert_by_seed = {"42": _expert(n, 4)}
    return oof_by_seed, expert_by_seed, y


def test_fit_seed_blend_selects_valid_weights(oof_setup):
    oof, exp, y = oof_setup
    spec = PostProcessSpec(expert_w_grid=(0.0, 0.25, 0.5), hard_cluster=HARD)
    sb = fit_seed_blend(oof["42"], exp["42"], y, spec, live=False)
    assert set(sb.keep) <= {"ft", "mixer", "glu"} and len(sb.keep) <= 3
    assert len(sb.weights) == len(sb.keep)
    np.testing.assert_allclose(sum(sb.weights), 1.0, atol=1e-6)
    assert sb.expert_w in spec.expert_w_grid


def test_fit_seed_blend_drops_non_finite_models(oof_setup):
    oof, exp, y = oof_setup
    broken = dict(oof["42"])
    broken["glu"] = broken["glu"].copy()
    broken["glu"][0, 0] = np.nan
    sb = fit_seed_blend(broken, exp["42"], y, PostProcessSpec(hard_cluster=HARD), live=False)
    assert "glu" not in sb.keep


def test_fit_postprocess_produces_valid_probs_and_labels(oof_setup):
    oof, exp, y = oof_setup
    spec = PostProcessSpec(hard_cluster=HARD)
    fit, probs = fit_postprocess(oof, exp, y, spec, live=False)
    np.testing.assert_allclose(probs.sum(1), 1.0, atol=1e-9)
    pred = fit.predict(probs)
    assert pred.shape == y.shape
    assert set(np.unique(pred)) <= set(range(K))


def test_postprocess_fit_json_roundtrip_gives_identical_predictions(oof_setup):
    oof, exp, y = oof_setup
    fit, probs = fit_postprocess(oof, exp, y, PostProcessSpec(hard_cluster=HARD), live=False)
    restored = PostProcessFit.from_dict(fit.to_dict())
    np.testing.assert_array_equal(fit.predict(probs), restored.predict(probs))
    np.testing.assert_allclose(fit.transform_all(oof, exp), restored.transform_all(oof, exp))


def test_transform_all_matches_fit_output(oof_setup):
    """저장된 파라미터로 다시 계산한 확률이 학습 때 값과 같아야 한다 (추론 경로 일치)."""
    oof, exp, y = oof_setup
    fit, probs = fit_postprocess(oof, exp, y, PostProcessSpec(hard_cluster=HARD), live=False)
    np.testing.assert_allclose(fit.transform_all(oof, exp), probs, rtol=1e-12, atol=1e-12)


def test_nested_estimate_is_not_higher_than_selection_score(oof_setup):
    """중첩 추정치는 선택 점수보다 높지 않아야 한다 (선택 편향의 방향)."""
    oof, exp, y = oof_setup
    spec = PostProcessSpec(hard_cluster=HARD)
    fit, probs = fit_postprocess(oof, exp, y, spec, live=False)
    selection = f1_score(y, fit.predict(probs), average="macro")
    est = nested_estimate(oof, exp, y, spec, n_splits=3, seed=0, live=False)
    assert est["nested_f1"] <= selection + 1e-9
    assert 0.0 <= est["nested_f1"] <= 1.0
    assert est["n_splits"] == 3


def test_nested_estimate_exposes_overfitting_on_pure_noise():
    """신호가 없는 확률에서는 후처리가 선택 점수만 올린다. 중첩 추정은 속지 않는다."""
    n = K * 30
    y = np.repeat(np.arange(K), n // K)
    oof = {"42": {"ft": _probs(n, seed=7), "mixer": _probs(n, seed=8)}}
    spec = PostProcessSpec(hard_cluster=HARD, use_expert=False)
    fit, probs = fit_postprocess(oof, None, y, spec, live=False)
    selection = f1_score(y, fit.predict(probs), average="macro")
    est = nested_estimate(oof, None, y, spec, n_splits=3, seed=0, live=False)
    assert selection > est["nested_f1"]


def test_expert_disabled_spec_ignores_expert_probs(oof_setup):
    oof, exp, y = oof_setup
    spec = PostProcessSpec(hard_cluster=HARD, use_expert=False)
    fit, _ = fit_postprocess(oof, exp, y, spec, live=False)
    assert all(sb.expert_w == 0.0 for sb in fit.seeds.values())


def test_multi_seed_average_uses_every_seed():
    n = K * 20
    y = np.repeat(np.arange(K), n // K)
    oof = {
        "42": {"ft": _probs(n, y, 1.5, 11), "mixer": _probs(n, y, 1.4, 12)},
        "52": {"ft": _probs(n, y, 1.5, 13), "mixer": _probs(n, y, 1.4, 14)},
    }
    fit, probs = fit_postprocess(oof, None, y, PostProcessSpec(use_expert=False), live=False)
    assert set(fit.seeds) == {"42", "52"}
    per_seed = [fit.blend_seed(s, oof[s]) for s in ("42", "52")]
    np.testing.assert_allclose(probs, np.mean(per_seed, axis=0), rtol=1e-9, atol=1e-9)
