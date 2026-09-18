import numpy as np
import pytest

from src.blend import combine, greedy_weight_search


def _two_views(probs_and_labels):
    P, y = probs_and_labels
    rng = np.random.default_rng(5)
    noise = rng.dirichlet(np.ones(P.shape[1]), size=len(P))
    P2 = 0.7 * P + 0.3 * noise
    return [P, P2], y


@pytest.mark.parametrize("mode", ["prob", "logit", "geomean"])
def test_combine_returns_valid_probs(probs_and_labels, mode):
    plist, _ = _two_views(probs_and_labels)
    out = combine(plist, [0.5, 0.5], mode)
    assert out.shape == plist[0].shape
    np.testing.assert_allclose(out.sum(1), 1.0, atol=1e-9)
    assert (out >= 0).all()


def test_combine_single_full_weight_is_identity(probs_and_labels):
    plist, _ = _two_views(probs_and_labels)
    np.testing.assert_allclose(combine(plist, [1.0, 0.0], "prob"), plist[0], atol=1e-12)
    np.testing.assert_allclose(combine(plist, [1.0, 0.0], "geomean"), plist[0], atol=1e-9)


def test_greedy_weight_search_normalized_and_not_worse_than_uniform(probs_and_labels):
    from sklearn.metrics import f1_score
    plist, y = _two_views(probs_and_labels)
    w = greedy_weight_search(plist, y, passes=1, live=False, mode="logit")
    assert w.shape == (2,)
    assert (w >= 0).all()
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-9)
    uniform = f1_score(y, combine(plist, [0.5, 0.5], "logit").argmax(1), average="macro")
    found = f1_score(y, combine(plist, w, "logit").argmax(1), average="macro")
    assert found >= uniform - 1e-12
