"""학습 루프 회귀 테스트.

핵심: SAM + R-Drop 동시 사용 시 첫 backward 후 해제된 그래프를 다시 타는 버그
(RuntimeError: Trying to backward through the graph a second time) 재발 방지.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models import GLUMLP, fit_torch_model, fit_ft, fit_tabmixer, fit_glumlp  # noqa: E402

N_CLASSES = 21


def _split(small_xy):
    X, y = small_xy
    n_va = len(y) // 4
    return X[n_va:], y[n_va:], X[:n_va], y[:n_va]


def _tiny_model(F_in):
    return GLUMLP(F_in, N_CLASSES, width=32, depth=1, p=0.1)


BASE_KW = dict(lr=1e-3, wd=1e-4, epochs=2, bs=64, warm=1, patience=5,
               ls=0.05, live=False, mc_dropout=False, amp_enabled=False)


@pytest.mark.parametrize("use_sam,rdrop_alpha,mixup_alpha", [
    (True, 0.5, 0.2),    # config.py 기본값 조합 — 과거 크래시 케이스
    (True, 0.5, 0.0),
    (True, 0.0, 0.2),
    (False, 0.5, 0.2),
    (False, 0.0, 0.0),
])
def test_fit_torch_model_all_regularizer_combinations(small_xy, use_sam, rdrop_alpha, mixup_alpha):
    Xtr, ytr, Xva, yva = _split(small_xy)
    model, predict = fit_torch_model(
        _tiny_model(Xtr.shape[1]), Xtr, ytr, Xva, yva,
        use_sam=use_sam, sam_rho=0.05, rdrop_alpha=rdrop_alpha, mixup_alpha=mixup_alpha,
        **BASE_KW)
    p = predict(Xva)
    assert p.shape == (len(yva), N_CLASSES)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)


def test_sam_rdrop_second_step_actually_updates_weights(small_xy):
    """SAM 2-step 경로에서 파라미터가 실제로 갱신되는지 (detach로 그래프만 끊고 학습은 유지)."""
    Xtr, ytr, Xva, yva = _split(small_xy)
    torch.manual_seed(0)
    model = _tiny_model(Xtr.shape[1])
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    kw = dict(BASE_KW, epochs=1, patience=100)
    model, _ = fit_torch_model(model, Xtr, ytr, Xva, yva,
                               use_sam=True, rdrop_alpha=0.5, mixup_alpha=0.0, **kw)
    changed = any(not torch.equal(before[k].cpu(), v.detach().cpu())
                  for k, v in model.state_dict().items())
    assert changed


def test_focal_loss_path_runs(small_xy):
    Xtr, ytr, Xva, yva = _split(small_xy)
    _, predict = fit_torch_model(
        _tiny_model(Xtr.shape[1]), Xtr, ytr, Xva, yva,
        loss_mode="focal", focal_gamma=2.0, use_sam=False, **BASE_KW)
    assert predict(Xva).shape == (len(yva), N_CLASSES)


def test_mc_dropout_and_tta_prediction(small_xy):
    Xtr, ytr, Xva, yva = _split(small_xy)
    _, predict = fit_torch_model(
        _tiny_model(Xtr.shape[1]), Xtr, ytr, Xva, yva, use_sam=False, **BASE_KW)
    p = predict(Xva, mc_passes=3, enable_dropout=True, tta_noise_std=0.05)
    assert p.shape == (len(yva), N_CLASSES)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-5)


@pytest.mark.parametrize("fitter", [fit_ft, fit_tabmixer, fit_glumlp])
def test_architecture_factories_forward(small_xy, fitter):
    """세 아키텍처가 실제 크기로 SAM+R-Drop+MixUp 1 epoch를 완주하는지."""
    Xtr, ytr, Xva, yva = _split(small_xy)
    kw = dict(BASE_KW, epochs=1, bs=128)
    _, predict = fitter(Xtr, ytr, Xva, yva, N_CLASSES,
                        use_sam=True, rdrop_alpha=0.5, mixup_alpha=0.2, **kw)
    p = predict(Xva)
    assert p.shape == (len(yva), N_CLASSES)
    assert np.isfinite(p).all()


def test_predictor_save_load_reproduces_probabilities(small_xy, tmp_path):
    """체크포인트 왕복이 확률을 정확히 재현해야 추론 경로를 신뢰할 수 있다."""
    from src.models import TorchPredictor
    Xtr, ytr, Xva, yva = _split(small_xy)
    kw = dict(BASE_KW, epochs=1)
    _, predictor = fit_glumlp(Xtr, ytr, Xva, yva, N_CLASSES, use_sam=False, **kw)
    predictor.temperature = 1.07
    before = predictor(Xva)

    path = tmp_path / "model_glu.pt"
    predictor.save(path)
    restored = TorchPredictor.load(path)
    assert restored.arch == "glu"
    assert restored.temperature == pytest.approx(1.07)
    np.testing.assert_allclose(restored(Xva), before, rtol=1e-6, atol=1e-6)


def test_predictor_temperature_changes_confidence_not_ranking(small_xy):
    Xtr, ytr, Xva, yva = _split(small_xy)
    kw = dict(BASE_KW, epochs=1)
    _, predictor = fit_glumlp(Xtr, ytr, Xva, yva, N_CLASSES, use_sam=False, **kw)
    raw = predictor(Xva, apply_temperature=False)
    predictor.temperature = 2.0
    warm = predictor(Xva)
    np.testing.assert_array_equal(raw.argmax(1), warm.argmax(1))   # 순위 불변
    assert warm.max(1).mean() < raw.max(1).mean()                  # 확신도는 낮아짐


def test_build_model_rejects_unknown_architecture():
    from src.models import build_model
    with pytest.raises(KeyError):
        build_model("no_such_arch", 10, N_CLASSES)


def test_predictor_without_arch_cannot_be_saved(small_xy, tmp_path):
    Xtr, ytr, Xva, yva = _split(small_xy)
    _, predictor = fit_torch_model(_tiny_model(Xtr.shape[1]), Xtr, ytr, Xva, yva,
                                   use_sam=False, **BASE_KW)
    with pytest.raises(ValueError):
        predictor.save(tmp_path / "x.pt")
