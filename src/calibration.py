"""온도 스케일링, 로짓 바이어스 튜닝, EM prior shift, Sinkhorn 균형 배정.

모든 후처리는 OOF 개선 시에만 적용하는 게이트 패턴을 전제로 한다.
"""
import numpy as np
from sklearn.metrics import f1_score

from .utils import ensure_prob_finite, safe_log, softmax_np


def temperature_scale_grid(probs, y, mode="soft"):
    probs = ensure_prob_finite(probs)
    if mode == "off":
        return 1.0
    grid = np.linspace(0.9, 1.1, 9) if mode == "soft" else np.linspace(0.7, 1.3, 13)
    bestT, bestF = 1.0, -1.0
    for T in grid:
        adj = softmax_np(safe_log(probs) / T)
        f1 = f1_score(y, adj.argmax(1), average="macro")
        if f1 > bestF:
            bestF, bestT = f1, float(T)
    return bestT


def apply_temperature(p, T):
    return softmax_np(safe_log(ensure_prob_finite(p)) / T)


def logit_bias_tune(probs, y, step=0.05, lim=0.30, passes=1):
    probs = ensure_prob_finite(probs)
    K = probs.shape[1]
    b = np.zeros(K, float)
    for _ in range(passes):
        for k in range(K):
            best, bf = -1.0, b[k]
            for v in np.arange(-lim, lim + 1e-12, step):
                bb = b.copy()
                bb[k] = v
                adj = softmax_np(safe_log(probs) + bb[None, :])
                f1 = f1_score(y, adj.argmax(1), average="macro")
                if f1 > best:
                    best = f1
                    bf = v
            b[k] = bf
    return b


def apply_logit_bias(p, b):
    return softmax_np(safe_log(ensure_prob_finite(p)) + b[None, :])


def em_prior_shift(P, prior_init, iters=50, eps=1e-6):
    pi = prior_init.astype(float)
    pi = pi / np.clip(pi.sum(), 1e-12, None)
    P = np.clip(P, 1e-12, 1.0)
    for _ in range(iters):
        denom = (P * pi[None, :]).sum(axis=1, keepdims=True)
        Q = (P * pi[None, :]) / np.clip(denom, 1e-12, None)
        new_pi = Q.mean(axis=0)
        if np.max(np.abs(new_pi - pi)) < eps:
            pi = new_pi
            break
        pi = new_pi
    return Q, pi


def sinkhorn_balanced_assign(P, iters=80):
    """예측 클래스 분포를 균등하게 유도하는 로짓 보정 후 argmax."""
    L = safe_log(ensure_prob_finite(P))
    N, K = P.shape
    u = np.zeros(K)
    for _ in range(iters):
        A = np.exp(L + u[None, :])
        A /= A.sum(1, keepdims=True)
        u -= 0.5 * np.log(np.clip(A.sum(0), 1e-12, None) / (N / K))
    return (L + u[None, :]).argmax(1)


def gated_balanced_assign(oof_probs, y_oof, test_probs, live=True):
    """OOF에서 균형 배정이 argmax보다 좋을 때만 test에 적용."""
    f1_argmax = f1_score(y_oof, oof_probs.argmax(1), average="macro")
    f1_bal = f1_score(y_oof, sinkhorn_balanced_assign(oof_probs), average="macro")
    use = f1_bal > f1_argmax + 1e-5
    if live:
        print(f"[BalancedAssign] argmax={f1_argmax:.5f} vs balanced={f1_bal:.5f} -> use={use}")
    return sinkhorn_balanced_assign(test_probs) if use else test_probs.argmax(1)


def gated_bias_tune(oof_probs, y_oof, test_probs, step=0.05, lim=0.30, passes=1, live=True):
    """OOF에서 바이어스 튜닝이 개선될 때만 test에 적용. (OOF, test) 확률 반환."""
    base = f1_score(y_oof, oof_probs.argmax(1), average="macro")
    b = logit_bias_tune(oof_probs, y_oof, step=step, lim=lim, passes=passes)
    adj = apply_logit_bias(oof_probs, b)
    tuned = f1_score(y_oof, adj.argmax(1), average="macro")
    use = tuned > base + 1e-5
    if live:
        print(f"[BiasTune] base={base:.5f} vs tuned={tuned:.5f} -> use={use}")
    if use:
        return adj, apply_logit_bias(test_probs, b)
    return oof_probs, test_probs
