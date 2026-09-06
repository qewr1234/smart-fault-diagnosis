"""OOF 그리디 가중치 탐색 및 확률 결합 (prob / logit / geomean)."""
import numpy as np
from sklearn.metrics import f1_score

from .utils import ensure_prob_finite, safe_log, softmax_np


def combine(plist, weights, mode="logit"):
    m = len(plist)
    if mode == "prob":
        out = sum(weights[i] * plist[i] for i in range(m))
    elif mode == "logit":
        out = softmax_np(sum(weights[i] * safe_log(plist[i]) for i in range(m)))
    elif mode == "geomean":
        G = np.exp(sum(weights[i] * safe_log(plist[i]) for i in range(m)))
        out = G / np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    else:
        out = sum(weights[i] * plist[i] for i in range(m))
    return ensure_prob_finite(out)


def greedy_weight_search(plist, y, steps=None, passes=2, live=False, mode="logit"):
    if steps is None:
        steps = np.round(np.arange(0.0, 1.01, 0.1), 2)
    m = len(plist)
    w = np.array([1.0 / m] * m)
    for p in range(passes):
        if live:
            print(f"[Blend] pass {p + 1}/{passes}")
        for i in range(m):
            best_w, best = -1, -1
            for wi in steps:
                ww = w.copy()
                ww[i] = wi
                s = ww.sum()
                if s == 0:
                    continue
                ww /= s
                f = f1_score(y, combine(plist, ww, mode).argmax(1), average="macro")
                if f > best:
                    best, best_w = f, wi
            if best_w >= 0:
                w[i] = best_w
                w /= w.sum()
                if live:
                    print(f"  - dim {i}: -> {best_w:.2f} (f1={best:.4f})")
    return w
