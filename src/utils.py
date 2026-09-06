import os
import random

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def infer_feature_cols(df):
    return [c for c in df.columns if c.startswith("X_")]


def softmax_np(logits, axis=-1):
    logits = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(logits)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-12, None)


def safe_log(p):
    return np.log(np.clip(p, 1e-12, 1.0))


def ensure_prob_finite(p):
    if not np.isfinite(p).all():
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum(axis=1, keepdims=True)
    p = np.where(s == 0, 1.0 / p.shape[1], p / s)
    return p


def compute_metrics(y_true, probas):
    probas = ensure_prob_finite(probas)
    pred = probas.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred, average="macro")
    loss = log_loss(y_true, probas, labels=np.unique(y_true))
    return acc, f1, loss
