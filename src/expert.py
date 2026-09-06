"""하드클러스터 {0,3,9,15,19} 전용 LightGBM 전문가 + 리랭킹.

전역 모델이 하드클러스터로 예측한 샘플에 한해, 클러스터 5개 클래스만으로
학습한 전문가의 로짓을 혼합해 클러스터 내부 확률 질량을 재분배한다.
혼합 가중치 w는 OOF Macro-F1로 선택하며, w=0이면 자동 비활성화.
"""
import numpy as np
from sklearn.metrics import f1_score


def fit_expert(Xtr, ytr, hard_cluster, n_estimators=600, seed=42):
    import lightgbm as lgb
    hard = np.asarray(hard_cluster)
    mask = np.isin(ytr, hard)
    y5 = np.searchsorted(hard, ytr[mask])
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        verbose=-1, n_jobs=-1)
    model.fit(Xtr[mask], y5)
    return model


def rerank_hard_cluster(global_probs, expert_probs_5, hard_cluster, w=0.5):
    """하드클러스터 내부 확률만 전문가 로짓과 혼합해 재분배 (클러스터 밖 확률 유지)."""
    if w <= 0:
        return global_probs
    hard = list(hard_cluster)
    P = global_probs.copy()
    mask = np.isin(P.argmax(1), hard)
    if mask.sum() == 0:
        return P
    logG = np.log(np.clip(P[np.ix_(np.where(mask)[0], hard)], 1e-12, 1))
    logE = np.log(np.clip(expert_probs_5[mask], 1e-12, 1))
    mix = (1 - w) * logG + w * logE
    mix = np.exp(mix - mix.max(1, keepdims=True))
    mix /= mix.sum(1, keepdims=True)
    cluster_mass = P[np.ix_(np.where(mask)[0], hard)].sum(1, keepdims=True)
    sub = P[mask].copy()
    sub[:, hard] = mix * cluster_mass
    P[mask] = sub
    return P


def select_expert_weight(oof_probs, oof_expert_5, y, hard_cluster, w_grid=(0.0, 0.25, 0.5), live=True):
    """OOF Macro-F1이 최대가 되는 혼합 가중치 선택."""
    best_w, best_f1 = 0.0, -1.0
    for w in w_grid:
        P = rerank_hard_cluster(oof_probs, oof_expert_5, hard_cluster, w=w)
        f1 = f1_score(y, P.argmax(1), average="macro")
        if live:
            print(f"[Expert] w={w:.2f} -> OOF macro_f1={f1:.5f}")
        if f1 > best_f1 + 1e-6:
            best_f1, best_w = f1, w
    if live:
        print(f"[Expert] selected w={best_w:.2f}")
    return best_w
