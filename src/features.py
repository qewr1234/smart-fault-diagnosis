"""Feature engineering.

기존 v16 FE(winsorize, pairwise, row stats)에 더해
FDI(analytical redundancy) 잔차 피처와 X_11 분포 shift 클리핑을 제공.
표준화/분위수 통계는 항상 train(또는 test 분위수 클리핑의 경우 test)에서만
산출해 리키지를 방지한다.
"""
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, skew

# 검증에서 확인된 준중복 센서쌍 (train |corr| > 0.95)
REDUNDANT_PAIRS = [
    ("X_04", "X_39"), ("X_05", "X_25"), ("X_26", "X_30"), ("X_38", "X_47"),
    ("X_07", "X_33"), ("X_12", "X_21"), ("X_09", "X_20"), ("X_20", "X_22"),
    ("X_05", "X_51"), ("X_09", "X_51"), ("X_22", "X_25"), ("X_05", "X_09"),
]
# 완전 중복 (corr = 1.0) — DL 입력에서 하나 제거
EXACT_DUP_DROP = ["X_45", "X_17"]


def winsorize_selected(Xtr, Xva, Xte, top_feats, qlo=0.005, qhi=0.995):
    for c in top_feats:
        lo, hi = Xtr[c].quantile(qlo), Xtr[c].quantile(qhi)
        if not np.isfinite(lo):
            lo = Xtr[c].min()
        if not np.isfinite(hi):
            hi = Xtr[c].max()
        if lo >= hi:
            lo, hi = Xtr[c].min(), Xtr[c].max()
        for df in (Xtr, Xva, Xte):
            df[c] = df[c].clip(lo, hi)
    return Xtr, Xva, Xte


def add_pairwise_features(Xtr, Xva, Xte, top_feats, limit_pairs=12):
    for a, b in list(combinations(top_feats, 2))[:limit_pairs]:
        for df in (Xtr, Xva, Xte):
            df[f"{a}_minus_{b}"] = df[a] - df[b]
            df[f"{a}_over_{b}"] = df[a] / (df[b].replace(0, np.nan) + 1e-9)
    return Xtr, Xva, Xte


def add_row_stats(Xtr, Xva, Xte, feat_cols):
    def enrich(df):
        A = df[feat_cols].values.astype(float)
        df["row_mean"] = np.nanmean(A, axis=1)
        df["row_std"] = np.nanstd(A, axis=1)
        df["row_min"] = np.nanmin(A, axis=1)
        df["row_max"] = np.nanmax(A, axis=1)
        q25 = np.nanquantile(A, 0.25, axis=1)
        q75 = np.nanquantile(A, 0.75, axis=1)
        df["row_iqr"] = q75 - q25
        df["row_energy"] = np.nanmean(A ** 2, axis=1)
        df["row_skew"] = skew(A, axis=1, bias=False, nan_policy="omit")
        df["row_kurt"] = kurtosis(A, axis=1, fisher=True, bias=False, nan_policy="omit")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return df

    return enrich(Xtr.copy()), enrich(Xva.copy()), enrich(Xte.copy())


def add_fdi_features(df_train, df_test, feat_cols, drop_exact_dup=True, live=False):
    """analytical-redundancy 잔차 + 편차 시그니처 피처."""
    out_tr, out_te = df_train.copy(), df_test.copy()
    mu = df_train[feat_cols].mean()
    sd = df_train[feat_cols].std().replace(0, 1.0)
    med = df_train[feat_cols].median()

    n_added = 0
    for a, b in REDUNDANT_PAIRS:
        if a not in feat_cols or b not in feat_cols:
            continue
        for df in (out_tr, out_te):
            df[f"r_{a}_{b}"] = (df[a] - mu[a]) / sd[a] - (df[b] - mu[b]) / sd[b]
        n_added += 1

    # 노이즈-레벨 시그니처 (클래스별 분산 차이가 큰 X_48 등에 대응)
    for df in (out_tr, out_te):
        mad = (df[feat_cols] - med).abs()
        df["dev_mean"] = mad.mean(axis=1)
        df["dev_max"] = mad.max(axis=1)
        if "X_48" in feat_cols:
            df["dev_X48"] = (df["X_48"] - med["X_48"]).abs()

    if drop_exact_dup:
        out_tr = out_tr.drop(columns=EXACT_DUP_DROP, errors="ignore")
        out_te = out_te.drop(columns=EXACT_DUP_DROP, errors="ignore")

    if live:
        print(f"[FDI] residual pairs added={n_added}, exact-dup dropped={drop_exact_dup}")
    return out_tr, out_te


def clip_train_to_test_range(df_train, df_test, cols=("X_11",), q=(0.001, 0.999), live=False):
    """train에만 존재하는 극단값을 test 분위수 범위로 클리핑 (X_11 shift 대응)."""
    out = df_train.copy()
    for c in cols:
        if c not in df_train.columns or c not in df_test.columns:
            continue
        lo, hi = df_test[c].quantile(q[0]), df_test[c].quantile(q[1])
        n_clip = int(((out[c] < lo) | (out[c] > hi)).sum())
        out[c] = out[c].clip(lo, hi)
        if live:
            print(f"[Clip] {c}: [{lo:.3f}, {hi:.3f}] clipped={n_clip}")
    return out
