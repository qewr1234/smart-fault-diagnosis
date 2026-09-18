"""Feature engineering (fit/transform 분리).

각 변환의 "학습된 통계"(분위수, 평균/표준편차, 중앙값, 선택된 컬럼)를 계산하는
`compute_*` 함수와, 그 통계를 새 데이터에 적용하는 `apply_*` 함수를 분리한다.
이 구조 덕분에 학습 시 산출한 통계를 저장해 두고 추론 시 그대로 재사용할 수 있으며,
test 통계가 train 변환에 새어 들어가는 것을 구조적으로 막는다.

`winsorize_selected` / `add_pairwise_features` / `add_row_stats` / `add_fdi_features` /
`clip_train_to_test_range`는 기존 호출부와의 호환을 위한 얇은 래퍼다.
"""
from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, skew

# 아래 상수는 원본 데이터에서 관측된 값으로, 자동 탐색(src/discovery.py)을 끄거나
# 탐색이 아무것도 찾지 못했을 때 쓰이는 폴백이다. 기본 경로에서는 사용되지 않는다.
REDUNDANT_PAIRS = [
    ("X_04", "X_39"), ("X_05", "X_25"), ("X_26", "X_30"), ("X_38", "X_47"),
    ("X_07", "X_33"), ("X_12", "X_21"), ("X_09", "X_20"), ("X_20", "X_22"),
    ("X_05", "X_51"), ("X_09", "X_51"), ("X_22", "X_25"), ("X_05", "X_09"),
]
# 완전 중복 (corr = 1.0) — DL 입력에서 하나 제거 (폴백)
EXACT_DUP_DROP = ["X_45", "X_17"]

ROW_STAT_COLS = ["row_mean", "row_std", "row_min", "row_max", "row_iqr",
                 "row_energy", "row_skew", "row_kurt"]


# --------------------------------------------------------------------------
# winsorize
# --------------------------------------------------------------------------
def compute_winsor_bounds(df, cols, qlo=0.005, qhi=0.995):
    """train 분위수 기반 클리핑 경계. {col: (lo, hi)}"""
    bounds = {}
    for c in cols:
        if c not in df.columns:
            continue
        lo, hi = df[c].quantile(qlo), df[c].quantile(qhi)
        if not np.isfinite(lo):
            lo = df[c].min()
        if not np.isfinite(hi):
            hi = df[c].max()
        if lo >= hi:
            lo, hi = df[c].min(), df[c].max()
        bounds[c] = (float(lo), float(hi))
    return bounds


def apply_winsor_bounds(df, bounds):
    out = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in out.columns:
            out[c] = out[c].clip(lo, hi)
    return out


def winsorize_selected(Xtr, Xva, Xte, top_feats, qlo=0.005, qhi=0.995):
    """호환 래퍼: train 분위수로 세 프레임을 모두 클리핑."""
    bounds = compute_winsor_bounds(Xtr, top_feats, qlo, qhi)
    for c, (lo, hi) in bounds.items():
        for df in (Xtr, Xva, Xte):
            if c in df.columns:
                df[c] = df[c].clip(lo, hi)
    return Xtr, Xva, Xte


# --------------------------------------------------------------------------
# pairwise
# --------------------------------------------------------------------------
def select_pairs(top_feats, limit_pairs=12):
    return [tuple(p) for p in list(combinations(top_feats, 2))[:limit_pairs]]


def apply_pairwise(df, pairs):
    out = df.copy()
    for a, b in pairs:
        if a not in out.columns or b not in out.columns:
            continue
        out[f"{a}_minus_{b}"] = out[a] - out[b]
        out[f"{a}_over_{b}"] = out[a] / (out[b].replace(0, np.nan) + 1e-9)
    return out


def add_pairwise_features(Xtr, Xva, Xte, top_feats, limit_pairs=12):
    """호환 래퍼."""
    pairs = select_pairs(top_feats, limit_pairs)
    return apply_pairwise(Xtr, pairs), apply_pairwise(Xva, pairs), apply_pairwise(Xte, pairs)


# --------------------------------------------------------------------------
# row statistics (행 단위 — 학습 통계 없음)
# --------------------------------------------------------------------------
def apply_row_stats(df, feat_cols):
    out = df.copy()
    cols = [c for c in feat_cols if c in out.columns]
    A = out[cols].values.astype(float)
    out["row_mean"] = np.nanmean(A, axis=1)
    out["row_std"] = np.nanstd(A, axis=1)
    out["row_min"] = np.nanmin(A, axis=1)
    out["row_max"] = np.nanmax(A, axis=1)
    q25 = np.nanquantile(A, 0.25, axis=1)
    q75 = np.nanquantile(A, 0.75, axis=1)
    out["row_iqr"] = q75 - q25
    out["row_energy"] = np.nanmean(A ** 2, axis=1)
    out["row_skew"] = skew(A, axis=1, bias=False, nan_policy="omit")
    out["row_kurt"] = kurtosis(A, axis=1, fisher=True, bias=False, nan_policy="omit")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def add_row_stats(Xtr, Xva, Xte, feat_cols):
    """호환 래퍼."""
    return (apply_row_stats(Xtr, feat_cols),
            apply_row_stats(Xva, feat_cols),
            apply_row_stats(Xte, feat_cols))


# --------------------------------------------------------------------------
# FDI (analytical redundancy) residuals
# --------------------------------------------------------------------------
def compute_fdi_stats(df, feat_cols, pairs=None, dev_cols=None, drop_cols=None):
    """잔차 표준화에 쓸 train 통계. 반드시 train에서만 호출한다.

    pairs / dev_cols / drop_cols 를 주지 않으면 모듈 상단의 폴백 상수를 쓴다.
    기본 경로에서는 FeaturePipeline 이 discovery 결과를 넣어준다.
    """
    cols = [c for c in feat_cols if c in df.columns]
    sd = df[cols].std().replace(0, 1.0)
    if pairs is None:
        pairs = REDUNDANT_PAIRS
    if dev_cols is None:
        dev_cols = ["X_48"]
    if drop_cols is None:
        drop_cols = EXACT_DUP_DROP
    return {
        "cols": cols,
        "mu": df[cols].mean().to_dict(),
        "sd": sd.to_dict(),
        "med": df[cols].median().to_dict(),
        "pairs": [(a, b) for a, b in pairs if a in cols and b in cols],
        "dev_cols": [c for c in dev_cols if c in cols],
        "drop_cols": [c for c in drop_cols if c in cols],
    }


def apply_fdi_features(df, stats, drop_exact_dup=True):
    out = df.copy()
    mu, sd, med = stats["mu"], stats["sd"], stats["med"]
    cols = [c for c in stats["cols"] if c in out.columns]

    for a, b in stats["pairs"]:
        if a in out.columns and b in out.columns:
            out[f"r_{a}_{b}"] = (out[a] - mu[a]) / sd[a] - (out[b] - mu[b]) / sd[b]

    med_s = np.array([med[c] for c in cols], dtype=float)
    mad = np.abs(out[cols].values.astype(float) - med_s[None, :])
    out["dev_mean"] = np.nanmean(mad, axis=1)
    out["dev_max"] = np.nanmax(mad, axis=1)
    for c in stats.get("dev_cols", []):
        if c in out.columns:
            out[f"dev_{c}"] = (out[c] - med[c]).abs()

    if drop_exact_dup:
        out = out.drop(columns=stats.get("drop_cols", EXACT_DUP_DROP), errors="ignore")
    return out


def add_fdi_features(df_train, df_test, feat_cols, drop_exact_dup=True, live=False):
    """호환 래퍼: train에서 통계를 산출해 train/test 양쪽에 적용."""
    stats = compute_fdi_stats(df_train, feat_cols)
    out_tr = apply_fdi_features(df_train, stats, drop_exact_dup)
    out_te = apply_fdi_features(df_test, stats, drop_exact_dup)
    if live:
        print(f"[FDI] residual pairs added={len(stats['pairs'])}, "
              f"exact-dup dropped={drop_exact_dup}")
    return out_tr, out_te


# --------------------------------------------------------------------------
# train -> test 범위 클리핑 (transductive: 참조 프레임의 분위수를 쓴다)
# --------------------------------------------------------------------------
def compute_clip_bounds(df_ref, cols=("X_11",), q=(0.001, 0.999)):
    bounds = {}
    for c in cols:
        if c in df_ref.columns:
            bounds[c] = (float(df_ref[c].quantile(q[0])), float(df_ref[c].quantile(q[1])))
    return bounds


def clip_train_to_test_range(df_train, df_test, cols=("X_11",), q=(0.001, 0.999), live=False):
    """train에만 존재하는 극단값을 test 분위수 범위로 클리핑 (X_11 shift 대응).

    test 라벨이 아니라 test 피처 분포만 사용하는 transductive 변환이다.
    추론 시에는 적용하지 않는다(test는 이미 그 범위 안에 있다).
    """
    bounds = compute_clip_bounds(df_test, [c for c in cols if c in df_train.columns], q)
    out = df_train.copy()
    for c, (lo, hi) in bounds.items():
        n_clip = int(((out[c] < lo) | (out[c] > hi)).sum())
        out[c] = out[c].clip(lo, hi)
        if live:
            print(f"[Clip] {c}: [{lo:.3f}, {hi:.3f}] clipped={n_clip}")
    return out
