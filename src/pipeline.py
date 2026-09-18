"""학습 시 산출한 전처리 통계를 담아 추론 시 그대로 재사용하는 피처 파이프라인.

`fit`은 train(그리고 X_11 클리핑용 참조 프레임)에서 모든 통계를 산출하고,
`transform`은 그 통계만으로 임의의 프레임을 행렬로 변환한다.
학습과 추론이 같은 객체를 쓰므로 두 경로가 갈라질 수 없다.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from .features import (
    apply_fdi_features, apply_pairwise, apply_row_stats, apply_winsor_bounds,
    compute_clip_bounds, compute_fdi_stats, compute_winsor_bounds, select_pairs,
)


@dataclass
class FeaturePipeline:
    """설정 + 학습된 통계. joblib으로 그대로 직렬화된다."""

    feat_cols: list
    use_fdi: bool = True
    clip_x11: bool = True
    add_pairdiff: bool = True
    topk_var: int = 16
    limit_pairs: int = 12

    # fit 에서 채워지는 학습된 상태
    clip_bounds: dict = field(default_factory=dict)
    winsor_bounds: dict = field(default_factory=dict)
    fdi_stats: dict = field(default_factory=dict)
    pairs: list = field(default_factory=list)
    row_stat_cols: list = field(default_factory=list)
    imputer: object = None
    variance_filter: object = None
    output_columns: list = field(default_factory=list)
    n_features_out: int = 0
    fitted: bool = False

    # ---------------- fit ----------------
    def fit(self, df_train, df_reference=None):
        """train에서 통계를 산출한다.

        df_reference: X_11 클리핑 경계를 얻을 프레임(보통 test). 피처 분포만 쓰며
        라벨은 전혀 보지 않는다. None이면 클리핑을 건너뛴다.
        """
        X = df_train[self.feat_cols].copy()

        if self.clip_x11 and df_reference is not None:
            self.clip_bounds = compute_clip_bounds(df_reference, ("X_11",))
            X = apply_winsor_bounds(X, self.clip_bounds)

        variances = X.var().sort_values(ascending=False)
        top_feats = list(variances.head(max(1, self.topk_var)).index)
        self.winsor_bounds = compute_winsor_bounds(X, top_feats)
        X = apply_winsor_bounds(X, self.winsor_bounds)

        if self.use_fdi:
            self.fdi_stats = compute_fdi_stats(X, self.feat_cols)
            X = apply_fdi_features(X, self.fdi_stats)

        if self.add_pairdiff:
            self.pairs = select_pairs([c for c in top_feats if c in X.columns], self.limit_pairs)
            X = apply_pairwise(X, self.pairs)

        self.row_stat_cols = [c for c in X.columns if c in self.feat_cols]
        X = apply_row_stats(X, self.row_stat_cols)
        self.output_columns = list(X.columns)

        self.imputer = SimpleImputer(strategy="median")
        M = self.imputer.fit_transform(X)
        self.variance_filter = VarianceThreshold(0.0)
        M = self.variance_filter.fit_transform(M)
        self.n_features_out = M.shape[1]
        self.fitted = True
        return M

    # ---------------- transform ----------------
    def transform(self, df):
        """학습된 통계만으로 변환. train 클리핑은 재적용하지 않는다."""
        if not self.fitted:
            raise RuntimeError("FeaturePipeline.fit() 을 먼저 호출해야 합니다.")
        X = df[self.feat_cols].copy()
        X = apply_winsor_bounds(X, self.winsor_bounds)
        if self.use_fdi:
            X = apply_fdi_features(X, self.fdi_stats)
        if self.add_pairdiff:
            X = apply_pairwise(X, self.pairs)
        X = apply_row_stats(X, self.row_stat_cols)

        missing = [c for c in self.output_columns if c not in X.columns]
        if missing:
            raise ValueError(f"변환 결과에 누락된 컬럼: {missing[:5]}")
        X = X[self.output_columns]

        M = self.imputer.transform(X)
        return self.variance_filter.transform(M)

    # ---------------- io ----------------
    def save(self, path):
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        import joblib
        obj = joblib.load(path)
        if not isinstance(obj, FeaturePipeline):
            raise TypeError(f"FeaturePipeline 이 아닙니다: {type(obj)}")
        return obj


def augment_with_anomaly_score(X, iso):
    """IsolationForest 이상치 스코어를 마지막 컬럼으로 붙인다 (None이면 그대로)."""
    if iso is None:
        return X
    return np.hstack([X, iso.score_samples(X).reshape(-1, 1)])
