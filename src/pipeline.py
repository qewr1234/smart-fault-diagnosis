"""학습 시 산출한 전처리 통계를 담아 추론 시 그대로 재사용하는 피처 파이프라인.

`fit`은 train(그리고 shift 탐지용 참조 프레임)에서 모든 구조와 통계를 산출한다.
중복 센서쌍, 완전 중복 컬럼, 분포 shift 컬럼, 분산 시그니처 컬럼은 상수가 아니라
`src/discovery.py`가 데이터에서 찾아낸 결과다. `transform`은 그 결과만으로 변환하므로
학습과 추론이 갈라질 수 없다.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from .discovery import (
    discover_sensor_redundancy, discover_shift_columns, discover_signature_columns,
)
from .features import (
    apply_fdi_features, apply_pairwise, apply_row_stats, apply_winsor_bounds,
    compute_fdi_stats, compute_winsor_bounds, select_pairs,
)


@dataclass
class FeaturePipeline:
    """설정 + 데이터에서 도출한 구조 + 학습된 통계. joblib으로 그대로 직렬화된다."""

    feat_cols: list
    use_fdi: bool = True
    clip_shift: bool = True
    add_pairdiff: bool = True
    topk_var: int = 16
    limit_pairs: int = 12

    # 자동 탐색 설정
    auto_discover: bool = True
    corr_threshold: float = 0.95
    exact_dup_threshold: float = 0.9999
    max_redundant_pairs: int = 12
    shift_excess_ratio: float = 1.0
    shift_outside_frac: float = 0.02
    max_shift_cols: int = 3
    n_signature_cols: int = 1

    # fit 에서 채워지는 도출 결과 + 학습된 상태
    redundancy: dict = field(default_factory=dict)
    shift_report: dict = field(default_factory=dict)
    signature_report: dict = field(default_factory=dict)
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
    def fit(self, df_train, df_reference=None, y=None):
        """train에서 구조와 통계를 산출한다.

        df_reference: 분포 shift 컬럼을 찾을 프레임(보통 test). 피처 분위수만 쓰며
            라벨은 전혀 보지 않는다. None이면 shift 클리핑을 건너뛴다.
        y: 분산 시그니처 컬럼 탐색에만 쓰이는 train 라벨. None이면 생략한다.
        """
        X = df_train[self.feat_cols].copy()

        if self.clip_shift and df_reference is not None:
            self.shift_report = discover_shift_columns(
                X, df_reference, self.feat_cols,
                min_excess_ratio=self.shift_excess_ratio,
                max_outside_frac=self.shift_outside_frac,
                max_cols=self.max_shift_cols) if self.auto_discover else {"columns": ["X_11"]}
            self.clip_bounds = {}
            for c in self.shift_report.get("columns", []):
                if c in df_reference.columns:
                    self.clip_bounds[c] = (float(df_reference[c].quantile(0.001)),
                                           float(df_reference[c].quantile(0.999)))
            X = apply_winsor_bounds(X, self.clip_bounds)

        variances = X.var().sort_values(ascending=False)
        top_feats = list(variances.head(max(1, self.topk_var)).index)
        self.winsor_bounds = compute_winsor_bounds(X, top_feats)
        X = apply_winsor_bounds(X, self.winsor_bounds)

        if self.use_fdi:
            pairs = dev_cols = drop_cols = None
            if self.auto_discover:
                self.redundancy = discover_sensor_redundancy(
                    X, self.feat_cols, corr_threshold=self.corr_threshold,
                    exact_threshold=self.exact_dup_threshold,
                    max_pairs=self.max_redundant_pairs)
                pairs = self.redundancy["pairs"]
                drop_cols = self.redundancy["drop_cols"]
                if y is not None and self.n_signature_cols > 0:
                    self.signature_report = discover_signature_columns(
                        X, y, self.feat_cols, top_k=self.n_signature_cols)
                    dev_cols = self.signature_report["columns"]
                else:
                    dev_cols = []
            self.fdi_stats = compute_fdi_stats(X, self.feat_cols, pairs=pairs,
                                               dev_cols=dev_cols, drop_cols=drop_cols)
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
        """학습된 통계만으로 변환. train 전용 shift 클리핑은 재적용하지 않는다."""
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

    # ---------------- 보고 ----------------
    def discovery_report(self):
        """metrics.json 에 남길 도출 결과 요약."""
        return {
            "auto_discover": bool(self.auto_discover),
            "exact_duplicate_groups": self.redundancy.get("exact_groups", []),
            "dropped_duplicate_cols": self.redundancy.get("drop_cols", []),
            "redundant_pairs": [list(p) for p in self.redundancy.get("pairs", [])],
            "pair_correlations": self.redundancy.get("correlations", {}),
            "shift_columns": self.shift_report.get("columns", []),
            "shift_excess_ratio": self.shift_report.get("excess_ratio", {}),
            "shift_runner_up_ratio": self.shift_report.get("runner_up_ratio"),
            "shift_columns_examined": self.shift_report.get("n_examined"),
            "signature_columns": self.signature_report.get("columns", []),
            "signature_scores": self.signature_report.get("scores", {}),
            "n_features_out": int(self.n_features_out),
        }

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
