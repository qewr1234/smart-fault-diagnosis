"""자동 탐색: 심어 둔 구조를 실제로 찾아내는지, 없는 구조를 지어내지 않는지."""
import numpy as np
import pandas as pd
import pytest

from src.discovery import (
    discover_hard_cluster, discover_sensor_redundancy, discover_shift_columns,
    discover_signature_columns,
)
from tests.conftest import FEAT_COLS, make_synthetic

K = 21


@pytest.fixture
def frame():
    X, y = make_synthetic(21 * 20, seed=31)
    return pd.DataFrame(X, columns=FEAT_COLS), y


# ---------------- 센서 중복 ----------------
def test_finds_planted_exact_duplicates(frame):
    """합성 데이터에 심은 X_45=X_06, X_17=X_10 을 그룹으로 묶어야 한다."""
    df, _ = frame
    r = discover_sensor_redundancy(df, FEAT_COLS)
    groups = {tuple(g) for g in r["exact_groups"]}
    assert ("X_06", "X_45") in groups
    assert ("X_10", "X_17") in groups
    # 그룹마다 대표 하나만 남기고 제거
    assert set(r["drop_cols"]) == {"X_45", "X_17"}


def test_exact_duplicate_group_merges_three_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(200, 52)), columns=FEAT_COLS)
    df["X_20"] = df["X_02"]
    df["X_30"] = df["X_02"]
    r = discover_sensor_redundancy(df, FEAT_COLS)
    group = [g for g in r["exact_groups"] if "X_02" in g][0]
    assert group == ["X_02", "X_20", "X_30"]
    assert set(r["drop_cols"]) >= {"X_20", "X_30"}


def test_redundant_pairs_are_sorted_by_correlation_and_capped():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(400, 52))
    df = pd.DataFrame(base, columns=FEAT_COLS)
    df["X_02"] = df["X_01"] + rng.normal(scale=0.05, size=400)   # 매우 강함
    df["X_04"] = df["X_03"] + rng.normal(scale=0.20, size=400)   # 약간 약함
    r = discover_sensor_redundancy(df, FEAT_COLS, corr_threshold=0.9, max_pairs=2)
    assert len(r["pairs"]) <= 2
    assert ("X_01", "X_02") in r["pairs"]
    corrs = list(r["correlations"].values())
    assert corrs == sorted(corrs, reverse=True)


def test_exact_duplicate_pairs_excluded_from_residual_pairs(frame):
    """잔차가 항상 0인 완전 중복 쌍은 잔차 피처 후보에서 빠져야 한다."""
    df, _ = frame
    r = discover_sensor_redundancy(df, FEAT_COLS, corr_threshold=0.5)
    assert ("X_06", "X_45") not in r["pairs"]
    assert all("X_45" not in p and "X_17" not in p for p in r["pairs"])


def test_no_redundancy_found_in_independent_columns():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(300, 52)), columns=FEAT_COLS)
    r = discover_sensor_redundancy(df, FEAT_COLS, corr_threshold=0.95)
    assert r["exact_groups"] == [] and r["pairs"] == []


# ---------------- 분포 shift ----------------
def test_finds_planted_shift_column():
    rng = np.random.default_rng(3)
    train = pd.DataFrame(rng.normal(size=(500, 52)), columns=FEAT_COLS)
    ref = pd.DataFrame(rng.normal(size=(300, 52)), columns=FEAT_COLS)
    train.loc[train.index[:3], "X_11"] = -60.0   # train 에만 있는 먼 꼬리
    r = discover_shift_columns(train, ref, FEAT_COLS)
    assert r["columns"] == ["X_11"]
    assert r["excess_ratio"]["X_11"] > 1.0


def test_ignores_wholesale_distribution_difference():
    """대부분의 행이 참조 범위 밖이면 클리핑 대상이 아니다(다른 문제다)."""
    rng = np.random.default_rng(4)
    train = pd.DataFrame(rng.normal(size=(400, 52)), columns=FEAT_COLS)
    ref = pd.DataFrame(rng.normal(size=(300, 52)), columns=FEAT_COLS)
    train["X_11"] = train["X_11"] + 50.0
    r = discover_shift_columns(train, ref, FEAT_COLS)
    assert "X_11" not in r["columns"]


def test_shift_detection_finds_nothing_on_matched_distributions(frame):
    df, _ = frame
    other, _ = make_synthetic(21 * 8, seed=32)
    r = discover_shift_columns(df, pd.DataFrame(other, columns=FEAT_COLS), FEAT_COLS)
    assert r["columns"] == []


# ---------------- 분산 시그니처 ----------------
def test_finds_column_whose_spread_depends_on_class():
    rng = np.random.default_rng(5)
    n = 21 * 30
    y = np.repeat(np.arange(K), n // K)
    df = pd.DataFrame(rng.normal(size=(n, 52)), columns=FEAT_COLS)
    scale = np.where(y < 5, 0.1, 6.0)            # 클래스에 따라 흔들림 크기가 다름
    df["X_48"] = rng.normal(size=n) * scale
    r = discover_signature_columns(df, y, FEAT_COLS, top_k=1)
    assert r["columns"] == ["X_48"]
    assert r["scores"]["X_48"] > 0.5


def test_signature_disabled_with_zero_top_k(frame):
    df, y = frame
    assert discover_signature_columns(df, y, FEAT_COLS, top_k=0)["columns"] == []


# ---------------- 하드클러스터 ----------------
def _probs_with_cluster(cluster, n_per=40, seed=0, confusion=0.9):
    """`cluster` 안의 클래스들끼리만 심하게 섞이는 예측을 만든다."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n_per)
    logits = rng.normal(scale=0.2, size=(len(y), K))
    for i, t in enumerate(y):
        if t in cluster:
            # 클러스터 내부에서 무작위로 선택 -> 상호 혼동
            pick = cluster[rng.integers(len(cluster))] if rng.random() < confusion else t
            logits[i, pick] += 6.0
        else:
            logits[i, t] += 8.0
    p = np.exp(logits - logits.max(1, keepdims=True))
    return y, p / p.sum(1, keepdims=True)


def test_recovers_planted_confusion_cluster():
    planted = [0, 3, 9, 15, 19]
    y, p = _probs_with_cluster(planted, seed=6)
    info = discover_hard_cluster(y, p)
    assert info["cluster"] == planted
    assert set(info["candidates"]) >= set(planted)


def test_returns_empty_when_no_classes_are_confused():
    """모든 클래스가 잘 분리되면 클러스터를 지어내지 않는다."""
    rng = np.random.default_rng(7)
    y = np.repeat(np.arange(K), 30)
    logits = rng.normal(scale=0.1, size=(len(y), K))
    logits[np.arange(len(y)), y] += 12.0
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    assert discover_hard_cluster(y, p)["cluster"] == []


def test_cluster_is_capped_at_max_size():
    planted = [0, 2, 4, 6, 8, 10, 12]
    y, p = _probs_with_cluster(planted, seed=8)
    info = discover_hard_cluster(y, p, f1_quantile=0.5, max_size=4)
    assert len(info["cluster"]) == 4
    assert set(info["cluster"]) <= set(planted)


def test_high_confusion_threshold_rejects_weak_links():
    planted = [1, 5, 11]
    y, p = _probs_with_cluster(planted, seed=9, confusion=0.3)
    assert discover_hard_cluster(y, p, min_confusion=0.95)["cluster"] == []


def test_class_f1_report_has_one_entry_per_class():
    y, p = _probs_with_cluster([0, 3], seed=10)
    info = discover_hard_cluster(y, p)
    assert len(info["class_f1"]) == K
    assert all(0.0 <= v <= 1.0 for v in info["class_f1"])


def test_small_reference_sample_does_not_create_false_shift():
    """참조 표본이 train 보다 작으면 정상 변동만으로 범위 밖 행이 생긴다.

    그 정도로는 shift 컬럼이 아니다 (경계를 살짝 넘는 값은 세지 않는다).
    """
    rng = np.random.default_rng(41)
    train = pd.DataFrame(rng.normal(size=(2000, 52)), columns=FEAT_COLS)
    ref = pd.DataFrame(rng.normal(size=(60, 52)), columns=FEAT_COLS)
    r = discover_shift_columns(train, ref, FEAT_COLS)
    assert r["columns"] == []


def test_far_outside_fraction_is_reported_for_found_column():
    rng = np.random.default_rng(42)
    train = pd.DataFrame(rng.normal(size=(500, 52)), columns=FEAT_COLS)
    ref = pd.DataFrame(rng.normal(size=(400, 52)), columns=FEAT_COLS)
    train.loc[train.index[:2], "X_07"] = 80.0
    r = discover_shift_columns(train, ref, FEAT_COLS)
    assert r["columns"] == ["X_07"]
    assert r["far_outside_frac"]["X_07"] == pytest.approx(2 / 500)


def test_shift_report_includes_separation_margin():
    """다중비교 판단용: 선택되지 않은 컬럼 중 최고 비율을 함께 보고한다."""
    rng = np.random.default_rng(43)
    train = pd.DataFrame(rng.normal(size=(800, 52)), columns=FEAT_COLS)
    ref = pd.DataFrame(rng.normal(size=(600, 52)), columns=FEAT_COLS)
    train.loc[train.index[0], "X_03"] = 200.0
    r = discover_shift_columns(train, ref, FEAT_COLS)
    assert r["columns"] == ["X_03"]
    assert r["n_examined"] == 52
    assert r["runner_up_ratio"] < r["excess_ratio"]["X_03"]
    assert r["runner_up_ratio"] < 1.0
