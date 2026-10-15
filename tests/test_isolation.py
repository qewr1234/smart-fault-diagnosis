"""ResidualIsolator: 잔차 발화 패턴으로 고장 센서를 격리하는지, 격리 불가를 솔직히 말하는지."""
import copy
import json
import logging

import numpy as np
import pandas as pd
import pytest

from evaluate_isolation import (
    REPORT_JSON, REPORT_MD, evaluate_run, render_markdown, run_evaluation, write_reports,
)
from src import artifacts
from src.isolation import DIAGNOSE_COLUMNS, ResidualIsolator
from src.pipeline import FeaturePipeline
from tests.conftest import FEAT_COLS

TWO_PAIR = ["X_05", "X_06", "X_07"]     # 삼각형 X_05-X_06-X_07: 각각 두 쌍에 포함
ONE_PAIR = ["X_01", "X_02", "X_03", "X_04"]


@pytest.fixture(scope="module")
def planted():
    """준중복 쌍을 심은 프레임. X_05 는 X_06, X_07 두 쌍에 들어가고 X_01-X_02, X_03-X_04 는 단일 쌍."""
    rng = np.random.default_rng(77)
    n = 600
    tr = pd.DataFrame(rng.normal(size=(n, 52)), columns=FEAT_COLS)
    tr["X_02"] = tr["X_01"] + rng.normal(scale=0.05, size=n)
    tr["X_04"] = tr["X_03"] + rng.normal(scale=0.05, size=n)
    tr["X_06"] = tr["X_05"] + rng.normal(scale=0.05, size=n)
    tr["X_07"] = tr["X_05"] + rng.normal(scale=0.05, size=n)
    te = tr.iloc[:100].copy().reset_index(drop=True)
    pipe = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    pipe.fit(tr, None)
    pairs = {frozenset(p) for p in pipe.fdi_stats["pairs"]}
    assert {frozenset(("X_05", "X_06")), frozenset(("X_05", "X_07")),
            frozenset(("X_01", "X_02")), frozenset(("X_03", "X_04"))} <= pairs
    return pipe, tr, te


@pytest.fixture(scope="module")
def hub():
    """허브-스포크 구조: X_06, X_07, X_08 이 모두 X_05 의 준중복이지만 서로는 쌍이 아니다.

    노이즈 0.25 면 corr(X_05, X_0k) ~ 0.97 (쌍), corr(X_0j, X_0k) ~ 0.94 (< 0.95, 쌍 아님).
    실제 데이터에서 흔한 "허브 센서 + 단일 쌍 파트너" 구조를 재현해, 허브의 작은 드리프트로
    시그니처가 일부만 발화할 때 파트너를 잘못 지목하지 않는지 본다.
    """
    rng = np.random.default_rng(11)
    n = 800
    tr = pd.DataFrame(rng.normal(size=(n, 52)), columns=FEAT_COLS)
    for c in ("X_06", "X_07", "X_08"):
        tr[c] = tr["X_05"] + rng.normal(scale=0.25, size=n)
    pipe = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    pipe.fit(tr, None)
    pairs = {frozenset(p) for p in pipe.fdi_stats["pairs"]}
    assert pairs == {frozenset(("X_05", "X_06")), frozenset(("X_05", "X_07")), frozenset(("X_05", "X_08"))}
    return pipe, tr.iloc[:300].copy().reset_index(drop=True)


def _inject(pipe, df, sensor, k):
    out = df.copy()
    out[sensor] = out[sensor] + k * pipe.fdi_stats["sd"][sensor]
    return out


def _iso(pairs, sensors):
    """파이프라인 없이 z-score 프레임을 그대로 받는 격리기 (순위 규칙 단위 테스트용)."""
    return ResidualIsolator(pairs=pairs, residual_stats={}, sensor_cols=list(sensors), zscore_fn=lambda df: df)


# --------------------------------------------------------------------------
# 구조 (coverage)
# --------------------------------------------------------------------------
def test_coverage_on_planted_frame(planted):
    pipe, _, _ = planted
    cov = ResidualIsolator.from_pipeline(pipe).coverage()
    assert cov["pairs_per_sensor"]["X_05"] == 2
    assert "X_05" in cov["uniquely_isolable"]
    # 쌍 하나를 서로만 공유하는 두 센서는 시그니처가 같아 단독 격리가 불가능하다
    assert "X_01" in cov["covered"] and "X_02" in cov["covered"]
    assert "X_01" not in cov["uniquely_isolable"] and "X_02" not in cov["uniquely_isolable"]
    assert ["X_01", "X_02"] in cov["ambiguous_groups"]
    expected_uncovered = sorted(set(FEAT_COLS) - set(cov["covered"]))
    assert cov["uncovered"] == expected_uncovered
    assert "X_08" in cov["uncovered"] and cov["pairs_per_sensor"]["X_08"] == 0
    assert cov["n_sensors"] == 52


def test_coverage_signature_logic_without_pipeline():
    """파이프라인 없이 쌍 목록만으로도 시그니처 고유성이 맞게 계산된다."""
    iso = _iso([("A", "B"), ("A", "C"), ("A", "D"), ("E", "F")], "ABCDEFG")
    cov = iso.coverage()
    assert cov["covered"] == ["A", "B", "C", "D", "E", "F"]
    assert cov["uncovered"] == ["G"]
    assert cov["uniquely_isolable"] == ["A", "B", "C", "D"]   # B/C/D 는 서로 다른 쌍에 있어 고유
    assert cov["ambiguous_groups"] == [["E", "F"]]
    assert cov["pairs_per_sensor"] == {"A": 3, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 0}


# --------------------------------------------------------------------------
# 순위 규칙 (docstring 의 예시를 그대로 고정)
# --------------------------------------------------------------------------
def test_ranking_prefers_full_signature_then_evidence():
    iso = _iso([("A", "B"), ("A", "C"), ("A", "D"), ("E", "F")], "ABCDEF")
    Z = pd.DataFrame({
        "r_A_B": [5.0, 5.0, 0.5, -4.0],
        "r_A_C": [0.0, 4.0, 0.0, 0.0],
        "r_A_D": [0.0, 6.0, 0.0, 0.0],
        "r_E_F": [0.0, 0.0, 0.0, 3.5],
    }, index=[10, 11, 12, 13])
    d = iso.diagnose(Z, threshold=3.0)
    assert list(d.columns) == DIAGNOSE_COLUMNS
    assert list(d.index) == [10, 11, 12, 13]
    # r_AB 만 발화: B(1/1) 가 A(1/3) 를 이긴다 — 조용한 r_AC, r_AD 가 A 의 알리바이
    assert d.loc[10, "suspect"] == "B" and not d.loc[10, "ambiguous"]
    assert d.loc[10, "top_candidate"] == "B" and d.loc[10, "suspects"] == ["B"]
    assert d.loc[10, "suspect_match_ratio"] == pytest.approx(1.0)
    assert d.loc[10, "suspect_evidence"] == pytest.approx(5.0)
    assert d.loc[10, "fired_residuals"] == ["r_A_B"] and d.loc[10, "unexplained_residuals"] == []
    # 세 잔차 모두 발화: A 만 셋을 모두 설명한다
    assert d.loc[11, "suspect"] == "A" and not d.loc[11, "ambiguous"]
    assert d.loc[11, "n_fired"] == 3
    assert d.loc[11, "suspect_evidence"] == pytest.approx(15.0)
    # 발화 없음
    assert not d.loc[12, "detected"] and d.loc[12, "suspect"] is None and d.loc[12, "top_candidate"] is None
    assert d.loc[12, "suspects"] == [] and not d.loc[12, "ambiguous"]
    assert d.loc[12, "suspect_match_ratio"] == 0.0 and d.loc[12, "suspect_evidence"] == 0.0
    assert d.loc[12, "note"] == "정상: 발화한 잔차 없음"
    # 두 독립 쌍이 동시에 발화: 어느 센서도 둘을 함께 설명하지 못하므로 단독 지목하지 않는다.
    # |z| 가 큰 쪽의 B 가 1위 후보로 남되, r_EF 는 설명되지 않은 잔차로 드러나야 한다.
    assert d.loc[13, "n_fired"] == 2 and d.loc[13, "suspect"] is None and d.loc[13, "ambiguous"]
    assert d.loc[13, "suspects"] == ["A", "B", "E", "F"]
    assert d.loc[13, "top_candidate"] == "B" and d.loc[13, "unexplained_residuals"] == ["r_E_F"]
    assert "다중 고장" in d.loc[13, "note"] and "r_E_F" in d.loc[13, "note"]


def test_exact_tie_is_reported_as_ambiguous():
    iso = _iso([("A", "B"), ("C", "D")], "ABCD")
    Z = pd.DataFrame({"r_A_B": [4.0, 4.0], "r_C_D": [0.0, 4.0 + 1e-12]})
    d = iso.diagnose(Z)
    assert d.loc[0, "ambiguous"] and d.loc[0, "suspect"] is None and d.loc[0, "suspects"] == ["A", "B"]
    assert d.loc[0, "top_candidate"] is None and d.loc[0, "unexplained_residuals"] == []
    assert "A / B 중 하나" in d.loc[0, "note"] and "구분 불가" in d.loc[0, "note"]
    # 허용오차 1e-9 안의 차이는 동점으로 본다. 두 독립 쌍이라 설명되지 않은 잔차도 남는다
    assert d.loc[1, "ambiguous"] and d.loc[1, "suspects"] == ["A", "B", "C", "D"]
    assert d.loc[1, "top_candidate"] is None and d.loc[1, "unexplained_residuals"] == ["r_C_D"]


def test_shared_residual_tie_with_quiet_alibis_reads_as_a_tie():
    """삼각형 X_05-X_06-X_07 에서 r_06_07 하나만 발화: X_06 과 X_07 이 동점이고 각자 다른 잔차는 조용하다.

    1위가 하나인 "부분 발화" 문구가 아니라 "둘 중 하나" 동점 문구로 읽혀야 하며, 시그니처가 절반만
    발화했다는 사실(탐지 한계 근처)도 함께 적는다.
    """
    iso = _iso([("X_05", "X_06"), ("X_05", "X_07"), ("X_06", "X_07")], ["X_05", "X_06", "X_07"])
    d = iso.diagnose(pd.DataFrame({"r_X_05_X_06": [0.0], "r_X_05_X_07": [0.0], "r_X_06_X_07": [-3.3]}))
    r = d.iloc[0]
    assert r["ambiguous"] and r["suspect"] is None and r["top_candidate"] is None
    assert r["suspects"] == ["X_06", "X_07"] and r["unexplained_residuals"] == []
    assert r["suspect_match_ratio"] == pytest.approx(0.5)
    assert "X_06 / X_07 중 하나" in r["note"] and "50%" in r["note"] and "부분 발화" not in r["note"]


def test_partial_signature_is_not_a_confident_single_suspect():
    """허브 a 의 시그니처가 일부만 발화 (r_ad 가 임계 직전): 파트너 b 를 단독 지목하면 안 된다.

    예전 규칙(match_ratio -> evidence)은 b(1/1) 를 확신 있게 지목해 a 의 드리프트를 b 로 잘못
    짚었다. a 는 발화 잔차를 모두 설명하되 자기 잔차가 2/3 만 발화했으므로 모호로 남긴다.
    """
    iso = _iso([("A", "B"), ("A", "C"), ("A", "D")], "ABCD")
    d = iso.diagnose(pd.DataFrame({"r_A_B": [5.0, 5.0], "r_A_C": [4.0, 4.0], "r_A_D": [2.9, 3.0]}))
    r = d.iloc[0]
    assert r["detected"] and r["n_fired"] == 2
    assert r["ambiguous"] and r["suspect"] is None
    assert r["suspects"] == ["A", "B", "C"]                # 발화 잔차에 등장한 센서 전체
    assert r["top_candidate"] == "A"                       # 가장 많이 설명하는 센서는 여전히 보여 준다
    assert r["suspect_match_ratio"] == pytest.approx(2 / 3)
    assert r["suspect_evidence"] == pytest.approx(9.0)
    assert r["unexplained_residuals"] == []
    assert "부분 발화" in r["note"] and "A" in r["note"] and "67%" in r["note"]
    # 세 번째 잔차가 임계에 닿으면 a 의 시그니처가 완성돼 단독 지목된다
    r2 = d.iloc[1]
    assert r2["suspect"] == "A" and not r2["ambiguous"] and r2["suspects"] == ["A"]
    assert "A 센서 이상 의심" in r2["note"]


def test_drift_plus_unrelated_residual_names_the_lead_but_stays_ambiguous():
    """A 드리프트(두 잔차 ~100σ) 와 무관한 쌍의 우연 발화(3.1σ)가 겹친 행.

    A 가 셋 중 둘을 설명하므로 1위 후보는 A 지만 r_EF 는 설명되지 않으므로 단독 지목은 하지 않고,
    메모에 "A 가 2/3개 설명" 과 설명되지 않은 잔차를 함께 적어 엔지니어가 바로 읽게 한다.
    """
    iso = _iso([("A", "B"), ("A", "C"), ("E", "F")], "ABCEF")
    d = iso.diagnose(pd.DataFrame({"r_A_B": [100.0], "r_A_C": [-98.0], "r_E_F": [3.1]}))
    r = d.iloc[0]
    assert r["ambiguous"] and r["suspect"] is None
    assert r["top_candidate"] == "A" and r["suspect_match_ratio"] == pytest.approx(1.0)
    assert r["unexplained_residuals"] == ["r_E_F"]
    assert r["suspects"] == ["A", "B", "C", "E", "F"]
    assert "A 가 2/3개 설명" in r["note"] and "r_E_F" in r["note"] and "다중 고장" in r["note"]


def test_non_finite_residual_does_not_break_ranking():
    """센서 값이 inf 면 잔차 z 도 inf 다. evidence 행렬곱에 NaN 이 번져 후보가 사라지면 안 된다."""
    iso = _iso([("A", "B"), ("A", "C")], "ABC")
    d = iso.diagnose(pd.DataFrame({"r_A_B": [np.inf, 6.0, -np.inf], "r_A_C": [4.0, 4.0, np.nan]}))
    assert list(d["suspect"]) == ["A", "A", "B"]           # inf 쌍만 발화한 행 2 는 B(1/1) 가 A(1/2) 를 이긴다
    assert not d["ambiguous"].any()
    assert np.isfinite(d["suspect_evidence"]).all()
    assert d.loc[1, "suspect_evidence"] == pytest.approx(10.0)


# --------------------------------------------------------------------------
# 합성 데이터에 고장 주입
# --------------------------------------------------------------------------
def test_two_pair_sensor_is_isolated_uniquely(planted):
    pipe, _, te = planted
    iso = ResidualIsolator.from_pipeline(pipe)
    d = iso.diagnose(_inject(pipe, te, "X_05", 5.0), threshold=3.0)
    assert d["detected"].mean() > 0.95
    hit = (~d["ambiguous"]) & (d["suspect"] == "X_05")
    assert hit.mean() > 0.95
    # 단독 지목이 틀리는 일은 없어야 한다 — 무관한 잔차가 우연히 함께 발화한 행은 모호로 남는다
    assert not ((~d["ambiguous"]) & d["detected"] & (d["suspect"] != "X_05")).any()
    assert all("X_05" in s for s in d.loc[d["detected"], "suspects"])
    assert (d.loc[d["detected"], "top_candidate"] == "X_05").all()
    msg = ResidualIsolator.explain(d[hit].iloc[0])
    assert "X_05 센서 이상 의심" in msg and msg == d[hit].iloc[0]["note"]
    for note in d.loc[d["ambiguous"], "note"]:
        assert "X_05 가" in note and "다중 고장" in note


def test_one_pair_sensor_is_ambiguous_between_partners(planted):
    pipe, _, te = planted
    iso = ResidualIsolator.from_pipeline(pipe)
    d = iso.diagnose(_inject(pipe, te, "X_01", 5.0), threshold=3.0)
    assert d["detected"].mean() > 0.95
    detected = d[d["detected"]]
    assert detected["ambiguous"].all()
    assert detected["suspect"].isna().all()
    assert all(set(s) >= {"X_01", "X_02"} for s in detected["suspects"])
    assert "격리 모호" in ResidualIsolator.explain(detected.iloc[0])
    summ = ResidualIsolator.summarize(d)
    assert summ["n_ambiguous"] == summ["n_detected"] > 0
    assert summ["suspect_counts"] == {}
    assert "X_01|X_02" in summ["ambiguous_set_counts"]


def test_hub_sensor_partial_fire_never_blames_a_partner(hub):
    """허브 X_05 의 1 sd 드리프트: 잔차 셋 중 둘만 발화하는 행이 흔하다.

    예전 규칙은 그런 행의 약 40% 를 파트너(X_06/X_07/X_08) 단독 지목으로 확신 있게 틀렸다.
    이제 단독 지목이 틀리는 행은 없어야 하고, 탐지된 모든 행의 후보에 X_05 가 들어야 한다.
    """
    pipe, te = hub
    iso = ResidualIsolator.from_pipeline(pipe)
    d = iso.diagnose(_inject(pipe, te, "X_05", 1.0), threshold=3.0)
    det = d[d["detected"]]
    assert len(det) > 0.9 * len(d)
    # 잔차가 둘 이상 발화한 행: 틀린 단독 지목이 없고, X_05 가 항상 후보이자 1위 후보다
    multi = det[det["n_fired"] >= 2]
    assert len(multi) > 0.5 * len(det)
    assert (multi.loc[~multi["ambiguous"], "suspect"] == "X_05").all()
    assert all("X_05" in s for s in multi["suspects"])
    assert (multi["top_candidate"] == "X_05").all()
    partial = multi[multi["ambiguous"]]
    assert len(partial) > 0                                          # 부분 발화(2/3) 행이 실제로 존재한다
    assert (partial["n_fired"] == 2).all()
    assert all("부분 발화" in n and "X_05" in n for n in partial["note"])
    assert np.allclose(partial["suspect_match_ratio"], 2 / 3)
    # 잔차 하나만 발화한 행은 스펙의 알리바이 규칙대로 파트너(1/1)를 지목한다 — 조용한 두 잔차가
    # X_05 의 알리바이이므로 발화 패턴만으로는 구분할 수 없는, 탐지 한계에 걸린 행이다.
    single = det[det["n_fired"] == 1]
    assert (single["suspect"] != "X_05").all() and not single["ambiguous"].any()
    # 파트너 하나의 드리프트는 그 파트너의 단일 잔차만 발화시켜 파트너를 지목한다
    d6 = iso.diagnose(_inject(pipe, te, "X_06", 5.0), threshold=3.0)
    assert ((d6["suspect"] == "X_06") & ~d6["ambiguous"]).mean() > 0.9

    result = run_evaluation(pipe, te, magnitudes=(1,), threshold=3.0)
    rec = [r for r in result["per_sensor"] if r["sensor"] == "X_05"][0]
    assert rec["detection_rate"] > 0.9
    # 후보집합 포함 = 단독 지목 정답 + 모호 (모호한 행은 모두 X_05 를 후보에 둔다); 나머지는 단일 발화 행
    assert rec["suspect_set_accuracy"] == pytest.approx(rec["top1_accuracy"] + rec["ambiguous_rate"])
    assert rec["suspect_set_accuracy"] > 0.85
    assert rec["detection_rate"] - rec["suspect_set_accuracy"] == pytest.approx(len(single) / len(d))


def test_clean_rows_have_low_false_alarm_rate(planted):
    pipe, tr, _ = planted
    iso = ResidualIsolator.from_pipeline(pipe)
    d = iso.diagnose(tr, threshold=3.0)
    assert d["detected"].mean() <= 0.05
    assert (d.loc[~d["detected"], "suspect"].isna()).all()
    assert list(d.index) == list(tr.index)


def test_detectability_is_threshold_times_residual_std(planted):
    pipe, _, _ = planted
    iso = ResidualIsolator.from_pipeline(pipe)
    assert iso.unstandardized_residuals == []
    det = iso.detectability(threshold=3.0)
    assert set(det) == set(iso.residual_names)
    for name, v in det.items():
        assert v == pytest.approx(3.0 * pipe.residual_stats[name]["std"])
        assert 0 < v < 1.0     # 노이즈 0.05 로 심은 쌍은 1 sd 미만의 편향도 잡아야 한다


def test_precomputed_zscores_give_identical_diagnosis(planted):
    """predict.py 처럼 z-score 를 한 번만 계산해 넘겨도 결과가 같아야 한다."""
    pipe, _, te = planted
    iso = ResidualIsolator.from_pipeline(pipe)
    faulty = _inject(pipe, te, "X_05", 5.0)
    a = iso.diagnose(faulty)
    b = iso.diagnose(faulty, zscores=pipe.residual_zscores(faulty))
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        iso.diagnose(faulty, zscores=pipe.residual_zscores(faulty).iloc[:10])


# --------------------------------------------------------------------------
# 구버전 실행 디렉터리 (residual_stats 없음)
# --------------------------------------------------------------------------
def test_pipeline_without_residual_stats_is_flagged_loudly(planted, tmp_path, caplog):
    """residual_stats 를 저장하지 않던 구버전 파이프라인은 z-score 가 표준화되지 않는다.

    탐지가 조용히 0 으로 떨어지는 대신 from_pipeline 이 경고하고, unstandardized_residuals 와
    평가 리포트에 그 사실이 드러나야 한다.
    """
    pipe, _, te = planted
    old = copy.deepcopy(pipe)
    del old.__dict__["residual_stats"]
    old.save(tmp_path / artifacts.PIPELINE_FILE)
    old = FeaturePipeline.load(tmp_path / artifacts.PIPELINE_FILE)
    assert not hasattr(old, "residual_stats")

    with caplog.at_level(logging.WARNING, logger="src.isolation"):
        iso = ResidualIsolator.from_pipeline(old)
    assert any("train 통계" in r.getMessage() and "표준화" in r.getMessage() for r in caplog.records)
    assert iso.unstandardized_residuals == iso.residual_names
    assert iso.detectability() == {}
    # 같은 2 sd 편향: 정상 파이프라인은 전부 잡고 구버전은 거의 못 잡는다 (이것이 경고의 이유다)
    faulty = _inject(pipe, te, "X_05", 2.0)
    assert ResidualIsolator.from_pipeline(pipe).diagnose(faulty)["detected"].mean() > 0.95
    assert iso.diagnose(faulty)["detected"].mean() < 0.5

    result = run_evaluation(old, te, magnitudes=(2,), threshold=3.0)
    assert result["unstandardized_residuals"] == iso.residual_names
    assert result["detectability_sd"] == {}
    md = render_markdown(result)
    assert "경고" in md and "표준화되지 않았다" in md and iso.residual_names[0] in md


# --------------------------------------------------------------------------
# 쌍이 없는 파이프라인
# --------------------------------------------------------------------------
@pytest.mark.parametrize("use_fdi", [False, True])
def test_zero_pairs_pipeline_detects_nothing(use_fdi):
    """FDI 를 끄거나(use_fdi=False) 준중복 쌍이 전혀 없는 데이터에서는 빈 결과를 내야 한다."""
    rng = np.random.default_rng(5)
    tr = pd.DataFrame(rng.normal(size=(300, 52)), columns=FEAT_COLS)
    pipe = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False, use_fdi=use_fdi)
    pipe.fit(tr, None)
    iso = ResidualIsolator.from_pipeline(pipe)
    assert iso.pairs == [] and iso.unstandardized_residuals == []
    cov = iso.coverage()
    assert cov["covered"] == [] and cov["uniquely_isolable"] == []
    assert cov["uncovered"] == FEAT_COLS
    d = iso.diagnose(tr.iloc[:20])
    assert list(d.columns) == DIAGNOSE_COLUMNS and len(d) == 20
    assert not d["detected"].any() and not d["ambiguous"].any()
    assert d["suspect"].isna().all() and all(s == [] for s in d["suspects"])
    assert (d["note"] == "정상: 발화한 잔차 없음").all()
    summ = ResidualIsolator.summarize(d)
    assert summ["n_rows"] == 20 and summ["n_detected"] == 0 and summ["n_unexplained"] == 0
    assert summ["detection_rate"] == 0.0 and summ["suspect_counts"] == {}
    assert iso.detectability() == {}


# --------------------------------------------------------------------------
# run_evaluation / 리포트
# --------------------------------------------------------------------------
def _record(result, sensor, magnitude):
    rows = [r for r in result["per_sensor"] if r["sensor"] == sensor and r["magnitude"] == magnitude]
    assert len(rows) == 1
    return rows[0]


def test_run_evaluation_on_planted_frame(planted, tmp_path):
    pipe, _, te = planted
    result = run_evaluation(pipe, te, magnitudes=(1, 5), threshold=3.0)
    assert result["n_rows"] == len(te)
    assert result["false_alarm_rate"] <= 0.05
    assert result["coverage"]["uniquely_isolable"] == TWO_PAIR
    assert result["unstandardized_residuals"] == []
    for s in TWO_PAIR:
        rec = _record(result, s, 5.0)
        assert rec["detection_rate"] >= 0.95
        # 무관한 잔차가 우연히 함께 발화한 행은 모호로 남으므로 100% 가 아닐 수 있지만,
        # 틀린 단독 지목은 없다: top-1 + 모호 = 탐지, 후보집합에는 항상 든다
        assert rec["top1_accuracy"] >= 0.95
        assert rec["top1_accuracy"] + rec["ambiguous_rate"] == pytest.approx(rec["detection_rate"])
        assert rec["suspect_set_accuracy"] == pytest.approx(rec["detection_rate"])
        assert rec["uniquely_isolable"] and rec["n_pairs"] == 2
    for s in ONE_PAIR:
        rec = _record(result, s, 5.0)
        assert rec["detection_rate"] >= 0.95
        assert rec["top1_accuracy"] == 0.0 and rec["suspect_set_accuracy"] >= 0.95
        assert rec["ambiguous_rate"] >= 0.95
    agg = {r["magnitude"]: r for r in result["per_magnitude"]}
    assert set(agg) == {1.0, 5.0}
    assert agg[5.0]["n_sensors"] == len(result["coverage"]["covered"])
    assert agg[5.0]["top1_accuracy"] == pytest.approx(
        np.mean([_record(result, s, 5.0)["top1_accuracy"] for s in TWO_PAIR + ONE_PAIR]))
    assert agg[5.0]["detection_rate"] >= agg[1.0]["detection_rate"]

    json_path, md_path = write_reports(result, tmp_path)
    assert json_path == tmp_path / REPORT_JSON and md_path == tmp_path / REPORT_MD
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["per_magnitude"] == json.loads(json.dumps(result["per_magnitude"]))
    md = md_path.read_text(encoding="utf-8")
    for heading in ("## 1. 격리 커버리지", "## 2. 오경보율", "## 3. 주입 크기별 성능", "## 4. 센서별 성능"):
        assert heading in md
    assert "격리 불가" in md and "X_08" in md
    assert "| 5 |" in md and "| X_05 | 2 | O |" in md and "| X_01 | 1 | X |" in md
    assert "경고" not in md                      # 정상 파이프라인에는 비표준화 경고가 없다


def test_run_evaluation_normalises_magnitudes():
    """빈 magnitudes 는 즉시 거절하고, 순서가 섞이거나 중복된 크기는 정렬·중복 제거한다."""
    iso_pipe = _FakePipe()
    df = pd.DataFrame({"A": np.zeros(5), "B": np.zeros(5)})
    with pytest.raises(ValueError):
        run_evaluation(iso_pipe, df, magnitudes=())
    result = run_evaluation(iso_pipe, df, magnitudes=(5, 1, 5))
    assert result["magnitudes"] == [1.0, 5.0]
    assert [r["magnitude"] for r in result["per_magnitude"]] == [1.0, 5.0]
    assert result["per_sensor"][-1]["magnitude"] == 5.0      # 로그에 찍히는 마지막 기록 = 가장 큰 크기 = 리포트 4절


class _FakePipe:
    """쌍 하나짜리 최소 파이프라인 흉내 (run_evaluation 의 입력 검증용)."""
    feat_cols = ["A", "B"]
    use_fdi = True
    fdi_stats = {"pairs": [("A", "B")], "mu": {"A": 0.0, "B": 0.0}, "sd": {"A": 1.0, "B": 1.0}}
    residual_stats = {"r_A_B": {"mean": 0.0, "std": 1.0}}

    def residual_zscores(self, df):
        return pd.DataFrame({"r_A_B": df["A"] - df["B"]}, index=df.index)


def test_run_evaluation_max_rows_samples(planted):
    pipe, _, te = planted
    result = run_evaluation(pipe, te, magnitudes=(5,), max_rows=30, seed=1)
    assert result["n_rows"] == 30
    assert all(r["magnitude"] == 5.0 for r in result["per_sensor"])


def test_evaluate_run_reads_only_the_run_directory(planted, tmp_path):
    """CLI 경로: feature_pipeline.joblib + config.json 만 있는 실행 디렉터리로 리포트를 만든다."""
    pipe, _, te = planted
    run_dir = tmp_path / "run"
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    run_dir.mkdir()
    data_dir.mkdir()
    pipe.save(run_dir / artifacts.PIPELINE_FILE)
    artifacts.write_json(run_dir / artifacts.CONFIG_FILE, {"DATA_DIR": str(data_dir), "VERSION": "t"})
    frame = te.copy()
    frame.insert(0, "ID", [f"TEST_{i:04d}" for i in range(len(frame))])
    frame.to_csv(data_dir / "test.csv", index=False)

    res = evaluate_run(run_dir, magnitudes=(5,), out_dir=out_dir, live=False)
    assert (out_dir / REPORT_JSON).exists() and (out_dir / REPORT_MD).exists()
    assert res["report_md"] == str(out_dir / REPORT_MD)
    payload = artifacts.read_json(out_dir / REPORT_JSON)
    assert payload["meta"]["run_dir"] == str(run_dir)
    rec = _record(payload, "X_05", 5.0)
    assert rec["top1_accuracy"] >= 0.95 and rec["suspect_set_accuracy"] == pytest.approx(rec["detection_rate"])
