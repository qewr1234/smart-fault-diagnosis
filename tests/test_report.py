"""src/report.py: 확률 행렬 -> 행 단위 진단 리포트 변환의 계약을 고정한다.

합성 확률만 쓰며 학습된 실행 디렉터리에 의존하지 않는다.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from src.isolation import ResidualIsolator
from src.pipeline import FeaturePipeline
from src.report import (
    ISOLATION_COLUMNS, REPORT_COLUMNS, REVIEW_REASONS, build_diagnosis_report, format_summary,
    sensors_from_residual, summarize_report,
)
from tests.conftest import FEAT_COLS

CLASSES = np.array([10, 20, 30, 40])          # 원본 라벨 != 인코딩 인덱스
IDS = np.array(["A", "B", "C", "D", "E"])


def _probs():
    """행마다 검토 사유가 하나씩만 켜지도록 설계한 확률 (4 클래스).

    0: 확신 높음, 2위와 격차 큼, 클러스터 밖            -> 사유 없음
    1: 1위 0.45 (<0.5), 격차 0.35                        -> low_confidence 만
    2: 1위 0.50 (not < 0.5), 2위 0.45 -> 격차 0.05       -> ambiguous_top2 만
    3: 1위 0.90, argmax = 인덱스 3 (하드클러스터)        -> confusable_class 만
    4: 확신 높음 (잔차 테스트에서 sensor_residual 용)    -> 기본적으로 사유 없음
    """
    return np.array([
        [0.90, 0.05, 0.03, 0.02],
        [0.45, 0.10, 0.25, 0.20],
        [0.50, 0.45, 0.03, 0.02],
        [0.02, 0.03, 0.05, 0.90],
        [0.80, 0.10, 0.05, 0.05],
    ])


def _report(**kw):
    kw.setdefault("hard_cluster", [3])
    return build_diagnosis_report(IDS, _probs(), CLASSES, **kw)


def test_column_order_is_exactly_the_contract():
    rep = _report()
    assert list(rep.columns) == REPORT_COLUMNS == [
        "ID", "pred", "confidence", "top2", "top2_prob", "top3", "top3_prob", "margin",
        "in_hard_cluster", "anomaly_score", "n_abnormal_residuals", "abnormal_residuals",
        "involved_sensors", "suspect_sensor", "suspect_candidates", "isolation_ambiguous",
        "isolation_note", "review_reasons", "needs_review",
    ]
    assert ISOLATION_COLUMNS == ["suspect_sensor", "suspect_candidates", "isolation_ambiguous", "isolation_note"]
    assert len(rep) == len(IDS)
    assert list(rep["ID"]) == list(IDS)


def test_labels_are_original_classes_not_encoded_indices():
    rep = _report()
    P = _probs()
    np.testing.assert_array_equal(rep["pred"].to_numpy(), CLASSES[P.argmax(1)])
    for col in ("pred", "top2", "top3"):
        assert set(rep[col]).issubset(set(CLASSES.tolist()))
    assert rep.loc[0, "pred"] == 10 and rep.loc[3, "pred"] == 40


def test_topn_and_margin_math():
    rep = _report()
    # 행 1: 0.45 / 0.25 / 0.20 / 0.10 -> top2 = 클래스 30 (idx 2), top3 = 40 (idx 3)
    r = rep.iloc[1]
    assert r["confidence"] == pytest.approx(0.45)
    assert r["top2"] == 30 and r["top2_prob"] == pytest.approx(0.25)
    assert r["top3"] == 40 and r["top3_prob"] == pytest.approx(0.20)
    assert r["margin"] == pytest.approx(0.45 - 0.25)
    # 모든 행에서 margin = confidence - top2_prob, 후보 확률은 내림차순
    np.testing.assert_allclose(rep["margin"], rep["confidence"] - rep["top2_prob"])
    assert (rep["confidence"] >= rep["top2_prob"]).all()
    assert (rep["top2_prob"] >= rep["top3_prob"]).all()


def test_each_review_reason_triggers_on_its_own():
    rep = _report()
    assert rep.loc[0, "review_reasons"] == "" and not rep.loc[0, "needs_review"]
    assert rep.loc[1, "review_reasons"] == "low_confidence"
    assert rep.loc[2, "review_reasons"] == "ambiguous_top2"
    assert rep.loc[3, "review_reasons"] == "confusable_class"
    assert bool(rep.loc[3, "in_hard_cluster"]) is True
    assert rep.loc[[1, 2, 3], "needs_review"].all()
    assert rep["needs_review"].dtype == bool
    assert rep["in_hard_cluster"].dtype == bool


def test_review_reasons_keep_canonical_order_when_combined():
    P = np.array([[0.30, 0.28, 0.22, 0.20]])      # 낮은 확신 + 애매 + 클러스터 밖
    Z = pd.DataFrame({"r_X_01_X_02": [4.0]})
    rep = build_diagnosis_report(["x"], P, CLASSES, hard_cluster=[0], residual_z=Z)
    assert rep.loc[0, "review_reasons"] == ";".join(REVIEW_REASONS)
    assert rep.loc[0, "review_reasons"].split(";") == list(REVIEW_REASONS)


def test_thresholds_are_configurable():
    rep = _report(low_confidence=0.95, low_margin=0.0)
    assert rep["review_reasons"].str.contains("low_confidence").all()
    assert not rep["review_reasons"].str.contains("ambiguous_top2").any()


def test_residual_string_formatting_and_sensor_isolation():
    """X_05 드리프트: X_05 가 든 두 잔차만 벌어지고, X_05 가 빈도 1위 의심 센서가 된다."""
    Z = pd.DataFrame({
        "r_X_05_X_07": [4.24, 0.1, 0.0, 0.2, -0.3],
        "r_X_05_X_06": [-3.51, 0.2, 0.0, 0.1, 0.4],
        "r_X_06_X_07": [0.10, 0.0, 0.0, 0.3, -0.1],
    }, index=[100, 101, 102, 103, 104])              # 인덱스 라벨은 무시되고 위치로 맞춘다
    rep = _report(residual_z=Z)
    r = rep.iloc[0]
    assert r["n_abnormal_residuals"] == 2
    assert r["abnormal_residuals"] == "r_X_05_X_07:+4.2;r_X_05_X_06:-3.5"   # |z| 내림차순, 부호 유지
    assert r["involved_sensors"] == "X_05;X_07;X_06"                        # 빈도 2 > 1, 동률은 등장순
    assert "sensor_residual" in r["review_reasons"] and r["needs_review"]
    assert (rep.iloc[1:]["n_abnormal_residuals"] == 0).all()
    assert (rep.iloc[1:]["abnormal_residuals"] == "").all()
    assert (rep.iloc[1:]["involved_sensors"] == "").all()
    assert rep["n_abnormal_residuals"].dtype.kind == "i"


def test_z_threshold_is_inclusive_and_nan_is_ignored():
    Z = pd.DataFrame({"r_X_01_X_02": [3.0, 2.99, np.nan, -3.0, 0.0]})
    rep = _report(residual_z=Z)
    assert list(rep["n_abnormal_residuals"]) == [1, 0, 0, 1, 0]
    assert rep.loc[3, "abnormal_residuals"] == "r_X_01_X_02:-3.0"


def test_non_finite_residual_is_rendered_readably():
    """센서 값이 inf 면 z 도 inf 다. 309자리 숫자 대신 '+inf' 로 적고 |z| 순서도 유지한다."""
    Z = pd.DataFrame({"r_X_01_X_02": [np.inf, 0.0, 0.0, 0.0, -np.inf],
                      "r_X_05_X_06": [4.0, 0.0, 0.0, 0.0, 0.0]})
    rep = _report(residual_z=Z)
    assert rep.loc[0, "abnormal_residuals"] == "r_X_01_X_02:+inf;r_X_05_X_06:+4.0"
    assert rep.loc[4, "abnormal_residuals"] == "r_X_01_X_02:-inf"
    assert list(rep["n_abnormal_residuals"]) == [2, 0, 0, 0, 1]


def test_sensors_from_residual_handles_underscored_names():
    assert sensors_from_residual("r_X_04_X_39") == ("X_04", "X_39")
    assert sensors_from_residual("r_X_1_X_2") == ("X_1", "X_2")
    assert sensors_from_residual("r_ab_cd") == ("ab", "cd")


def test_pairs_argument_resolves_sensor_names_without_parsing(caplog):
    """센서 이름이 X_nn 꼴이 아니면 이름 파싱은 틀릴 수 있다. 쌍 목록을 주면 파싱하지 않는다."""
    Z = pd.DataFrame({"r_temp_1_press": [4.0, 0.0, 0.0, 0.0, 0.0]})
    with_pairs = _report(residual_z=Z, pairs=[("temp_1", "press")])
    assert with_pairs.loc[0, "involved_sensors"] == "temp_1;press"
    with caplog.at_level(logging.WARNING, logger="src.report"):
        without = _report(residual_z=Z)
    assert without.loc[0, "involved_sensors"] == "temp_1_press"            # 파싱 불가 -> 그대로 두고 경고
    assert any("나눌 수 없어" in r.getMessage() for r in caplog.records)
    # 기본 스키마(X_nn)에서는 두 경로의 결과가 같다
    Z2 = pd.DataFrame({"r_X_05_X_07": [4.2, 0, 0, 0, 0], "r_X_05_X_06": [-3.5, 0, 0, 0, 0]})
    a = _report(residual_z=Z2, pairs=[("X_05", "X_07"), ("X_05", "X_06")])
    b = _report(residual_z=Z2)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[0, "involved_sensors"] == "X_05;X_07;X_06"


def test_empty_residual_frame_and_missing_anomaly_are_handled():
    rep = _report(residual_z=pd.DataFrame(index=range(len(IDS))), anomaly_score=None)
    assert (rep["n_abnormal_residuals"] == 0).all()
    assert (rep["abnormal_residuals"] == "").all()
    assert rep["anomaly_score"].isna().all()
    assert not rep["review_reasons"].str.contains("sensor_residual").any()
    rep2 = _report(residual_z=None)
    pd.testing.assert_frame_equal(rep, rep2)


def test_anomaly_score_is_passed_through():
    score = np.array([-0.5, -0.4, -0.6, -0.3, -0.45])
    rep = _report(anomaly_score=score)
    np.testing.assert_allclose(rep["anomaly_score"], score)
    assert rep["anomaly_score"].dtype == float
    # 요약에는 범위와 방향(낮을수록 이상)이 함께 나온다 — 원시 점수만 보면 어느 쪽이 나쁜지 알 수 없다
    s = summarize_report(rep)
    assert s["anomaly_score_range"] == pytest.approx([-0.6, -0.3])
    text = format_summary(s)
    assert "이상치 점수 범위 -0.600 ~ -0.300" in text and "낮을수록" in text
    assert summarize_report(_report())["anomaly_score_range"] is None
    assert "이상치 점수 없음" in format_summary(summarize_report(_report()))


def test_k_below_three_fills_missing_top_columns_with_nan():
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    rep = build_diagnosis_report(["a", "b"], P, np.array(["ok", "fault"]), hard_cluster=[])
    assert list(rep.columns) == REPORT_COLUMNS
    assert list(rep["pred"]) == ["ok", "fault"]
    assert list(rep["top2"]) == ["fault", "ok"]
    assert rep["top3"].isna().all() and rep["top3_prob"].isna().all()
    np.testing.assert_allclose(rep["margin"], [0.4, 0.2])
    assert not rep["in_hard_cluster"].any()


def test_pred_idx_override_matches_submission_and_flags_ambiguity():
    """균형 배정이 argmax 와 다른 클래스를 골랐을 때: pred 는 그 클래스, margin 은 음수."""
    P = np.array([[0.55, 0.40, 0.03, 0.02]])
    rep = build_diagnosis_report(["a"], P, CLASSES, hard_cluster=[], pred_idx=[1])
    r = rep.iloc[0]
    assert r["pred"] == 20 and r["confidence"] == pytest.approx(0.40)
    assert r["top2"] == 10 and r["top2_prob"] == pytest.approx(0.55)
    assert r["top3"] == 30
    assert r["margin"] == pytest.approx(-0.15)
    assert "ambiguous_top2" in r["review_reasons"]
    # argmax 와 같은 pred_idx 는 기본 동작과 완전히 같다
    default = build_diagnosis_report(IDS, _probs(), CLASSES, hard_cluster=[3])
    same = build_diagnosis_report(IDS, _probs(), CLASSES, hard_cluster=[3], pred_idx=_probs().argmax(1))
    pd.testing.assert_frame_equal(default, same)


def test_shape_mismatches_raise():
    with pytest.raises(ValueError):
        build_diagnosis_report(IDS[:3], _probs(), CLASSES, hard_cluster=[])
    with pytest.raises(ValueError):
        build_diagnosis_report(IDS, _probs(), CLASSES[:3], hard_cluster=[])
    with pytest.raises(ValueError):
        build_diagnosis_report(IDS, _probs(), CLASSES, hard_cluster=[],
                               residual_z=pd.DataFrame({"r_X_01_X_02": [0.0, 0.0]}))
    with pytest.raises(ValueError):
        build_diagnosis_report(IDS, _probs(), CLASSES, hard_cluster=[], anomaly_score=[0.0])
    with pytest.raises(ValueError):
        build_diagnosis_report(IDS, _probs(), CLASSES, hard_cluster=[], top_k=1)


def test_summarize_and_format_summary_content():
    Z = pd.DataFrame({
        "r_X_05_X_07": [4.0, 0.0, 0.0, 0.0, 3.5],
        "r_X_05_X_06": [-3.5, 0.0, 0.0, 0.0, 0.0],
    })
    rep = _report(residual_z=Z)
    s = summarize_report(rep)
    assert s["n_rows"] == 5
    assert s["n_needs_review"] == 5                     # 행 1/2/3 은 각자 사유, 행 0/4 는 잔차 이상
    assert s["reasons_count"] == {"low_confidence": 1, "ambiguous_top2": 1,
                                  "confusable_class": 1, "sensor_residual": 2}
    assert s["predicted_class_counts"] == {10: 4, 40: 1}
    assert s["n_with_abnormal_residuals"] == 2
    assert s["n_in_hard_cluster"] == 1
    assert s["involved_sensor_counts"] == {"X_05": 2, "X_07": 2, "X_06": 1}
    assert s["mean_confidence"] == pytest.approx(rep["confidence"].mean())

    import json
    json.dumps(s)                                        # 직렬화 가능해야 한다

    text = format_summary(s)
    lines = text.splitlines()
    assert lines[0] == "[진단 요약]"
    assert "전체 5행" in lines[1] and "검토 필요 5행" in lines[1] and "100.0%" in lines[1]
    assert "센서 잔차 이상(sensor_residual)=2" in text and "확신도 낮음(low_confidence)=1" in text
    assert "X_05x2" in text
    assert "10:4" in text and "40:1" in text


def test_summary_of_empty_report_does_not_crash():
    rep = build_diagnosis_report([], np.zeros((0, 4)), CLASSES, hard_cluster=[])
    assert list(rep.columns) == REPORT_COLUMNS and len(rep) == 0
    s = summarize_report(rep)
    assert s["n_rows"] == 0 and s["n_needs_review"] == 0
    assert s["n_with_suspect_sensor"] == 0 and s["suspect_sensor_counts"] == {}
    assert format_summary(s).startswith("[진단 요약]")


# --------------------------------------------------------------------------
# 센서 격리(ResidualIsolator) 판정 병합
# --------------------------------------------------------------------------
def _isolation_frame():
    """ResidualIsolator.diagnose() 가 내는 모양을 그대로 흉내 낸 프레임 (인덱스 라벨은 무시된다)."""
    frame = pd.DataFrame({
        "detected": [True, True, False, True, False],
        "n_fired": [2, 1, 0, 1, 0],
        "fired_residuals": [["r_X_05_X_07", "r_X_05_X_06"], ["r_X_01_X_02"], [], ["r_X_03_X_04"], []],
        "suspects": [["X_05"], ["X_01", "X_02"], [], ["X_03", "X_04"], []],
        "suspect_match_ratio": [1.0, 1.0, 0.0, 1.0, 0.0],
        "suspect_evidence": [7.7, 4.0, 0.0, 3.2, 0.0],
        "ambiguous": [False, True, False, True, False],
        "note": ["X_05 센서 이상 의심 — ...", "격리 모호: X_01 / X_02 중 하나 — ...", "정상: 발화한 잔차 없음",
                 "격리 모호: X_03 / X_04 중 하나 — ...", "정상: 발화한 잔차 없음"],
    }, index=[100, 101, 102, 103, 104])
    # diagnose() 와 같이 None 을 유지하는 object 열 (pandas 3 의 str 추론 방지)
    frame.insert(3, "suspect", pd.Series(["X_05", None, None, None, None], index=frame.index, dtype=object))
    return frame


def test_isolation_columns_are_empty_without_isolation_frame():
    """쌍이 없는 파이프라인(격리 정보 없음)에서도 열은 존재하되 비어 있어야 한다."""
    rep = _report()
    assert (rep["suspect_sensor"] == "").all()
    assert (rep["suspect_candidates"] == "").all()
    assert (rep["isolation_note"] == "").all()
    assert not rep["isolation_ambiguous"].any()
    assert rep["isolation_ambiguous"].dtype == bool
    s = summarize_report(rep)
    assert s["n_with_suspect_sensor"] == 0 and s["n_isolation_ambiguous"] == 0
    assert s["suspect_sensor_counts"] == {}
    assert "격리 단독 지목 0행" in format_summary(s)


def test_isolation_frame_is_merged_by_row_position():
    rep = _report(isolation=_isolation_frame())
    assert list(rep.columns) == REPORT_COLUMNS
    # 열 위치: involved_sensors 바로 뒤, review_reasons 바로 앞
    cols = list(rep.columns)
    assert cols[cols.index("involved_sensors") + 1:cols.index("review_reasons")] == ISOLATION_COLUMNS
    assert list(rep["suspect_sensor"]) == ["X_05", "", "", "", ""]
    assert list(rep["suspect_candidates"]) == ["X_05", "X_01;X_02", "", "X_03;X_04", ""]
    assert list(rep["isolation_ambiguous"]) == [False, True, False, True, False]
    assert rep["isolation_ambiguous"].dtype == bool
    assert list(rep["isolation_note"]) == list(_isolation_frame()["note"])      # 한 줄 판정이 그대로 실린다
    # 격리 열은 review_reasons 를 바꾸지 않는다 (사유 계약은 네 가지 그대로)
    assert rep.loc[0, "review_reasons"] == ""
    assert list(rep["review_reasons"]) == list(_report()["review_reasons"])

    s = summarize_report(rep)
    assert s["n_with_suspect_sensor"] == 1
    assert s["n_isolation_ambiguous"] == 2
    assert s["suspect_sensor_counts"] == {"X_05": 1}
    text = format_summary(s)
    assert "격리 단독 지목 1행" in text and "X_05x1" in text and "격리 모호 2행" in text


def test_isolation_summary_lists_top3_suspects_by_frequency():
    iso = pd.DataFrame({
        "suspect": pd.Series(["X_07", "X_05", "X_05", "X_06", "X_05"], dtype=object),
        "suspects": [["X_07"], ["X_05"], ["X_05"], ["X_06"], ["X_05"]],
        "ambiguous": [False] * 5,
    })
    rep = _report(isolation=iso)
    assert (rep["isolation_note"] == "").all()            # note 열이 없는 프레임이면 빈 문자열
    s = summarize_report(rep)
    assert list(s["suspect_sensor_counts"].items()) == [("X_05", 3), ("X_06", 1), ("X_07", 1)]
    text = format_summary(s)
    assert "격리 단독 지목 5행 (상위 3: X_05x3, X_06x1, X_07x1)" in text
    # 네 번째 센서는 top-3 에서 잘린다 (빈도 같으면 이름순)
    iso4 = iso.copy()
    iso4.loc[4, "suspect"] = "X_08"            # X_05x2, X_06x1, X_07x1, X_08x1
    text4 = format_summary(summarize_report(_report(isolation=iso4)))
    assert "격리 단독 지목 5행 (상위 3: X_05x2, X_06x1, X_07x1)" in text4
    assert "X_08" not in text4


def test_isolation_summary_survives_csv_roundtrip(tmp_path):
    """CSV 로 저장하면 빈 문자열이 NaN 이 되는데, 요약은 그래도 같은 숫자를 내야 한다."""
    rep = _report(isolation=_isolation_frame())
    path = tmp_path / "diagnosis.csv"
    rep.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert list(back.columns) == REPORT_COLUMNS
    assert back["isolation_ambiguous"].dtype == bool
    s1, s2 = summarize_report(rep), summarize_report(back)
    for key in ("n_with_suspect_sensor", "n_isolation_ambiguous", "suspect_sensor_counts",
                "involved_sensor_counts", "reasons_count"):
        assert s1[key] == s2[key]


def test_isolation_length_mismatch_raises():
    with pytest.raises(ValueError):
        _report(isolation=_isolation_frame().iloc[:3])
    with pytest.raises(ValueError):
        _report(isolation=pd.DataFrame({"suspect": [None] * 5}))     # suspects / ambiguous 열 없음


def test_isolation_end_to_end_on_planted_pairs():
    """심은 준중복 쌍 + 센서 바이어스 주입: 격리기 판정이 리포트 열로 그대로 들어가야 한다.

    X_05 는 두 쌍(X_05-X_06, X_05-X_07)에 들어 있어 단독 격리되고, X_01 은 X_02 와 쌍 하나를
    공유할 뿐이라 후보집합 {X_01, X_02} 로 모호하게 남는다. 그 차이가 리포트에 보여야 한다.
    """
    rng = np.random.default_rng(77)
    n = 600
    tr = pd.DataFrame(rng.normal(size=(n, 52)), columns=FEAT_COLS)
    tr["X_02"] = tr["X_01"] + rng.normal(scale=0.05, size=n)
    tr["X_06"] = tr["X_05"] + rng.normal(scale=0.05, size=n)
    tr["X_07"] = tr["X_05"] + rng.normal(scale=0.05, size=n)
    pipe = FeaturePipeline(feat_cols=FEAT_COLS, clip_shift=False)
    pipe.fit(tr, None)
    assert {frozenset(p) for p in pipe.fdi_stats["pairs"]} >= {
        frozenset(("X_01", "X_02")), frozenset(("X_05", "X_06")), frozenset(("X_05", "X_07"))}

    te = tr.iloc[:30].copy().reset_index(drop=True)
    te.insert(0, "ID", [f"TEST_{i:04d}" for i in range(len(te))])
    sd = pipe.fdi_stats["sd"]
    te.loc[:9, "X_05"] += 5.0 * sd["X_05"]        # 행 0-9: X_05 단독 격리 가능
    te.loc[10:19, "X_01"] += 5.0 * sd["X_01"]      # 행 10-19: X_01/X_02 모호
    # 행 20-29: 정상

    k = 4
    probs = np.full((len(te), k), 1.0 / k)
    probs[:, 0] = 0.7
    probs /= probs.sum(1, keepdims=True)
    isolator = ResidualIsolator.from_pipeline(pipe)
    residual_z = pipe.residual_zscores(te)
    isolation = isolator.diagnose(te, threshold=3.0, zscores=residual_z)
    rep = build_diagnosis_report(te["ID"].values, probs, CLASSES, hard_cluster=[],
                                 residual_z=residual_z, isolation=isolation, pairs=isolator.pairs)

    assert list(rep.columns) == REPORT_COLUMNS
    first = rep.iloc[:10]
    # 이 시드에서는 행 0 에 무관한 r_X_01_X_02 가 우연히 3.1σ 로 함께 발화한다. 그 행은 X_05 를
    # 단독 지목하지 않고 "X_05 가 2/3개 설명, 설명되지 않은 잔차 r_X_01_X_02" 로 모호 처리된다.
    assert (first["suspect_sensor"] == "X_05").mean() >= 0.9
    assert not ((first["suspect_sensor"] != "") & (first["suspect_sensor"] != "X_05")).any()   # 틀린 단독 지목 없음
    assert first["suspect_candidates"].str.contains("X_05").all()
    assert first["involved_sensors"].str.startswith("X_05").all()
    assert first["review_reasons"].str.contains("sensor_residual").all()
    confident = first[~first["isolation_ambiguous"]]
    assert confident["isolation_note"].str.startswith("X_05 센서 이상 의심").all()
    for note in first.loc[first["isolation_ambiguous"], "isolation_note"]:
        assert "X_05 가 2/3개 설명" in note and "r_X_01_X_02" in note

    second = rep.iloc[10:20]
    assert (second["suspect_sensor"] == "").all()
    assert (second["suspect_candidates"] == "X_01;X_02").mean() > 0.9
    assert second["isolation_ambiguous"].mean() > 0.9
    assert second["isolation_note"].str.contains("격리 모호: X_01 / X_02 중 하나").mean() > 0.9

    clean = rep.iloc[20:]
    assert (clean["suspect_sensor"] == "").all() and not clean["isolation_ambiguous"].any()
    assert (clean["isolation_note"] == "정상: 발화한 잔차 없음").all()
    # 격리 모호 행은 잔차 자체는 이상이므로 여전히 sensor_residual 로 검토 대상이다
    assert second["review_reasons"].str.contains("sensor_residual").mean() > 0.9

    s = summarize_report(rep)
    assert s["suspect_sensor_counts"].get("X_05", 0) >= 9
    assert s["n_isolation_ambiguous"] >= 9
    assert "X_05x" in format_summary(s)
