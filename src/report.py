"""행 단위 진단 리포트 — 모델 확률을 현장 엔지니어가 바로 판독할 수 있는 표로 바꾼다.

제출 파일(ID, target)은 "무슨 고장인가"만 답한다. 장비 지원 엔지니어가 실제로 묻는 것은
그 다음이다.

  - 이 판정을 얼마나 믿을 수 있는가?            -> confidence / margin / top2·top3 후보
  - 모델이 원래 잘 헷갈리는 클래스 집합인가?    -> in_hard_cluster (confusable_class)
  - 센서 자체가 이상한 건 아닌가?               -> abnormal_residuals / involved_sensors
  - 전체적으로 학습 분포에서 벗어난 샘플인가?   -> anomaly_score (IsolationForest)
  - 그래서 사람이 다시 봐야 하는 행은 어느 것인가? -> review_reasons / needs_review

센서 격리(FDI isolation)는 analytical redundancy 의 고전적 논리를 그대로 따른다.
준중복 센서쌍 (a, b) 의 표준화 잔차 r_a_b 는 정상 상태에서 ~N(0, 1) 이고, 센서 a 에
드리프트/편향이 생기면 a 가 들어간 잔차만 벌어진다. 따라서 임계를 넘은 잔차들에
공통으로 등장하는 센서가 가장 유력한 고장 센서다. `involved_sensors` 는 그 빈도순이다.

여기에 `src/isolation.py` 의 `ResidualIsolator` 가 내린 격리 판정을 네 열로 덧붙인다.
발화한 잔차만 세는 빈도순과 달리, 격리기는 발화하지 않은 잔차까지 알리바이로 쓴다
(signature matching). 단독 지목된 센서가 `suspect_sensor`, 후보 전체가 `suspect_candidates`,
단독 지목에 실패했다는 표시(쌍 하나를 둘이서만 공유했거나, 발화 잔차를 센서 하나로 설명할 수
없었거나)가 `isolation_ambiguous`, 그 판정을 한 줄 한국어로 풀어 쓴 것이 `isolation_note` 다.
리포트의 열 정의는 이 모듈 한 곳(REPORT_COLUMNS)에만 둔다.

잔차 이름 -> 센서 이름 복원은 파이프라인이 아는 쌍 목록(`pairs` 인자)으로 하고, 쌍 목록이 없을
때만 이름 파싱(`sensors_from_residual`)으로 대신한다.

이 모듈은 pandas/numpy 만 쓰는 순수 함수로 구성되며 torch 를 import 하지 않는다.
"""
import logging
import re
from collections import Counter

import numpy as np
import pandas as pd

from .features import residual_name

log = logging.getLogger(__name__)

# 검토 사유 — 리포트 문자열에 이 순서로 기록된다.
REVIEW_REASONS = ("low_confidence", "ambiguous_top2", "confusable_class", "sensor_residual")
# 요약 출력용 한국어 라벨 (CSV 에는 코드 그대로 기록된다)
REASON_LABELS = {
    "low_confidence": "확신도 낮음",
    "ambiguous_top2": "1·2위 근접",
    "confusable_class": "혼동 클러스터",
    "sensor_residual": "센서 잔차 이상",
}

# 센서 격리(src/isolation.py) 판정 열. involved_sensors 바로 뒤에 온다.
ISOLATION_COLUMNS = ["suspect_sensor", "suspect_candidates", "isolation_ambiguous", "isolation_note"]

# 기본(top_k=3) 리포트의 열 순서. predict.py 가 쓰는 CSV 계약이다.
REPORT_COLUMNS = [
    "ID", "pred", "confidence", "top2", "top2_prob", "top3", "top3_prob", "margin",
    "in_hard_cluster", "anomaly_score", "n_abnormal_residuals", "abnormal_residuals",
    "involved_sensors", *ISOLATION_COLUMNS, "review_reasons", "needs_review",
]

_SENSOR_TOKEN = re.compile(r"X_\d+")


# --------------------------------------------------------------------------
# 잔차 이름 <-> 센서
# --------------------------------------------------------------------------
def sensors_from_residual(name):
    """잔차 열 이름 `r_{a}_{b}` 에서 두 센서 이름 (a, b) 를 복원한다.

    센서 이름 자체가 `X_04` 처럼 밑줄을 포함하므로 단순 split 으로는 나눌 수 없다.
    1) `X_\\d+` 패턴이 정확히 두 개면 그것을 쓴다 (기본 데이터 스키마).
    2) 아니면 접두어를 뗀 토큰을 반으로 나눈다 (같은 형식의 이름 두 개가 붙은 경우).
    3) 그래도 안 되면 본문 전체를 하나의 이름으로 돌려주고 경고한다.

    파이프라인의 쌍 목록을 아는 호출자는 `residual_pair_map` 을 써서 파싱 자체를 피한다.
    """
    body = name[2:] if name.startswith("r_") else name
    found = _SENSOR_TOKEN.findall(body)
    if len(found) == 2 and "_".join(found) == body:
        return found[0], found[1]
    tokens = body.split("_")
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        return "_".join(tokens[:half]), "_".join(tokens[half:])
    log.warning("[Report] 잔차 이름 %r 을 센서 두 개로 나눌 수 없어 그대로 둡니다 — "
                "build_diagnosis_report 에 pairs 를 넘기면 파싱 없이 복원됩니다.", name)
    return (body,)


def residual_pair_map(pairs):
    """[(a, b), ...] -> {잔차 이름: (a, b)}. features.residual_name 과 같은 규칙을 쓴다."""
    return {residual_name(a, b): (a, b) for a, b in pairs}


def _format_z(z):
    """부호를 항상 붙인 소수 1자리 (예: +4.2 / -3.7, 비유한은 +inf).

    부호는 r_a_b = z_a - z_b 기준이다 — 양수면 a 가 높거나 b 가 낮다는 뜻이고, 어느 쪽 센서가
    움직였는지는 부호가 아니라 suspect_sensor(격리 판정)로 읽어야 한다.
    """
    return f"{float(z):+.1f}"


def _rank_involved_sensors(entries, pair_map=None):
    """비정상 잔차 목록(|z| 내림차순)에서 센서를 빈도순으로 나열한다.

    빈도가 같으면 더 큰 |z| 잔차에 먼저 등장한 센서를 앞에 둔다. pair_map 에 있는 잔차는
    그 (a, b) 를 쓰고, 없는 잔차만 이름을 파싱한다.
    """
    pair_map = pair_map or {}
    counts, first_seen = Counter(), {}
    for name in entries:
        for s in pair_map.get(name) or sensors_from_residual(name):
            counts[s] += 1
            first_seen.setdefault(s, len(first_seen))
    return sorted(counts, key=lambda s: (-counts[s], first_seen[s]))


# --------------------------------------------------------------------------
# 리포트
# --------------------------------------------------------------------------
def _candidate_ranking(probs, pred_idx):
    """확률 내림차순 클래스 순위. pred_idx 가 주어지면 그 클래스를 1위로 끌어올린다.

    후처리(균형 배정)가 argmax 와 다른 클래스를 골랐을 때 리포트의 pred 가 제출 파일과
    일치하도록 하기 위한 것이다. 이 경우 margin 이 음수가 되어 ambiguous_top2 로 잡힌다.
    """
    order = np.argsort(-probs, axis=1, kind="stable")
    if pred_idx is None:
        return order
    pred_idx = np.asarray(pred_idx, dtype=int)
    if pred_idx.shape != (probs.shape[0],):
        raise ValueError(f"pred_idx 길이 {pred_idx.shape} 가 행 수 {probs.shape[0]} 와 다릅니다.")
    for i in np.flatnonzero(order[:, 0] != pred_idx):
        rest = order[i][order[i] != pred_idx[i]]
        order[i] = np.concatenate([[pred_idx[i]], rest])
    return order


def _abnormal_residual_columns(residual_z, n_rows, z_threshold, pair_map=None):
    """행별 비정상 잔차 수 / 문자열 / 관련 센서 문자열을 만든다."""
    n_abn = np.zeros(n_rows, dtype=int)
    abn_str = [""] * n_rows
    sensors_str = [""] * n_rows
    if residual_z is None or residual_z.shape[1] == 0:
        return n_abn, abn_str, sensors_str
    if residual_z.shape[0] != n_rows:
        raise ValueError(f"residual_z 행 수 {residual_z.shape[0]} 가 행 수 {n_rows} 와 다릅니다.")

    names = [str(c) for c in residual_z.columns]
    # NaN 잔차는 판단 보류(정상 취급). ±inf 는 그대로 두어 "+inf" 로 읽히게 한다 (309자리 숫자 방지).
    Z = np.nan_to_num(residual_z.to_numpy(dtype=float), nan=0.0, posinf=np.inf, neginf=-np.inf)
    flagged = np.abs(Z) >= float(z_threshold)
    n_abn = flagged.sum(axis=1).astype(int)
    for i in np.flatnonzero(n_abn):
        cols = np.flatnonzero(flagged[i])
        cols = cols[np.argsort(-np.abs(Z[i, cols]), kind="stable")]
        entries = [names[j] for j in cols]
        abn_str[i] = ";".join(f"{names[j]}:{_format_z(Z[i, j])}" for j in cols)
        sensors_str[i] = ";".join(_rank_involved_sensors(entries, pair_map))
    return n_abn, abn_str, sensors_str


def _is_missing(v):
    """None 또는 NaN 스칼라인지 (리스트/문자열은 값으로 본다)."""
    if v is None:
        return True
    return isinstance(v, float) and np.isnan(v)


def _isolation_columns(isolation, n_rows):
    """ResidualIsolator.diagnose() 결과를 리포트 열 네 개로 바꾼다.

    isolation 은 suspect(str|None) / suspects(list) / ambiguous(bool) 열을 가진 프레임이며
    잔차와 마찬가지로 행 위치로 맞춘다. note 열(한 줄 판정)은 있으면 isolation_note 로 옮기고
    없으면 빈 문자열이다. None 이면 격리 정보 없음 (빈 문자열 / False).
    """
    suspect = [""] * n_rows
    candidates = [""] * n_rows
    ambiguous = np.zeros(n_rows, dtype=bool)
    note = [""] * n_rows
    if isolation is None:
        return suspect, candidates, ambiguous, note
    if len(isolation) != n_rows:
        raise ValueError(f"isolation 행 수 {len(isolation)} 가 행 수 {n_rows} 와 다릅니다.")
    missing = [c for c in ("suspect", "suspects", "ambiguous") if c not in isolation.columns]
    if missing:
        raise ValueError(f"isolation 프레임에 열이 없습니다: {missing}")

    for i, (s, cands) in enumerate(zip(isolation["suspect"].tolist(), isolation["suspects"].tolist())):
        if not _is_missing(s):
            suspect[i] = str(s)
        if isinstance(cands, str):
            candidates[i] = cands
        elif not _is_missing(cands):
            candidates[i] = ";".join(str(c) for c in cands)
    ambiguous = isolation["ambiguous"].fillna(False).to_numpy(dtype=bool)
    if "note" in isolation.columns:
        note = ["" if _is_missing(v) else str(v) for v in isolation["note"].tolist()]
    return suspect, candidates, ambiguous, note


def build_diagnosis_report(ids, probs, classes, hard_cluster, residual_z=None, anomaly_score=None,
                           top_k=3, z_threshold=3.0, low_confidence=0.5, low_margin=0.10,
                           pred_idx=None, isolation=None, pairs=None):
    """행 단위 진단 리포트 DataFrame 을 만든다.

    인자
      ids:           행 ID (길이 n)
      probs:         (n, K) 후처리된 확률 (바이어스 적용 후)
      classes:       원본 라벨 배열 (label_classes.json). pred = classes[argmax]
      hard_cluster:  서로 혼동되는 클래스의 인코딩 인덱스 목록 (빈 리스트 가능)
      residual_z:    FeaturePipeline.residual_zscores(test) 결과 (열 = r_X_a_X_b). None 가능
      anomaly_score: IsolationForest score_samples 의 fold 평균 (낮을수록 이상). None 가능
      top_k:         후보 클래스 열 수 (top2 .. top{top_k}). 기본 3 -> top2, top3
      z_threshold:   |z| 가 이 값 이상이면 비정상 잔차
      low_confidence: 1위 확률이 이 값 미만이면 low_confidence
      low_margin:    1위-2위 확률 차가 이 값 미만이면 ambiguous_top2
      pred_idx:      최종 예측의 인코딩 인덱스 (제출 파일과 동일하게 맞출 때). None 이면 argmax
      isolation:     ResidualIsolator.diagnose(test) 결과 (suspect / suspects / ambiguous [/ note] 열).
                     None 이면 격리 열은 빈 문자열 / False 로 채운다 (쌍이 없는 파이프라인 포함)
      pairs:         파이프라인의 센서쌍 [(a, b), ...]. 주면 involved_sensors 의 센서 이름을 잔차
                     이름 파싱 대신 이 목록으로 복원한다 (pipe.fdi_stats["pairs"] / isolator.pairs)

    출력 열 (top_k=3 기준, 이 순서)
      ID, pred, confidence, top2, top2_prob, top3, top3_prob, margin, in_hard_cluster,
      anomaly_score, n_abnormal_residuals, abnormal_residuals, involved_sensors,
      suspect_sensor, suspect_candidates, isolation_ambiguous, isolation_note,
      review_reasons, needs_review

      anomaly_score       IsolationForest score_samples 평균 — 낮을수록 학습 분포에서 먼 샘플
                          (모두 음수인 것이 정상). 저장된 isoforest 가 없으면 NaN
      abnormal_residuals  "잔차:z;..." (|z| 내림차순). z 의 부호는 r_a_b = z_a - z_b 기준이라
                          어느 센서가 움직였는지는 suspect_sensor 로 읽는다
      suspect_sensor      격리기가 단독으로 지목한 센서 (없으면 "")
      suspect_candidates  후보 센서 전체, ";" 구분 (예: "X_01;X_02")
      isolation_ambiguous 잔차는 발화했지만 단독 지목에 실패했는가 (동점 후보, 또는 발화 잔차를
                          센서 하나로 설명할 수 없는 다중 고장 / 부분 발화)
      isolation_note      격리 판정을 풀어 쓴 한 줄 한국어 메시지 (ResidualIsolator.explain)

    K < top_k 이면 없는 후보 열은 NaN 으로 채운다. pred/topN 은 인코딩 인덱스가 아니라
    원본 라벨이다. 잔차, 이상치 점수, 격리 결과는 행 위치로 맞추며 인덱스 라벨은 보지 않는다.
    """
    if top_k < 2:
        raise ValueError("top_k 는 2 이상이어야 합니다 (margin 계산에 2위가 필요).")
    probs = np.asarray(probs, dtype=float)
    classes = np.asarray(classes)
    ids = np.asarray(ids)
    if probs.ndim != 2:
        raise ValueError(f"probs 는 (n, K) 2차원이어야 합니다: {probs.shape}")
    n, K = probs.shape
    if len(ids) != n:
        raise ValueError(f"ids 길이 {len(ids)} 가 probs 행 수 {n} 와 다릅니다.")
    if len(classes) != K:
        raise ValueError(f"classes 길이 {len(classes)} 가 probs 열 수 {K} 와 다릅니다.")
    if anomaly_score is not None and len(anomaly_score) != n:
        raise ValueError(f"anomaly_score 길이 {len(anomaly_score)} 가 행 수 {n} 와 다릅니다.")

    rows = np.arange(n)
    order = _candidate_ranking(probs, pred_idx)
    top1 = order[:, 0]
    confidence = probs[rows, top1]

    out = {"ID": ids, "pred": classes[top1], "confidence": confidence}
    for j in range(2, top_k + 1):
        if K >= j:
            idx = order[:, j - 1]
            out[f"top{j}"] = classes[idx]
            out[f"top{j}_prob"] = probs[rows, idx]
        else:
            out[f"top{j}"] = np.full(n, np.nan)
            out[f"top{j}_prob"] = np.full(n, np.nan)
    margin = confidence - probs[rows, order[:, 1]] if K >= 2 else np.full(n, np.nan)
    out["margin"] = margin

    hard = np.asarray(list(hard_cluster), dtype=int)
    in_hard = np.isin(top1, hard) if hard.size else np.zeros(n, dtype=bool)
    out["in_hard_cluster"] = in_hard
    out["anomaly_score"] = (np.full(n, np.nan) if anomaly_score is None
                            else np.asarray(anomaly_score, dtype=float))

    pair_map = residual_pair_map(pairs) if pairs else None
    n_abn, abn_str, sensors_str = _abnormal_residual_columns(residual_z, n, z_threshold, pair_map)
    out["n_abnormal_residuals"] = n_abn
    out["abnormal_residuals"] = abn_str
    out["involved_sensors"] = sensors_str

    suspect, candidates, iso_ambiguous, iso_note = _isolation_columns(isolation, n)
    out["suspect_sensor"] = suspect
    out["suspect_candidates"] = candidates
    out["isolation_ambiguous"] = iso_ambiguous
    out["isolation_note"] = iso_note

    flags = {
        "low_confidence": confidence < float(low_confidence),
        "ambiguous_top2": (np.nan_to_num(margin, nan=np.inf) < float(low_margin)),
        "confusable_class": in_hard,
        "sensor_residual": n_abn > 0,
    }
    reason_matrix = np.column_stack([flags[r] for r in REVIEW_REASONS])
    out["review_reasons"] = [";".join(r for r, on in zip(REVIEW_REASONS, row) if on)
                             for row in reason_matrix]
    out["needs_review"] = reason_matrix.any(axis=1)

    report = pd.DataFrame(out)
    report["in_hard_cluster"] = report["in_hard_cluster"].astype(bool)
    report["isolation_ambiguous"] = report["isolation_ambiguous"].astype(bool)
    report["needs_review"] = report["needs_review"].astype(bool)
    report["n_abnormal_residuals"] = report["n_abnormal_residuals"].astype(int)
    return report


# --------------------------------------------------------------------------
# 요약
# --------------------------------------------------------------------------
def _native(v):
    """numpy 스칼라를 json 직렬화 가능한 파이썬 값으로."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _split_counts(series):
    """';' 로 이어진 문자열 열에서 항목별 등장 행 수를 센다."""
    counts = Counter()
    for s in series.fillna("").astype(str):
        for item in s.split(";"):
            if item:
                counts[item] += 1
    return counts


def summarize_report(report):
    """리포트 전체를 집계한 dict (json 직렬화 가능).

    키: n_rows, n_needs_review, reasons_count {사유: 행 수}, predicted_class_counts {라벨: 행 수},
        n_with_abnormal_residuals, n_in_hard_cluster, mean_confidence,
        involved_sensor_counts {센서: 비정상 잔차에 등장한 행 수, 빈도 내림차순},
        n_with_suspect_sensor (격리기가 단독 지목한 행 수), n_isolation_ambiguous,
        suspect_sensor_counts {센서: 단독 지목된 행 수, 빈도 내림차순},
        anomaly_score_range [min, max] (IsolationForest 점수, 낮을수록 이상; 점수가 없으면 None)

    CSV 에서 다시 읽은 리포트(빈 문자열이 NaN 으로 바뀐 것)도 그대로 받는다.
    """
    n = int(len(report))
    anomaly = report["anomaly_score"].astype(float) if n and "anomaly_score" in report.columns else pd.Series(dtype=float)
    anomaly_range = ([float(anomaly.min()), float(anomaly.max())]
                     if anomaly.notna().any() else None)
    reasons = _split_counts(report["review_reasons"]) if n else Counter()
    sensors = _split_counts(report["involved_sensors"]) if n else Counter()

    has_isolation = all(c in report.columns for c in ISOLATION_COLUMNS)
    suspects = Counter()
    n_ambiguous = 0
    if n and has_isolation:
        suspects = _split_counts(report["suspect_sensor"])      # 행마다 센서 하나이므로 행 수와 같다
        n_ambiguous = int(report["isolation_ambiguous"].fillna(False).astype(bool).sum())

    class_counts = {}
    if n:
        vc = report["pred"].value_counts()
        keys = list(vc.index)
        try:
            keys = sorted(keys)
        except TypeError:
            pass
        class_counts = {_native(k): int(vc[k]) for k in keys}

    return {
        "n_rows": n,
        "n_needs_review": int(report["needs_review"].sum()) if n else 0,
        "reasons_count": {r: int(reasons.get(r, 0)) for r in REVIEW_REASONS},
        "predicted_class_counts": class_counts,
        "n_with_abnormal_residuals": int((report["n_abnormal_residuals"] > 0).sum()) if n else 0,
        "n_in_hard_cluster": int(report["in_hard_cluster"].sum()) if n else 0,
        "mean_confidence": float(report["confidence"].mean()) if n else float("nan"),
        "involved_sensor_counts": {s: int(c) for s, c in
                                   sorted(sensors.items(), key=lambda kv: (-kv[1], kv[0]))},
        "n_with_suspect_sensor": int(sum(suspects.values())),
        "n_isolation_ambiguous": n_ambiguous,
        "suspect_sensor_counts": {s: int(c) for s, c in
                                  sorted(suspects.items(), key=lambda kv: (-kv[1], kv[0]))},
        "anomaly_score_range": anomaly_range,
    }


def format_summary(summary, max_sensors=5, max_suspects=3):
    """summarize_report 결과를 predict.py 끝에 출력할 한국어 여러 줄 문자열로 만든다.

    max_sensors:  잔차 이상에 등장한 센서를 몇 개까지 보일지 (빈도순)
    max_suspects: 격리기가 단독 지목한 센서를 몇 개까지 보일지 (빈도순, 기본 top-3)
    """
    n = summary["n_rows"]
    n_rev = summary["n_needs_review"]
    pct = (100.0 * n_rev / n) if n else 0.0
    reasons = ", ".join(f"{REASON_LABELS[r]}({r})={summary['reasons_count'].get(r, 0)}"
                        for r in REVIEW_REASONS)
    classes = ", ".join(f"{k}:{v}" for k, v in summary["predicted_class_counts"].items())

    lines = [
        "[진단 요약]",
        f"  전체 {n}행 중 검토 필요 {n_rev}행 ({pct:.1f}%)",
        f"  검토 사유: {reasons}",
        f"  평균 확신도 {summary['mean_confidence']:.3f} / 혼동 클러스터 예측 {summary['n_in_hard_cluster']}행",
    ]
    sensors = summary.get("involved_sensor_counts", {})
    top = list(sensors.items())[:max_sensors]
    suspect = ", ".join(f"{s}x{c}" for s, c in top) if top else "없음"
    lines.append(f"  센서 잔차 이상 {summary['n_with_abnormal_residuals']}행 (의심 센서: {suspect})")
    isolated = summary.get("suspect_sensor_counts", {})
    top_iso = list(isolated.items())[:max_suspects]
    iso_str = ", ".join(f"{s}x{c}" for s, c in top_iso) if top_iso else "없음"
    lines.append(f"  격리 단독 지목 {summary.get('n_with_suspect_sensor', 0)}행 "
                 f"(상위 {max_suspects}: {iso_str}) / 격리 모호 {summary.get('n_isolation_ambiguous', 0)}행")
    rng = summary.get("anomaly_score_range")
    if rng:
        lines.append(f"  이상치 점수 범위 {rng[0]:.3f} ~ {rng[1]:.3f} "
                     "(IsolationForest, 낮을수록 학습 분포에서 멂)")
    else:
        lines.append("  이상치 점수 없음 (저장된 isoforest 없음)")
    lines.append(f"  예측 클래스 분포: {classes if classes else '없음'}")
    return "\n".join(lines)
