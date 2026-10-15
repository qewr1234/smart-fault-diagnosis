# ============================================
# Smart Fault Diagnosis — 센서 고장 격리(FDI) 평가 진입점
#
# train.py 가 남긴 실행 디렉터리의 feature_pipeline.joblib 만 읽어,
# 잔차 기반 센서 격리가 "어느 센서를, 얼마나 작은 편향부터, 얼마나 정확히"
# 짚어내는지 고장 주입(fault injection)으로 측정한다. 모델 체크포인트는 쓰지 않는다.
#
#   python evaluate_isolation.py outputs/<VERSION>_<ts>
#   python evaluate_isolation.py outputs/<VERSION>_<ts> --test-csv new_batch.csv --magnitudes 1 2 3 5
# ============================================
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import artifacts
from src.isolation import ResidualIsolator
from src.pipeline import FeaturePipeline

log = logging.getLogger("evaluate_isolation")

REPORT_JSON = "isolation_report.json"
REPORT_MD = "isolation_report.md"
SENSOR_METRICS = ["detection_rate", "top1_accuracy", "suspect_set_accuracy", "ambiguous_rate"]


# --------------------------------------------------------------------------
# 평가 본체
# --------------------------------------------------------------------------
def run_evaluation(pipe, df, magnitudes=(1, 2, 3, 5), threshold=3.0, max_rows=None, seed=0):
    """고장 주입으로 잔차 격리 성능을 잰다.

    pipe:       학습된 FeaturePipeline
    df:         원본 센서 프레임 (X_01~X_52 포함). 주입 전 상태를 "정상" 기준선으로 본다.
    magnitudes: 주입할 편향 크기 (센서별 train 표준편차의 배수). 비어 있으면 ValueError.
                오름차순으로 정렬·중복 제거해 쓰므로 결과의 per_magnitude 는 작은 크기부터 온다.
    threshold:  잔차 z-score 발화 임계 (|z| >= threshold)
    max_rows:   None 이면 전체, 아니면 seed 로 표본 추출해 속도를 맞춘다

    절차: 커버된 센서 s 와 크기 k 마다 df 를 복사해 s 컬럼에 k·sd_s 를 더하고 diagnose()
    를 돌려 탐지율 / top-1 격리 정확도(suspect == s) / 후보집합 포함률(s in suspects) /
    모호 비율을 기록한다. 크기별 집계는 센서 평균이다.

    주의: df 에 실제 고장 행이 섞여 있으면 false_alarm_rate 는 순수 오경보율이 아니라
    "주입 전 탐지율(기준선)" 로 읽어야 하며, 주입 후 탐지율도 그만큼 바닥이 올라간다.

    결과의 unstandardized_residuals 가 비어 있지 않으면 파이프라인에 train 잔차 통계가 없는
    구버전 실행 디렉터리다. 그때 |z| 임계는 sigma 단위가 아니므로 탐지율은 실제 감도를
    크게 과소평가한다 (리포트에도 같은 경고를 남긴다).
    """
    magnitudes = sorted({float(k) for k in magnitudes})
    if not magnitudes:
        raise ValueError("magnitudes 가 비어 있습니다 — 주입 크기를 하나 이상 지정하세요.")
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=int(max_rows), random_state=seed)
    df = df.reset_index(drop=True)

    iso = ResidualIsolator.from_pipeline(pipe)
    cov = iso.coverage()
    sd = pipe.fdi_stats.get("sd", {})
    uniquely = set(cov["uniquely_isolable"])
    unstandardized = iso.unstandardized_residuals

    clean = iso.diagnose(df, threshold=threshold)
    clean_summary = ResidualIsolator.summarize(clean)
    false_alarm_rate = clean_summary["detection_rate"]
    log.info("[Isolation] 행 %d, 쌍 %d, 커버 센서 %d/%d, 단독 격리 가능 %d, 임계 |z|>=%.1f, "
             "기준선 탐지율 %.2f%%", len(df), cov["n_pairs"], len(cov["covered"]),
             cov["n_sensors"], len(uniquely), threshold, 100 * false_alarm_rate)
    if unstandardized:
        log.warning("[Isolation] 잔차 %d개의 z-score 가 표준화되지 않아 아래 탐지율/격리 정확도는 "
                    "실제 감도를 크게 과소평가합니다 (구버전 실행 디렉터리).", len(unstandardized))

    per_sensor = []
    for s in cov["covered"]:
        if s not in df.columns:
            log.warning("[Isolation] %s 컬럼이 데이터에 없어 건너뜁니다.", s)
            continue
        scale = float(sd.get(s, 1.0))
        for k in magnitudes:
            faulty = df.copy()
            faulty[s] = faulty[s] + k * scale
            diag = iso.diagnose(faulty, threshold=threshold)
            n = len(diag)
            rec = {
                "sensor": s,
                "magnitude": k,
                "n_pairs": int(cov["pairs_per_sensor"].get(s, 0)),
                "uniquely_isolable": s in uniquely,
                "detection_rate": float(diag["detected"].mean()) if n else 0.0,
                "top1_accuracy": float((diag["suspect"] == s).mean()) if n else 0.0,
                "suspect_set_accuracy": float(np.mean([s in c for c in diag["suspects"]])) if n else 0.0,
                "ambiguous_rate": float(diag["ambiguous"].mean()) if n else 0.0,
            }
            per_sensor.append(rec)
        last = per_sensor[-1]            # magnitudes 가 정렬돼 있으므로 가장 큰 주입 크기 = 리포트 4절과 동일
        log.info("[Isolation] %s (쌍 %d%s) k=%g: 탐지 %.1f%% / top-1 %.1f%% / 후보집합 %.1f%% / 모호 %.1f%%",
                 s, last["n_pairs"], ", 단독 격리 가능" if last["uniquely_isolable"] else "",
                 last["magnitude"], 100 * last["detection_rate"], 100 * last["top1_accuracy"],
                 100 * last["suspect_set_accuracy"], 100 * last["ambiguous_rate"])

    per_magnitude = []
    for k in magnitudes:
        rows = [r for r in per_sensor if r["magnitude"] == k]
        agg = {"magnitude": k, "n_sensors": len(rows)}
        for m in SENSOR_METRICS:
            agg[m] = float(np.mean([r[m] for r in rows])) if rows else None
        per_magnitude.append(agg)

    return {
        "n_rows": int(len(df)),
        "threshold": float(threshold),
        "magnitudes": magnitudes,
        "false_alarm_rate": float(false_alarm_rate),
        "baseline": clean_summary,
        "coverage": cov,
        "detectability_sd": iso.detectability(threshold),
        "unstandardized_residuals": list(unstandardized),
        "per_magnitude": per_magnitude,
        "per_sensor": per_sensor,
    }


# --------------------------------------------------------------------------
# 리포트
# --------------------------------------------------------------------------
def _pct(v):
    return "-" if v is None else f"{100 * v:.1f}%"


def per_magnitude_table(result):
    lines = ["| 주입 크기 (×sd) | 탐지율 | top-1 격리 | 후보집합 포함 | 모호 비율 |",
             "|---|---|---|---|---|"]
    for r in result["per_magnitude"]:
        lines.append(f"| {r['magnitude']:g} | {_pct(r['detection_rate'])} | {_pct(r['top1_accuracy'])} "
                     f"| {_pct(r['suspect_set_accuracy'])} | {_pct(r['ambiguous_rate'])} |")
    return "\n".join(lines)


def per_sensor_table(result, magnitude=None):
    if magnitude is None:
        magnitude = max(result["magnitudes"]) if result["magnitudes"] else None
    rows = [r for r in result["per_sensor"] if r["magnitude"] == magnitude]
    lines = ["| 센서 | 쌍 수 | 단독 격리 | 탐지율 | top-1 격리 | 후보집합 포함 | 모호 비율 |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['sensor']} | {r['n_pairs']} | {'O' if r['uniquely_isolable'] else 'X'} "
                     f"| {_pct(r['detection_rate'])} | {_pct(r['top1_accuracy'])} "
                     f"| {_pct(r['suspect_set_accuracy'])} | {_pct(r['ambiguous_rate'])} |")
    return "\n".join(lines)


def render_markdown(result, meta=None):
    """isolation_report.md 본문. 현장 엔지니어가 그대로 읽을 수 있게 한국어로 쓴다."""
    meta = meta or {}
    cov = result["coverage"]
    k_max = max(result["magnitudes"]) if result["magnitudes"] else None
    out = ["# 센서 고장 격리 평가 (잔차 기반 FDI)", ""]
    for key, label in (("run_dir", "실행 디렉터리"), ("data_path", "평가 데이터")):
        if meta.get(key):
            out.append(f"- {label}: `{meta[key]}`")
    out += [f"- 평가 행 수: {result['n_rows']}",
            f"- 발화 임계: |z| >= {result['threshold']:g} (train 잔차 분포 기준 z-score)",
            f"- 주입 크기: {', '.join(f'{k:g}' for k in result['magnitudes'])} × 센서별 train 표준편차",
            ""]

    out += ["## 1. 격리 커버리지", "",
            f"- 잔차 쌍: {cov['n_pairs']}개 — " + ", ".join(f"{a}-{b}" for a, b in cov["pairs"]),
            f"- 커버 센서 (쌍에 포함): {len(cov['covered'])}/{cov['n_sensors']}",
            f"- 단독 격리 가능 (시그니처 고유): {len(cov['uniquely_isolable'])}개 — "
            + (", ".join(cov["uniquely_isolable"]) or "없음"),
            f"- 격리 불가 (어느 쌍에도 없음): {len(cov['uncovered'])}개"]
    if cov["uncovered"]:
        out.append(f"  - {', '.join(cov['uncovered'])}")
    if cov["ambiguous_groups"]:
        out.append("- 서로 구분 불가 묶음 (시그니처 동일): "
                   + "; ".join(" / ".join(g) for g in cov["ambiguous_groups"]))
    unstd = result.get("unstandardized_residuals", [])
    if unstd:
        out += ["", f"> **경고**: 잔차 {len(unstd)}개에 train 잔차 통계가 없어 z-score 가 표준화되지 않았다 "
                "(residual_stats 를 저장하지 않던 구버전 실행 디렉터리). 발화 임계 |z| 가 sigma 단위가 "
                "아니므로 아래 탐지율·격리 정확도는 실제 감도를 크게 과소평가한다. train.py 로 다시 "
                "학습한 실행 디렉터리로 평가해야 한다.",
                "> 해당 잔차: " + ", ".join(unstd)]
    det = result.get("detectability_sd", {})
    if det:
        out += ["", f"잔차별 최소 탐지 가능 편향 (|z| >= {result['threshold']:g} 기준, 센서 sd 단위):", "",
                "| 잔차 | 최소 편향 (×sd) |", "|---|---|"]
        out += [f"| {n} | {v:.3f} |" for n, v in det.items()]
    out.append("")

    out += ["## 2. 오경보율 (주입 전 기준선)", "",
            f"- 주입 전 데이터에서 탐지된 행: {result['baseline']['n_detected']}/{result['n_rows']} "
            f"= **{_pct(result['false_alarm_rate'])}**",
            "- 평가 데이터에 실제 고장 행이 섞여 있다면 이 값은 순수 오경보율이 아니라 "
            "주입 전 탐지율(기준선)로 읽어야 한다.", ""]

    out += ["## 3. 주입 크기별 성능 (커버 센서 평균)", "", per_magnitude_table(result), "",
            "- 탐지율: 주입 행 중 잔차가 하나라도 발화한 비율",
            "- top-1 격리: 주입한 센서를 단독으로 지목한 비율 (모호한 행은 실패로 센다)",
            "- 후보집합 포함: 주입한 센서가 후보 목록에 든 비율",
            "- 모호 비율: 단독 지목에 실패한 비율 — 최상위 후보가 둘 이상(동점)이거나, 발화 잔차를 "
            "센서 하나로 설명할 수 없어(허브 센서의 부분 발화 / 다중 고장) 관련 센서 전체를 후보로 남긴 행", ""]

    if k_max is not None:
        out += [f"## 4. 센서별 성능 (주입 크기 {k_max:g} × sd)", "", per_sensor_table(result, k_max), "",
                "- 단독 격리 O: 포함된 쌍 집합이 고유해 발화 패턴만으로 지목 가능",
                "- 단독 격리 X: 쌍 하나를 상대 센서와만 공유 — 후보집합까지는 좁혀도 둘 중 하나는 "
                "추가 계측/점검으로 확정해야 한다", ""]
    return "\n".join(out)


def write_reports(result, out_dir, meta=None):
    """isolation_report.json / .md 를 out_dir 에 쓰고 두 경로를 돌려준다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    if meta:
        payload["meta"] = dict(meta)
    json_path = out_dir / REPORT_JSON
    md_path = out_dir / REPORT_MD
    artifacts.write_json(json_path, payload)
    md_path.write_text(render_markdown(result, meta), encoding="utf-8")
    return json_path, md_path


# --------------------------------------------------------------------------
# 실행 디렉터리 기준 진입점
# --------------------------------------------------------------------------
def evaluate_run(run_dir, data_dir=None, test_csv=None, magnitudes=(1, 2, 3, 5), threshold=3.0,
                 max_rows=None, out_dir=None, seed=0, live=True):
    """실행 디렉터리의 파이프라인으로 평가를 돌리고 리포트를 쓴다. 결과 dict 를 돌려준다."""
    logging.basicConfig(level=logging.INFO if live else logging.WARNING,
                        format="%(message)s", force=True)
    run_dir = Path(run_dir)
    cfg = artifacts.read_json(run_dir / artifacts.CONFIG_FILE)
    data_dir = Path(data_dir or cfg["DATA_DIR"])
    test_path = Path(test_csv) if test_csv else data_dir / "test.csv"

    pipe = FeaturePipeline.load(run_dir / artifacts.PIPELINE_FILE)
    df = pd.read_csv(test_path)
    log.info("[Isolation] %s -> %s", test_path.name, df.shape)

    result = run_evaluation(pipe, df, magnitudes=magnitudes, threshold=threshold,
                            max_rows=max_rows, seed=seed)
    meta = {"run_dir": str(run_dir), "data_path": str(test_path),
            "version": cfg.get("VERSION", "run")}
    json_path, md_path = write_reports(result, out_dir or run_dir, meta)
    result["report_json"] = str(json_path)
    result["report_md"] = str(md_path)
    log.info("[Isolation] 리포트 저장: %s", md_path)
    return result


def parse_args():
    ap = argparse.ArgumentParser(description="저장된 파이프라인으로 잔차 기반 센서 격리 성능을 평가한다.")
    ap.add_argument("run_dir", help="train.py 가 남긴 outputs/<VERSION>_<timestamp> 경로")
    ap.add_argument("--data-dir", default=None, help="test.csv 위치 (기본: 실행 config.json 의 DATA_DIR)")
    ap.add_argument("--test-csv", default=None, help="평가에 쓸 csv 경로를 직접 지정")
    ap.add_argument("--magnitudes", type=float, nargs="+", default=[1, 2, 3, 5],
                    help="주입 편향 크기 (센서별 train sd 배수), 예: --magnitudes 1 2 3 5")
    ap.add_argument("--threshold", type=float, default=3.0, help="잔차 z-score 발화 임계 (기본 3.0)")
    ap.add_argument("--max-rows", type=int, default=None, help="평가에 쓸 최대 행 수 (표본 추출)")
    ap.add_argument("--seed", type=int, default=0, help="행 표본 추출 시드")
    ap.add_argument("--out-dir", default=None, help="리포트 저장 위치 (기본: 실행 디렉터리)")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    res = evaluate_run(a.run_dir, data_dir=a.data_dir, test_csv=a.test_csv, magnitudes=a.magnitudes,
                       threshold=a.threshold, max_rows=a.max_rows, out_dir=a.out_dir,
                       seed=a.seed, live=not a.quiet)
    print()
    print(per_magnitude_table(res))
    print()
    print("SAVED report:", res["report_md"])
