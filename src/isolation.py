"""잔차 기반 센서 고장 격리 (FDI 의 'I', Fault Isolation).

`FeaturePipeline` 이 train 에서 찾아낸 준중복 센서쌍 (a, b) 마다 표준화 잔차
r_ab = (x_a - mu_a)/sd_a - (x_b - mu_b)/sd_b 가 있고, train 잔차 분포로 z-score 를 매기면
정상 상태에서는 |z| 가 작고 어느 한쪽 센서가 편향/드리프트하면 그 센서가 들어간 잔차만
커진다. 이 모듈은 그 "발화 패턴" 을 센서 단위로 되돌려 읽는다.

격리 원리 (signature matching)
------------------------------
센서 s 의 시그니처 = s 가 포함된 잔차(쌍)의 집합. s 가 고장이면 그 시그니처의 잔차가
모두 발화하고 나머지는 조용해야 한다. 그래서 센서마다 세 점수를 매긴다.

  explained(s)   = 발화한 쌍 중 s 를 포함한 수 (s 하나로 설명되는 발화 잔차 수)
  match_ratio(s) = explained(s) / (s 를 포함한 전체 쌍 수)
  evidence(s)    = s 를 포함한 발화 쌍들의 |z| 합

explained 내림차순 -> match_ratio 내림차순 -> evidence 내림차순으로 순위를 매긴다
(동점 허용오차 1e-9). 1위 센서가 `top_candidate` 다.

  예) a 가 (a,b), (a,c), (a,d) 세 쌍에 들어 있을 때
      - r_ab 하나만 발화: a, b 모두 1개를 설명하지만 b 는 1/1, a 는 1/3 -> b 를 지목한다.
        a 가 고장이라면 r_ac, r_ad 도 함께 커졌어야 하므로, 조용한 두 잔차가 a 의 알리바이다.
      - 세 잔차가 모두 발화: a 만 3개를 설명하므로 a 를 지목한다.

단독 지목(`suspect`)은 1위가 하나뿐이고 그 시그니처가 발화 패턴과 정확히 같을 때만 한다 —
즉 1위 센서가 발화 잔차를 모두 담고(explained == n_fired), 자기 잔차가 모두 발화했을 때
(match_ratio == 1). 그 외는 모두 `ambiguous=True` 이며 종류가 셋이다.

  - 동점: 쌍을 하나만 가진 두 센서 (예: X_01-X_02 만 존재) 는 시그니처가 같으므로 r_01_02 가
    발화해도 둘 중 누구인지 구분할 수 없다. `suspects` 에 동점 후보를 모두 남긴다.
  - 설명되지 않은 잔차: 1위 센서가 담지 못한 발화 잔차가 있다 (`unexplained_residuals`).
    독립된 두 쌍 (a,b), (c,d) 가 동시에 발화한 다중 고장이 전형이며, 발화 잔차에 등장한 센서
    전체를 `suspects` 로 남긴다.
  - 부분 발화: 1위 센서가 발화 잔차를 모두 담지만 자기 잔차 일부는 조용하다 (match_ratio < 1).
    예) 위의 a 에서 r_ab=5, r_ac=4 만 발화 (r_ad=2.9 로 임계 직전). a 의 드리프트가 시그니처를
    일부만 발화시킨 것일 수도, b 와 c 가 동시에 고장난 것일 수도 있어 a 를 단독 지목하지 않고
    후보 {a, b, c} 를 남긴다. (예전처럼 b 를 지목하면 a 의 드리프트를 b 로 잘못 짚는 셈이다.)

어느 쌍에도 들어가지 않은 센서는 잔차로는 격리할 수 없는 "격리 불가" 센서다
(`coverage()["uncovered"]`).

z-score 의 전제: 파이프라인이 train 잔차 통계(residual_stats)를 갖고 있어야 |z| 임계가
sigma 단위가 된다. 구버전 실행 디렉터리처럼 통계가 없으면 `unstandardized_residuals` 에
해당 잔차가 남고 `from_pipeline` 이 경고한다 (탐지 감도가 크게 떨어진 상태).

torch 에 의존하지 않으며, 저장된 `feature_pipeline.joblib` 만으로 동작한다.
"""
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .features import residual_name

log = logging.getLogger(__name__)

DIAGNOSE_COLUMNS = ["detected", "n_fired", "fired_residuals", "suspect", "suspects", "top_candidate",
                    "suspect_match_ratio", "suspect_evidence", "ambiguous",
                    "unexplained_residuals", "note"]
TIE_TOL = 1e-9
# ±inf 잔차(센서 값이 inf)는 이 값으로 잘라 순위 계산에 NaN 이 번지지 않게 한다.
Z_CAP = 1e6
NORMAL_NOTE = "정상: 발화한 잔차 없음"


@dataclass
class ResidualIsolator:
    """센서쌍 구조 + 잔차 z-score 계산기를 묶어 행 단위 센서 격리를 수행한다.

    pairs:          [(a, b), ...] train 에서 도출한 준중복 센서쌍
    residual_stats: {잔차이름: {"mean", "std"}} train 잔차 통계 (탐지 감도 보고용)
    sensor_cols:    전체 센서 컬럼 (커버리지 계산 기준)
    zscore_fn:      원본 프레임 -> 잔차 z-score 프레임 (열 = 잔차 이름)
    """

    pairs: list
    residual_stats: dict
    sensor_cols: list
    zscore_fn: Callable

    residual_names: list = field(init=False, repr=False)
    _sensor_signature: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.pairs = [tuple(p) for p in self.pairs]
        self.residual_names = [residual_name(a, b) for a, b in self.pairs]
        sig = {}
        for (a, b), name in zip(self.pairs, self.residual_names):
            sig.setdefault(a, set()).add(name)
            sig.setdefault(b, set()).add(name)
        self._sensor_signature = {s: frozenset(v) for s, v in sig.items()}

    @classmethod
    def from_pipeline(cls, pipe):
        """학습된 FeaturePipeline 의 쌍/잔차 통계/z-score 계산기를 그대로 연결한다.

        파이프라인에 train 잔차 통계가 없으면 (residual_stats 를 저장하지 않던 구버전 실행
        디렉터리) z-score 가 평균 0 / 표준편차 1 로 대체돼 |z| 임계가 sigma 단위가 아니게 된다.
        그 상태를 조용히 지나치지 않도록 경고를 남긴다.
        """
        pairs = list(pipe.fdi_stats.get("pairs", [])) if getattr(pipe, "use_fdi", True) else []
        iso = cls(pairs=pairs,
                  residual_stats=dict(getattr(pipe, "residual_stats", {}) or {}),
                  sensor_cols=list(pipe.feat_cols),
                  zscore_fn=pipe.residual_zscores)
        missing = iso.unstandardized_residuals
        if missing:
            log.warning("[Isolation] 잔차 %d/%d개에 train 통계가 없어 z-score 가 표준화되지 않습니다 — "
                        "구버전 실행 디렉터리(residual_stats 없음)로 보입니다. |z| 임계가 sigma 단위가 "
                        "아니라 탐지 감도가 크게 낮아지니 train.py 로 다시 학습하세요: %s",
                        len(missing), len(iso.residual_names), missing[:5])
        return iso

    # ---------------- 구조 보고 ----------------
    @property
    def covered(self):
        return sorted(self._sensor_signature)

    @property
    def unstandardized_residuals(self):
        """train 통계(residual_stats)가 없어 z-score 가 표준화되지 않는 잔차 이름 목록."""
        return [n for n in self.residual_names if n not in self.residual_stats]

    def coverage(self):
        """잔차 구조만으로 결정되는 격리 가능성 요약.

        covered:           하나 이상의 쌍에 들어간 센서 (정렬)
        uncovered:         어느 쌍에도 없는 센서 -> 잔차로는 격리 불가
        pairs_per_sensor:  센서별 쌍 수 (uncovered 는 0)
        uniquely_isolable: 시그니처(포함된 쌍 집합)가 다른 모든 센서와 다른 센서.
                           쌍 하나를 공유하고 다른 쌍이 없는 두 센서는 시그니처가 같아
                           포함되지 않는다.
        ambiguous_groups:  시그니처가 완전히 같은 센서 묶음 (서로 구분 불가)
        """
        covered = self.covered
        all_sensors = list(self.sensor_cols) + [s for s in covered if s not in self.sensor_cols]
        pairs_per_sensor = {s: len(self._sensor_signature.get(s, ())) for s in all_sensors}
        uncovered = sorted(s for s in self.sensor_cols if s not in self._sensor_signature)

        by_sig = {}
        for s in covered:
            by_sig.setdefault(self._sensor_signature[s], []).append(s)
        uniquely = sorted(g[0] for g in by_sig.values() if len(g) == 1)
        groups = sorted(sorted(g) for g in by_sig.values() if len(g) > 1)
        return {
            "n_pairs": len(self.pairs),
            "pairs": [list(p) for p in self.pairs],
            "n_sensors": len(self.sensor_cols),
            "covered": covered,
            "uncovered": uncovered,
            "pairs_per_sensor": pairs_per_sensor,
            "uniquely_isolable": uniquely,
            "ambiguous_groups": groups,
        }

    def detectability(self, threshold=3.0):
        """잔차별 최소 탐지 가능 편향 (train sd 단위).

        센서 a 에 k·sd_a 의 편향이 생기면 r_ab 는 정확히 k 만큼 움직이므로, z 가
        threshold 를 넘으려면 k >= threshold * std(r_ab) 여야 한다. 잔차 표준편차가
        작을수록(쌍의 상관이 강할수록) 작은 편향까지 잡아낸다. train 통계가 없는 잔차
        (`unstandardized_residuals`) 는 계산할 수 없어 빠진다.
        """
        out = {}
        for name in self.residual_names:
            st = self.residual_stats.get(name)
            if st is None:
                continue
            out[name] = float(threshold * st["std"])
        return out

    # ---------------- 행 단위 진단 ----------------
    def diagnose(self, df, threshold=3.0, zscores=None):
        """원본 프레임의 행마다 센서 고장을 탐지(D)하고 격리(I)한다.

        zscores 를 주면 (predict.py 처럼 리포트용으로 이미 계산한 z-score 프레임) 다시 계산하지
        않는다. df 와 행 위치가 같아야 한다.

        반환 DataFrame (index = df.index) 의 열:
          detected              bool   발화한 잔차가 하나라도 있는가 (n_fired > 0)
          n_fired               int    |z| >= threshold 인 잔차 수
          fired_residuals       list   발화한 잔차 이름
          suspect               str|None  단독으로 지목된 센서 (1위가 하나뿐이고 시그니처가 발화 패턴과
                                       정확히 일치). 모호하거나 발화가 없으면 None
          suspects              list   후보 센서 (이름순). 단독 지목이면 [suspect], 동점이면 동점 전체,
                                       설명되지 않은 잔차/부분 발화면 발화 잔차에 등장한 센서 전체.
                                       발화 없으면 []
          top_candidate         str|None  순위 1위 센서 (발화 잔차를 가장 많이 설명). 1위가 동점이거나
                                       발화가 없으면 None. 모호한 행에서도 "가장 유력한" 센서를 보여 준다
          suspect_match_ratio   float  1위 센서의 match_ratio (발화 없으면 0.0)
          suspect_evidence      float  1위 센서의 |z| 합 (발화 없으면 0.0)
          ambiguous             bool   단독 지목 실패 (동점 / 설명되지 않은 잔차 / 부분 발화).
                                       발화 없으면 False
          unexplained_residuals list   1위 센서의 시그니처 밖에서 발화한 잔차 (비어 있지 않으면 다중 고장
                                       가능; 1위가 동점이면 이름순 첫 번째 기준)
          note                  str    현장 엔지니어용 한 줄 판정 (explain())
        """
        index = df.index
        if not self.pairs:
            return self._empty(index)

        Z = self.zscore_fn(df) if zscores is None else zscores
        if len(Z) != len(index):
            raise ValueError(f"z-score 행 수 {len(Z)} 가 입력 행 수 {len(index)} 와 다릅니다.")
        present = [i for i, n in enumerate(self.residual_names) if n in Z.columns]
        if len(present) < len(self.residual_names):
            missing = [n for n in self.residual_names if n not in Z.columns]
            log.warning("[Isolation] 잔차 %d개가 입력에 없어 제외합니다: %s",
                        len(missing), missing[:5])
        if not present:
            return self._empty(index)

        names = [self.residual_names[i] for i in present]
        pairs = [self.pairs[i] for i in present]
        sensors = sorted({s for p in pairs for s in p})
        col_of = {s: j for j, s in enumerate(sensors)}

        # 발생(incidence) 행렬: 잔차 x 센서
        M = np.zeros((len(pairs), len(sensors)), dtype=float)
        for i, (a, b) in enumerate(pairs):
            M[i, col_of[a]] = 1.0
            M[i, col_of[b]] = 1.0
        n_pairs_of = M.sum(axis=0)

        A = np.abs(Z[names].to_numpy(dtype=float))
        # NaN 잔차는 판단 보류(발화 아님). ±inf 는 큰 유한값으로 잘라 evidence 행렬곱에 NaN 이 생기지 않게 한다.
        A = np.nan_to_num(A, nan=0.0, posinf=Z_CAP, neginf=Z_CAP)
        fired = A >= threshold                                   # (n, m)
        n_fired = fired.sum(axis=1)
        detected = n_fired > 0

        cnt = fired.astype(float) @ M                            # 센서별 설명되는 발화 잔차 수
        ratio = cnt / n_pairs_of[None, :]
        evidence = np.where(fired, A, 0.0) @ M
        cand = cnt > 0                                           # 발화 잔차에 등장한 센서

        # 순위: 설명되는 발화 잔차 수 -> match_ratio -> evidence (각 단계 동점은 다음 단계로)
        cnt_m = np.where(cand, cnt, -np.inf)
        best_cnt = cnt_m.max(axis=1)
        tied = cand & (cnt_m >= best_cnt[:, None] - TIE_TOL)
        ratio_m = np.where(tied, ratio, -np.inf)
        best_ratio = ratio_m.max(axis=1)
        tied &= ratio_m >= best_ratio[:, None] - TIE_TOL
        ev_m = np.where(tied, evidence, -np.inf)
        best_ev = ev_m.max(axis=1)
        top = tied & (ev_m >= best_ev[:, None] - TIE_TOL)
        n_top = top.sum(axis=1)

        # 단일 고장 가설의 일관성: 1위 시그니처 == 발화 패턴 (발화 잔차를 모두 담고, 자기 잔차가 모두 발화)
        rep = top.argmax(axis=1)                                 # 행별 대표 1위 (동점이면 이름순 첫 번째)
        covers_all = best_cnt >= n_fired - 0.5                   # cnt == n_fired (정수 비교)
        full_signature = best_ratio >= 1.0 - TIE_TOL
        inconsistent = detected & ~(covers_all & full_signature)
        unexplained = fired & (M[:, rep].T == 0)                 # 대표 1위 시그니처 밖의 발화 잔차
        ambiguous = detected & ((n_top != 1) | inconsistent)

        n = len(index)
        sensors_arr = np.array(sensors, dtype=object)
        names_arr = np.array(names, dtype=object)
        fired_lists = [list(names_arr[row]) for row in fired]
        unexplained_lists = [list(names_arr[row]) for row in unexplained]
        suspects, top_candidate = [], []
        for i in range(n):
            if not detected[i]:
                suspects.append([])
            elif inconsistent[i]:
                suspects.append(sorted(sensors_arr[cand[i]]))
            else:
                suspects.append(sorted(sensors_arr[top[i]]))
            top_candidate.append(str(sensors_arr[rep[i]]) if (detected[i] and n_top[i] == 1) else None)
        suspect = [s[0] if (d and not a) else None for s, d, a in zip(suspects, detected, ambiguous)]

        out = pd.DataFrame({
            "detected": detected.astype(bool),
            "n_fired": n_fired.astype(int),
            "fired_residuals": pd.Series(fired_lists, index=index, dtype=object),
            "suspect": pd.Series(suspect, index=index, dtype=object),   # pandas 3 의 str 추론 방지 (None 유지)
            "suspects": pd.Series(suspects, index=index, dtype=object),
            "top_candidate": pd.Series(top_candidate, index=index, dtype=object),
            "suspect_match_ratio": np.where(detected, best_ratio, 0.0).astype(float),
            "suspect_evidence": np.where(detected, best_ev, 0.0).astype(float),
            "ambiguous": ambiguous.astype(bool),
            "unexplained_residuals": pd.Series(unexplained_lists, index=index, dtype=object),
        }, index=index)
        out["note"] = pd.Series([self.explain(r) for r in out.to_dict("records")], index=index, dtype=object)
        return out[DIAGNOSE_COLUMNS]

    @staticmethod
    def _empty(index):
        n = len(index)
        return pd.DataFrame({
            "detected": np.zeros(n, dtype=bool),
            "n_fired": np.zeros(n, dtype=int),
            "fired_residuals": pd.Series([[] for _ in range(n)], index=index, dtype=object),
            "suspect": pd.Series([None] * n, index=index, dtype=object),
            "suspects": pd.Series([[] for _ in range(n)], index=index, dtype=object),
            "top_candidate": pd.Series([None] * n, index=index, dtype=object),
            "suspect_match_ratio": np.zeros(n, dtype=float),
            "suspect_evidence": np.zeros(n, dtype=float),
            "ambiguous": np.zeros(n, dtype=bool),
            "unexplained_residuals": pd.Series([[] for _ in range(n)], index=index, dtype=object),
            "note": pd.Series([NORMAL_NOTE] * n, index=index, dtype=object),
        }, index=index, columns=DIAGNOSE_COLUMNS)

    # ---------------- 요약 ----------------
    @staticmethod
    def summarize(diag):
        """diagnose() 결과를 한 덩어리로 요약한다 (리포트/로그용).

        n_unexplained: 발화는 했지만 1위 센서가 담지 못한 잔차가 있는 행 수
                       (다중 고장 의심; ambiguous 에 포함된다)
        """
        n = int(len(diag))
        n_det = int(diag["detected"].sum()) if n else 0
        n_amb = int(diag["ambiguous"].sum()) if n else 0
        unexpl = diag["unexplained_residuals"] if "unexplained_residuals" in diag.columns else []
        n_unexpl = int(sum(1 for u in unexpl if isinstance(u, (list, tuple)) and len(u) > 0))
        suspect_counts = Counter(s for s in diag["suspect"] if s is not None)
        amb_sets = Counter("|".join(s) for s, a in zip(diag["suspects"], diag["ambiguous"]) if a and s)
        return {
            "n_rows": n,
            "n_detected": n_det,
            "detection_rate": (n_det / n) if n else 0.0,
            "n_ambiguous": n_amb,
            "ambiguous_rate": (n_amb / n) if n else 0.0,
            "n_unexplained": n_unexpl,
            "suspect_counts": dict(sorted(suspect_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
            "ambiguous_set_counts": dict(sorted(amb_sets.items(), key=lambda kv: (-kv[1], kv[0]))),
        }

    @staticmethod
    def explain(row):
        """diagnose() 의 한 행을 현장 엔지니어가 읽을 한 줄 메시지로 바꾼다 (note 열의 원천)."""
        if not bool(row["detected"]):
            return NORMAL_NOTE
        fired_list = list(row["fired_residuals"])
        fired = ", ".join(fired_list)
        candidates = " / ".join(row["suspects"])
        if not row["ambiguous"]:
            return (f"{row['suspect']} 센서 이상 의심 — 해당 센서 잔차 "
                    f"{row['suspect_match_ratio']:.0%} 발화, |z| 합 {row['suspect_evidence']:.1f} "
                    f"(발화: {fired})")
        unexplained = _as_list(row.get("unexplained_residuals") if hasattr(row, "get") else None)
        lead = row.get("top_candidate") if hasattr(row, "get") else None
        lead = None if (lead is None or (isinstance(lead, float) and np.isnan(lead))) else str(lead)
        ratio = float(row.get("suspect_match_ratio", 1.0)) if hasattr(row, "get") else 1.0
        if unexplained:
            n_expl = len(fired_list) - len(unexplained)
            lead_txt = f"{lead} 가 {n_expl}/{len(fired_list)}개 설명, " if lead else ""
            return (f"격리 모호: 발화 잔차를 센서 하나로 설명 불가 — 다중 고장 가능. {lead_txt}"
                    f"설명되지 않은 잔차: {', '.join(unexplained)}; 관련 센서 {candidates} (발화: {fired})")
        if lead and ratio < 1.0 - TIE_TOL:
            # 1위가 하나뿐인데 자기 시그니처가 일부만 발화 (허브 센서의 부분 발화 / 파트너 동시 고장)
            return (f"격리 모호: {lead} 가 발화 잔차를 모두 설명하지만 자기 잔차는 {ratio:.0%} 만 발화 — "
                    f"{lead} 부분 발화 또는 상대 센서 동시 고장 가능. 관련 센서 {candidates} (발화: {fired})")
        if row["suspects"]:
            # 동점: 발화 잔차를 공유하는 센서들 — 각자의 다른 잔차가 조용하면 그 비율도 적는다
            quiet = f" (각자 시그니처의 {ratio:.0%} 만 발화)" if ratio < 1.0 - TIE_TOL else ""
            return f"격리 모호: {candidates} 중 하나 — 공유 잔차만 발화해 구분 불가{quiet} (발화: {fired})"
        return f"탐지됐으나 후보 없음 (발화: {fired})"


def _as_list(v):
    """리스트/튜플이면 list, 그 외(None/NaN/문자열)는 빈 리스트."""
    return list(v) if isinstance(v, (list, tuple)) else []
