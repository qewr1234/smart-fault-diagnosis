"""데이터에서 파이프라인 상수를 직접 도출한다.

원래 이 값들은 코드에 박혀 있었다 — 중복 센서쌍 12개, 완전 중복 컬럼 2개,
분포 shift 컬럼 `X_11`, 분산 시그니처 컬럼 `X_48`, 하드클러스터 {0,3,9,15,19}.
모두 특정 데이터셋을 들여다본 결과라 다른 데이터에서는 조용히 틀린다.
여기서는 같은 결론을 train 데이터(그리고 라벨을 쓰지 않는 참조 분포)에서
재현 가능한 규칙으로 뽑아낸다.

라벨을 쓰는 함수는 `discover_signature_columns`(train 라벨)와
`discover_hard_cluster`(검증 예측)뿐이며, 나머지는 피처 분포만 본다.
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


# --------------------------------------------------------------------------
# 센서 중복 (analytical redundancy)
# --------------------------------------------------------------------------
def _correlation(df, cols):
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr().abs()
    return corr.fillna(0.0)


def discover_sensor_redundancy(df, feat_cols, corr_threshold=0.95,
                               exact_threshold=0.9999, max_pairs=12):
    """중복 센서 구조를 상관행렬에서 찾는다.

    반환 dict:
      - `exact_groups`: |corr| >= exact_threshold 로 묶인 컬럼 그룹
      - `drop_cols`: 각 그룹에서 대표 하나만 남기고 제거할 컬럼
      - `pairs`: 잔차 피처를 만들 준중복 쌍 (|corr| >= corr_threshold, 강한 순)
      - `correlations`: 각 쌍의 |corr|

    완전 중복 그룹 내부의 쌍은 잔차가 항상 0이라 `pairs`에서 제외한다.
    """
    corr = _correlation(df, feat_cols)
    cols = list(corr.columns)

    # --- 완전 중복: union-find 로 그룹화 ---
    parent = {c: c for c in cols}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if corr.at[a, b] >= exact_threshold:
                union(a, b)

    groups = {}
    for c in cols:
        groups.setdefault(find(c), []).append(c)
    exact_groups = [sorted(g) for g in groups.values() if len(g) > 1]
    exact_groups.sort()
    drop_cols = sorted(c for g in exact_groups for c in g[1:])
    same_group = {c: find(c) for c in cols}

    # --- 준중복 쌍: 상관 강한 순, 제거될 컬럼과 완전중복 내부 쌍은 제외 ---
    candidates = []
    keep = [c for c in cols if c not in drop_cols]
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            r = float(corr.at[a, b])
            if r >= corr_threshold and same_group[a] != same_group[b]:
                candidates.append((r, a, b))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    if max_pairs and max_pairs > 0:
        candidates = candidates[:max_pairs]

    return {
        "exact_groups": exact_groups,
        "drop_cols": drop_cols,
        "pairs": [(a, b) for _, a, b in candidates],
        "correlations": {f"{a}|{b}": r for r, a, b in candidates},
    }


# --------------------------------------------------------------------------
# 분포 shift 컬럼 (train 에만 있는 극단 꼬리)
# --------------------------------------------------------------------------
def discover_shift_columns(df_train, df_reference, feat_cols, q=(0.001, 0.999),
                           min_excess_ratio=1.0, max_outside_frac=0.02, max_cols=3):
    """train 에만 존재하는 극단값 꼬리를 가진 컬럼을 찾는다.

    참조 분포의 분위수 구간 [lo, hi] 와 그 폭 w 를 기준으로, 구간에서 w 의
    `min_excess_ratio` 배 이상 더 떨어진 값을 "멀리 벗어난 값"으로 본다.
    구간을 살짝 넘는 값은 표본 크기 차이에서 오는 정상적인 변동이므로 세지 않는다.

    두 조건을 모두 만족해야 shift 컬럼으로 본다.
      1. 가장 먼 값이 참조 폭의 `min_excess_ratio` 배 이상 벗어나 있다.
      2. 그렇게 멀리 벗어난 행의 비율이 `max_outside_frac` 이하다.
         (대부분의 행이 멀리 있다면 그건 드문 이상값이 아니라 전면적인 분포 차이라,
          클리핑이 아니라 다른 대응이 필요하다.)

    참조(test) 분포의 분위수만 사용하며 라벨은 보지 않는다.
    """
    found, all_ratios = [], {}
    for c in feat_cols:
        if c not in df_train.columns or c not in df_reference.columns:
            continue
        lo = float(df_reference[c].quantile(q[0]))
        hi = float(df_reference[c].quantile(q[1]))
        width = hi - lo
        if not np.isfinite(width) or width <= 0:
            continue
        col = df_train[c].astype(float)
        excess = max(max(0.0, lo - float(col.min())), max(0.0, float(col.max()) - hi))
        ratio = excess / width
        all_ratios[c] = ratio
        if ratio < min_excess_ratio:
            continue
        margin = min_excess_ratio * width
        far = float(((col < lo - margin) | (col > hi + margin)).mean())
        if 0.0 < far <= max_outside_frac:
            found.append((ratio, far, c))

    found.sort(key=lambda t: (-t[0], t[2]))
    if max_cols and max_cols > 0:
        found = found[:max_cols]
    selected = [c for _, _, c in found]

    # 여러 컬럼을 동시에 보므로 우연히 임계를 넘는 컬럼이 나올 수 있다.
    # 선택되지 않은 컬럼 중 최고 비율을 함께 남겨, 실제로 도드라지는지 판단하게 한다.
    others = [r for c, r in all_ratios.items() if c not in selected]
    return {
        "columns": selected,
        "excess_ratio": {c: r for r, _, c in found},
        "far_outside_frac": {c: f for _, f, c in found},
        "n_examined": len(all_ratios),
        "runner_up_ratio": float(max(others)) if others else 0.0,
    }


# --------------------------------------------------------------------------
# 분산 시그니처 컬럼 (클래스마다 노이즈 수준이 다른 센서)
# --------------------------------------------------------------------------
def discover_signature_columns(df, y, feat_cols, top_k=1, min_class_count=2):
    """클래스별 산포(std)가 클래스에 따라 크게 달라지는 컬럼을 찾는다.

    고장 유형에 따라 '값' 대신 '흔들림의 크기'가 바뀌는 센서를 잡아낸다.
    점수 = 클래스별 std 의 표준편차 / 평균 (변동계수).
    """
    y = np.asarray(y)
    classes = [c for c in np.unique(y) if (y == c).sum() >= min_class_count]
    if len(classes) < 2 or top_k <= 0:
        return {"columns": [], "scores": {}}

    scores = {}
    for c in feat_cols:
        if c not in df.columns:
            continue
        per_class = np.array([float(df[c].values[y == k].std(ddof=1)) for k in classes])
        per_class = per_class[np.isfinite(per_class)]
        mean = per_class.mean() if per_class.size else 0.0
        if per_class.size < 2 or mean <= 1e-12:
            continue
        scores[c] = float(per_class.std(ddof=1) / mean)

    ranked = sorted(scores, key=lambda c: (-scores[c], c))[:top_k]
    return {"columns": ranked, "scores": {c: scores[c] for c in ranked}}


# --------------------------------------------------------------------------
# 하드클러스터 (서로 혼동되는 클래스 집합)
# --------------------------------------------------------------------------
def discover_hard_cluster(y_true, probs, f1_quantile=0.35, min_confusion=0.05,
                          min_size=2, max_size=8):
    """상호 혼동되는 저성능 클래스 집합을 혼동행렬에서 찾는다.

    1. 클래스별 F1 하위 `f1_quantile` 분위를 후보로 둔다.
    2. 두 클래스가 서로 오분류되는 비율이 `min_confusion` 이상이면 간선을 잇는다.
       비율 = (C[i,j] + C[j,i]) / (n_i + n_j)
    3. 후보들의 연결요소 중 가장 큰 것을 클러스터로 삼는다.
       `max_size`를 넘으면 클러스터 내부 혼동이 큰 순으로 자른다.

    조건을 만족하는 집합이 없으면 빈 리스트를 돌려준다(전문가 모델 비활성화).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs.argmax(1)
    K = probs.shape[1]
    labels = np.arange(K)

    class_f1 = f1_score(y_true, pred, average=None, labels=labels, zero_division=0)
    threshold = float(np.quantile(class_f1, f1_quantile))
    candidates = [int(k) for k in labels if class_f1[k] <= threshold]
    info = {"cluster": [], "class_f1": [float(v) for v in class_f1],
            "f1_threshold": threshold, "candidates": candidates}
    if len(candidates) < min_size:
        return info

    C = confusion_matrix(y_true, pred, labels=labels)
    support = C.sum(axis=1)

    rate = {}
    adj = {k: set() for k in candidates}
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            denom = support[a] + support[b]
            if denom == 0:
                continue
            r = float((C[a, b] + C[b, a]) / denom)
            rate[(a, b)] = r
            if r >= min_confusion:
                adj[a].add(b)
                adj[b].add(a)

    seen, components = set(), []
    for k in candidates:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(comp))

    components = [c for c in components if len(c) >= min_size]
    if not components:
        return info

    best = max(components, key=lambda c: (len(c), -sum(class_f1[k] for k in c)))
    if max_size and len(best) > max_size:
        strength = {k: sum(rate.get(tuple(sorted((k, o))), 0.0) for o in best if o != k)
                    for k in best}
        best = sorted(sorted(best, key=lambda k: -strength[k])[:max_size])

    info["cluster"] = [int(k) for k in best]
    info["confusion_rates"] = {f"{a}|{b}": r for (a, b), r in rate.items()
                               if a in best and b in best}
    return info
