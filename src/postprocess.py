"""블렌드 가중치 / 전문가 가중치 / 로짓 바이어스 / 균형 배정을 하나의 후처리 객체로 묶는다.

두 가지 역할이 있다.

1. `fit_postprocess` — OOF에서 모든 후처리 파라미터를 고르고, 그 결과를 저장·재사용
   가능한 `PostProcessFit`으로 돌려준다. 추론은 이 객체만 쓴다.
2. `nested_estimate` — 같은 선택 절차를 OOF 행에 대한 K-fold로 반복해, "선택에 쓰지 않은
   행"에서만 점수를 매긴 **정직한 추정치**를 준다.

`fit_postprocess`가 보고하는 점수는 자기가 고른 파라미터로 자기를 채점한 값이라 낙관적이다.
후처리의 실제 기여를 읽을 때는 항상 `nested_estimate` 쪽을 봐야 한다.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .blend import combine, greedy_weight_search
from .calibration import apply_logit_bias, logit_bias_tune, sinkhorn_balanced_assign
from .expert import rerank_hard_cluster, select_expert_weight
from .utils import ensure_prob_finite


@dataclass(frozen=True)
class PostProcessSpec:
    """후처리 탐색 공간 (config에서 온 설정)."""
    blend_mode: str = "logit"
    blend_passes: int = 2
    max_models: int = 3
    use_expert: bool = True
    expert_w_grid: tuple = (0.0, 0.25, 0.5)
    hard_cluster: tuple = (0, 3, 9, 15, 19)
    smooth_eps: float = 0.02
    use_bias_tune: bool = True
    bias_lim: float = 0.30
    bias_step: float = 0.05
    use_balanced_assign: bool = True

    def to_dict(self):
        d = asdict(self)
        d["expert_w_grid"] = list(self.expert_w_grid)
        d["hard_cluster"] = list(self.hard_cluster)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["expert_w_grid"] = tuple(d.get("expert_w_grid", (0.0, 0.25, 0.5)))
        d["hard_cluster"] = tuple(d.get("hard_cluster", (0, 3, 9, 15, 19)))
        return PostProcessSpec(**d)


@dataclass
class SeedBlend:
    """한 시드의 블렌드 구성."""
    keep: list
    weights: list
    expert_w: float = 0.0


@dataclass
class PostProcessFit:
    """학습에서 고른 모든 후처리 파라미터. 추론은 이 객체만 사용한다."""
    spec: PostProcessSpec
    seeds: dict = field(default_factory=dict)      # seed(str) -> SeedBlend
    bias: list = None                              # 전역 로짓 바이어스 (미적용이면 None)
    use_balanced: bool = False

    # ---------- 적용 ----------
    def blend_seed(self, seed, probs_by_name, expert_probs=None):
        sb = self.seeds[str(seed)]
        plist = [ensure_prob_finite(probs_by_name[n]) for n in sb.keep]
        P = plist[0] if len(plist) == 1 else combine(plist, sb.weights, self.spec.blend_mode)
        if expert_probs is not None and sb.expert_w > 0:
            P = rerank_hard_cluster(P, expert_probs, list(self.spec.hard_cluster), w=sb.expert_w)
        eps = max(0.0, float(self.spec.smooth_eps))
        if eps > 0:
            P = (1.0 - eps) * P + eps / float(P.shape[1])
        return P

    def average_seeds(self, per_seed_probs):
        return ensure_prob_finite(np.mean(np.stack(per_seed_probs, axis=0), axis=0))

    def apply_bias(self, probs):
        if self.bias is None:
            return ensure_prob_finite(probs)
        return apply_logit_bias(probs, np.asarray(self.bias, dtype=float))

    def predict(self, probs):
        P = self.apply_bias(probs)
        return sinkhorn_balanced_assign(P) if self.use_balanced else P.argmax(1)

    def transform_all(self, oof_by_seed, expert_by_seed=None):
        """시드별 모델 확률 -> 후처리된 최종 확률 (전역 바이어스 적용 전 평균)."""
        per_seed = []
        for seed in self.seeds:
            exp = None if expert_by_seed is None else expert_by_seed.get(seed)
            per_seed.append(self.blend_seed(seed, oof_by_seed[seed], exp))
        return self.average_seeds(per_seed)

    # ---------- 직렬화 ----------
    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "seeds": {k: asdict(v) for k, v in self.seeds.items()},
            "bias": None if self.bias is None else list(map(float, self.bias)),
            "use_balanced": bool(self.use_balanced),
        }

    @staticmethod
    def from_dict(d):
        return PostProcessFit(
            spec=PostProcessSpec.from_dict(d["spec"]),
            seeds={k: SeedBlend(**v) for k, v in d["seeds"].items()},
            bias=d.get("bias"),
            use_balanced=bool(d.get("use_balanced", False)),
        )


# --------------------------------------------------------------------------
# 선택 (fit)
# --------------------------------------------------------------------------
def _slice(probs_by_name, rows):
    return {n: p[rows] for n, p in probs_by_name.items()}


def fit_seed_blend(probs_by_name, expert_probs, y, spec, live=False):
    """한 시드에 대해 사용할 모델·블렌드 가중치·전문가 가중치를 고른다."""
    scores = {}
    for name, p in probs_by_name.items():
        if not np.isfinite(p).all():
            continue
        scores[name] = f1_score(y, p.argmax(1), average="macro")
    if not scores:
        raise ValueError("유효한 모델 OOF 확률이 없습니다.")
    keep = sorted(scores, key=scores.get, reverse=True)[:spec.max_models]

    plist = [ensure_prob_finite(probs_by_name[n]) for n in keep]
    if len(plist) == 1:
        weights = [1.0]
    else:
        w = greedy_weight_search(plist, y, passes=spec.blend_passes, live=live, mode=spec.blend_mode)
        weights = [float(v) for v in w]

    expert_w = 0.0
    if spec.use_expert and expert_probs is not None:
        P = plist[0] if len(plist) == 1 else combine(plist, weights, spec.blend_mode)
        expert_w = float(select_expert_weight(P, expert_probs, y, list(spec.hard_cluster),
                                              w_grid=spec.expert_w_grid, live=live))
    return SeedBlend(keep=keep, weights=weights, expert_w=expert_w)


def fit_postprocess(oof_by_seed, expert_by_seed, y, spec, live=True):
    """모든 후처리 파라미터를 OOF에서 고른다. (fit, 후처리된 OOF 확률) 반환."""
    fit = PostProcessFit(spec=spec, seeds={})
    per_seed = []
    for seed, probs_by_name in oof_by_seed.items():
        exp = None if expert_by_seed is None else expert_by_seed.get(seed)
        sb = fit_seed_blend(probs_by_name, exp, y, spec, live=live)
        fit.seeds[str(seed)] = sb
        if live:
            print(f"[Blend seed={seed}] keep={sb.keep} "
                  f"weights={[round(w, 3) for w in sb.weights]} expert_w={sb.expert_w:.2f}")
        per_seed.append(fit.blend_seed(seed, probs_by_name, exp))

    blended = fit.average_seeds(per_seed)

    if spec.use_bias_tune:
        base = f1_score(y, blended.argmax(1), average="macro")
        b = logit_bias_tune(blended, y, step=spec.bias_step, lim=spec.bias_lim)
        tuned = f1_score(y, apply_logit_bias(blended, b).argmax(1), average="macro")
        if tuned > base + 1e-5:
            fit.bias = [float(v) for v in b]
        if live:
            print(f"[BiasTune] base={base:.5f} vs tuned={tuned:.5f} -> use={fit.bias is not None}")

    if spec.use_balanced_assign:
        P = fit.apply_bias(blended)
        f1_argmax = f1_score(y, P.argmax(1), average="macro")
        f1_bal = f1_score(y, sinkhorn_balanced_assign(P), average="macro")
        fit.use_balanced = bool(f1_bal > f1_argmax + 1e-5)
        if live:
            print(f"[BalancedAssign] argmax={f1_argmax:.5f} vs balanced={f1_bal:.5f} "
                  f"-> use={fit.use_balanced}")

    return fit, blended


# --------------------------------------------------------------------------
# 정직한 추정 (nested)
# --------------------------------------------------------------------------
def nested_estimate(oof_by_seed, expert_by_seed, y, spec, n_splits=5, seed=0, live=True):
    """후처리 파라미터 선택을 K-fold로 감싸, 선택에 쓰이지 않은 행에서만 채점한다.

    반환 dict:
      - `selection_f1`: 전체 OOF에서 고르고 전체 OOF에서 잰 점수 (낙관적, 참고용)
      - `nested_f1`: 선택에 쓰지 않은 행에서만 잰 점수 (테스트 성능의 정직한 추정)
      - `raw_blend_f1`: 후처리 전 단순 시드 평균 점수 (후처리 기여 비교 기준)
    """
    y = np.asarray(y)
    n = len(y)
    pred = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fi, (sel_idx, eval_idx) in enumerate(skf.split(np.zeros(n), y), 1):
        sel_oof = {s: _slice(p, sel_idx) for s, p in oof_by_seed.items()}
        sel_exp = None if expert_by_seed is None else {s: p[sel_idx] for s, p in expert_by_seed.items()}
        fold_fit, _ = fit_postprocess(sel_oof, sel_exp, y[sel_idx], spec, live=False)

        eval_oof = {s: _slice(p, eval_idx) for s, p in oof_by_seed.items()}
        eval_exp = None if expert_by_seed is None else {s: p[eval_idx] for s, p in expert_by_seed.items()}
        P = fold_fit.transform_all(eval_oof, eval_exp)
        pred[eval_idx] = fold_fit.predict(P)
        if live:
            print(f"[Nested {fi}/{n_splits}] eval_f1="
                  f"{f1_score(y[eval_idx], pred[eval_idx], average='macro'):.5f}")

    raw = ensure_prob_finite(np.mean(np.stack(
        [np.mean(np.stack(list(p.values()), axis=0), axis=0) for p in oof_by_seed.values()],
        axis=0), axis=0))
    return {
        "nested_f1": float(f1_score(y, pred, average="macro")),
        "raw_blend_f1": float(f1_score(y, raw.argmax(1), average="macro")),
        "n_splits": int(n_splits),
    }
