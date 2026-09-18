# ============================================
# Smart Fault Diagnosis — 학습 전용 진입점
#
# 학습 결과(전처리 통계, fold별 모델, 후처리 파라미터, 지표)를 실행 디렉터리에 저장한다.
# 제출 파일 생성은 predict.py 가 이 디렉터리만 읽어서 수행한다.
#
# OOF 편향 방지:
#   - 각 fold의 train 부분에서 다시 내부 홀드아웃을 떼어 early stopping과 온도
#     스케일링에 사용한다. 바깥 fold는 오직 채점에만 쓰이므로 OOF가 낙관 편향되지 않는다.
#   - 후처리(블렌드/전문가/바이어스/균형배정) 파라미터는 OOF에서 고르되, 그 기여도는
#     선택에 쓰지 않은 행에서만 매기는 중첩 추정치(metrics.json의 nested_f1)로 보고한다.
# ============================================
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from config import CONFIG
from src import artifacts
from src.calibration import temperature_scale_grid
from src.discovery import discover_hard_cluster
from src.models import fit_architecture
from src.pipeline import FeaturePipeline, augment_with_anomaly_score
from src.postprocess import PostProcessSpec, fit_postprocess, nested_estimate
from src.utils import compute_metrics, infer_feature_cols, set_seed

log = logging.getLogger("train")

MODEL_FLAGS = [("ft", "USE_FT"), ("mixer", "USE_MIXER"), ("glu", "USE_GLU")]


def _setup_logging(live):
    logging.basicConfig(level=logging.INFO if live else logging.WARNING,
                        format="%(message)s", force=True)


def _inner_split(y_tr, frac, seed):
    """fold의 train 부분에서 early stopping / 온도용 내부 홀드아웃을 뗀다.

    클래스당 표본이 부족해 층화 분할이 불가능하면 None을 돌려준다(호출부가 폴백).
    """
    counts = np.bincount(y_tr, minlength=int(y_tr.max()) + 1)
    n_classes = int((counts > 0).sum())
    n_inner = int(round(len(y_tr) * frac))
    if counts[counts > 0].min() < 2 or n_inner < n_classes:
        return None
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_idx, inner_idx = next(sss.split(np.zeros(len(y_tr)), y_tr))
    return fit_idx, inner_idx


def _make_run_dir(cfg):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg["OUT_DIR"]) / f"{cfg['VERSION']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(cfg):
    """학습 후 실행 디렉터리 경로를 돌려준다."""
    live = cfg["LIVE"]
    _setup_logging(live)
    set_seed(cfg["CV_SEED"])

    data_dir = Path(cfg["DATA_DIR"])
    train = pd.read_csv(data_dir / "train.csv")
    if "target" not in train.columns:
        raise ValueError("train.csv 에 target 컬럼이 없습니다.")
    feat = infer_feature_cols(train)

    le = LabelEncoder()
    y_all = le.fit_transform(train["target"].values)
    K = len(le.classes_)

    # 분포 shift 컬럼 탐지에 쓸 참조 분포 (피처 분위수만 사용, 라벨은 보지 않음).
    reference = None
    if cfg["CLIP_SHIFT_COLS"]:
        test_path = data_dir / "test.csv"
        if test_path.exists():
            reference = pd.read_csv(test_path)[feat]
        else:
            log.info("[Shift] test.csv 없음 -> shift 클리핑 생략")

    pipe = FeaturePipeline(
        feat_cols=feat, use_fdi=cfg["USE_FDI_FEATURES"],
        clip_shift=cfg["CLIP_SHIFT_COLS"], add_pairdiff=cfg["ADD_PAIRDIFF"],
        topk_var=cfg["TOPK_VAR"], auto_discover=cfg["AUTO_DISCOVER"],
        corr_threshold=cfg["CORR_THRESHOLD"], exact_dup_threshold=cfg["EXACT_DUP_THRESHOLD"],
        max_redundant_pairs=cfg["MAX_REDUNDANT_PAIRS"],
        shift_excess_ratio=cfg["SHIFT_EXCESS_RATIO"],
        shift_outside_frac=cfg["SHIFT_OUTSIDE_FRAC"], max_shift_cols=cfg["MAX_SHIFT_COLS"],
        n_signature_cols=cfg["N_SIGNATURE_COLS"])
    X_all = pipe.fit(train, reference, y=y_all)
    report = pipe.discovery_report()
    log.info("[Pipeline] features=%d (원본 %d)", X_all.shape[1], len(feat))
    if cfg["AUTO_DISCOVER"]:
        log.info("[Discover] 완전중복 그룹=%s -> 제거 %s",
                 report["exact_duplicate_groups"], report["dropped_duplicate_cols"])
        log.info("[Discover] 준중복 쌍 %d개: %s",
                 len(report["redundant_pairs"]), report["redundant_pairs"][:4])
        log.info("[Discover] shift 컬럼=%s | 분산 시그니처=%s",
                 report["shift_columns"], report["signature_columns"])

    class_weights = None
    if cfg["USE_CLASS_WEIGHT"]:
        cnt = np.bincount(y_all, minlength=K).astype(float)
        w = cnt.sum() / np.clip(cnt, 1e-12, None)
        class_weights = w / w.mean()

    use_expert = cfg["USE_EXPERT"]
    if use_expert:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            log.warning("[Expert] lightgbm 미설치 -> 전문가 비활성화")
            use_expert = False

    # 하드클러스터: "auto" 면 검증 예측의 혼동 구조에서 도출하고,
    # 리스트를 주면 그 값을 그대로 쓴다.
    configured_cluster = cfg["HARD_CLUSTER"]
    auto_cluster = isinstance(configured_cluster, str) and configured_cluster.lower() == "auto"
    hard = None if auto_cluster else sorted(configured_cluster)
    cluster_info = {"cluster": hard, "source": "config" if not auto_cluster else None}

    run_dir = _make_run_dir(cfg)
    pipe.save(run_dir / artifacts.PIPELINE_FILE)
    artifacts.write_json(run_dir / artifacts.CONFIG_FILE, cfg)
    artifacts.write_json(run_dir / artifacts.LABELS_FILE, list(le.classes_))

    model_names = [n for n, flag in MODEL_FLAGS if cfg[flag]]
    if not model_names:
        raise ValueError("활성화된 모델이 없습니다 (USE_FT / USE_MIXER / USE_GLU).")

    oof_by_seed, expert_by_seed, per_model_scores = {}, {}, {}
    nested_calibration_used = []

    for si, seed in enumerate(cfg["SEEDS"], 1):
        log.info("\n========== SEED %d/%d (seed=%d) ==========", si, len(cfg["SEEDS"]), seed)
        set_seed(seed)
        skf = StratifiedKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=seed)
        folds = list(skf.split(X_all, y_all))

        oof = {n: np.zeros((X_all.shape[0], K)) for n in model_names}
        fold_iso = {}
        inner_probs, inner_targets = [], []

        # ---------- pass 1: 딥러닝 모델 ----------
        for fi, (tr_idx, va_idx) in enumerate(folds, 1):
            log.info("\n===== FOLD %d/%d (seed=%d) =====", fi, cfg["CV_FOLDS"], seed)
            Xtr, Xva = X_all[tr_idx], X_all[va_idx]
            ytr, yva = y_all[tr_idx], y_all[va_idx]
            artifacts.fold_dir(run_dir, seed, fi, create=True)

            iso = None
            if cfg["USE_ISOFOREST"]:
                iso = IsolationForest(n_estimators=200, contamination="auto",
                                      random_state=seed + fi, n_jobs=1).fit(Xtr)
                import joblib
                joblib.dump(iso, artifacts.isoforest_path(run_dir, seed, fi))
            fold_iso[fi] = iso
            Xtr_aug = augment_with_anomaly_score(Xtr, iso)
            Xva_aug = augment_with_anomaly_score(Xva, iso)

            # ---- 내부 홀드아웃: early stopping + 온도 스케일링 전용 ----
            split = _inner_split(ytr, cfg["INNER_VAL_FRAC"], seed + fi) \
                if cfg["NESTED_CALIBRATION"] else None
            if split is None:
                if cfg["NESTED_CALIBRATION"]:
                    log.warning("[Nested] fold %d: 층화 내부 분할 불가 -> 바깥 fold로 폴백 "
                                "(이 fold의 OOF는 낙관 편향됨)", fi)
                X_fit, y_fit = Xtr_aug, ytr
                X_inner, y_inner = Xva_aug, yva
                nested_calibration_used.append(False)
            else:
                fit_idx, inner_idx = split
                X_fit, y_fit = Xtr_aug[fit_idx], ytr[fit_idx]
                X_inner, y_inner = Xtr_aug[inner_idx], ytr[inner_idx]
                nested_calibration_used.append(True)

            common_kw = dict(
                lr=cfg["LR"], wd=1e-4, epochs=cfg["EPOCHS"], bs=cfg["BS"],
                warm=cfg["WARM"], patience=cfg["PATIENCE"], ls=cfg["LS"],
                live=live, class_weights=class_weights, loss_mode=cfg["LOSS"],
                focal_gamma=cfg["FOCAL_GAMMA"], mixup_alpha=cfg["MIXUP_ALPHA"],
                rdrop_alpha=cfg["RDROP_ALPHA"], use_sam=cfg["USE_SAM"], sam_rho=cfg["SAM_RHO"],
                mc_dropout=cfg["MC_DROPOUT"], tta_noise_std=cfg["TTA_NOISE_STD"],
                amp_enabled=cfg["AMP_ENABLED"],
            )
            mc_kw = dict(mc_passes=max(1, cfg["MC_PASSES"]), enable_dropout=cfg["MC_DROPOUT"],
                         tta_noise_std=cfg["TTA_NOISE_STD"])

            fold_inner = []
            for name in model_names:
                _, predictor = fit_architecture(name, X_fit, y_fit, X_inner, y_inner, K, **common_kw)
                # 온도는 내부 홀드아웃에서만 고른다 -> 바깥 fold OOF는 깨끗하다.
                p_inner = predictor(X_inner, **mc_kw, apply_temperature=False)
                predictor.temperature = temperature_scale_grid(p_inner, y_inner, mode=cfg["TEMP_MODE"])
                oof[name][va_idx] = predictor(Xva_aug, **mc_kw)
                predictor.save(artifacts.model_path(run_dir, seed, fi, name))
                fold_inner.append(p_inner)

            inner_probs.append(np.mean(np.stack(fold_inner, axis=0), axis=0))
            inner_targets.append(y_inner)

        # ---------- 하드클러스터 도출 ----------
        if use_expert and hard is None:
            source = "inner_holdout" if all(nested_calibration_used) else "outer_fold"
            cluster_info = discover_hard_cluster(
                np.concatenate(inner_targets), np.vstack(inner_probs),
                f1_quantile=cfg["CLUSTER_F1_QUANTILE"],
                min_confusion=cfg["CLUSTER_MIN_CONFUSION"],
                max_size=cfg["CLUSTER_MAX_SIZE"])
            cluster_info["source"] = source
            hard = sorted(cluster_info["cluster"])
            if hard:
                log.info("[Cluster] 도출된 하드클러스터=%s (후보=%s, F1 임계=%.3f, 출처=%s)",
                         hard, cluster_info["candidates"], cluster_info["f1_threshold"], source)
            else:
                log.warning("[Cluster] 상호 혼동 클러스터를 찾지 못함 -> 전문가 모델 비활성화")
                use_expert = False

        # ---------- pass 2: 하드클러스터 전문가 ----------
        oof_expert = None
        if use_expert and hard:
            from src.expert import fit_expert
            import joblib
            oof_expert = np.zeros((X_all.shape[0], len(hard)))
            for fi, (tr_idx, va_idx) in enumerate(folds, 1):
                Xtr_aug = augment_with_anomaly_score(X_all[tr_idx], fold_iso[fi])
                Xva_aug = augment_with_anomaly_score(X_all[va_idx], fold_iso[fi])
                exp_model = fit_expert(Xtr_aug, y_all[tr_idx], hard,
                                       n_estimators=cfg["EXPERT_N_ESTIMATORS"], seed=seed + fi)
                oof_expert[va_idx] = exp_model.predict_proba(Xva_aug)
                joblib.dump(exp_model, artifacts.expert_path(run_dir, seed, fi))

        seed_key = str(seed)
        oof_by_seed[seed_key] = oof
        if oof_expert is not None:
            expert_by_seed[seed_key] = oof_expert
        per_model_scores[seed_key] = {}
        for name in model_names:
            acc, f1, loss = compute_metrics(y_all, oof[name])
            per_model_scores[seed_key][name] = {"macro_f1": f1, "accuracy": acc, "log_loss": loss}
            log.info("[OOF %-6s seed=%d] f1=%.6f | acc=%.6f | loss=%.6f", name, seed, f1, acc, loss)

    # ---------- 후처리 파라미터 선택 ----------
    spec = PostProcessSpec(
        blend_mode=cfg["BLEND_MODE"], use_expert=use_expert,
        expert_w_grid=tuple(cfg["EXPERT_W_GRID"]), hard_cluster=tuple(hard or ()),
        smooth_eps=cfg["SMOOTH_EPS"], use_bias_tune=cfg["USE_BIAS_TUNE"],
        bias_lim=cfg["BIAS_LIM"], use_balanced_assign=cfg["USE_BALANCED_ASSIGN"],
    )
    expert_arg = expert_by_seed if use_expert else None
    log.info("\n---------- Post-processing 선택 ----------")
    pp_fit, oof_probs = fit_postprocess(oof_by_seed, expert_arg, y_all, spec, live=live)

    acc_s, f1_s, loss_s = compute_metrics(y_all, pp_fit.apply_bias(oof_probs))
    selection_f1 = float(f1_s)

    # ---------- 정직한 추정 ----------
    nested = {"nested_f1": None, "raw_blend_f1": None, "n_splits": 0}
    if cfg["NESTED_POSTPROCESS_FOLDS"] and cfg["NESTED_POSTPROCESS_FOLDS"] > 1:
        log.info("\n---------- Post-processing 중첩 검증 ----------")
        nested = nested_estimate(oof_by_seed, expert_arg, y_all, spec,
                                 n_splits=cfg["NESTED_POSTPROCESS_FOLDS"],
                                 seed=cfg["CV_SEED"], live=live)

    metrics = {
        "version": cfg["VERSION"],
        "n_train": int(X_all.shape[0]),
        "n_features": int(X_all.shape[1]),
        "n_classes": int(K),
        "per_model_oof": per_model_scores,
        "selection_f1": selection_f1,
        "selection_accuracy": float(acc_s),
        "selection_log_loss": float(loss_s),
        "nested_f1": nested["nested_f1"],
        "raw_blend_f1": nested["raw_blend_f1"],
        "nested_folds": nested["n_splits"],
        "nested_calibration": bool(all(nested_calibration_used)) if nested_calibration_used else False,
        "inner_val_frac": cfg["INNER_VAL_FRAC"],
        "discovery": report,
        "hard_cluster": list(hard or []),
        "hard_cluster_source": cluster_info.get("source"),
        "hard_cluster_class_f1": cluster_info.get("class_f1"),
        "hard_cluster_candidates": cluster_info.get("candidates"),
        "expert_enabled": bool(use_expert and hard),
    }
    artifacts.write_json(run_dir / artifacts.METRICS_FILE, metrics)
    artifacts.write_json(run_dir / artifacts.POSTPROCESS_FILE, pp_fit.to_dict())
    np.save(run_dir / artifacts.OOF_PROBS_FILE, oof_probs)
    np.save(run_dir / artifacts.OOF_TARGET_FILE, y_all)

    print(f"\n[OOF] selection macro_f1={selection_f1:.6f}  (후처리를 고른 행에서 잰 값, 낙관적)")
    if nested["nested_f1"] is not None:
        print(f"[OOF] nested   macro_f1={nested['nested_f1']:.6f}  (선택에 쓰지 않은 행, 정직한 추정)")
        print(f"[OOF] 후처리 전 시드평균 macro_f1={nested['raw_blend_f1']:.6f}")
    if not metrics["nested_calibration"]:
        print("[경고] 일부 fold에서 내부 홀드아웃을 만들지 못해 OOF가 낙관 편향되었습니다.")
    print("SAVED run:", run_dir)
    return str(run_dir)


def parse_args():
    ap = argparse.ArgumentParser(description="학습 후 실행 디렉터리를 남긴다 (제출은 predict.py).")
    ap.add_argument("--fast", action="store_true", help="스모크 테스트 (시드 1개, epoch 축소)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--no-predict", action="store_true",
                    help="학습만 하고 제출 파일은 만들지 않는다")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = dict(CONFIG)
    if args.fast:
        cfg.update({"SEEDS": [42], "EPOCHS": 12, "CV_FOLDS": 3, "PATIENCE": 5,
                    "MC_PASSES": 1, "EXPERT_N_ESTIMATORS": 150,
                    "NESTED_POSTPROCESS_FOLDS": 3})
        print("[FAST] smoke-test mode")
    if args.seeds:
        cfg["SEEDS"] = args.seeds
    if args.data_dir:
        cfg["DATA_DIR"] = args.data_dir
    if args.out_dir:
        cfg["OUT_DIR"] = args.out_dir

    run_dir = run(cfg)
    if not args.no_predict:
        from predict import predict_run
        sub = predict_run(run_dir, data_dir=cfg["DATA_DIR"], live=cfg["LIVE"])
        print("SAVED submission:", sub)
    return run_dir


if __name__ == "__main__":
    main()
