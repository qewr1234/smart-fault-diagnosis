# ============================================
# Smart Fault Diagnosis — 원클릭 학습/추론
# DL 3종 (FT/Mixer/GLU) + 하드클러스터 전문가(LGBM)
# 5-Fold CV | OOF 로짓 블렌드 | 게이트형 후처리
# ============================================
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import CONFIG
from src.utils import set_seed, infer_feature_cols, compute_metrics, ensure_prob_finite
from src.features import (
    winsorize_selected, add_pairwise_features, add_row_stats,
    add_fdi_features, clip_train_to_test_range,
)
from src.calibration import (
    temperature_scale_grid, apply_temperature,
    gated_bias_tune, gated_balanced_assign,
)
from src.blend import greedy_weight_search, combine
from src.expert import fit_expert, rerank_hard_cluster, select_expert_weight
from src.models import fit_ft, fit_tabmixer, fit_glumlp

# Colab이면 드라이브 마운트
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
except Exception:
    pass


def run(cfg):
    set_seed(cfg["CV_SEED"])
    live = cfg["LIVE"]

    # ---------- Load ----------
    data_dir = Path(cfg["DATA_DIR"])
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    feat = infer_feature_cols(train)
    assert "target" in train.columns
    le = LabelEncoder()
    y_all = le.fit_transform(train["target"].values)
    K = len(le.classes_)

    train_X = train[feat].copy()
    test_X = test[feat].copy()

    # ---------- FE ----------
    if cfg["CLIP_X11"]:
        train_X = clip_train_to_test_range(train_X, test_X, cols=("X_11",), live=live)

    vars_ = train_X.var().sort_values(ascending=False)
    top_feats = list(vars_.head(max(1, cfg["TOPK_VAR"])).index)
    train_X, _, test_X = winsorize_selected(train_X, train_X, test_X, top_feats)

    if cfg["USE_FDI_FEATURES"]:
        train_X, test_X = add_fdi_features(train_X, test_X, feat, live=live)

    if cfg["ADD_PAIRDIFF"]:
        top_feats = [c for c in top_feats if c in train_X.columns]  # FDI 중복 제거분 가드
        train_X, _, test_X = add_pairwise_features(train_X, train_X, test_X, top_feats, limit_pairs=12)
        if live:
            print(f"[Pairwise] added features between top-{len(top_feats)}")

    base_cols = [c for c in train_X.columns if c in feat]
    train_X, _, test_X = add_row_stats(train_X, train_X, test_X, base_cols)

    imp = SimpleImputer(strategy="median")
    X_all = imp.fit_transform(train_X)
    Xte = imp.transform(test_X)

    vt = VarianceThreshold(0.0)
    X_all = vt.fit_transform(X_all)
    Xte = vt.transform(Xte)
    if live:
        print("[VT] kept dims:", X_all.shape[1])

    # class weights (optional)
    class_weights = None
    if cfg["USE_CLASS_WEIGHT"]:
        cnt = np.bincount(y_all, minlength=K).astype(float)
        w = cnt.sum() / np.clip(cnt, 1e-12, None)
        class_weights = w / w.mean()

    hard = sorted(cfg["HARD_CLUSTER"])
    use_expert = cfg["USE_EXPERT"]
    if use_expert:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            print("[Expert] lightgbm 미설치 -> 전문가 비활성화")
            use_expert = False

    # ---------- Seed loop ----------
    blend_val_accum, test_accum = None, None

    for si, seed in enumerate(cfg["SEEDS"], 1):
        if live:
            print(f"\n========== SEED RUN {si}/{len(cfg['SEEDS'])} (seed={seed}) ==========")
        set_seed(seed)
        skf = StratifiedKFold(n_splits=cfg["CV_FOLDS"], shuffle=True, random_state=seed)

        model_names = [n for n, use in
                       [("ft", cfg["USE_FT"]), ("mixer", cfg["USE_MIXER"]), ("glu", cfg["USE_GLU"])] if use]
        fitters = {"ft": fit_ft, "mixer": fit_tabmixer, "glu": fit_glumlp}

        oof = {n: np.zeros((X_all.shape[0], K)) for n in model_names}
        tpred = {n: [] for n in model_names}
        oof_expert = np.zeros((X_all.shape[0], len(hard)))
        tpred_expert = []

        for fi, (tr_idx, va_idx) in enumerate(skf.split(X_all, y_all), 1):
            if live:
                print(f"\n===== FOLD {fi}/{cfg['CV_FOLDS']} (seed={seed}) =====")
            Xtr, Xva = X_all[tr_idx], X_all[va_idx]
            ytr, yva = y_all[tr_idx], y_all[va_idx]

            if cfg["USE_ISOFOREST"]:
                iso = IsolationForest(n_estimators=200, contamination="auto",
                                      random_state=seed + fi, n_jobs=1)
                iso.fit(Xtr)
                Xtr_aug = np.hstack([Xtr, iso.score_samples(Xtr).reshape(-1, 1)])
                Xva_aug = np.hstack([Xva, iso.score_samples(Xva).reshape(-1, 1)])
                Xte_aug = np.hstack([Xte, iso.score_samples(Xte).reshape(-1, 1)])
            else:
                Xtr_aug, Xva_aug, Xte_aug = Xtr, Xva, Xte

            common_kw = dict(
                lr=cfg["LR"], wd=1e-4, epochs=cfg["EPOCHS"], bs=cfg["BS"],
                warm=cfg["WARM"], patience=cfg["PATIENCE"], ls=cfg["LS"],
                live=live, class_weights=class_weights, loss_mode=cfg["LOSS"],
                focal_gamma=cfg["FOCAL_GAMMA"], mixup_alpha=cfg["MIXUP_ALPHA"],
                rdrop_alpha=cfg["RDROP_ALPHA"], use_sam=cfg["USE_SAM"], sam_rho=cfg["SAM_RHO"],
                mc_dropout=cfg["MC_DROPOUT"], tta_noise_std=cfg["TTA_NOISE_STD"],
                amp_enabled=cfg["AMP_ENABLED"],
            )

            for name in model_names:
                _, pred_fn = fitters[name](Xtr_aug, ytr, Xva_aug, yva, K, **common_kw)
                pva = pred_fn(Xva_aug, mc_passes=max(1, cfg["MC_PASSES"]),
                              enable_dropout=cfg["MC_DROPOUT"], tta_noise_std=cfg["TTA_NOISE_STD"])
                T = temperature_scale_grid(pva, yva, mode=cfg["TEMP_MODE"])
                oof[name][va_idx] = apply_temperature(pva, T)
                pte = pred_fn(Xte_aug, mc_passes=max(1, cfg["MC_PASSES"]),
                              enable_dropout=cfg["MC_DROPOUT"], tta_noise_std=cfg["TTA_NOISE_STD"])
                tpred[name].append(apply_temperature(pte, T))

            if use_expert:
                exp_model = fit_expert(Xtr_aug, ytr, hard,
                                       n_estimators=cfg["EXPERT_N_ESTIMATORS"], seed=seed + fi)
                oof_expert[va_idx] = exp_model.predict_proba(Xva_aug)
                tpred_expert.append(exp_model.predict_proba(Xte_aug))

        # ---------- OOF scores / blend ----------
        scores = {}
        for name in model_names:
            acc, f1, loss = compute_metrics(y_all, oof[name])
            scores[name] = f1
            if live:
                print(f"[OOF {name:6s}] f1={f1:.6f} | acc={acc:.6f} | loss={loss:.6f}")
            tpred[name] = np.mean(tpred[name], axis=0)

        keep = sorted(scores, key=scores.get, reverse=True)
        keep = [n for n in keep if np.isfinite(oof[n]).all()][:3]
        if live:
            print("[Blend] candidates:", keep)

        val_list = [oof[n] for n in keep]
        if len(val_list) == 1:
            final_va, weights = ensure_prob_finite(val_list[0]), [1.0]
        else:
            w = greedy_weight_search(val_list, y_all, passes=2, live=live, mode=cfg["BLEND_MODE"])
            final_va, weights = combine(val_list, w, cfg["BLEND_MODE"]), w.tolist()

        test_list = [tpred[n] for n in keep]
        final_te = ensure_prob_finite(test_list[0]) if len(test_list) == 1 \
            else combine(test_list, weights, cfg["BLEND_MODE"])

        # ---------- Expert rerank (OOF 선택) ----------
        if use_expert:
            te_expert = np.mean(tpred_expert, axis=0)
            w_exp = select_expert_weight(final_va, oof_expert, y_all, hard,
                                         w_grid=cfg["EXPERT_W_GRID"], live=live)
            final_va = rerank_hard_cluster(final_va, oof_expert, hard, w=w_exp)
            final_te = rerank_hard_cluster(final_te, te_expert, hard, w=w_exp)

        # ---------- Smoothing ----------
        eps = max(0.0, float(cfg["SMOOTH_EPS"]))
        if eps > 0:
            final_va = (1.0 - eps) * final_va + eps / float(K)
            final_te = (1.0 - eps) * final_te + eps / float(K)

        acc_f, f1_f, loss_f = compute_metrics(y_all, final_va)
        print(f"[Final OOF seed={seed}] loss={loss_f:.6f} | acc={acc_f:.6f} | macro_f1={f1_f:.6f}")
        print("[Blend] weights:", {n: round(weights[i], 3) for i, n in enumerate(keep)})

        blend_val_accum = final_va if blend_val_accum is None else blend_val_accum + final_va
        test_accum = final_te if test_accum is None else test_accum + final_te

    # ---------- Seed average ----------
    blend_val_accum /= float(len(cfg["SEEDS"]))
    test_accum /= float(len(cfg["SEEDS"]))
    acc_f, f1_f, loss_f = compute_metrics(y_all, blend_val_accum)
    print(f"\n[Final OOF (seed-avg)] loss={loss_f:.6f} | acc={acc_f:.6f} | macro_f1={f1_f:.6f}")

    # ---------- Gated post-processing ----------
    if cfg["USE_BIAS_TUNE"]:
        blend_val_accum, test_accum = gated_bias_tune(
            blend_val_accum, y_all, test_accum, lim=cfg["BIAS_LIM"], live=True)

    if cfg["USE_BALANCED_ASSIGN"]:
        pred = gated_balanced_assign(blend_val_accum, y_all, test_accum, live=True)
    else:
        pred = test_accum.argmax(1)

    pred_labels = le.inverse_transform(pred)

    # ---------- Save ----------
    out_dir = Path(cfg["OUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    sub = pd.DataFrame({"ID": test["ID"].values, "target": pred_labels})
    if list(sample.columns) != ["ID", "target"]:
        sub = sample[["ID"]].merge(sub, on="ID", how="left")
    sub_path = out_dir / f"submission_{cfg['VERSION']}_{ts}.csv"
    sub.to_csv(sub_path, index=False)

    oof_path = out_dir / f"oof_{cfg['VERSION']}_{ts}.npy"
    np.save(oof_path, blend_val_accum)
    print("SAVED:", sub_path)
    print("SAVED:", oof_path)
    return str(sub_path)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="스모크 테스트 (시드 1개, epoch 축소)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(CONFIG)
    if args.fast:
        cfg.update({"SEEDS": [42], "EPOCHS": 12, "CV_FOLDS": 3, "PATIENCE": 5,
                    "MC_PASSES": 1, "EXPERT_N_ESTIMATORS": 150})
        print("[FAST] smoke-test mode")
    if args.seeds:
        cfg["SEEDS"] = args.seeds
    if args.data_dir:
        cfg["DATA_DIR"] = args.data_dir
    if args.out_dir:
        cfg["OUT_DIR"] = args.out_dir
    run(cfg)
