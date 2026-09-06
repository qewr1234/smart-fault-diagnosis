# ============================================
# 전체 파이프라인 설정
# ============================================

CONFIG = {
    # Paths
    "DATA_DIR": "data",
    "OUT_DIR": "outputs",
    "VERSION": "v17_fdi_expert_gated",

    # CV / seeds
    "CV_FOLDS": 5,
    "CV_SEED": 42,
    "SEEDS": [42, 52, 62],   # --fast 시 [42]로 축소

    # Feature Eng
    "TOPK_VAR": 16,
    "ADD_PAIRDIFF": True,
    "USE_ISOFOREST": True,
    "USE_FDI_FEATURES": True,   # 잔차 + 편차 시그니처 (핵심 추가)
    "CLIP_X11": True,           # X_11 train 극단값 -> test 분위수 클리핑
    "PCA_COMPONENTS": 0,

    # Training
    "EPOCHS": 140,
    "BS": 512,
    "LR": 3e-3,
    "WARM": 8,
    "PATIENCE": 16,
    "LS": 0.05,
    "LOSS": "ce",            # ce | focal
    "FOCAL_GAMMA": 2.0,
    "MIXUP_ALPHA": 0.2,
    "RDROP_ALPHA": 0.5,
    "USE_SAM": True,
    "SAM_RHO": 0.05,
    "USE_CLASS_WEIGHT": False,

    # Inference / TTA
    "MC_DROPOUT": True,
    "MC_PASSES": 2,
    "TTA_NOISE_STD": 0.0,

    # Calibration / Blend
    "TEMP_MODE": "soft",       # soft | grid | off
    "BLEND_MODE": "logit",     # prob | logit | geomean
    "SMOOTH_EPS": 0.02,
    "USE_BIAS_TUNE": True,     # OOF 게이트 내장, 범위 확대(lim=0.3)
    "BIAS_LIM": 0.30,
    "USE_EM_PRIOR": False,
    "EM_ALPHA": 0.0,

    # Models
    "USE_FT": True,
    "USE_MIXER": True,
    "USE_GLU": True,

    # Hard-cluster expert (LightGBM)
    "USE_EXPERT": True,
    "HARD_CLUSTER": [0, 3, 9, 15, 19],
    "EXPERT_W_GRID": [0.0, 0.25, 0.5],   # OOF Macro-F1로 선택 (0.0 = 미적용)
    "EXPERT_N_ESTIMATORS": 600,

    # Post-processing (OOF 게이트)
    "USE_BALANCED_ASSIGN": True,

    # Devices/AMP
    "AMP_ENABLED": True,
    "LIVE": True,
}
