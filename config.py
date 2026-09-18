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

    # 검증 정직성 (OOF 편향 방지)
    "NESTED_CALIBRATION": True,   # early stopping/온도를 fold 내부 홀드아웃에서 결정
    "INNER_VAL_FRAC": 0.15,       # 내부 홀드아웃 비율
    "NESTED_POSTPROCESS_FOLDS": 5,  # 후처리 기여도의 정직한 추정에 쓸 fold 수 (0=생략)

    # Feature Eng
    "TOPK_VAR": 16,
    "ADD_PAIRDIFF": True,
    "USE_ISOFOREST": True,
    "USE_FDI_FEATURES": True,   # 잔차 + 편차 시그니처 (핵심 추가)
    "CLIP_SHIFT_COLS": True,    # train 극단 꼬리 -> 참조(test) 분위수로 클리핑
    "PCA_COMPONENTS": 0,

    # 자동 탐색 (하드코딩된 센서쌍/컬럼/클러스터 대체)
    "AUTO_DISCOVER": True,      # False면 features.py 의 폴백 상수를 쓴다
    "CORR_THRESHOLD": 0.95,     # 준중복 센서쌍 기준 |corr|
    "EXACT_DUP_THRESHOLD": 0.9999,  # 완전 중복으로 묶어 하나만 남길 기준
    "MAX_REDUNDANT_PAIRS": 12,  # 잔차 피처를 만들 쌍 수 상한
    "SHIFT_EXCESS_RATIO": 1.0,  # 꼬리 길이 / 참조 분포 폭 (이상이면 shift 컬럼)
    "SHIFT_OUTSIDE_FRAC": 0.02, # 참조 범위 밖 train 행 비율 상한
    "MAX_SHIFT_COLS": 3,
    "N_SIGNATURE_COLS": 1,      # 클래스별 산포가 크게 다른 컬럼 수 (0=비활성)

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
    # "auto" = 검증 예측의 혼동 구조에서 도출. 리스트를 주면 그 값을 고정으로 쓴다.
    "HARD_CLUSTER": "auto",
    "CLUSTER_F1_QUANTILE": 0.35,   # 후보로 삼을 저성능 클래스 분위
    "CLUSTER_MIN_CONFUSION": 0.05, # 두 클래스를 이을 상호 오분류율 하한
    "CLUSTER_MAX_SIZE": 8,
    "EXPERT_W_GRID": [0.0, 0.25, 0.5],   # OOF Macro-F1로 선택 (0.0 = 미적용)
    "EXPERT_N_ESTIMATORS": 600,

    # Post-processing (OOF 게이트)
    "USE_BALANCED_ASSIGN": True,

    # Devices/AMP
    "AMP_ENABLED": True,
    "LIVE": True,
}
