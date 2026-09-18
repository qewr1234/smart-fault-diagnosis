"""train.run() 엔드투엔드 스모크 테스트 (CPU, 소형 설정).

config.py 기본값의 정규화 조합(SAM + R-Drop + MixUp)을 그대로 두고 크기만 줄여,
'기본 설정이 완주하는가'를 검증한다.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from config import CONFIG  # noqa: E402
from train import run  # noqa: E402


@pytest.mark.slow
def test_run_end_to_end_with_default_regularizers(data_dir, tmp_path):
    cfg = dict(CONFIG)
    cfg.update({
        "DATA_DIR": str(data_dir),
        "OUT_DIR": str(tmp_path / "out"),
        "SEEDS": [42],
        "CV_FOLDS": 2,
        "EPOCHS": 2,
        "WARM": 1,
        "PATIENCE": 5,
        "BS": 128,
        "MC_PASSES": 1,
        "EXPERT_N_ESTIMATORS": 10,
        "LIVE": False,
        "AMP_ENABLED": False,
    })
    assert cfg["USE_SAM"] and cfg["RDROP_ALPHA"] > 0 and cfg["MIXUP_ALPHA"] > 0

    sub_path = run(cfg)

    sub = pd.read_csv(sub_path)
    test = pd.read_csv(data_dir / "test.csv")
    assert list(sub.columns) == ["ID", "target"]
    assert len(sub) == len(test)
    assert (sub["ID"].values == test["ID"].values).all()
    assert sub["target"].between(0, 20).all()

    oof_files = list((tmp_path / "out").glob("oof_*.npy"))
    assert len(oof_files) == 1
    oof = np.load(oof_files[0])
    assert oof.shape == (len(pd.read_csv(data_dir / "train.csv")), 21)
    np.testing.assert_allclose(oof.sum(1), 1.0, atol=1e-6)
