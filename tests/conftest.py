"""공용 fixture: 소형 합성 센서 데이터 (X_01~X_52, 21 클래스).

실제 DACON 데이터와 같은 스키마를 갖고, 완전 중복 센서쌍(X_06=X_45, X_10=X_17)을
포함해 FDI 피처 경로가 실제 데이터와 동일하게 동작하도록 만든다.
"""
import numpy as np
import pandas as pd
import pytest

N_CLASSES = 21
FEAT_COLS = [f"X_{i:02d}" for i in range(1, 53)]


def make_synthetic(n_rows, seed=0, n_classes=N_CLASSES):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_rows // n_classes)
    rng.shuffle(y)
    n = len(y)
    shift = rng.normal(size=(n_classes, 52))
    X = rng.normal(size=(n, 52)) + 0.6 * shift[y]
    # 완전 중복 센서쌍 (README/features.py 가정과 동일)
    X[:, 44] = X[:, 5]   # X_45 = X_06
    X[:, 16] = X[:, 9]   # X_17 = X_10
    return X, y


@pytest.fixture(scope="session")
def small_xy():
    """numpy (X, y) — 단위 테스트용."""
    return make_synthetic(21 * 20, seed=1)


@pytest.fixture(scope="session")
def probs_and_labels():
    """라벨과 어느 정도 상관이 있는 확률 행렬 — 후처리/블렌드 테스트용."""
    rng = np.random.default_rng(2)
    n, k = 21 * 30, N_CLASSES
    y = np.repeat(np.arange(k), n // k)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 2.0
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return p, y


@pytest.fixture
def data_dir(tmp_path):
    """train.csv / test.csv / sample_submission.csv 가 있는 임시 데이터 디렉터리."""
    Xtr, ytr = make_synthetic(21 * 20, seed=3)
    Xte, _ = make_synthetic(21 * 6, seed=4)
    train = pd.DataFrame(Xtr, columns=FEAT_COLS)
    train.insert(0, "ID", [f"TRAIN_{i:04d}" for i in range(len(train))])
    train["target"] = ytr
    test = pd.DataFrame(Xte, columns=FEAT_COLS)
    test.insert(0, "ID", [f"TEST_{i:04d}" for i in range(len(test))])
    sample = pd.DataFrame({"ID": test["ID"], "target": 0})

    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    sample.to_csv(tmp_path / "sample_submission.csv", index=False)
    return tmp_path
