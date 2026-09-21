"""Data loading and leakage-safe grouped splits for the phishing dataset."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_arff(path: str | Path):
    """Read this simple nominal ARFF without relying on an external parser."""
    names, rows, in_data = [], [], False
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if line.lower().startswith("@attribute"):
            names.append(line.split()[1])
        elif line.lower() == "@data":
            in_data = True
        elif in_data:
            rows.append([int(v.strip()) for v in line.split(",")])
    frame = pd.DataFrame(rows, columns=names)
    X = frame.iloc[:, :-1].copy()
    # Dataset convention: -1 is phishing, 1 is legitimate.
    y = (frame.iloc[:, -1].to_numpy() == -1).astype(int)
    return X, y


def vector_groups(X: pd.DataFrame) -> np.ndarray:
    """Stable group IDs: identical 30-feature vectors always share a group."""
    return np.array([
        hashlib.sha256(",".join(map(str, row)).encode()).hexdigest()
        for row in X.to_numpy()
    ])


def outer_split(X, y, groups, seed=2026):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(X, y, groups))
    assert not set(groups[train_idx]).intersection(groups[test_idx])
    return train_idx, test_idx


def inner_splits(X, y, groups, seed=2026):
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups))
