from __future__ import annotations

import numpy as np
import pandas as pd


def modified_zscore(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    median = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - median))
    score = np.abs(x - median) / (mad + 1e-8)
    return (score > k).astype(int)


def prob_interval_anomalies(eval_df: pd.DataFrame, model: str, level: int = 99) -> pd.Series:
    lo, hi = f"{model}-lo-{level}", f"{model}-hi-{level}"
    if lo not in eval_df.columns:
        raise KeyError(f"Колонка {lo} не найдена; передайте forecast с level=[{level}]")
    return ((eval_df["y"] < eval_df[lo]) | (eval_df["y"] > eval_df[hi])).astype(int)


def _lag_matrix(g: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    X = pd.DataFrame(index=g.index)
    for lag in lags:
        X[f"lag{lag}"] = g["y"].shift(lag)
    return X.bfill().ffill()


def isolation_forest_panel(
    panel: pd.DataFrame,
    contamination: float = 0.05,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 288),
) -> pd.Series:
    from sklearn.ensemble import IsolationForest

    out = pd.Series(0, index=panel.index)
    for _, g in panel.groupby("unique_id"):
        g = g.sort_values("ds")
        X = _lag_matrix(g, lags)
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        out.loc[g.index] = (clf.fit_predict(X.values) == -1).astype(int)
    return out


def lof_panel(
    panel: pd.DataFrame,
    contamination: float = 0.05,
    n_neighbors: int = 50,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 288),
) -> pd.Series:
    from sklearn.neighbors import LocalOutlierFactor

    out = pd.Series(0, index=panel.index)
    for _, g in panel.groupby("unique_id"):
        g = g.sort_values("ds")
        X = _lag_matrix(g, lags)
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
        out.loc[g.index] = (clf.fit_predict(X.values) == -1).astype(int)
    return out


def matrix_profile_anomalies(
    panel: pd.DataFrame,
    window: int = 288,
    z_threshold: float = 2.5,
) -> pd.Series:
    import stumpy

    out = pd.Series(0, index=panel.index)
    for _, g in panel.groupby("unique_id"):
        g = g.sort_values("ds")
        if len(g) < 2 * window:
            continue
        mp = stumpy.stump(g["y"].values, m=window)
        score = np.zeros(len(g))
        score[window - 1:] = mp[:, 0]
        out.loc[g.index] = modified_zscore(score, k=z_threshold)
    return out


def score_anomalies(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float("nan")
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "n_predicted": int(y_pred.sum()),
        "n_actual": int(y_true.sum()),
    }


def ks_drift_detection(
    residuals: np.ndarray,
    window_size: int = 200,
    threshold: float = 0.01,
) -> list[int]:
    from scipy.stats import ks_2samp

    if len(residuals) < 2 * window_size:
        return []
    drift_idx: list[int] = []
    ref = residuals[:window_size]
    last_idx = -window_size
    for i in range(window_size, len(residuals) - window_size):
        cur = residuals[i:i + window_size]
        _, p = ks_2samp(ref, cur)
        if p < threshold and i - last_idx > window_size:
            drift_idx.append(i)
            ref = cur
            last_idx = i
    return drift_idx
