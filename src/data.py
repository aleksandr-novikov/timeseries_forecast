from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

NAB_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
DEFAULT_SERIES = (
    "ec2_cpu_utilization_24ae8d",
    "ec2_cpu_utilization_53ea38",
    "ec2_cpu_utilization_5f5533",
)
FREQ = "5min"
DAILY_SEASON = 288
WEEKLY_SEASON = 2016


def _csv_path(name: str, raw_dir: Path) -> Path:
    return raw_dir / f"{name}.csv"


def download_nab(
    series: Iterable[str] = DEFAULT_SERIES,
    raw_dir: str | Path = "data/raw/NAB",
) -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    import urllib.request as ur

    for fname in ("combined_windows.json", "combined_labels.json"):
        dst = raw_dir / fname
        if not dst.exists():
            ur.urlretrieve(f"{NAB_BASE}/labels/{fname}", dst)

    for name in series:
        dst = _csv_path(name, raw_dir)
        if not dst.exists():
            ur.urlretrieve(f"{NAB_BASE}/data/realAWSCloudwatch/{name}.csv", dst)
    return raw_dir


def load_panel(
    series: Iterable[str] = DEFAULT_SERIES,
    raw_dir: str | Path = "data/raw/NAB",
    freq: str = FREQ,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    frames = []
    for name in series:
        df = pd.read_csv(_csv_path(name, raw_dir), parse_dates=["timestamp"])
        df = df.set_index("timestamp").sort_index()
        # 5f5533 имеет 3-минутный сдвиг от остальных - снапим к общей сетке
        df.index = df.index.round(freq)
        df = df[~df.index.duplicated(keep="first")]
        df = df.asfreq(freq)
        df["value"] = df["value"].ffill().bfill()
        frames.append(pd.DataFrame({
            "unique_id": name,
            "ds": df.index,
            "y": df["value"].to_numpy(dtype=float),
        }))
    panel = pd.concat(frames, ignore_index=True)
    common_start = panel.groupby("unique_id")["ds"].min().max()
    common_end = panel.groupby("unique_id")["ds"].max().min()
    panel = panel[(panel["ds"] >= common_start) & (panel["ds"] <= common_end)]
    return panel[["unique_id", "ds", "y"]].reset_index(drop=True)


def load_anomaly_labels(
    series: Iterable[str] = DEFAULT_SERIES,
    raw_dir: str | Path = "data/raw/NAB",
) -> tuple[dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]], dict[str, list[pd.Timestamp]]]:
    raw_dir = Path(raw_dir)
    with open(raw_dir / "combined_windows.json") as f:
        cw = json.load(f)
    with open(raw_dir / "combined_labels.json") as f:
        cl = json.load(f)

    windows, points = {}, {}
    for name in series:
        key = f"realAWSCloudwatch/{name}.csv"
        windows[name] = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in cw.get(key, [])]
        points[name] = [pd.Timestamp(t) for t in cl.get(key, [])]
    return windows, points


def make_anomaly_mask(
    panel: pd.DataFrame,
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> pd.Series:
    mask = pd.Series(False, index=panel.index)
    for uid, intervals in windows.items():
        idx = panel["unique_id"] == uid
        for start, end in intervals:
            mask |= idx & (panel["ds"] >= start) & (panel["ds"] <= end)
    return mask.astype(int)


def train_test_split_panel(
    panel: pd.DataFrame,
    fh_frac: float = 0.42,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n = panel.groupby("unique_id").size().iloc[0]
    fh = int(fh_frac * n)
    df_test = panel.groupby("unique_id").tail(fh)
    df_train = panel.drop(df_test.index).reset_index(drop=True)
    return df_train, df_test.reset_index(drop=True), fh
