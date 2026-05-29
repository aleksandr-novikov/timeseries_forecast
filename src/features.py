from __future__ import annotations

import numpy as np
import pandas as pd


def _idx(dates):
    return pd.RangeIndex(len(dates))


def hour_sin(dates):
    h = pd.to_datetime(dates).hour + pd.to_datetime(dates).minute / 60
    return pd.Series(np.sin(2 * np.pi * h / 24), index=_idx(dates))


def hour_cos(dates):
    h = pd.to_datetime(dates).hour + pd.to_datetime(dates).minute / 60
    return pd.Series(np.cos(2 * np.pi * h / 24), index=_idx(dates))


def dow_sin(dates):
    d = pd.to_datetime(dates).dayofweek
    return pd.Series(np.sin(2 * np.pi * d / 7), index=_idx(dates))


def dow_cos(dates):
    d = pd.to_datetime(dates).dayofweek
    return pd.Series(np.cos(2 * np.pi * d / 7), index=_idx(dates))


def is_weekend(dates):
    arr = (pd.to_datetime(dates).dayofweek >= 5).astype(int)
    return pd.Series(np.asarray(arr), index=_idx(dates))


DEFAULT_DATE_FEATURES = [hour_sin, hour_cos, dow_sin, dow_cos, is_weekend]
