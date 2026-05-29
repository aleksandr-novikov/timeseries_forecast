"""CLI: python -m src.pipeline --forecast-model MSTL --anomaly-method matrix_profile"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.anomaly import (
    isolation_forest_panel,
    lof_panel,
    matrix_profile_anomalies,
    modified_zscore,
    score_anomalies,
)
from src.data import (
    DAILY_SEASON,
    FREQ,
    WEEKLY_SEASON,
    load_anomaly_labels,
    load_panel,
    make_anomaly_mask,
)


def _build_factory():
    from statsforecast.models import AutoETS, MSTL, SeasonalNaive, TBATS

    return {
        "SeasonalNaive": SeasonalNaive(season_length=DAILY_SEASON, alias="SNaive"),
        "MSTL": MSTL(season_length=[DAILY_SEASON, WEEKLY_SEASON], alias="MSTL"),
        "AutoETS": AutoETS(season_length=DAILY_SEASON, model="ZZA", alias="ETS"),
        "TBATS": TBATS(
            season_length=[DAILY_SEASON, WEEKLY_SEASON],
            use_trend=True, use_damped_trend=True, use_arma_errors=False,
            alias="TBATS",
        ),
    }


def build_forecast(
    panel: pd.DataFrame,
    horizon: int,
    model_name: str,
    level: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    from statsforecast import StatsForecast

    factory = _build_factory()
    if model_name not in factory:
        raise KeyError(f"Unknown forecast model: {model_name}; choose from {list(factory)}")

    sf = StatsForecast(models=[factory[model_name]], freq=FREQ, n_jobs=-1)
    t0 = time.perf_counter()
    fcst = sf.forecast(df=panel, h=horizon, level=level, fitted=True)
    fit_time = time.perf_counter() - t0

    try:
        fitted = sf.forecast_fitted_values()
    except Exception:
        fitted = pd.DataFrame()
    return fcst, fitted, fit_time


def detect_anomalies(panel_with_resid: pd.DataFrame, method: str) -> pd.Series:
    if method == "zscore_resid":
        out = pd.Series(0, index=panel_with_resid.index)
        for _, g in panel_with_resid.groupby("unique_id"):
            out.loc[g.index] = modified_zscore(g["residual"].fillna(0).values, k=3.0)
        return out
    if method == "isolation_forest":
        return isolation_forest_panel(panel_with_resid, contamination=0.05)
    if method == "lof":
        return lof_panel(panel_with_resid, contamination=0.05)
    if method == "matrix_profile":
        return matrix_profile_anomalies(panel_with_resid, window=DAILY_SEASON)
    raise KeyError(f"Unknown anomaly method: {method}")


def run_pipeline(
    forecast_model: str = "MSTL",
    anomaly_method: str = "zscore_resid",
    horizon: int | None = None,
    out_dir: str | Path = "data/processed",
) -> dict:
    from src.eval import evaluate_panel, make_metrics

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    windows, _ = load_anomaly_labels()
    h = horizon or DAILY_SEASON

    test_raw = panel.groupby("unique_id").tail(h)
    train = panel.drop(test_raw.index).reset_index(drop=True)
    test = test_raw.reset_index(drop=True)

    t0 = time.perf_counter()
    fcst, fitted, fit_time = build_forecast(train, horizon=h, model_name=forecast_model, level=[99])
    fcst_time = time.perf_counter() - t0 - fit_time
    eval_df = test.merge(fcst, on=["unique_id", "ds"])

    metrics, _ = make_metrics(seasonality=DAILY_SEASON)
    model_alias = next(
        c for c in fcst.columns
        if c not in ("unique_id", "ds") and "-lo-" not in c and "-hi-" not in c
    )
    metrics_df = evaluate_panel(
        eval_df=eval_df, train_df=train,
        model_names=[model_alias], metrics=metrics,
    )

    full = panel.copy()
    if not fitted.empty:
        full = full.merge(
            fitted[["unique_id", "ds", model_alias]].rename(columns={model_alias: "fit"}),
            on=["unique_id", "ds"], how="left",
        )
        full["fit"] = full["fit"].ffill()
        full["residual"] = full["y"] - full["fit"]
    else:
        full["residual"] = full["y"]

    t1 = time.perf_counter()
    anom_pred = detect_anomalies(full, anomaly_method)
    detect_time = time.perf_counter() - t1

    anom_true = make_anomaly_mask(full, windows)
    anom_scores = score_anomalies(anom_true.values, anom_pred.values)

    eval_df.to_parquet(out_dir / "pipeline_forecast.parquet", index=False)
    full.assign(anom_pred=anom_pred, anom_true=anom_true).to_parquet(
        out_dir / "pipeline_anomalies.parquet", index=False
    )
    metrics_df.to_csv(out_dir / "pipeline_forecast_metrics.csv")

    summary = {
        "forecast_model": forecast_model,
        "anomaly_method": anomaly_method,
        "horizon": h,
        "fit_time_s": round(fit_time, 3),
        "forecast_time_s": round(fcst_time, 3),
        "detect_time_s": round(detect_time, 3),
        "forecast_metrics_avg": metrics_df[model_alias].groupby(level=0).mean().to_dict(),
        "anomaly_scores": anom_scores,
        "panel_rows": int(len(panel)),
    }
    with open(out_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-model", default="MSTL",
                   choices=["SeasonalNaive", "MSTL", "AutoETS", "TBATS"])
    p.add_argument("--anomaly-method", default="zscore_resid",
                   choices=["zscore_resid", "isolation_forest", "lof", "matrix_profile"])
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()

    summary = run_pipeline(
        forecast_model=args.forecast_model,
        anomaly_method=args.anomaly_method,
        horizon=args.horizon,
        out_dir=args.out_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
