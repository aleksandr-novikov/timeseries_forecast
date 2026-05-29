from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd


def make_metrics(seasonality: int = 288):
    from utilsforecast.losses import mae, rmse, smape, mase

    metrics = [mae, rmse, smape, partial(mase, seasonality=seasonality)]
    return metrics, ["mae", "rmse", "smape", "mase"]


def make_probabilistic_metrics(seasonality: int = 288):
    from utilsforecast.losses import (
        mae, rmse, smape, mase, scaled_crps, coverage, scaled_quantile_loss,
    )

    metrics = [
        mae, rmse, smape,
        partial(mase, seasonality=seasonality),
        scaled_crps, coverage,
        partial(scaled_quantile_loss, seasonality=seasonality),
    ]
    return metrics, ["mae", "rmse", "smape", "mase", "crps", "coverage", "qloss"]


def evaluate_panel(
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    model_names: list[str],
    metrics,
    level: list[int] | None = None,
) -> pd.DataFrame:
    from utilsforecast.evaluation import evaluate

    df = evaluate(
        df=eval_df, train_df=train_df,
        models=model_names, metrics=metrics, level=level,
    )
    return df.set_index("metric")


def pivot_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.reset_index()
    melted = df.melt(id_vars=["metric", "unique_id"], var_name="model", value_name="value")
    return melted.pivot_table(index="unique_id", columns=["metric", "model"], values="value")


def residual_diagnostics(residuals: np.ndarray) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
    from scipy.stats import jarque_bera
    from statsmodels.tsa.stattools import kpss

    r = pd.Series(residuals).dropna()
    n = len(r)

    lb = acorr_ljungbox(r, lags=[min(20, n // 5)], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])

    x = np.column_stack([np.ones(n), np.arange(n)])
    try:
        bp_p = float(het_breuschpagan(r.values, x)[1])
    except Exception:
        bp_p = float("nan")

    _, jb_p = jarque_bera(r.values)

    try:
        _, kpss_p, *_ = kpss(r.values, regression="c", nlags="auto")
    except Exception:
        kpss_p = float("nan")

    return {
        "ljung_box_p": lb_p,
        "breusch_pagan_p": bp_p,
        "jarque_bera_p": float(jb_p),
        "kpss_p": float(kpss_p),
        "n": n,
    }


def plot_residual_diagnostics(residuals: np.ndarray, ax_array=None, title: str = ""):
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from scipy import stats as sps

    diag = residual_diagnostics(residuals)
    r = pd.Series(residuals).dropna()

    if ax_array is None:
        _, ax_array = plt.subplots(2, 2, figsize=(14, 7))

    ax_array[0, 0].plot(r.values, linewidth=0.6, color="steelblue")
    ax_array[0, 0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    ax_array[0, 0].set_title(f"Residuals (KPSS p={diag['kpss_p']:.3f})")

    plot_acf(r.values, lags=min(40, len(r) // 4), ax=ax_array[0, 1])
    ax_array[0, 1].set_title(f"ACF (Ljung-Box p={diag['ljung_box_p']:.3f})")

    ax_array[1, 0].hist(r.values, bins=50, density=True, alpha=0.7, color="steelblue")
    ax_array[1, 0].set_title(f"Histogram (Breusch-Pagan p={diag['breusch_pagan_p']:.3f})")

    sps.probplot(r.values, dist="norm", plot=ax_array[1, 1])
    ax_array[1, 1].set_title(f"Q-Q plot (Jarque-Bera p={diag['jarque_bera_p']:.3f})")

    if title:
        try:
            ax_array[0, 0].figure.suptitle(title, y=1.02, fontweight="bold")
        except Exception:
            pass
    return diag


def dm_test(
    y_true: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    h: int = 1,
    loss: str = "mae",
) -> dict:
    from scipy import stats as sps

    y_true = np.asarray(y_true, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)

    if loss == "mae":
        e1, e2 = np.abs(y_true - f1), np.abs(y_true - f2)
    elif loss == "mse":
        e1, e2 = (y_true - f1) ** 2, (y_true - f2) ** 2
    else:
        raise ValueError(f"Unknown loss: {loss}")

    d = e1 - e2
    n = len(d)
    mean_d = d.mean()

    lrv = ((d - mean_d) ** 2).mean()
    for k in range(1, h):
        cov = ((d[k:] - mean_d) * (d[:-k] - mean_d)).mean()
        lrv += 2 * cov

    dm = mean_d / np.sqrt(lrv / n)
    hln_factor = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln_factor
    p_value = 2 * (1 - sps.t.cdf(abs(dm_hln), df=n - 1))

    if dm_hln > 0 and p_value < 0.05:
        verdict = "f1 хуже"
    elif dm_hln < 0 and p_value < 0.05:
        verdict = "f2 хуже"
    else:
        verdict = "разница не значима"

    return {
        "dm_stat": float(dm),
        "dm_stat_hln": float(dm_hln),
        "p_value": float(p_value),
        "n": n,
        "mean_diff": float(mean_d),
        "verdict": verdict,
    }
