"""Extended evaluation metrics for direction model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ── Calibration ───────────────────────────────────────────────────────────────

def calibration_metrics(
    y_true: pd.Series,
    proba: pd.Series,
    n_bins: int = 10,
) -> dict:
    """Reliability diagram data and summary calibration statistics.

    Returns ECE, mean/std of predicted probabilities, coverage above common
    thresholds, and a per-bin table of (mean_predicted, actual_freq, n).
    """
    p = proba.values
    y = y_true.values

    # Expected Calibration Error — equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_num = 0.0
    bin_rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() == 0:
            continue
        mean_pred = float(p[mask].mean())
        actual_freq = float(y[mask].mean())
        n = int(mask.sum())
        ece_num += n * abs(mean_pred - actual_freq)
        bin_rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "mean_pred": round(mean_pred, 4),
            "actual_freq": round(actual_freq, 4),
            "gap": round(mean_pred - actual_freq, 4),
            "n": n,
        })
    ece = ece_num / len(p)

    return {
        "ece": round(ece, 5),
        "mean_proba": round(float(p.mean()), 5),
        "std_proba": round(float(p.std()), 5),
        "pct_above_52": round(float((p >= 0.52).mean() * 100), 2),
        "pct_above_55": round(float((p >= 0.55).mean() * 100), 2),
        "pct_above_60": round(float((p >= 0.60).mean() * 100), 2),
        "bin_table": pd.DataFrame(bin_rows),
    }


# ── Signal quality ────────────────────────────────────────────────────────────

def spearman_ic(y_true: pd.Series, proba: pd.Series) -> tuple[float, float]:
    """Spearman IC between predicted probability and realised direction."""
    ic, p = stats.spearmanr(proba, y_true)
    return float(ic), float(p)


def brier_score(y_true: pd.Series, proba: pd.Series) -> float:
    """Mean squared error between predicted probability and binary outcome.
    Baseline (always predict 0.5) = 0.25; lower is better.
    """
    return float(np.mean((proba.values - y_true.values) ** 2))


def precision_at_thresholds(
    y_true: pd.Series,
    proba: pd.Series,
    thresholds: tuple[float, ...] = (0.52, 0.55, 0.58, 0.60),
) -> pd.DataFrame:
    """Hit rate and coverage when model confidence exceeds each threshold."""
    rows = []
    base_rate = float(y_true.mean())
    for thr in thresholds:
        mask = proba >= thr
        n = int(mask.sum())
        hit = float(y_true[mask].mean()) if n > 0 else float("nan")
        rows.append({
            "threshold": thr,
            "n_predictions": n,
            "coverage_pct": round(n / len(y_true) * 100, 1),
            "hit_rate": round(hit, 4),
            "lift_vs_base": round(hit - base_rate, 4) if not np.isnan(hit) else float("nan"),
        })
    return pd.DataFrame(rows)


# ── Quintile analysis ─────────────────────────────────────────────────────────

def quintile_returns(
    df: pd.DataFrame,
    prob_col: str = "proba",
    return_col: str = "return_fwd_1d",
    n_bins: int = 5,
) -> pd.DataFrame:
    """Mean forward return per predicted-probability quintile.

    Monotonic increase from Q1 → Q5 indicates a useful ranking signal.
    """
    out = df[[prob_col, return_col]].copy()
    codes, bins = pd.qcut(out[prob_col], n_bins, labels=False, duplicates="drop", retbins=True)
    actual_bins = len(bins) - 1
    out["quintile"] = pd.Categorical(codes + 1, categories=list(range(1, actual_bins + 1)))
    summary = (
        out.groupby("quintile", observed=True)[return_col]
        .agg(mean_return="mean", n="count", std="std")
        .assign(
            mean_return=lambda x: x["mean_return"].round(6),
            t_stat=lambda x: (x["mean_return"] / (x["std"] / np.sqrt(x["n"]))).round(3),
        )
    )
    summary["spread_vs_q1"] = (summary["mean_return"] - summary["mean_return"].iloc[0]).round(6)
    return summary


# ── Long-short portfolio ──────────────────────────────────────────────────────

def long_short_metrics(
    df: pd.DataFrame,
    prob_col: str = "proba",
    return_col: str = "return_fwd_1d",
    top_n: int = 2,
    bottom_n: int = 2,
    periods_per_year: int = 252,
) -> dict:
    """Daily equal-weight long top-N / short bottom-N portfolio.

    Requires columns: date, ticker, prob_col, return_col.
    Returns annualised return, Sharpe, max drawdown, and the daily series.
    """
    daily: list[float] = []
    dates: list[pd.Timestamp] = []

    for date, group in df.groupby("date"):
        if len(group) < top_n + bottom_n:
            continue
        ranked = group.sort_values(prob_col)
        short_ret = ranked.iloc[:bottom_n][return_col].mean()
        long_ret = ranked.iloc[-top_n:][return_col].mean()
        daily.append(long_ret - short_ret)
        dates.append(date)

    if not daily:
        return {}

    series = pd.Series(daily, index=dates)
    ann_ret = float(series.mean() * periods_per_year)
    ann_vol = float(series.std() * np.sqrt(periods_per_year))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    cum = (1 + series).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    win_rate = float((series > 0).mean())
    t_stat = float(series.mean() / (series.std() / np.sqrt(len(series)))) if series.std() > 0 else float("nan")

    return {
        "ann_return": round(ann_ret, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(win_rate, 4),
        "t_stat": round(t_stat, 3),
        "n_days": len(series),
        "daily_series": series,
    }


# ── Rolling IC ────────────────────────────────────────────────────────────────

def rolling_ic_monthly(
    df: pd.DataFrame,
    prob_col: str = "proba",
    target_col: str = "direction_fwd_1d",
) -> pd.DataFrame:
    """Spearman IC computed per calendar month."""
    out = df.copy()
    out["month"] = out["date"].dt.to_period("M")
    rows = []
    for period, grp in out.groupby("month"):
        if len(grp) < 5:
            continue
        ic, p = stats.spearmanr(grp[prob_col], grp[target_col])
        rows.append({"month": str(period), "ic": round(float(ic), 4), "p": round(float(p), 4), "n": len(grp)})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["ic_positive"] = result["ic"] > 0
    return result


# ── Conditional performance ───────────────────────────────────────────────────

def conditional_ic(
    df: pd.DataFrame,
    prob_col: str = "proba",
    target_col: str = "direction_fwd_1d",
    mention_col: str = "mentions_abnormal",
    volatility_col: str = "volatility_20d",
) -> pd.DataFrame:
    """IC split by mention regime and volatility regime."""
    rows = []

    def _ic_row(label: str, mask: pd.Series) -> dict:
        sub = df[mask]
        if len(sub) < 10:
            return {}
        ic, p = stats.spearmanr(sub[prob_col], sub[target_col])
        return {"condition": label, "ic": round(float(ic), 4), "p": round(float(p), 4), "n": len(sub)}

    med_abn = df[mention_col].median()
    rows.append(_ic_row("mentions_high", df[mention_col] > med_abn))
    rows.append(_ic_row("mentions_low", df[mention_col] <= med_abn))

    med_vol = df[volatility_col].median()
    rows.append(_ic_row("vol_high", df[volatility_col] > med_vol))
    rows.append(_ic_row("vol_low", df[volatility_col] <= med_vol))

    return pd.DataFrame([r for r in rows if r])
