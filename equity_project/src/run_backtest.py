import os
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt

from equity_project.src.utils import load_config, save_dict

project_path = Path(__file__).parent.parent

TOP_N = 35
MIN_SCORE = 0.03
MAX_WEIGHT = 0.07
REBALANCE_EVERY_N_DAYS = 5
VOL_FEATURE_NAME = "vol22"
MIN_POSITIONS = 10
REBALANCE_BLEND_ALPHA = 0.3


def get_class_probability(preds: pd.DataFrame, class_label: int) -> pd.DataFrame:
    """Extract probability column for a given class label."""
    if class_label not in preds.columns:
        return pd.DataFrame(index=preds.index)
    return preds[class_label].unstack(level="Ticker")


def cap_and_normalize_weights(weights: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Cap single-name weights and normalize daily portfolio weights."""
    weights = weights.clip(upper=max_weight)
    row_sum = weights.sum(axis=1)
    weights = weights.div(row_sum.replace(0, np.nan), axis=0).fillna(0)
    return weights


def cap_and_normalize_targets(target_weights: pd.DataFrame, max_weight: float) -> pd.DataFrame:
    """Cap/normalize only on rebalance rows; keep NaNs elsewhere."""
    target_weights = target_weights.copy()
    rebalance_rows = target_weights.notna().any(axis=1)
    if not rebalance_rows.any():
        return target_weights

    tmp = target_weights.loc[rebalance_rows].fillna(0).clip(upper=max_weight)
    row_sum = tmp.sum(axis=1).replace(0, np.nan)
    tmp = tmp.div(row_sum, axis=0).fillna(0)
    target_weights.loc[rebalance_rows] = tmp
    return target_weights


def blend_rebalance_targets(target_weights: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Reduce turnover by blending new targets with previous rebalance targets.

    alpha=1.0 means "use full new target".
    Smaller alpha makes rebalancing more gradual and can reduce fees.
    """
    if alpha is None or alpha >= 1.0:
        return target_weights
    if alpha <= 0:
        raise ValueError("alpha must be in (0, 1].")

    out = target_weights.copy()
    rebalance_dates = out.index[out.notna().any(axis=1)]

    prev = None
    for dt in rebalance_dates:
        row = out.loc[dt].fillna(0)
        if prev is None:
            blended = row
        else:
            blended = (1 - alpha) * prev + alpha * row
        out.loc[dt] = blended
        prev = blended

    return out


def keep_top_holdings_on_rebalance(
    target_weights: pd.DataFrame,
    max_holdings: int,
) -> pd.DataFrame:
    """On rebalance rows, keep only the largest weights and drop the rest."""
    if max_holdings is None or max_holdings <= 0:
        return target_weights

    out = target_weights.copy()
    rebalance_dates = out.index[out.notna().any(axis=1)]

    for dt in rebalance_dates:
        row = out.loc[dt].fillna(0)
        if row.gt(0).sum() <= max_holdings:
            continue
        keep = row.nlargest(max_holdings).index
        row.loc[~row.index.isin(keep)] = 0.0
        out.loc[dt] = row

    return out


def apply_rebalance_frequency(weights: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Keep target weights only every n trading days.

    Important: We return NaNs on non-rebalance days so the backtest engine
    doesn't attempt to rebalance daily just to maintain exact target percents.
    """
    if n_days <= 1:
        return weights

    rebalanced = weights.copy()
    rebalanced.iloc[:, :] = np.nan
    rebalanced.iloc[::n_days, :] = weights.iloc[::n_days, :]
    return rebalanced


def generate_weights(preds: pd.DataFrame, x_backtest: pd.DataFrame) -> pd.DataFrame:
    """Convert class probabilities into risk-adjusted long-only portfolio weights.

    Strategy:
        score = P(up) - P(down)
        keep only positive high-confidence scores
        select top-N stocks daily
        scale by inverse volatility
        cap max weight
        rebalance weekly
    """
    p_down = get_class_probability(preds, -1)
    p_up = get_class_probability(preds, 1)

    # Base signal: model confidence spread.
    # Fill missing probabilities with 0 so we can fall back gracefully.
    score = (p_up - p_down).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Keep only positive-confidence names; if too few, fall back to equal-weight/risk-parity.
    score = score.where(score >= MIN_SCORE, 0)

    top_mask = score.rank(axis=1, ascending=False, method="first") <= TOP_N
    score = score.where(top_mask, 0)

    # Long-only: ignore negative/zero scores after filtering.
    score = score.where(score > 0, 0)

    n_pos = score.gt(0).sum(axis=1)
    need_fallback = n_pos < MIN_POSITIONS
    if need_fallback.any():
        # When signals are weak (or missing), stay invested via diversified fallback.
        score = score.copy()
        score.loc[need_fallback, :] = 1.0

    if VOL_FEATURE_NAME in x_backtest.columns:
        vol = x_backtest[VOL_FEATURE_NAME].unstack(level="Ticker")
        inv_vol = 1 / vol.replace(0, np.nan)
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        # If vol is missing for some names on some dates, use cross-sectional median.
        inv_vol = inv_vol.fillna(inv_vol.median(axis=1), axis=0).fillna(1.0)
        raw_weights = score * inv_vol
    else:
        raw_weights = score

    raw_weights = raw_weights.replace([np.inf, -np.inf], np.nan).fillna(0)

    weights = cap_and_normalize_weights(raw_weights, MAX_WEIGHT)
    target_weights = apply_rebalance_frequency(weights, REBALANCE_EVERY_N_DAYS)
    target_weights = blend_rebalance_targets(target_weights, REBALANCE_BLEND_ALPHA)
    target_weights = keep_top_holdings_on_rebalance(target_weights, TOP_N)
    target_weights = cap_and_normalize_targets(target_weights, MAX_WEIGHT)

    return target_weights


def calculate_performance_metrics(returns: pd.Series) -> Dict:
    """Calculate custom performance metrics."""
    returns = returns.dropna()

    if len(returns) == 0:
        return {}

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1
    annual_return = equity.iloc[-1] ** (252 / len(returns)) - 1

    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar),
    }


def calculate_ipc(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Calculate rolling intra-portfolio correlation."""
    returns = close.pct_change(fill_method=None)

    ipc_values = {}

    for date in weights.index:
        selected = weights.loc[date]
        selected = selected[selected > 0].index.tolist()

        if len(selected) < 2:
            ipc_values[date] = np.nan
            continue

        hist_returns = returns.loc[:date, selected].tail(window)

        if len(hist_returns) < 20:
            ipc_values[date] = np.nan
            continue

        corr = hist_returns.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        ipc_values[date] = upper.stack().mean()

    return pd.Series(ipc_values, name="ipc")


def calculate_equal_weight_benchmark(close: pd.DataFrame) -> pd.Series:
    """Build equal-weight benchmark from available active S&P500 universe."""
    returns = close.pct_change(fill_method=None)
    benchmark_returns = returns.mean(axis=1).fillna(0)
    return benchmark_returns


def run_backtest() -> None:
    """Run ML strategy backtest and save metrics, plots and diversification stats."""
    os.makedirs(project_path / "artifacts/plots", exist_ok=True)
    os.makedirs(project_path / "artifacts/metrics", exist_ok=True)

    cfg = load_config((project_path.parent / "config.yaml").as_posix())

    x_backtest = pd.read_parquet(project_path / "data/processed/X_backtest.parquet")
    backtest_data = pd.read_parquet(
        project_path / "data/raw/backtest_data.parquet",
        engine="pyarrow",
    )

    bundle = joblib.load(project_path / "models/model.joblib")
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    classes = bundle["classes"]

    x_backtest = x_backtest.replace([np.inf, -np.inf], np.nan)
    x_model = x_backtest[feature_columns].dropna()

    pred_array = model.predict_proba(x_model)
    preds = pd.DataFrame(pred_array, index=x_model.index, columns=classes)

    close = backtest_data["Close"].dropna(axis=1, how="all")
    open_ = backtest_data["Open"].dropna(axis=1, how="all")

    target_weights = generate_weights(preds, x_model)

    common_tickers = sorted(
        set(close.columns) & set(open_.columns) & set(target_weights.columns)
    )

    close = close[common_tickers]
    open_ = open_[common_tickers]
    target_weights = target_weights[common_tickers]

    # Signals at date t are executed at next trading day's open.
    price = open_.shift(-1)

    # Orders: targets only on rebalance dates; NaN elsewhere means "no new order".
    target_weights = target_weights.reindex(close.index)
    target_weights = target_weights.where(price.notna() & close.notna(), np.nan)

    # Diagnostics / IPC: approximate held weights by carrying the last target.
    held_weights = target_weights.ffill().fillna(0)
    held_weights = held_weights.where(price.notna() & close.notna(), 0)
    held_weights = cap_and_normalize_weights(held_weights, MAX_WEIGHT)

    init_cash = cfg["init_cash"]
    fees = cfg["fees"]

    pf = vbt.Portfolio.from_orders(
        close=close,
        price=price,
        size=target_weights,
        size_type="targetpercent",
        group_by=True,
        cash_sharing=True,
        freq="1d",
        init_cash=init_cash,
        fees=fees,
    )

    pf.plot().write_image(project_path / "artifacts/plots/pnl.png")

    strategy_returns = pf.returns()
    strategy_metrics = calculate_performance_metrics(strategy_returns)

    benchmark_returns = calculate_equal_weight_benchmark(close)
    benchmark_metrics = calculate_performance_metrics(benchmark_returns)

    ipc = calculate_ipc(close=close, weights=held_weights)
    ipc.to_frame().to_parquet(project_path / "artifacts/metrics/ipc.parquet")

    benchmark_ipc = calculate_ipc(
        close=close,
        weights=close.notna().astype(float).div(close.notna().sum(axis=1), axis=0).fillna(0),
    )

    vectorbt_stats = pf.stats().to_dict()

    backtest_metrics = {
        "strategy_metrics": strategy_metrics,
        "benchmark_equal_weight_sp500_like_metrics": benchmark_metrics,
        "strategy_minus_benchmark_sharpe": (
            strategy_metrics.get("sharpe_ratio", np.nan)
            - benchmark_metrics.get("sharpe_ratio", np.nan)
        ),
        "ipc_mean": float(ipc.mean()),
        "ipc_median": float(ipc.median()),
        "benchmark_ipc_mean": float(benchmark_ipc.mean()),
        "benchmark_ipc_median": float(benchmark_ipc.median()),
        "vectorbt_stats": vectorbt_stats,
        "strategy_parameters": {
            "top_n": TOP_N,
            "min_score": MIN_SCORE,
            "min_positions": MIN_POSITIONS,
            "max_weight": MAX_WEIGHT,
            "rebalance_every_n_days": REBALANCE_EVERY_N_DAYS,
            "vol_feature_name": VOL_FEATURE_NAME,
        },
    }

    save_dict(
        backtest_metrics,
        project_path / "artifacts/metrics/backtest_metrics.json",
    )

    print("Бэктест завершён.")
    print(backtest_metrics["strategy_metrics"])


if __name__ == "__main__":
    run_backtest()
