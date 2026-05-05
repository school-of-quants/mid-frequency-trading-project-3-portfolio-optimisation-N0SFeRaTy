import datetime as dt
import json
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from yaml import safe_load


def load_config(config_path: str) -> Dict:
    """Загружает yaml конфиг в виде python словаря

    Args:
        config_path (str): Путь до конфига

    Returns:
        Dict: Словарь с параметрами конфига
    """
    with open(config_path) as file:
        config = safe_load(file)
    return config


def save_dict(dict_, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(dict_, f, indent=4, default=str)


def applyPtSlOnT1(close, events, ptSl):
    """Tripple barrier method adjusted for Low and High prices

    Args:
        close (pd.Series): Close prices.
        events (pd.DataFrame): A pandas dataframe, with columns:
            - t1: The timestamp of vertical barrier. When the value is np.nan, there will not be a vertical barrier.
            - trgt: The unit width of the horizontal barriers.
        ptSl (list):
            - ptSl[0]: The factor that multiplies trgt to set the width of the upper barrier.
                If 0, there will not be an upper barrier.
            - ptSl[1]: The factor that multiplies trgt to set the width of the lower barrier.
                If 0, there will not be a lower barrier.

    Returns:
        pd.DataFrame: Timestamps of each barrier touch
    """

    # apply stop loss/profit taking, if it takes place before t1 (end of event)

    out = events[["t1"]].copy(deep=True)
    if ptSl[0] > 0:
        pt = ptSl[0] * events["trgt"]
    else:
        pt = pd.Series(index=events.index)  # NaNs
    if ptSl[1] > 0:
        sl = -ptSl[1] * events["trgt"]
    else:
        sl = pd.Series(index=events.index)  # NaNs
    for loc, t1 in events["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]  # path prices
        df0 = df0 / close[loc] - 1  # path returns
        out.loc[loc, "sl"] = df0[df0 < sl[loc]].index.min()  # earliest stop loss.
        out.loc[loc, "pt"] = df0[df0 > pt[loc]].index.min()  # earliest profit taking.
    return out


def three_barrier(
    close,
    ptSl=[1, 1],
    rolling_n=50,
    scaling_factor=2.0,
    horizon: int = 10,
    min_trgt: float = 0.01,
    max_trgt: float = 0.08,
    neutral_threshold_mult: float = 0.25,
    progress: bool = False,
    progress_step_pct: float = 5.0,
):
    """Labeling based on triple-barrier method.

    This is a performance-oriented implementation. It avoids per-timestamp pandas
    slicing (which is very slow) and instead works on numpy arrays with index
    positions.

        Notes:
                - Horizontal barrier width is volatility-scaled:
                    trgt[t] = clip(scaling_factor * rolling_std(returns, rolling_n), min_trgt, max_trgt)
                - Vertical barrier is set in trading days: +horizon rows forward.

    Args:
        close (pd.Series | pd.DataFrame): Close prices.
        ptSl (list, optional): applyPtSlOnT1 ptSl parameter. Defaults to [1, 1].
        rolling_n (int, optional): Rolling window to calculate standart deviation. Defaults to 50.
        scaling_factor (float, optional): Multiplier of standart deviation to calculate horizontal barrier size. Defaults to 2.0.
                horizon (int, optional): Vertical barrier in trading days. Defaults to 10.
                min_trgt (float, optional): Minimum horizontal barrier width (return space). Defaults to 0.01.
                max_trgt (float, optional): Maximum horizontal barrier width (return space). Defaults to 0.08.
                neutral_threshold_mult (float, optional): If no barrier is hit by horizon, label is:
                    - 0 if |return_horizon| < neutral_threshold_mult * trgt
                    - sign(return_horizon) otherwise
                    Defaults to 0.25.

    Returns:
        pd.Series | pd.DataFrame: 3-class labels (1, 0, -1).
    """

    total_iters: Optional[int] = None
    done_iters = 0
    next_report_at: Optional[int] = None

    def _init_progress(total: int, n_dates: int, n_tickers: int) -> None:
        nonlocal total_iters, done_iters, next_report_at
        total_iters = int(total)
        done_iters = 0

        step = float(progress_step_pct)
        if step <= 0:
            step = 5.0

        next_report_at = max(1, int(np.ceil(total_iters * (step / 100.0))))
        print(
            f"[three_barrier] start; dates={n_dates}, tickers={n_tickers}, total_iters={total_iters}",
            flush=True,
        )
        print("[three_barrier] 0%", flush=True)

    def _tick_progress(increment: int = 1) -> None:
        nonlocal done_iters, next_report_at
        if total_iters is None or next_report_at is None:
            return

        done_iters += int(increment)

        if done_iters >= next_report_at:
            pct = min(100.0, 100.0 * done_iters / total_iters)
            print(f"[three_barrier] {pct:.1f}%", flush=True)

            step = float(progress_step_pct)
            if step <= 0:
                step = 5.0
            next_report_at = done_iters + max(1, int(np.ceil(total_iters * (step / 100.0))))

    def _finish_progress() -> None:
        if total_iters is None:
            return
        print("[three_barrier] 100%", flush=True)
        print("[three_barrier] done", flush=True)

    def _label_series(series: pd.Series, *, with_progress: bool) -> pd.Series:
        series = series.astype(float)
        index = series.index
        values = series.to_numpy(copy=False)
        n = len(values)

        if n == 0:
            return pd.Series(dtype=int, index=index)

        # Vertical barrier positions for each timestamp: +horizon trading days.
        h = int(horizon)
        if h < 1:
            h = 1
        t1_pos = np.minimum(np.arange(n, dtype=np.int64) + h, n - 1)

        # Volatility-scaled horizontal barrier width.
        # Use pct_change(fill_method=None) semantics: no forward-fill of NaNs.
        # We compute on numpy arrays for speed.
        rets1 = np.empty(n, dtype=np.float64)
        rets1[:] = np.nan
        rets1[1:] = values[1:] / values[:-1] - 1.0

        roll = int(rolling_n)
        if roll < 2:
            roll = 2

        # Rolling std via pandas for correctness on NaNs, then back to numpy.
        vol = pd.Series(rets1, index=index).rolling(roll, min_periods=roll).std(ddof=0).to_numpy()
        trgt = scaling_factor * vol
        trgt = np.clip(trgt, float(min_trgt), float(max_trgt))

        pt_mult = float(ptSl[0]) if len(ptSl) > 0 else 1.0
        sl_mult = float(ptSl[1]) if len(ptSl) > 1 else 1.0

        labels = np.full(n, np.nan, dtype=np.float32)

        for i in range(n):
            base = values[i]
            if not np.isfinite(base) or base == 0.0:
                labels[i] = np.nan
                continue

            # Need a full horizon window; otherwise label is undefined.
            if i + h >= n:
                labels[i] = np.nan
                continue

            t = trgt[i]
            if not np.isfinite(t) or t <= 0:
                labels[i] = np.nan
                continue

            pt_level: Optional[float] = (pt_mult * t) if pt_mult > 0 else None
            sl_level: Optional[float] = (-sl_mult * t) if sl_mult > 0 else None

            j = int(t1_pos[i])
            if j < i:
                j = i

            window = values[i : j + 1]

            # If we don't have a finite window, skip. This avoids injecting noisy 0 labels
            # from missing prices (e.g., around membership gaps).
            if not np.isfinite(window).all():
                labels[i] = np.nan
                if with_progress:
                    _tick_progress(1)
                continue

            # If window has NaNs, comparisons will be False
            rets = window / base - 1.0

            pt_hit = None
            if pt_level is not None:
                pt_idx = np.flatnonzero(rets > pt_level)
                if pt_idx.size:
                    pt_hit = int(pt_idx[0])

            sl_hit = None
            if sl_level is not None:
                sl_idx = np.flatnonzero(rets < sl_level)
                if sl_idx.size:
                    sl_hit = int(sl_idx[0])

            if pt_hit is None and sl_hit is None:
                # If no horizontal barrier was hit, use the sign of the horizon return.
                ret_end = float(rets[-1])
                thr = float(neutral_threshold_mult) * float(t)
                if not np.isfinite(ret_end):
                    labels[i] = np.nan
                elif abs(ret_end) < thr:
                    labels[i] = 0
                else:
                    labels[i] = 1 if ret_end > 0 else -1
            elif sl_hit is None or (pt_hit is not None and pt_hit < sl_hit):
                labels[i] = 1
            else:
                labels[i] = -1

            if with_progress:
                _tick_progress(1)

        return pd.Series(labels, index=index, dtype="float").astype("Int64")

    if isinstance(close, pd.Series):
        if progress:
            _init_progress(total=len(close), n_dates=len(close), n_tickers=1)
        result = _label_series(close, with_progress=progress)
        if progress:
            _finish_progress()
        return result

    if isinstance(close, pd.DataFrame):
        n_dates = len(close.index)
        n_tickers = len(close.columns)
        if progress:
            _init_progress(total=n_dates * n_tickers, n_dates=n_dates, n_tickers=n_tickers)

        out = {}
        for col in close.columns:
            out[col] = _label_series(close[col], with_progress=progress)

        result = pd.DataFrame(out, index=close.index)
        if progress:
            _finish_progress()
        return result

    raise TypeError("close must be a pandas Series or DataFrame")
