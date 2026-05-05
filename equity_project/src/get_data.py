import warnings
from pathlib import Path
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from equity_project.src.utils import load_config, save_dict, three_barrier

warnings.filterwarnings("ignore")

project_path = Path(__file__).parent.parent

YAHOO_TICKER_MAPPING = {
    "BF.B": "BF-B",
    "BRK.B": "BRK-B",
}

MIN_VALID_CLOSE_DAYS = 252
FEATURE_LOOKBACK_DAYS = 370

# Triple-barrier labeling parameters (kept here so you can tweak without touching utils).
LABEL_HORIZON_DAYS = 10
LABEL_ROLLING_N = 50
LABEL_SCALING_FACTOR = 1.5
LABEL_MIN_TRGT = 0.008
LABEL_MAX_TRGT = 0.06
LABEL_NEUTRAL_THRESHOLD_MULT = 0.25


def normalize_ticker(ticker: str) -> str:
    """Convert raw ticker to Yahoo-compatible format."""
    ticker = ticker.strip()
    return YAHOO_TICKER_MAPPING.get(ticker, ticker)


def parse_components(value: str) -> Set[str]:
    """Parse comma-separated S&P500 components."""
    if pd.isna(value):
        return set()
    return {normalize_ticker(ticker) for ticker in str(value).split(",")}


def load_historical_components(path: Path, start_date: str, end_date: str) -> pd.Series:
    """Load historical S&P500 components as sets of active tickers by date."""
    components = pd.read_csv(path, index_col=0)
    components.index = pd.to_datetime(components.index)
    components = components.sort_index()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    components = components[components.index <= end_date]
    parsed = components.iloc[:, 0].apply(parse_components)

    before_start = parsed[parsed.index <= start_date]
    after_start = parsed[parsed.index > start_date]

    if len(before_start) > 0:
        parsed = pd.concat([before_start.iloc[[-1]], after_start])
    else:
        parsed = after_start

    return parsed


def build_membership_mask(
    trading_dates: pd.DatetimeIndex,
    historical_components: pd.Series,
    tickers: List[str],
) -> pd.DataFrame:
    """Build daily S&P500 membership mask."""
    tickers = sorted(tickers)
    components_daily = historical_components.reindex(trading_dates, method="ffill")

    mask = pd.DataFrame(False, index=trading_dates, columns=tickers)

    for date, active_tickers in components_daily.items():
        if isinstance(active_tickers, set):
            active = list(set(tickers) & active_tickers)
            if active:
                mask.loc[date, active] = True

    return mask


def apply_membership_mask(data: pd.DataFrame, membership_mask: pd.DataFrame) -> pd.DataFrame:
    """Set OHLCV values to NaN when ticker was not in S&P500."""
    masked = data.copy()
    idx = pd.IndexSlice

    for ticker in membership_mask.columns:
        if ticker in masked["Close"].columns:
            masked.loc[~membership_mask[ticker], idx[:, ticker]] = np.nan

    return masked


def filter_tickers_by_data_availability(
    data: pd.DataFrame,
    min_valid_close_days: int = MIN_VALID_CLOSE_DAYS,
) -> Tuple[List[str], Dict[str, int]]:
    """Remove tickers with too few valid Close observations."""
    close = data["Close"]

    valid_tickers = []
    excluded = {}

    for ticker in close.columns:
        n_valid = int(close[ticker].notna().sum())

        if n_valid >= min_valid_close_days:
            valid_tickers.append(ticker)
        else:
            excluded[ticker] = n_valid

    return sorted(valid_tickers), excluded


def generate_features(data: pd.DataFrame) -> pd.DataFrame:
    """Generate cross-sectional and time-series ML features from OHLCV data."""
    t0 = time.perf_counter()

    def log(step: str) -> None:
        elapsed = time.perf_counter() - t0
        print(f"[features] {step} (elapsed: {elapsed:.1f}s)", flush=True)

    log(f"start; input shape={data.shape}, tickers={data['Close'].shape[1]}")
    x = data.copy()

    close = x["Close"]
    open_ = x["Open"]
    high = x["High"]
    low = x["Low"]
    volume = x["Volume"]

    tickers = close.columns

    log("compute returns")

    ret1 = close.pct_change(1, fill_method=None)
    ret5 = close.pct_change(5, fill_method=None)
    ret10 = close.pct_change(10, fill_method=None)
    ret22 = close.pct_change(22, fill_method=None)
    ret63 = close.pct_change(63, fill_method=None)
    ret126 = close.pct_change(126, fill_method=None)
    ret252 = close.pct_change(252, fill_method=None)

    market_ret = ret1.mean(axis=1)

    def add_feature(name: str, values: pd.DataFrame) -> None:
        x[[(name, ticker) for ticker in tickers]] = values

    # Returns / momentum.
    log("add return features")
    add_feature("ret1", ret1)
    add_feature("ret5", ret5)
    add_feature("ret10", ret10)
    add_feature("ret22", ret22)
    add_feature("ret63", ret63)
    add_feature("ret126", ret126)
    add_feature("ret252", ret252)

    log("add momentum ranks")
    add_feature("mom5_rank", ret5.rank(axis=1, pct=True))
    add_feature("mom22_rank", ret22.rank(axis=1, pct=True))
    add_feature("mom63_rank", ret63.rank(axis=1, pct=True))
    add_feature("mom252_rank", ret252.rank(axis=1, pct=True))

    # Relative returns versus equal-weight S&P500 universe proxy.
    log("add relative returns")
    add_feature("rel_ret5", ret5.sub(market_ret.rolling(5).sum(), axis=0))
    add_feature("rel_ret22", ret22.sub(market_ret.rolling(22).sum(), axis=0))
    add_feature("rel_ret63", ret63.sub(market_ret.rolling(63).sum(), axis=0))

    # Moving average deviations.
    log("add moving-average deviations")
    add_feature("dev5", (close - close.rolling(5).mean()) / close)
    add_feature("dev22", (close - close.rolling(22).mean()) / close)
    add_feature("dev63", (close - close.rolling(63).mean()) / close)
    add_feature("dev252", (close - close.rolling(252).mean()) / close)

    add_feature(
        "ma200vs50",
        (close.rolling(200).mean() - close.rolling(50).mean()) / close,
    )

    # Volatility.
    log("compute/add vol features")
    vol5 = ret1.rolling(5).std()
    vol22 = ret1.rolling(22).std()
    vol63 = ret1.rolling(63).std()
    vol252 = ret1.rolling(252).std()

    add_feature("vol5", vol5)
    add_feature("vol22", vol22)
    add_feature("vol63", vol63)
    add_feature("vol252", vol252)
    add_feature("vol22_rank", vol22.rank(axis=1, pct=True))
    add_feature("vol63_rank", vol63.rank(axis=1, pct=True))

    # Rolling Sharpe-like features.
    log("add sharpe-like features")
    add_feature("sharpe22", ret1.rolling(22).mean() / vol22)
    add_feature("sharpe63", ret1.rolling(63).mean() / vol63)

    # Market beta.
    log("compute/add beta63")
    market_var63 = market_ret.rolling(63).var()
    beta63 = ret1.rolling(63).cov(market_ret).div(market_var63, axis=0)
    add_feature("beta63", beta63)

    # Intraday / range features.
    log("add intraday/range features")
    add_feature("open_to_close", close / open_ - 1)
    add_feature("high_low_range", high / low - 1)

    # Volume / liquidity.
    log("add volume/liquidity features")
    dollar_volume = close * volume
    add_feature("log_dollar_volume", np.log1p(dollar_volume))
    add_feature("dollar_volume_rank", dollar_volume.rank(axis=1, pct=True))
    add_feature("volume_change22", volume / volume.rolling(22).mean() - 1)

    log("drop raw OHLCV columns")
    x = x.drop(columns=["Close", "High", "Low", "Open", "Volume"], errors="ignore")

    # Remove initial rows with NaNs.
    log("cut cold start rows")
    x = x.iloc[260:, :]

    log(f"done; output shape={x.shape}")

    return x


def get_label(data: pd.DataFrame) -> pd.DataFrame:
    """Generate triple-barrier labels."""
    return three_barrier(
        data["Close"],
        rolling_n=LABEL_ROLLING_N,
        scaling_factor=LABEL_SCALING_FACTOR,
        horizon=LABEL_HORIZON_DAYS,
        min_trgt=LABEL_MIN_TRGT,
        max_trgt=LABEL_MAX_TRGT,
        neutral_threshold_mult=LABEL_NEUTRAL_THRESHOLD_MULT,
        progress=True,
        progress_step_pct=5.0,
    )


def _print_label_distribution(y: pd.Series, name: str) -> None:
    vc = y.value_counts(dropna=False).sort_index()
    n = int(len(y))
    print(f"[labels] {name}: n={n}")
    for k, v in vc.items():
        pct = float(v) / n if n else 0.0
        print(f"[labels]   {k}: {int(v)} ({pct:.2%})")


def load_cached_raw_data_if_available() -> Tuple[pd.DataFrame, pd.DataFrame] | None:
    """Load cached raw parquets to avoid re-downloading from Yahoo.

    Returns:
        (data, membership_mask) if cache exists, else None.
    """
    raw_dir = project_path / "data" / "raw"
    train_path = raw_dir / "train_data.parquet"
    backtest_path = raw_dir / "backtest_data.parquet"

    if not train_path.exists() or not backtest_path.exists():
        return None

    train_data = pd.read_parquet(train_path, engine="pyarrow")
    backtest_data = pd.read_parquet(backtest_path, engine="pyarrow")

    data = pd.concat([train_data, backtest_data], axis=0)
    data = data[~data.index.duplicated(keep="last")].sort_index()

    membership_mask = data["Close"].notna()
    return data, membership_mask


def get_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Download Yahoo data, apply S&P500 membership mask and filter low-quality tickers."""
    cfg = load_config((project_path.parent / "config.yaml").as_posix())

    train_start_date = pd.to_datetime(cfg["train_start_date"])
    backtest_end_date = pd.to_datetime(cfg["backtest_end_date"])
    download_start_date = train_start_date - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)

    pony_dir = project_path / "data" / "pony"
    pony_dir.mkdir(parents=True, exist_ok=True)

    components_path = pony_dir / "S&P_500_Historical_Components.csv"

    historical_components = load_historical_components(
        path=components_path,
        start_date=download_start_date,
        end_date=backtest_end_date,
    )

    all_tickers = sorted(set().union(*historical_components.values))

    print(f"Тикеров в историческом составе S&P500: {len(all_tickers)}")
    print(f"Скачиваем период: {download_start_date.date()} — {backtest_end_date.date()}")

    data = yf.download(
        all_tickers,
        start=download_start_date,
        end=backtest_end_date,
        group_by="column",
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    if data.empty:
        raise ValueError("Yahoo Finance не вернул данные.")

    if "Adj Close" in data.columns:
        data = data.drop(columns="Adj Close")

    data.index = pd.to_datetime(data.index)
    data = data.astype(float)

    yahoo_downloaded_tickers = sorted(
        ticker for ticker in data["Close"].columns if not data["Close"][ticker].isna().all()
    )

    yahoo_missing_tickers = sorted(set(all_tickers) - set(yahoo_downloaded_tickers))

    data = data.loc[:, pd.IndexSlice[:, yahoo_downloaded_tickers]]

    membership_mask = build_membership_mask(
        trading_dates=data.index,
        historical_components=historical_components,
        tickers=yahoo_downloaded_tickers,
    )

    data = apply_membership_mask(data, membership_mask)

    valid_tickers, excluded_low_data_tickers = filter_tickers_by_data_availability(data)

    data = data.loc[:, pd.IndexSlice[:, valid_tickers]]
    membership_mask = membership_mask[valid_tickers]

    report = {
        "download_start_date": str(download_start_date.date()),
        "train_start_date": str(train_start_date.date()),
        "backtest_end_date": str(backtest_end_date.date()),
        "n_tickers_in_historical_components": len(all_tickers),
        "n_yahoo_downloaded_tickers": len(yahoo_downloaded_tickers),
        "n_yahoo_missing_tickers": len(yahoo_missing_tickers),
        "yahoo_missing_tickers": yahoo_missing_tickers,
        "min_valid_close_days": MIN_VALID_CLOSE_DAYS,
        "n_excluded_low_data_tickers": len(excluded_low_data_tickers),
        "excluded_low_data_tickers": excluded_low_data_tickers,
        "n_final_tickers": len(valid_tickers),
        "final_tickers": valid_tickers,
        "note": (
            "Only Yahoo Finance is used. OHLCV values are set to NaN outside actual "
            "S&P500 membership dates. Tickers missing in Yahoo or having too few "
            "valid observations during membership are excluded."
        ),
    }

    save_dict(report, (pony_dir / "data_quality_report.json").as_posix())

    print(f"Yahoo скачал тикеров: {len(yahoo_downloaded_tickers)}")
    print(f"Yahoo не скачал тикеров: {len(yahoo_missing_tickers)}")
    print(f"Исключено по минимуму данных: {len(excluded_low_data_tickers)}")
    print(f"Финальное число тикеров: {len(valid_tickers)}")

    return data, membership_mask, report


def get_data() -> None:
    """Download data, generate features/labels and save train/backtest datasets."""
    cfg = load_config((project_path.parent / "config.yaml").as_posix())

    train_start_date = pd.to_datetime(cfg["train_start_date"])
    train_end_date = pd.to_datetime(cfg["train_end_date"])
    backtest_start_date = pd.to_datetime(cfg["backtest_start_date"])
    backtest_end_date = pd.to_datetime(cfg["backtest_end_date"])

    raw_dir = project_path / "data" / "raw"
    processed_dir = project_path / "data" / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cached = load_cached_raw_data_if_available()
    if cached is not None:
        print("[pipeline] using cached raw parquets (skip Yahoo download)", flush=True)
        data, membership_mask = cached
    else:
        data, membership_mask, _ = get_raw_data()

    print("[pipeline] start feature generation", flush=True)
    x = generate_features(data)
    print("[pipeline] features ready", flush=True)

    print("[pipeline] start label generation", flush=True)
    y = get_label(data)
    print("[pipeline] labels ready", flush=True)

    x = x.stack(level=1)
    x.index.names = ["Date", "Ticker"]

    y = y.stack(level=0)
    y.index.names = ["Date", "Ticker"]
    y.name = "target"

    active_mask = membership_mask.stack()
    active_mask.index.names = ["Date", "Ticker"]

    common_index = x.index.intersection(y.index).intersection(active_mask.index)

    x = x.loc[common_index]
    y = y.loc[common_index]
    active_mask = active_mask.loc[common_index]

    active_index = active_mask[active_mask].index

    x = x.loc[active_index]
    y = y.loc[active_index]

    _print_label_distribution(y, name="full")

    tickers_to_keep = sorted(x.index.get_level_values("Ticker").unique())
    data = data.loc[:, pd.IndexSlice[:, tickers_to_keep]]

    train_data = data.loc[
        (data.index >= train_start_date) & (data.index <= train_end_date)
    ]

    backtest_data = data.loc[
        (data.index >= backtest_start_date) & (data.index <= backtest_end_date)
    ]

    x_train = x.loc[
        (x.index.get_level_values("Date") >= train_start_date)
        & (x.index.get_level_values("Date") <= train_end_date)
    ]

    y_train = y.to_frame().loc[
        (y.index.get_level_values("Date") >= train_start_date)
        & (y.index.get_level_values("Date") <= train_end_date)
    ]

    _print_label_distribution(y_train["target"], name="train")

    x_backtest = x.loc[
        (x.index.get_level_values("Date") >= backtest_start_date)
        & (x.index.get_level_values("Date") <= backtest_end_date)
    ]

    y_backtest = y.to_frame().loc[
        (y.index.get_level_values("Date") >= backtest_start_date)
        & (y.index.get_level_values("Date") <= backtest_end_date)
    ]

    _print_label_distribution(y_backtest["target"], name="backtest")

    train_data.to_parquet(raw_dir / "train_data.parquet", engine="pyarrow")
    backtest_data.to_parquet(raw_dir / "backtest_data.parquet", engine="pyarrow")

    x_train.to_parquet(processed_dir / "X_train.parquet", engine="pyarrow")
    y_train.to_parquet(processed_dir / "y_train.parquet", engine="pyarrow")

    x_backtest.to_parquet(processed_dir / "X_backtest.parquet", engine="pyarrow")
    y_backtest.to_parquet(processed_dir / "y_backtest.parquet", engine="pyarrow")

    print("Данные успешно сохранены.")
    print(f"train_data: {train_data.shape}")
    print(f"backtest_data: {backtest_data.shape}")
    print(f"X_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_backtest: {x_backtest.shape}")
    print(f"y_backtest: {y_backtest.shape}")


if __name__ == "__main__":
    get_data()
