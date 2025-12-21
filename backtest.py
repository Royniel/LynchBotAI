# backtest.py
"""
Fully stable backtesting engine for Lynch Long/Short Strategy.

Features:
- Safe ticker downloading (drops missing / delisted / broken tickers)
- Safe benchmark downloading
- Equal weight & value-weighted portfolios
- Daily, weekly, monthly rebalancing
- Transaction cost modeling
- Rolling Sharpe
- Correlation matrix
- Fully Streamlit compatible
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Literal
import pandas as pd
import numpy as np
import yfinance as yf

Weighting = Literal["equal", "value"]
RebalanceFreq = Literal["daily", "weekly", "monthly"]


# =============================================================================
# CONFIG STRUCTURES
# =============================================================================

@dataclass
class BacktestConfig:
    tickers: List[str]
    long_list: List[str]
    short_list: List[str]
    start_date: str
    end_date: str
    weighting: Weighting = "equal"
    rebalance_freq: RebalanceFreq = "monthly"
    transaction_cost_bps: float = 10.0
    benchmark: str = "^GSPC"  # S&P 500 default


@dataclass
class BacktestResult:
    config: BacktestConfig
    price_df: pd.DataFrame
    asset_returns: pd.DataFrame
    portfolio_returns: pd.Series
    portfolio_equity: pd.Series
    benchmark_returns: Optional[pd.Series]
    benchmark_equity: Optional[pd.Series]
    weights: pd.DataFrame
    metrics: Dict[str, float]
    rolling_sharpe: pd.Series
    corr_matrix: pd.DataFrame


# =============================================================================
# SAFE PRICE DOWNLOAD
# =============================================================================

def _safe_download(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    """Download a single ticker safely."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )
        if df is None or df.empty or "Adj Close" not in df:
            print(f" ✗ Failed {ticker}")
            return None
        print(f" ✓ Loaded {ticker}")
        return df["Adj Close"]
    except Exception:
        print(f" ✗ Failed {ticker}")
        return None


def _download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """
    Safe, robust price download function.
    Drops tickers that fail; never crashes.
    """
    if not tickers:
        raise ValueError("No tickers provided for price download.")

    print(f"\nDownloading price data for: {tickers}\n")

    # Try bulk download first
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False
    )

    # If bulk works normally
    if raw is not None and not raw.empty and "Adj Close" in raw:
        prices = raw["Adj Close"]

        # If returned as Series → convert to DF
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Drop broken tickers
        bad = [c for c in prices.columns if prices[c].isna().all()]
        if bad:
            print(f" ⚠️ Dropping invalid tickers: {bad}")
            prices = prices.drop(columns=bad)

        if prices.empty:
            raise ValueError("All tickers invalid after cleanup.")

        return prices

    # Bulk failed → fallback to ticker-by-ticker
    print(" ⚠️ Bulk download failed. Trying individual tickers...\n")

    clean = {}
    for t in tickers:
        ser = _safe_download(t, start, end)
        if ser is not None:
            clean[t] = ser

    if not clean:
        raise ValueError("No valid price data found for any ticker.")

    return pd.DataFrame(clean)


# =============================================================================
# RETURNS
# =============================================================================

def _compute_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna(how="all")


# =============================================================================
# REBALANCING
# =============================================================================

def _get_rebalance_dates(index: pd.DatetimeIndex, freq: RebalanceFreq):
    if freq == "daily":
        return index

    if freq == "weekly":
        return index.to_series().resample("W-FRI").first().dropna().index

    if freq == "monthly":
        return index.to_series().resample("MS").first().dropna().index

    raise ValueError(f"Invalid rebalance freq: {freq}")


# =============================================================================
# WEIGHTS
# =============================================================================

def _compute_weights(
    all_assets: List[str],
    long_list: List[str],
    short_list: List[str],
    dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    weighting: Weighting,
    fin_df: Optional[pd.DataFrame]
) -> pd.DataFrame:

    assets = [a for a in all_assets if a in long_list or a in short_list]

    if not assets:
        raise ValueError("No overlap between selected tickers and price data.")

    weights = pd.DataFrame(0.0, index=dates, columns=assets)

    long_assets = [a for a in long_list if a in assets]
    short_assets = [a for a in short_list if a in assets]

    value_scores = None
    if weighting == "value" and fin_df is not None and "value_score" in fin_df.columns:
        value_scores = fin_df["value_score"].reindex(assets).fillna(0.0)

    prev = pd.Series(0.0, index=assets)

    for dt in dates:
        if dt in rebalance_dates:

            if weighting == "equal" or value_scores is None:
                w = pd.Series(0.0, index=assets)

                if long_assets:
                    lw = 0.5 / len(long_assets)
                    w[long_assets] = lw

                if short_assets:
                    sw = -0.5 / len(short_assets)
                    w[short_assets] = sw

                prev = w

            else:
                # Value-weighting logic
                w = pd.Series(0.0, index=assets)

                # Long side
                vl = value_scores.reindex(long_assets).clip(lower=0)
                if vl.sum() > 0:
                    w_long = 0.5 * (vl / vl.sum())
                else:
                    w_long = pd.Series(0.0, index=long_assets)

                # Short side
                vs = value_scores.reindex(short_assets).clip(lower=0)
                if (1 / vs.replace(0, np.nan)).sum(skipna=True) > 0:
                    inv = 1 / vs.replace(0, np.nan)
                    inv = inv.fillna(inv.mean())
                    w_short = -0.5 * (inv / inv.sum())
                else:
                    w_short = pd.Series(-0.5 / len(short_assets), index=short_assets)

                w[w_long.index] = w_long
                w[w_short.index] = w_short

                prev = w

        weights.loc[dt] = prev

    return weights


# =============================================================================
# TRANSACTION COSTS
# =============================================================================

def _compute_transaction_costs(weights: pd.DataFrame, bps: float) -> pd.Series:
    if bps <= 0:
        return pd.Series(0, index=weights.index)

    rate = bps / 10000
    turnover = weights.diff().abs().sum(axis=1)
    return turnover * rate


# =============================================================================
# METRICS
# =============================================================================

def _metrics(returns: pd.Series, equity: pd.Series):
    if returns.empty:
        return {}

    ann = 252
    total_return = float(equity.iloc[-1] - 1)
    years = len(returns) / ann
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    vol = float(returns.std() * np.sqrt(ann))
    sharpe = float((returns.mean() * ann) / (returns.std() + 1e-9))

    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1).min())

    return {
        "total_return": total_return,
        "CAGR": cagr,
        "annual_volatility": vol,
        "Sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


# =============================================================================
# MAIN BACKTEST ENGINE (BENCHMARK FIXED)
# =============================================================================

def run_long_short_backtest(config: BacktestConfig, fin_df: Optional[pd.DataFrame] = None) -> BacktestResult:

    all_tickers = list(set(config.tickers))

    # Add benchmark if provided
    if config.benchmark:
        all_tickers.append(config.benchmark)

    price_df = _download_prices(all_tickers, config.start_date, config.end_date)

    # Separate benchmark safely
    benchmark_returns = None
    benchmark_equity = None

    if config.benchmark in price_df.columns:
        bench_series = price_df[config.benchmark]
        benchmark_returns = bench_series.pct_change().fillna(0)
        benchmark_equity = (1 + benchmark_returns).cumprod()

        # Remove benchmark from asset universe
        asset_price_df = price_df.drop(columns=[config.benchmark])
    else:
        print(f" ⚠️ Benchmark '{config.benchmark}' unavailable. Continuing without it.")
        asset_price_df = price_df

    # Compute returns
    asset_returns = _compute_returns(asset_price_df)

    dates = asset_returns.index
    rebalance_dates = _get_rebalance_dates(dates, config.rebalance_freq)

    # Compute weights
    weights = _compute_weights(
        all_assets=list(asset_returns.columns),
        long_list=config.long_list,
        short_list=config.short_list,
        dates=dates,
        rebalance_dates=rebalance_dates,
        weighting=config.weighting,
        fin_df=fin_df
    )

    weights = weights.reindex(asset_returns.index).fillna(method="ffill").fillna(0)

    # Raw portfolio returns
    raw_returns = (weights * asset_returns).sum(axis=1)

    # Apply transaction costs
    costs = _compute_transaction_costs(weights, config.transaction_cost_bps)
    portfolio_returns = raw_returns - costs

    portfolio_equity = (1 + portfolio_returns).cumprod()

    metrics = _metrics(portfolio_returns, portfolio_equity)

    rolling_sharpe = (
        portfolio_returns.rolling(63).mean()
        / (portfolio_returns.rolling(63).std() + 1e-9)
        * np.sqrt(252)
    )

    corr_matrix = asset_returns.corr()

    return BacktestResult(
        config=config,
        price_df=price_df,
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        benchmark_returns=benchmark_returns,
        benchmark_equity=benchmark_equity,
        weights=weights,
        metrics=metrics,
        rolling_sharpe=rolling_sharpe,
        corr_matrix=corr_matrix,
    )
