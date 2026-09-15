# financial_data.py
import numpy as np
import pandas as pd
from yahooquery import Ticker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import streamlit as st

RATIO_DESCRIPTIONS = {
    "price": "Latest market price per share.",
    "pe_ratio": "Price-to-Earnings ratio — lower may mean cheaper valuation.",
    "pb_ratio": "Price-to-Book ratio — compares market cap to net assets.",
    "roe": "Return on Equity — profitability relative to shareholder equity.",
    "profit_margin": "Net profit margin — percentage of revenue converted to profit.",
    "debt_to_equity": "Debt-to-Equity — leverage level; lower is safer.",
    "dividend_yield": "Dividend yield — annual dividends as a % of share price.",
    "value_score": "Composite metric of valuation (P/E, P/B, dividend).",
    "quality_score": "Composite metric of business quality (ROE + margins).",
}

# Default Dow Jones tickers

@st.cache_resource
def load_tickers():
    return [
        "AAPL","MSFT","GS","JPM","AMGN","AXP","BA","CAT","CRM","CSCO",
        "CVX","DIS","DOW","HD","HON","IBM","INTC","JNJ","KO","MCD",
        "MMM","MRK","NKE","PG","TRV","UNH","V","VZ","WBA","WMT"
    ]


# Fetch financials using yahooquery
def safe_get(d, key, default=np.nan):
    if not isinstance(d, dict):
        # yahooquery returns a plain error string instead of a dict for bad tickers.
        return np.nan
    v = d.get(key)
    if v is None or isinstance(v, bool):
        # Fall back when the key is absent *or* present-but-null; dict.get's own
        # default only covers the absent case. bool is excluded because it is a
        # subclass of int and would otherwise coerce to 0.0/1.0.
        v = default
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return np.nan


def fetch_one_ticker(ticker):
    t = Ticker(ticker)

    summary = t.summary_detail.get(ticker, {})
    fin = t.financial_data.get(ticker, {})
    key = t.key_stats.get(ticker, {})
    price = t.price.get(ticker, {})

    return {
        "price": safe_get(price, "regularMarketPrice"),
        "pe_ratio": safe_get(summary, "trailingPE", safe_get(key, "trailingPE")),
        "pb_ratio": safe_get(summary, "priceToBook", safe_get(key, "priceToBook")),
        "roe": safe_get(fin, "returnOnEquity", safe_get(key, "returnOnEquity")),
        "profit_margin": safe_get(fin, "profitMargins", safe_get(key, "profitMargins")),
        "debt_to_equity": safe_get(key, "debtToEquity", safe_get(fin, "debtToEquity")),
        "dividend_yield": safe_get(summary, "dividendYield", safe_get(key, "dividendYield")),
    }


@st.cache_data(show_spinner=False)
def get_financial_df(tickers):
    rows = []
    for t in tickers:
        try:
            d = fetch_one_ticker(t)
        except Exception:
            continue
        d["ticker"] = t
        rows.append(d)

    df = pd.DataFrame(rows).set_index("ticker")

    # -------- FIX: If profit_margin missing → set 0 instead of NaN
    df["profit_margin"] = df["profit_margin"].fillna(0)

    # -------- Compute Value Score
    df["value_score"] = (
        (1 / df["pe_ratio"]).replace([np.inf, -np.inf], 0).fillna(0)
        + (0.5 / df["pb_ratio"]).replace([np.inf, -np.inf], 0).fillna(0)
        + df["dividend_yield"].fillna(0)
    )

    # -------- Compute Quality Score
    df["quality_score"] = (
        df["roe"].fillna(0) +
        df["profit_margin"].fillna(0)
    )

    return df

# K-Means Value/Quality Clustering (NaN-safe)

def _run_kmeans(fin_df, n_clusters=3):
    if fin_df.empty:
        return fin_df.assign(cluster=0), [], []

    features = fin_df[["value_score", "quality_score"]].copy()

    # Replace any NaNs with column median, then fallback to 0
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    features = features.fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    cluster_df = fin_df.copy()
    cluster_df["cluster"] = labels

    stats = (
        cluster_df.groupby("cluster")[["value_score", "quality_score"]]
        .mean()
        .assign(total=lambda d: d["value_score"] + d["quality_score"])
    )

    long_cluster = int(stats["total"].idxmax())
    short_cluster = int(stats["total"].idxmin())

    long_list = cluster_df[cluster_df["cluster"] == long_cluster].index.tolist()
    short_list = cluster_df[cluster_df["cluster"] == short_cluster].index.tolist()

    return cluster_df, long_list, short_list


def get_kmeans_results(tickers):
    fin_df = get_financial_df(tickers)
    cluster_df, long_list, short_list = _run_kmeans(fin_df)
    return fin_df, cluster_df, long_list, short_list


def get_cluster_labels(tickers):
    _, cluster_df, _, _ = get_kmeans_results(tickers)
    return cluster_df["cluster"].to_dict()
