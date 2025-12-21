import yfinance as yf
import pandas as pd
import numpy as np

TICKERS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "JPM"]

def get_raw_info(ticker):
    print("=" * 80)
    print(f"Fetching: {ticker}")
    print("=" * 80)

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"ERROR fetching {ticker}: {e}")
        return None

    # Print raw keys available
    print("\nAvailable keys:", list(info.keys())[:30], "...")
    print("\nRaw snapshot:")
    for key in ["currentPrice", "regularMarketPrice", "trailingPE", "priceToBook",
                "returnOnEquity", "profitMargin", "debtToEquity", "dividendYield"]:
        print(f"{key}: {info.get(key)}")

    return info


def compute_scores(info):
    if info is None:
        print("No info → skipping\n")
        return

    def g(key):
        val = info.get(key)
        return float(val) if isinstance(val, (int, float)) else np.nan

    price = g("currentPrice") if not np.isnan(g("currentPrice")) else g("regularMarketPrice")
    pe = g("trailingPE")
    pb = g("priceToBook")
    roe = g("returnOnEquity")
    pm = g("profitMargin")
    de = g("debtToEquity")
    dy = g("dividendYield")

    print("\n--- Cleaned numeric values ---")
    print(f"price: {price}")
    print(f"pe_ratio: {pe}")
    print(f"pb_ratio: {pb}")
    print(f"roe: {roe}")
    print(f"profit_margin: {pm}")
    print(f"debt_to_equity: {de}")
    print(f"dividend_yield: {dy}")

    # Compute value score
    value_score = 0
    if pe and pe > 0:
        value_score += 1/pe
    if pb and pb > 0:
        value_score += 0.5/pb
    if dy and dy > 0:
        value_score += dy

    # Compute quality score
    quality_score = 0
    if roe and not np.isnan(roe):
        quality_score += roe
    if pm and not np.isnan(pm):
        quality_score += pm

    print("\n--- Scores ---")
    print(f"value_score: {value_score}")
    print(f"quality_score: {quality_score}\n")


if __name__ == "__main__":
    print("\n====== BEGIN DEBUG ======\n")

    for ticker in TICKERS:
        raw = get_raw_info(ticker)
        compute_scores(raw)

    print("\n====== END DEBUG ======\n")
