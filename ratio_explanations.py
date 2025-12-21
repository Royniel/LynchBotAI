# ratio_explanations.py

RATIO_DESCRIPTIONS = {
    "pe_ratio": (
        "Price-to-earnings ratio. Shows how much investors are paying for $1 of earnings. "
        "Lower can signal value; very high can signal growth expectations or overvaluation."
    ),
    "pb_ratio": (
        "Price-to-book ratio. Compares market value to the book value of equity. "
        "Useful in asset-heavy sectors; low P/B can indicate value, but also possible trouble."
    ),
    "roe": (
        "Return on equity. Measures how efficiently a company uses shareholder capital to generate profit. "
        "High and stable ROE often reflects strong fundamentals."
    ),
    "profit_margin": (
        "Net profit margin. How much profit the company keeps from each dollar of revenue. "
        "Steady or improving margins signal quality and competitive advantage."
    ),
    "debt_to_equity": (
        "Debt-to-equity ratio. Indicates financial leverage. "
        "High leverage increases risk, especially in recessions or rising-rate environments."
    ),
    "dividend_yield": (
        "Dividend yield. Annual dividends divided by share price. "
        "A moderate, sustainable yield is healthy; extremely high yields can indicate distress."
    ),
    "value_score": (
        "Combined valuation measure built from P/E, P/B, and other metrics to estimate 'cheapness'. "
        "Higher suggests better value relative to peers."
    ),
    "quality_score": (
        "Quality score derived from ROE, margins, leverage, and stability metrics. "
        "Higher means financially stronger companies."
    ),
}
