import streamlit as st
import pandas as pd
import plotly.express as px
from backtest import BacktestConfig, run_long_short_backtest
from datetime import date
import time

# RAG Pipeline (Part I)
from rag_pipeline import answer_conversational, _retrieve_relevant_qa

# Financial data & clustering (Part II)
from financial_data import (
    load_tickers,
    get_financial_df,
    get_kmeans_results,
    RATIO_DESCRIPTIONS  
)

# Page Config
st.set_page_config(
    page_title="Peter Lynch Investment Assistant",
    page_icon="💹",
    layout="wide",
)

st.title("💹 Peter Lynch Investment Assistant")
st.caption(
    "Chat about Peter Lynch’s philosophy, inspect fundamentals, and see a simple "
    "value–quality long/short recommendation based on K-Means clustering."
)

# -------------------------------------------------------------------
# Session state: chat history
# -------------------------------------------------------------------
if "chat_history" not in st.session_state:
    # Start with LynchBot intro message
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "text": (
                "Hi, I’m **LynchBot**, your Peter Lynch–style investment assistant. "
                "I only use our curated Q&A dataset about Peter Lynch’s philosophy, "
                "so everything I say is grounded in that data. What would you like to discuss?"
            ),
        }
    ]

# -------------------------------------------------------------------
# Sidebar – Portfolio selection
# -------------------------------------------------------------------
st.sidebar.header("Stock Universe")

# Default = Dow Jones tickers
default_dow = load_tickers()

# Keep track of extra tickers the user adds
if "extra_tickers" not in st.session_state:
    st.session_state.extra_tickers = []

# Keep track of what is currently selected in the multiselect
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = default_dow.copy()

# Input box to add new ticker
new_ticker = st.sidebar.text_input("Add a custom ticker (e.g. TSLA):")

# Button to add ticker to universe + selection
if st.sidebar.button("➕ Add ticker"):
    t = new_ticker.strip().upper()
    if t:
        # Add to extra universe if not already there
        if t not in default_dow and t not in st.session_state.extra_tickers:
            st.session_state.extra_tickers.append(t)
        # Also auto-select it in the portfolio
        if t not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(t)

# Build universe: Dow + extra tickers
all_choices = sorted(set(default_dow + st.session_state.extra_tickers))

# Multiselect bound to session_state
selected_tickers = st.sidebar.multiselect(
    "Selected portfolio:",
    options=all_choices,
    key="selected_tickers",  # binds to st.session_state["selected_tickers"]
    default=st.session_state.selected_tickers,
    help="Dow Jones constituents are pre-selected. You can uncheck any of them "
         "and add new tickers above.",
)

if not selected_tickers:
    st.sidebar.warning("Please select at least one ticker to analyze.")
    st.stop()

st.sidebar.markdown("**Current universe:**")
st.sidebar.write(", ".join(selected_tickers))

# -------------------------------------------------------------------
# Main layout – Tabs
# -------------------------------------------------------------------
tab_chat, tab_fundamentals, tab_clusters, tab_backtest = st.tabs(
    [
        "💬 Chat-Bot",
        "📊 Financial Ratios Dashboard",
        "📈 Value–Quality Clusters",
        "📉 Strategy Backtest",
    ]
)

# -------------------------------------------------------------------
# TAB 1 — Conversational LynchBot
# -------------------------------------------------------------------
with tab_chat:
    st.subheader("Conversational Peter Lynch Bot")

    top_col, reset_col = st.columns([4, 1])
    with top_col:
        st.caption("Ask follow-up questions — LynchBot remembers the conversation.")
    with reset_col:
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "text": (
                        "Hi, I’m **LynchBot**, your Peter Lynch–style investment assistant. "
                        "I only use our curated Q&A dataset about Peter Lynch’s philosophy, "
                        "so everything I say is grounded in that data. What would you like to discuss?"
                    ),
                }
            ]
            st.rerun()

    # Show chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**LynchBot:** {msg['text']}")

    st.markdown("---")

    # Input for new message
    user_input = st.text_input(
        "Type your question or follow-up:",
        placeholder="e.g. How would Peter Lynch think about buying a beaten-down stock?",
        key="chat_input",
    )

    send_col, _ = st.columns([1, 4])
    with send_col:
        send_clicked = st.button("Send")

    if send_clicked and user_input.strip():
        user_text = user_input.strip()
        user_text_lower = user_text.lower()

        # Goodbye / reset commands
        reset_commands = ["goodbye", "bye", "reset", "clear", "start over", "new conversation", "restart"]

        if user_text_lower in reset_commands:
            # Optional: clear RAG session state if you implemented clear_session
            try:
                from rag_pipeline import clear_session
                clear_session("streamlit_chat")
            except ImportError:
                pass

            st.session_state.chat_history = [
                {
                    "role": "assistant",
                    "text": (
                        "Conversation reset! Hi, I'm **LynchBot**, your Peter Lynch–style investment assistant. "
                        "I only use our curated Q&A dataset about Peter Lynch's philosophy, "
                        "so everything I say is grounded in that data. What would you like to discuss?"
                    ),
                }
            ]
            st.success("👋 Goodbye! Starting fresh conversation...")
            time.sleep(1)
            st.rerun()

        else:
            # 1) Append user message to history
            st.session_state.chat_history.append({"role": "user", "text": user_text})

            # 2) Get LynchBot reply using multi-turn RAG (session_id managed inside rag_pipeline)
            with st.spinner("LynchBot is thinking..."):
                reply = answer_conversational(user_text, session_id="streamlit_chat")

            # 3) Append bot reply to history
            st.session_state.chat_history.append({"role": "assistant", "text": reply})

            # 4) Typing animation for the latest reply
            st.markdown("**You:** " + user_text)
            placeholder = st.empty()
            displayed = ""
            for ch in reply:
                displayed += ch
                placeholder.markdown(f"**LynchBot:** {displayed}▌")
                time.sleep(0.01)
            placeholder.markdown(f"**LynchBot:** {reply}")

    st.markdown("---")

    # Optional: show retrieved Q&A context for last user message
    if st.session_state.chat_history and any(m["role"] == "user" for m in st.session_state.chat_history):
        last_user_msg = [m["text"] for m in st.session_state.chat_history if m["role"] == "user"][-1]

        with st.expander("🔎 Show supporting Q&A snippets for the last question"):
            raw = _retrieve_relevant_qa(last_user_msg.strip(), top_k=5)

            # Handle both possible return types:
            # - (results, cache_hit)
            # - results
            if isinstance(raw, tuple):
                retrieved = raw[0]
            else:
                retrieved = raw

            if not retrieved:
                st.write("No relevant Q&A snippets found in the dataset.")
            else:
                for i, item in enumerate(retrieved, start=1):
                    st.markdown(f"**Snippet {i}:**")
                    st.markdown(f"> **Q:** {item['question']}")
                    st.markdown(f"> **A:** {item['answer']}")

    with st.expander("ℹ️ Quick primer on key financial ratios"):
        for col, desc in RATIO_DESCRIPTIONS.items():
            st.markdown(f"- **{col}** – {desc}")

# -------------------------------------------------------------------
# TAB 2 — Financial Ratios Dashboard
# -------------------------------------------------------------------
with tab_fundamentals:
    st.subheader("Fundamental snapshot of selected stocks")

    with st.spinner("Loading financial data from Yahoo Finance..."):
        fin_df = get_financial_df(selected_tickers)

    if fin_df.empty:
        st.error("Could not load financial data.")
    else:
        st.dataframe(
            fin_df.style.format(
                {
                    "price": "{:,.2f}",
                    "pe_ratio": "{:,.2f}",
                    "pb_ratio": "{:,.2f}",
                    "roe": "{:.2%}",
                    "profit_margin": "{:.2%}",
                    "debt_to_equity": "{:,.2f}",
                    "dividend_yield": "{:.2%}",
                    "value_score": "{:,.3f}",
                    "quality_score": "{:,.3f}",
                }
            ),
            use_container_width=True,
        )

        metric = st.selectbox(
            "Sort by metric:",
            options=[
                "price",
                "pe_ratio",
                "pb_ratio",
                "roe",
                "profit_margin",
                "debt_to_equity",
                "dividend_yield",
                "value_score",
                "quality_score",
            ],
            index=0,
        )
        top_n = st.slider("Show top N stocks:", 5, len(fin_df), 10)

        sorted_df = fin_df.sort_values(metric, ascending=False).head(top_n)
        st.table(sorted_df[[metric]])

# -------------------------------------------------------------------
# TAB 3 — Clusters
# -------------------------------------------------------------------
with tab_clusters:
    st.subheader("K-Means value–quality clusters & long/short ideas")

    with st.spinner("Running K-Means clustering..."):
        fin_df, cluster_df, long_list, short_list = get_kmeans_results(selected_tickers)

    if cluster_df.empty:
        st.error("No data available to cluster.")
    else:
        fig = px.scatter(
            cluster_df.reset_index(),
            x="value_score",
            y="quality_score",
            color="cluster",
            text="ticker",
            title="Value vs. Quality — K-Means Clusters",
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, key="cluster_chart", use_container_width=True)

        col_long, col_short = st.columns(2)
        with col_long:
            st.markdown("### ✅ Long candidates")
            for t in long_list:
                st.markdown(f"- **{t}**")

        with col_short:
            st.markdown("### ⚠️ Short candidates")
            for t in short_list:
                st.markdown(f"- **{t}**")

        st.dataframe(cluster_df, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4 — Strategy Backtest
# -------------------------------------------------------------------
with tab_backtest:
    st.subheader("Backtest: Value–Quality Long/Short Strategy")

    try:
        bt_fin_df, bt_cluster_df, bt_long_list, bt_short_list = get_kmeans_results(selected_tickers)
    except Exception as e:
        st.error(f"Cluster error: {e}")
        st.stop()

    st.markdown("### Long / Short Baskets")
    col_lon, col_sho = st.columns(2)

    with col_lon:
        chosen_long = st.multiselect(
            "Long tickers",
            options=sorted(bt_long_list),
            default=sorted(bt_long_list),
        )

    with col_sho:
        chosen_short = st.multiselect(
            "Short tickers",
            options=sorted(bt_short_list),
            default=sorted(bt_short_list),
        )

    today = date.today()
    default_start = today.replace(year=today.year - 3)

    col_dates, col_params = st.columns(2)

    with col_dates:
        start_date = st.date_input("Start Date", value=default_start)
        end_date = st.date_input("End Date", value=today)

        benchmark_choice = st.selectbox(
            "Benchmark",
            ["^GSPC (S&P 500)", "^DJI (Dow Jones)", "None"],
            index=0,
        )
        if benchmark_choice.startswith("^GSPC"):
            benchmark_ticker = "^GSPC"
        elif benchmark_choice.startswith("^DJI"):
            benchmark_ticker = "^DJI"
        else:
            benchmark_ticker = ""

    with col_params:
        weighting = st.selectbox("Weighting Scheme", ["equal", "value"])
        rebalance_freq = st.selectbox("Rebalance Frequency", ["monthly", "weekly", "daily"])
        tc_bps = st.slider("Transaction Costs (bps)", 0.0, 100.0, 10.0)

    run_bt = st.button("▶ Run Backtest")

    if run_bt:
        with st.spinner("Running backtest..."):
            config = BacktestConfig(
                tickers=selected_tickers,
                long_list=chosen_long,
                short_list=chosen_short,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                weighting=weighting,
                rebalance_freq=rebalance_freq,
                transaction_cost_bps=tc_bps,
                benchmark=benchmark_ticker,
            )

            try:
                bt_result = run_long_short_backtest(config, fin_df=bt_fin_df)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

        st.success("Backtest Completed Successfully ✔")

        # --- Metrics ---
        metrics_series = pd.Series(bt_result.metrics)
        percent_metrics = ["total_return", "CAGR", "annual_volatility", "max_drawdown"]
        for m in percent_metrics:
            if m in metrics_series:
                metrics_series[m] *= 100

        st.dataframe(metrics_series.to_frame("Value").style.format("{:.2f}"), use_container_width=True)

        # --- Equity Curve ---
        st.markdown("### Equity Curve (Strategy vs Benchmark)")

        eq_df = pd.DataFrame({"Strategy": bt_result.portfolio_equity})
        if bt_result.benchmark_equity is not None:
            eq_df["Benchmark"] = bt_result.benchmark_equity

        eq_df.index.name = "Date"
        eq_long = eq_df.reset_index().melt(id_vars="Date", var_name="Series", value_name="Equity")

        fig_eq = px.line(eq_long, x="Date", y="Equity", color="Series")
        st.plotly_chart(fig_eq, key="equity_curve_chart", use_container_width=True)

        # --- Rolling Sharpe ---
        st.markdown("### Rolling Sharpe Ratio (63-day window)")

        rs = bt_result.rolling_sharpe.dropna()
        if rs.empty:
            st.info("Not enough data for rolling Sharpe.")
        else:
            rs_df = rs.reset_index()
            rs_df.columns = ["Date", "rolling_sharpe"]

            fig_rs = px.line(rs_df, x="Date", y="rolling_sharpe", labels={"rolling_sharpe": "Sharpe (63D)"})
            st.plotly_chart(fig_rs, key="rolling_sharpe_chart", use_container_width=True)

        # --- Correlation Heatmap ---
        st.markdown("### Correlation Heatmap (Asset Returns)")

        corr = bt_result.corr_matrix
        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(fig_corr, key="corr_heatmap_chart", use_container_width=True)

        # --- Portfolio Composition ---
        st.markdown("### Portfolio Composition (Latest Weights)")

        latest_w = bt_result.weights.iloc[-1]
        latest_w = latest_w[latest_w != 0]

        comp_df = (
            latest_w.abs()
            .sort_values(ascending=False)
            .to_frame("Absolute Weight")
            .reset_index()
            .rename(columns={"index": "Ticker"})
        )

        fig_comp = px.bar(comp_df, x="Ticker", y="Absolute Weight")
        st.plotly_chart(fig_comp, key="portfolio_weights_chart", use_container_width=True)
