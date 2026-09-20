import streamlit as st
import pandas as pd
from analytics import (
    load_returns,
    wealth_index,
    drawdown_fun,
    max_drawdown,
    CAGR_fun,
    sharpe_fun,
    sortino_fun,
    CAPM_regression,
    fama_french,
    fama_french_regression,
    rolling,
)

st.set_page_config(page_title="Quant Attribution Dashboard", layout="wide")
st.title("Quantitative Performance & Factor Attribution Dashboard")

# Cache network calls so Streamlit doesn't re-download factors on every user click
@st.cache_data(show_spinner="Fetching Fama-French Factor Data...")
def get_cached_factors(start_date, end_date):
    return fama_french(start_date, end_date)


st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Strategy CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # 1. Ingestion
        clean_df, start, end = load_returns(uploaded_file)
        returns = clean_df["returns"]
        spy = clean_df["spy"]

        # 2. Factors
        factors_df, mkt_rf, smb, hml, rf = get_cached_factors(start, end)

        # 3. High-Level Summary Metrics
        st.subheader("Performance & Risk Scorecard")
        cagr = CAGR_fun(returns)
        dd = drawdown_fun(returns)
        mdd = max_drawdown(dd)
        sharpe = sharpe_fun(returns, rf=rf)
        sortino = sortino_fun(returns, rf=rf)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CAGR", f"{cagr:.2%}")
        col2.metric("Max Drawdown", f"{mdd:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col4.metric("Sortino Ratio", f"{sortino:.2f}")

        # 4. Wealth & Drawdown Curves
        st.subheader("Equity Curve & Drawdowns")
        wealth = wealth_index(returns)
        bench_wealth = wealth_index(spy)
        chart_data = pd.DataFrame({"Portfolio": wealth, "Benchmark (SPY)": bench_wealth})
        st.line_chart(chart_data)

        st.line_chart(dd.rename("Drawdown"))

        # 5. Factor Attribution (Static)
        st.subheader("Fama-French 3-Factor Attribution")
        ff_alpha, b_mkt, b_smb, b_hml, ff_r2 = fama_french_regression(returns, factors_df)

        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Annual Alpha", f"{ff_alpha:.2%}")
        col_b.metric("Market Beta (Mkt-RF)", f"{b_mkt:.2f}")
        col_c.metric("Size Beta (SMB)", f"{b_smb:.2f}")
        col_d.metric("Value Beta (HML)", f"{b_hml:.2f}")
        col_e.metric("R-Squared", f"{ff_r2:.2%}")

        # 6. Rolling Betas (Style Drift)
        st.subheader("Trailing 63-Day Rolling Factor Exposures (Style Drift)")
        with st.spinner("Calculating rolling factor exposures..."):
            rolling_df = rolling(returns, factors_df, window=63)

        if not rolling_df.empty:
            betas_to_plot = rolling_df[["rolling_beta_mkt", "rolling_beta_smb", "rolling_beta_hml"]]
            st.line_chart(betas_to_plot)
        else:
            st.warning("Dataset too short to generate 63-day rolling betas.")

    except Exception as e:
        st.error(f"Processing Error: {e}")

else:
    st.info("Upload a CSV file containing your date, portfolio returns, and benchmark columns to start.")
