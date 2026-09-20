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

st.set_page_config(page_title="Quantitative Attribution Dashboard", layout="wide")
st.title("📊 Quantitative Performance & Factor Attribution Dashboard")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Strategy CSV")
if uploaded_file is not None:
    try:
        # 1. Parse uploaded CSV
        clean_df, start, end = load_returns(uploaded_file)
        returns = clean_df["returns"]
        spy = clean_df["spy"]

        st.sidebar.success(f"Loaded {len(clean_df)} trading days")
        st.sidebar.caption(f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

        # 2. Fetch Factor Library (anchored with SPY)
        factors_df, mkt_rf, smb, hml, rf = fama_french(start, end, spy_series=spy)

        # 3. Performance & Risk Scorecard
        st.subheader("Performance & Risk Scorecard")
        cagr = CAGR_fun(returns)
        dd = drawdown_fun(returns)
        mdd = max_drawdown(dd)
        sharpe = sharpe_fun(returns, rf=rf)
        sortino = sortino_fun(returns, rf=rf)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CAGR", f"{cagr:.2%}")
        c2.metric("Max Drawdown", f"{mdd:.2%}")
        c3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        c4.metric("Sortino Ratio", f"{sortino:.2f}")

        # 4. Wealth & Drawdown Curves
        st.subheader("Equity Curve & Drawdowns")
        wealth = wealth_index(returns)
        bench_wealth = wealth_index(spy)
        wealth_df = pd.DataFrame({"Portfolio": wealth, "Benchmark (SPY)": bench_wealth})

        st.line_chart(wealth_df)
        st.line_chart(dd.rename("Portfolio Drawdown"))

        # --- DIAGNOSTIC BLOCK ---
        st.error("### DIAGNOSTIC CHECK")
        st.write("1. Length of Returns:", len(returns))
        st.write("2. Length of Factors:", len(factors_df))
        
        # Merge them exactly as the regression does
        test_merge = pd.concat([returns.rename("returns"), factors_df], axis=1, join="inner").dropna()
        st.write("3. Length after Merge (If this drops, dates are misaligned):", len(test_merge))
        
        # Show the raw numbers going into the regression
        st.write("4. First 5 rows being regressed:")
        st.dataframe(test_merge.head())
        # ------------------------
        # --- BYPASS MATH CHECK ---
        import statsmodels.api as sm
        
        # Calculate exactly from the verified diagnostic table
        test_y = test_merge["returns"] - test_merge["RF"]
        test_X = sm.add_constant(test_merge[["Mkt-RF", "SMB", "HML"]])
        
        test_model = sm.OLS(test_y, test_X).fit()
        true_beta = test_model.params["Mkt-RF"]
        
        st.success(f"🔥 MATHEMATICAL TRUTH (BETA): {true_beta:.2f}")
        # -------------------------
        # 5. Factor Attribution Model
        st.subheader("Fama-French 3-Factor Attribution")
        ff_alpha, b_mkt, b_smb, b_hml, ff_r2 = fama_french_regression(returns, factors_df)

        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Annual Alpha", f"{ff_alpha:.2%}")
        f2.metric("Market Beta (Mkt-RF)", f"{b_mkt:.2f}")
        f3.metric("Size Beta (SMB)", f"{b_smb:.2f}")
        f4.metric("Value Beta (HML)", f"{b_hml:.2f}")
        f5.metric("R-Squared", f"{ff_r2:.2%}")

        # 6. Style Drift / Rolling Factors
        st.subheader("Trailing 63-Day Rolling Betas (Style Drift)")
        if len(clean_df) >= 63:
            rolling_df = rolling(returns, factors_df, window=63)
            if not rolling_df.empty:
                st.line_chart(rolling_df[["rolling_beta_mkt", "rolling_beta_smb", "rolling_beta_hml"]])
            else:
                st.warning("Insufficient overlapping dates for rolling regression.")
        else:
            st.info("Requires at least 63 trading days to display rolling attribution.")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload a portfolio CSV from the sidebar to view attribution.")
