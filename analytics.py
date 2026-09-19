import io
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import getFamaFrenchFactors as gff

# Make a group of possible Aliases
spy_alias = {
    "spy_return", "spy", "sp", "benchmark", "mkt",
    "benchmark_return", "market", "spy_returns"
}

returns_alias = {
    "port", "portfolio", "returns", "portfolio_returns",
    "return", "data", "strategy", "strategy_return", "port_return"
}

date_alias = {
    "date", "dates", "time", "date_occured",
    "datetime", "day", "timestamp"
}


def normalize(name):
    return str(name).strip().lower().replace(" ", "_")

# Handle bytes from uploader or direct string/buffer
def load_returns(input_file):
    
    if isinstance(input_file, bytes):
        input_file = io.BytesIO(input_file)
    elif isinstance(input_file, str):
        input_file = io.StringIO(input_file)

    df = pd.read_csv(input_file)

    lookup = {normalize(col): col for col in df.columns}

    def find(aliases, label):
        for alias in aliases:
            if alias in lookup:
                return lookup[alias]
        raise ValueError(f"Needs relevant column titles for {label} (Example: dates, returns, spy)")

    date_col = find(date_alias, "date")
    returns_col = find(returns_alias, "returns")
    spy_col = find(spy_alias, "spy")

    # Turn all data to numbers; non-numeric values become NaN
    clean_df = pd.DataFrame({
        "returns": pd.to_numeric(df[returns_col], errors="coerce"),
        "spy": pd.to_numeric(df[spy_col], errors="coerce")
    })
    clean_df.index = pd.to_datetime(df[date_col], errors="coerce")

    # Drop NaNs and sort chronologically
    clean_df = clean_df[~clean_df.index.isna()].dropna().sort_index()

    if len(clean_df) < 30:
        raise ValueError(f"Needs at least 30 valid days; found {len(clean_df)}.")

    start = clean_df.index.min()
    end = clean_df.index.max()

    return clean_df, start, end


def fama_french(start, end):
    factors = gff.famaFrench3Factor(frequency="d")

    # 1. Fix date parsing: turn non-date string rows (like 'Mkt-RF' or headers) into NaT
    factors["date"] = pd.to_datetime(factors["date_ff_factors"], errors="coerce")
    factors = factors.dropna(subset=["date"]).set_index("date").sort_index()

    factor_cols = ["Mkt-RF", "SMB", "HML", "RF"]

    # 2. Ensure all factor columns are pure floats
    for col in factor_cols:
        factors[col] = pd.to_numeric(factors[col], errors="coerce")
    factors = factors.dropna(subset=factor_cols)

    # 3. Convert percentages (1.0 -> 0.01) to match decimal portfolio returns
    factors[factor_cols] = factors[factor_cols] / 100.0

    # 4. Safe string slicing for dates
    start_str = pd.to_datetime(start).strftime("%Y-%m-%d")
    end_str = pd.to_datetime(end).strftime("%Y-%m-%d")
    factors_sliced = factors.loc[start_str:end_str]

    if factors_sliced.empty:
        raise ValueError(
            f"No factor data available between {start_str} and {end_str}. "
            "Note: Kenneth French data has a ~1-2 month reporting lag."
        )

    rf = factors_sliced["RF"]
    mkt_rf = factors_sliced["Mkt-RF"]
    smb = factors_sliced["SMB"]
    hml = factors_sliced["HML"]

    return factors_sliced, mkt_rf, smb, hml, rf

def fama_french_regression(returns, factors_df):
    merged = pd.concat([returns.rename("returns"), factors_df], axis=1, join="inner").dropna()
    excessy = merged["returns"] - merged["RF"]
    x = merged[["Mkt-RF", "SMB", "HML"]]

    x_const = sm.add_constant(x)
    model = sm.OLS(excessy, x_const).fit()

    alpha = float(model.params["const"]) * 252
    beta_mkt = float(model.params["Mkt-RF"])
    beta_smb = float(model.params["SMB"])
    beta_hml = float(model.params["HML"])
    r_squared = float(model.rsquared)
    return alpha, beta_mkt, beta_smb, beta_hml, r_squared


def wealth_index(returns):
    initial = 1.0
    return initial * (1.0 + returns).cumprod()


def drawdown_fun(returns):
    wealth = wealth_index(returns)
    peak = wealth.cummax()
    return (wealth - peak) / peak


def max_drawdown(drawdown):
    return float(drawdown.min())


def CAGR_fun(returns):
    n = len(returns)
    if n == 0:
        return np.nan
    years = n / 252
    total_growth = (1.0 + returns).prod()
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (1.0 / years) - 1.0)


def sharpe_fun(returns, rf):
    excess = returns - rf
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    daily_sharpe = excess.mean() / sd
    return float(daily_sharpe * np.sqrt(252))


def sortino_fun(returns, rf):
    excess = returns - rf
    sortino_excess = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(sortino_excess ** 2))
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float((excess.mean() / dd) * np.sqrt(252))


def CAPM_regression(returns, spy, rf):
    y = returns - rf
    x = spy - rf
    beta = float(y.cov(x) / x.var())
    daily_alpha = y.mean() - (beta * x.mean())
    alpha = float(daily_alpha * 252)
    r = float(y.corr(x))
    r_squared = r ** 2
    return beta, alpha, r_squared


def rolling(returns, factors_df, window=63):
    merged = pd.concat([returns.rename("returns"), factors_df], axis=1, join="inner").dropna()
    records = []

    for i in range(window, len(merged)):
        sub = merged.iloc[i - window : i]
        y = sub["returns"] - sub["RF"]
        x = sub[["Mkt-RF", "SMB", "HML"]]
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()

        records.append({
            "date": sub.index[-1],
            "rolling_alpha": float(model.params["const"]) * 252,
            "rolling_beta_mkt": float(model.params["Mkt-RF"]),
            "rolling_beta_smb": float(model.params["SMB"]),
            "rolling_beta_hml": float(model.params["HML"]),
            "rolling_r_squared": float(model.rsquared)
        })

    rolling_df = pd.DataFrame(records)
    if not rolling_df.empty:
        rolling_df = rolling_df.set_index("date")

    return rolling_df
