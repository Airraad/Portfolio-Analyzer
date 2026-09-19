import argparse
import random
import numpy as np
import pandas as pd


def generate_portfolio_csv(filename="test_portfolio.csv", years=2, messy=False):
    # Generate business days (Mon-Fri)
    n_days = int(252 * years)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    # 1. Simulate Market (SPY): ~10% annual return, ~16% annual volatility
    mkt_drift = 0.10 / 252
    mkt_vol = 0.16 / np.sqrt(252)
    spy_daily = np.random.normal(mkt_drift, mkt_vol, n_days)

    # 2. Simulate Portfolio: Beta = 1.15, Alpha = +3% annual, Idiosyncratic Vol = ~10%
    true_alpha = 0.03 / 252
    beta = 1.15
    idio_vol = 0.10 / np.sqrt(252)
    port_daily = true_alpha + (beta * spy_daily) + np.random.normal(0, idio_vol, n_days)

    # 3. Column names (tests your alias mapping!)
    if messy:
        date_col = random.choice(["Date", "timestamp", "datetime", "day"])
        port_col = random.choice(["strategy_return", "portfolio_returns", "port_return", "returns"])
        spy_col = random.choice(["SPY", "spy_returns", "benchmark_return", "mkt"])
    else:
        date_col = "Date"
        port_col = "portfolio_returns"
        spy_col = "spy_returns"

    df = pd.DataFrame({
        date_col: dates.strftime("%Y-%m-%d"),
        port_col: port_daily,
        spy_col: spy_daily
    })

    # Optional real-world noise: sprinkle a couple NaNs to test .dropna()
    if messy:
        drop_indices = np.random.choice(df.index, size=max(1, int(len(df) * 0.01)), replace=False)
        df.loc[drop_indices, port_col] = np.nan

    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} trading days ({years} years) of portfolio data.")
    print(f"File saved to: {filename}")
    print(f"Columns: ['{date_col}', '{port_col}', '{spy_col}']")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate realistic portfolio CSV files.")
    parser.add_argument("-o", "--output", default="test_portfolio.csv", help="Output file name")
    parser.add_argument("-y", "--years", type=float, default=2.0, help="Number of years of data")
    parser.add_argument("--messy", action="store_true", help="Randomize column aliases and inject minor NaNs")

    args = parser.parse_args()
    generate_portfolio_csv(filename=args.output, years=args.years, messy=args.messy)
