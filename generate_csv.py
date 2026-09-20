"""
generate_csv.py / generate_sample_data.py
==========================================
Simulates realistic portfolio and SPY benchmark daily returns.
Supports command-line arguments:
  -o, --output: Output filename (default: test_portfolio.csv)
  -y, --years:  Number of trading years to generate (default: 2)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def generate_sample_returns(
    n_days: int = 504,
    start_date: str = "2023-01-03",
    market_mu_annual: float = 0.08,
    market_vol_annual: float = 0.16,
    portfolio_beta: float = 1.10,
    portfolio_alpha_annual: float = 0.02,
    idiosyncratic_vol_annual: float = 0.08,
    seed: int = 75,
) -> pd.DataFrame:
    """Simulate daily SPY and portfolio simple returns via a one-factor model."""
    rng = np.random.default_rng(seed)

    mkt_mu_daily = market_mu_annual / TRADING_DAYS_PER_YEAR
    mkt_vol_daily = market_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_daily = portfolio_alpha_annual / TRADING_DAYS_PER_YEAR
    idio_vol_daily = idiosyncratic_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)

    spy = rng.normal(loc=mkt_mu_daily, scale=mkt_vol_daily, size=n_days)
    epsilon = rng.normal(loc=0.0, scale=idio_vol_daily, size=n_days)
    portfolio = alpha_daily + portfolio_beta * spy + epsilon

    dates = pd.bdate_range(start=start_date, periods=n_days)

    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "portfolio_returns": np.round(portfolio, 6),
            "spy_returns": np.round(spy, 6),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic returns CSV for Portfolio Analyzer.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="test_portfolio.csv",
        help="Path or name of output CSV file (default: test_portfolio.csv)"
    )
    parser.add_argument(
        "-y", "--years",
        type=float,
        default=2.0,
        help="Number of years of trading data to simulate (default: 2.0)"
    )
    args = parser.parse_args()

    n_days = int(args.years * TRADING_DAYS_PER_YEAR)
    df = generate_sample_returns(n_days=n_days)

    df.to_csv(args.output, index=False)
    print(f"Generated {len(df)} trading days ({args.years} years) of portfolio data.")
    print(f"File saved to: {args.output}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
