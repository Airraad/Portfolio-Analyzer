"""
generate_csv.py
================
Simulates realistic portfolio and SPY benchmark daily returns under the
Fama-French 3-Factor specification:

    R_p - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + epsilon

Allows explicit calibration of style tilts (Small-Cap, Growth, Value) to
test factor attribution models and rolling style drift.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def generate_factor_returns(
    n_days: int = 504,
    start_date: str = "2023-01-03",
    market_mu_annual: float = 0.08,
    market_vol_annual: float = 0.16,
    smb_vol_annual: float = 0.07,
    hml_vol_annual: float = 0.07,
    portfolio_beta_mkt: float = 1.10,
    portfolio_beta_smb: float = 0.40,      # Default: Small-cap tilt (+0.40)
    portfolio_beta_hml: float = -0.30,     # Default: Growth tilt (-0.30)
    portfolio_alpha_annual: float = 0.02,  # 2% annual true manager alpha
    idiosyncratic_vol_annual: float = 0.06,
    rf_annual: float = 0.045,              # 4.5% risk-free rate proxy
    seed: int = 42,
) -> pd.DataFrame:
    """Generate daily portfolio and benchmark returns aligned to 3-factor exposures."""
    rng = np.random.default_rng(seed)

    # Convert annualized metrics to daily steps
    mkt_mu_daily = market_mu_annual / TRADING_DAYS_PER_YEAR
    mkt_vol_daily = market_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    smb_vol_daily = smb_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    hml_vol_daily = hml_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    alpha_daily = portfolio_alpha_annual / TRADING_DAYS_PER_YEAR
    idio_vol_daily = idiosyncratic_vol_annual / np.sqrt(TRADING_DAYS_PER_YEAR)
    rf_daily = rf_annual / TRADING_DAYS_PER_YEAR

    # 1. Simulate Factor Returns
    mkt_rf = rng.normal(loc=mkt_mu_daily, scale=mkt_vol_daily, size=n_days)
    smb = rng.normal(loc=0.0001, scale=smb_vol_daily, size=n_days)
    hml = rng.normal(loc=0.0001, scale=hml_vol_daily, size=n_days)

    # SPY benchmark gross return = (Mkt - RF) + RF
    spy = mkt_rf + rf_daily

    # 2. Simulate Portfolio Excess Returns
    epsilon = rng.normal(loc=0.0, scale=idio_vol_daily, size=n_days)
    excess_portfolio = (
        alpha_daily
        + portfolio_beta_mkt * mkt_rf
        + portfolio_beta_smb * smb
        + portfolio_beta_hml * hml
        + epsilon
    )
    portfolio = excess_portfolio + rf_daily

    # 3. Time Series Alignment
    dates = pd.bdate_range(start=start_date, periods=n_days)

    return pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "portfolio_returns": np.round(portfolio, 6),
            "spy_returns": np.round(spy, 6),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate returns with controllable Fama-French style factors.")
    parser.add_argument("-o", "--output", type=str, default="test_portfolio.csv", help="Output file path.")
    parser.add_argument("-y", "--years", type=float, default=2.0, help="Number of trading years (default: 2.0).")
    parser.add_argument("--beta", type=float, default=1.15, help="Market Beta exposure (default: 1.15).")
    parser.add_argument("--smb", type=float, default=0.45, help="Size factor exposure (+ small cap, - large cap).")
    parser.add_argument("--hml", type=float, default=-0.35, help="Value factor exposure (+ value, - growth).")
    parser.add_argument("--alpha", type=float, default=0.03, help="Annual excess alpha (default: 0.03 = 3%%).")
    parser.add_argument("--start", type=str, default="2023-01-03", help="Start date (YYYY-MM-DD).")
    args = parser.parse_args()

    n_days = int(args.years * TRADING_DAYS_PER_YEAR)
    df = generate_factor_returns(
        n_days=n_days,
        start_date=args.start,
        portfolio_beta_mkt=args.beta,
        portfolio_beta_smb=args.smb,
        portfolio_beta_hml=args.hml,
        portfolio_alpha_annual=args.alpha,
    )

    df.to_csv(args.output, index=False)
    print(f"Successfully generated {len(df)} rows ({args.years} years) -> {args.output}")
    print(f"Target Configuration: Beta={args.beta}, SMB={args.smb}, HML={args.hml}, Alpha={args.alpha:.2%}")


if __name__ == "__main__":
    main()
