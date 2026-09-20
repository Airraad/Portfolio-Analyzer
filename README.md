**Quantitative Performance & Factor Attribution Dashboard

A lightweight Streamlit web app built to analyze trading strategy performance and run Fama-French 3-factor regressions. Instead of just looking at raw returns, this tool breaks down exactly where a portfolio's performance is coming from by separating true Alpha from basic market exposure.


To maximize my learning, I focused entirely on writing the mathematical backend (analytics.py) from scratch. My priority was deeply understanding the core quantitative fundamentals. To make my life easier, I leveraged AI to generate the Streamlit frontend (app.py) and the random data generator (generate_csv.py).

Features

Performance Scorecard: Automatically calculates CAGR, Max Drawdown, Sharpe, and Sortino ratios.

Factor Calculation: Runs OLS regressions to extract Annual Alpha, Market Beta, Size (SMB), and Value (HML) factors.

Style Drift: Tracks trailing 63-day rolling betas to show how a strategy's market correlation changes over time.

How to Run
Clone the repository and navigate to the project folder.

Install the required dependencies:

(pip install streamlit pandas numpy statsmodels)

Launch the application:

(streamlit run app.py)

Upload a CSV or TXT file with your daily returns through the sidebar.

(Note: The file needs three columns: Date, Portfolio Return, and SPY. Returns should be formatted as decimals, e.g., 0.015 for 1.5%).
