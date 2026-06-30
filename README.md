# Quant Research Dashboard

A Streamlit quant research dashboard using free market data. The app now has three modes:

- **Stage A2 Research Lab:** ML-powered multi-asset white-box research pipeline.
- **Stage A1 Research Lab:** linear benchmark research pipeline for the capstone.
- **Regime-Aware Portfolio Dashboard:** the original ETF/stock strategy dashboard.

## Features

- Run Stage A2 white-box ML models: Decision Tree, Random Forest, Gradient Boosting, and Elastic Net
- Build HRP-style / risk-parity fallback, Ledoit-Wolf mean-variance, fractional Kelly, and beta-neutral ML portfolios
- Visualize rule-based regime proxy states, factor exposure heatmaps, stress tests, execution costs, and feature importance drift
- Run Stage A1 linear benchmark models: OLS, Ridge, LASSO, and Elastic Net
- Build 12-1 momentum, low-volatility, beta, drawdown, and trend features
- Evaluate 60/20/20 train/validation/test splits and combinatorial purged CV
- Backtest equal-weight, inverse-variance risk parity, market-neutral long/short, and 130/30 portfolios with true walk-forward OOS signals
- Display performance first with a $1,000,000 initial-capital equity curve
- Apply transaction costs, slippage, and a monthly turnover cap
- Track Sharpe, Sortino, Calmar, information ratio, alpha, beta, drawdown, and hit rate
- Select ETFs from a predefined universe
- View price charts and daily returns by category
- Compare performance over 1D, 5D, 1M, and 3M windows
- See annualized volatility and a correlation heatmap
- Show 20-day vs 50-day moving average signals
- Generate a rule-based market summary
- Run a monthly ETF momentum rotation backtest against SPY

## Research Lab Plain-English Summary

**Stage A1** is the linear benchmark lab. It builds transparent price-based features, trains OLS/Ridge/LASSO/Elastic Net models, and runs true walk-forward portfolio tests. The goal is to create a simple baseline that later ML systems must beat.

**Stage A2** is the white-box ML lab. It adds tree models, feature importance tracking, macro fields from FRED when available, global ETF proxies, HRP-style allocation, Ledoit-Wolf shrinkage, fractional Kelly, beta-neutral long/short, factor exposure checks, stress tests, and execution-cost estimates.

Both labs use walk-forward logic: each prediction month trains only on data before that month, then trades the following period. Dashboard performance is intended for research and education, not live investment advice.

## Limitations and Fallbacks

- Historical index membership is not point-in-time survivorship-bias-free.
- Yahoo Finance and FRED data can have missing values, revisions, ticker gaps, and delayed updates.
- A2 regime detection is a **rule-based / Gaussian-mixture regime proxy**, not a true Hidden Markov Model.
- A2 HRP is labeled **HRP-style / risk-parity fallback** because it falls back to simpler risk-parity behavior when the clustering input is too sparse.
- A2 can fall back from the requested ML model to a simpler model or a white-box momentum/risk score if the selected model cannot produce enough walk-forward predictions.
- A2 uses a Decision Tree as the default white-box model because it is faster and has been more stable in recent walk-forward tests than the default Random Forest. Random Forest and Gradient Boosting remain available as heavier alternatives.
- A2 includes risk controls: volatility targeting, optional regime exposure scaling, weight smoothing, and a daily-bar max drawdown guard. These are designed to reduce drawdown and turnover, not to guarantee a higher Sharpe ratio in every sample.
- A1 and A2 performance shown in the dashboard is net of estimated costs; full gross-vs-net attribution is still a known extension.
- Stress windows show "Not enough data" when the selected history does not overlap 2008, 2020, or 2022.

## ETF Universe

- Broad Market: `SPY`, `QQQ`, `IWM`, `DIA`
- Sectors: `XLK`, `XLF`, `XLE`, `XLV`, `XLY`, `XLP`, `XLU`, `XLI`, `XLB`, `XLRE`, `XLC`
- Asset Classes: `TLT`, `SHY`, `AGG`, `TIP`, `HYG`, `LQD`, `GLD`, `SLV`, `DBC`, `USO`, `VNQ`
- Geography: `EFA`, `EEM`, `FXI`, `VGK`, `EWJ`, `INDA`

## Requirements

- Python 3.11 or 3.12 recommended
- Internet connection for Yahoo Finance data

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit. By default it is usually:

```text
http://localhost:8501
```

## Project Structure

```text
Dashboard/
├── app.py
├── src/
│   └── stage_a1.py
│   └── stage_a2.py
├── reports/
│   └── working_paper_1.md
│   └── working_paper_2.md
├── data/
│   └── stock_universe.csv
├── README.md
└── requirements.txt
```

## Notes

- No API key is required.
- No FastAPI backend, database, React app, Docker setup, or authentication is used.
- Price data comes from Yahoo Finance via `yfinance`.
