# Quant Research Dashboard

A Streamlit quant research dashboard using free market data. The app now has two modes:

- **Stage A1 Research Lab:** linear benchmark research pipeline for the capstone.
- **Regime-Aware Portfolio Dashboard:** the original ETF/stock strategy dashboard.

## Features

- Run Stage A1 linear benchmark models: OLS, Ridge, LASSO, and Elastic Net
- Build 12-1 momentum, low-volatility, beta, drawdown, and trend features
- Evaluate 60/20/20 train/validation/test splits and combinatorial purged CV
- Backtest equal-weight, inverse-variance risk parity, market-neutral long/short, and 130/30 portfolios
- Apply transaction costs, slippage, and a monthly turnover cap
- Track Sharpe, Sortino, Calmar, information ratio, alpha, beta, drawdown, and hit rate
- Select ETFs from a predefined universe
- View price charts and daily returns by category
- Compare performance over 1D, 5D, 1M, and 3M windows
- See annualized volatility and a correlation heatmap
- Show 20-day vs 50-day moving average signals
- Generate a rule-based market summary
- Run a monthly ETF momentum rotation backtest against SPY

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
├── reports/
│   └── working_paper_1.md
├── data/
│   └── stock_universe.csv
├── README.md
└── requirements.txt
```

## Notes

- No API key is required.
- No FastAPI backend, database, React app, Docker setup, or authentication is used.
- Price data comes from Yahoo Finance via `yfinance`.
