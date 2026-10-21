# Quant Research Dashboard

A Streamlit quant research dashboard using free market data. The app now has three modes:

- **Stage A2 Research Lab:** presentation-focused white-box ML long-only stock selection dashboard.
- **Stage A1 Research Lab:** linear benchmark research pipeline for the capstone.
- **Regime-Aware Portfolio Dashboard:** the original ETF/stock strategy dashboard.

## Features

- Run and compare Stage A2 white-box ML models automatically using walk-forward OOS metrics
- Build Equal Weight, Score Weighted, and Inverse-Vol / Risk-Parity Style long-only research portfolios
- Visualize regime diagnostics, factor exposure heatmaps, stress tests, execution costs, and model feature importance
- Run Stage A1 linear benchmark models: OLS, Ridge, LASSO, and Elastic Net
- Build 12-1 momentum, low-volatility, beta, drawdown, and trend features
- Evaluate 60/20/20 train/validation/test splits and combinatorial purged CV
- Backtest equal-weight, inverse-variance risk parity, market-neutral long/short, and 130/30 portfolios with true walk-forward OOS signals
- Display performance first with a $1,000,000 initial-capital equity curve
- Apply fixed turnover-based transaction costs and a monthly turnover cap
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

**Stage A2** is now a presentation-focused white-box ML long-only stock selection dashboard. It compares fast white-box ML models automatically, selects the best model using walk-forward out-of-sample metrics, chooses a 10/15/20 stock holding count, compares portfolio construction methods, and presents the recommended stock portfolio first.

Both labs use walk-forward logic: each prediction month trains only on data before that month, then trades the following period. Dashboard performance is intended for research and education, not live investment advice.

## Stage A2 ML Ranking Pipeline

Stage A2 ranks stocks each month by predicted next-month excess return versus SPY, then passes the ranking scores into Equal Weight, Score Weighted, and Inverse-Vol / Risk-Parity Style long-only research portfolios. It does not use random K-fold validation. The final audience is not expected to tune model parameters from the UI.

The dashboard automatically:

- Runs Elastic Net, Random Forest, and XGBoost as the presentation model set.
- Ranks the first 200 tickers from `data/stock_universe.csv` by default, plus SPY and risk-proxy ETFs for features and benchmarking.
- Supports an optional point-in-time universe file at `data/stock_universe_point_in_time.csv` with `Ticker`, `StartDate`, and `EndDate` columns to avoid using stocks before they entered the investable universe or after they exited.
- Selects the model by OOS Sharpe, then drawdown, signal spread, and turnover.
- Selects the number of stocks to hold from 10, 15, or 20 using walk-forward OOS diagnostics, because a 5% minimum position rule implies at most 20 holdings.
- Compares Equal Weight Top N, Score Weighted Top N, and Inverse-Vol / Risk-Parity Style Top N while keeping every name at least 5% of the invested equity sleeve.
- Recommends the portfolio method by OOS Sharpe, then drawdown, turnover, and cost drag.
- Targets 10% annualized volatility, reduces equity exposure to at most 50% after a 10% drawdown, and retains the disclosed 15% hard drawdown research guard.
- Shows current holdings, model explanation, performance diagnostics, risk diagnostics, stress tests, and methodology in presentation tabs.

The Stage A2 **Performance Diagnostics** tab is deliberately honest when ML underperforms SPY. It compares the recommended ML strategy against SPY buy-and-hold, equal-weight stocks, 12-month momentum Top 30, and dual momentum; reports ML ranking spread and prediction IC; separates gross and net performance; checks rank turnover, target choice, feature ablations, overfitting risk, and portfolio construction effects.

XGBoost and SHAP are included in the default requirements. Stage A2 uses SHAP for feature importance and labels any native/sensitivity fallback if explanation fails. `hmmlearn` remains an optional regime-detection extension.

Stage A2 feature groups include:

- Momentum and relative strength: 1M/3M/6M/12M returns, returns relative to SPY, cross-sectional momentum rank.
- Risk and drawdown: rolling volatility, drawdown, volatility rank, drawdown rank.
- Trend and regime: SPY above 200-day MA, asset above 200-day MA, 50D/200D ratio, market drawdown, risk-off dummy.
- Correlation and beta: rolling beta/correlation to SPY and correlation to TLT.
- Macro when available: Treasury yields, 2Y-10Y spread, Fed funds, CPI YoY, unemployment, industrial production YoY, financial stress. Macro fields are lagged by one month as a conservative timing proxy.

Supported targets:

- Next-month raw return
- Next-month excess return versus SPY
- Next-month outperform-SPY classification
- Next-month cross-sectional rank percentile

The diagnostics tab shows top-minus-bottom prediction spread, monthly Rank IC, IC volatility, positive-IC month rate, ICIR, 6- and 12-month rolling IC, gross-vs-net performance, transaction-cost drag, turnover, fallback months, selected hyperparameters, and feature-importance stability.

Current reasons Stage A2 ML may underperform simple ETF momentum strategies:

- Monthly returns are noisy and hard to predict.
- The default universe can still be sector-concentrated depending on `stock_universe.csv`.
- Strict walk-forward validation removes look-ahead benefits that weaker backtests sometimes contain.
- Fixed transaction costs and turnover controls reduce upside.
- ML ranking alpha may be weaker than simple trend/beta exposure in strong bull markets.

## Limitations and Fallbacks

- The default `stock_universe.csv` is a static research universe. Survivorship-bias-free results require a populated `data/stock_universe_point_in_time.csv` file from a historical constituent source such as CRSP/Compustat, WRDS, Bloomberg, or another point-in-time membership database.
- Yahoo Finance and FRED data can have missing values, revisions, ticker gaps, and delayed updates.
- A2 regime detection uses a true 3-state Gaussian HMM when `hmmlearn` is installed. Otherwise it is clearly labeled as a Gaussian-mixture regime proxy.
- A2 inverse-vol / risk-parity style weighting is a simplified risk allocation method, not a full institutional HRP implementation.
- A2 can fall back from the requested ML model to a simpler model or a white-box momentum/risk score if the selected model cannot produce enough walk-forward predictions.
- A2 compares Elastic Net, Random Forest, and XGBoost; model selection remains a research-selection step and is not an independent final holdout.
- A2 targets 10% annualized volatility and may hold cash. It cuts equity exposure after a 10% drawdown and uses an idealized daily-bar 15% hard drawdown guard, so results must disclose that mechanical protection.
- GDELT, Google Trends, SEC EDGAR, Black-Litterman, market-impact costs, and TWAP/VWAP simulation are not implemented and must not be presented as completed results.
- A1 and A2 headline performance is net of estimated costs. A2 also shows gross-vs-net attribution in the ML diagnostics tab.
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

Optional Stage A2 extensions:

```bash
pip install lightgbm hmmlearn
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
│   └── stock_universe_point_in_time_template.csv
├── README.md
└── requirements.txt
```

## Notes

- No API key is required.
- No FastAPI backend, database, React app, Docker setup, or authentication is used.
- Price data comes from Yahoo Finance via `yfinance`.
