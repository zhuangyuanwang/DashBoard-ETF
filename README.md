# Quant Research Dashboard

A Streamlit quant research dashboard using free market data. The app now has three modes:

- **Stage A2 Research Lab:** presentation-focused white-box ML long-only stock selection dashboard.
- **Stage A1 Research Lab:** linear benchmark research pipeline for the capstone.
- **Regime-Aware Portfolio Dashboard:** the original ETF/stock strategy dashboard.

## Features

- Run and compare Stage A2 white-box ML models automatically using walk-forward OOS metrics
- Build Equal Weight, Score Weighted, Inverse-Vol / Risk-Parity Style, and beta-neutral research portfolios
- Visualize rule-based regime proxy states, factor exposure heatmaps, stress tests, execution costs, and SHAP/native feature importance drift
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

**Stage A2** is now a presentation-focused white-box ML long-only stock selection dashboard. It compares fast white-box ML models automatically, selects the best model using walk-forward out-of-sample metrics, chooses a 30/40/50 stock holding count, compares portfolio construction methods, and presents the recommended stock portfolio first.

Both labs use walk-forward logic: each prediction month trains only on data before that month, then trades the following period. Dashboard performance is intended for research and education, not live investment advice.

## Stage A2 ML Ranking Pipeline

Stage A2 ranks stocks each month by predicted next-month excess return versus SPY, then passes the ranking scores into Equal Weight, Score Weighted, Inverse-Vol / Risk-Parity Style, and beta-neutral research portfolios. It does not use random K-fold validation. The final audience is not expected to tune model parameters from the UI.

The dashboard automatically:

- Runs the default fast model set: Elastic Net, Decision Tree, and Random Forest. Heavier engines such as Gradient Boosting, XGBoost, and LightGBM remain supported in code but are not run by default on Streamlit Cloud.
- Selects the model by OOS Sharpe, then drawdown, signal spread, and turnover.
- Selects the number of stocks to hold from 30, 40, or 50 using walk-forward OOS diagnostics.
- Compares Equal Weight Top N, Score Weighted Top N, Inverse-Vol / Risk-Parity Style Top N, and Beta-Neutral Long/Short.
- Recommends the portfolio method by OOS Sharpe, then drawdown, turnover, and cost drag.
- Shows current holdings, a live-period monitor, model explanation, performance diagnostics, risk diagnostics, stress tests, and methodology in presentation tabs.

The Stage A2 **Live Monitor** tab uses the latest Yahoo Finance daily bars available to the app to show the recommended strategy from June 1 of the current year through the latest loaded date, compared with SPY. It is a research monitor, not broker-confirmed live execution P&L.

The Stage A2 **Performance Diagnostics** tab is deliberately honest when ML underperforms SPY. It compares the recommended ML strategy against SPY buy-and-hold, equal-weight stocks, 12-month momentum Top 30, and dual momentum; reports ML ranking spread and prediction IC; separates gross and net performance; checks rank turnover, target choice, feature ablations, overfitting risk, and portfolio construction effects.

XGBoost, LightGBM, SHAP, and hmmlearn are supported as optional engines. If those packages are not installed or are incompatible with the runtime, the dashboard clearly reports the fallback engine and keeps the research app running with sklearn histogram gradient boosting, native/sensitivity importance, or a Gaussian-mixture regime proxy.

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

The diagnostics tab shows top-minus-bottom prediction spread for all ranked assets and the ETF/global-proxy subset, prediction IC, gross-vs-net performance, transaction-cost drag, monthly turnover, fallback months, selected hyperparameters, and feature-importance stability.

Current reasons Stage A2 ML may underperform simple ETF momentum strategies:

- Monthly returns are noisy and hard to predict.
- The default universe can still be sector-concentrated depending on `stock_universe.csv`.
- Strict walk-forward validation removes look-ahead benefits that weaker backtests sometimes contain.
- Transaction costs, square-root impact, turnover controls, and drawdown guards reduce upside.
- ML ranking alpha may be weaker than simple trend/beta exposure in strong bull markets.

## Limitations and Fallbacks

- Historical index membership is not point-in-time survivorship-bias-free.
- Yahoo Finance and FRED data can have missing values, revisions, ticker gaps, and delayed updates.
- A2 regime detection uses a true 3-state Gaussian HMM when `hmmlearn` is installed. Otherwise it is clearly labeled as a Gaussian-mixture regime proxy.
- A2 inverse-vol / risk-parity style weighting is a simplified risk allocation method, not a full institutional HRP implementation.
- A2 can fall back from the requested ML model to a simpler model or a white-box momentum/risk score if the selected model cannot produce enough walk-forward predictions.
- A2 uses a Decision Tree as the default white-box model because it is faster and has been more stable in recent walk-forward tests than the default Random Forest. Random Forest, Gradient Boosting, optional XGBoost, and optional LightGBM remain available as heavier alternatives.
- A2 includes risk controls: volatility targeting, optional regime exposure scaling, weight smoothing, and a daily-bar max drawdown guard. These are designed to reduce drawdown and turnover, not to guarantee a higher Sharpe ratio in every sample.
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

Optional Stage A2 engines:

```bash
pip install xgboost lightgbm shap hmmlearn
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
