# Working Paper #2: White-Box ML Stock Selection

## 1. Abstract

Stage A2 builds a presentation-ready, white-box machine learning stock selection strategy. The system ranks stocks by predicted next-month excess return versus SPY, constructs a long-only portfolio from the highest-ranked names, and evaluates the strategy with walk-forward out-of-sample validation.

## 2. Introduction

The research question is whether transparent ML models can add stock-selection value beyond SPY, equal-weight stocks, and simple momentum after transaction costs and turnover.

## 3. Data and Universe

The dashboard uses `data/stock_universe.csv` as the default static research universe, with SPY and ETF proxies used for benchmark, sector context, and risk monitoring. If `data/stock_universe_point_in_time.csv` is populated with `Ticker`, `StartDate`, and `EndDate`, Stage A2 filters training rows and portfolio candidates to names active as of each signal date. Prices come from Yahoo Finance. FRED macro data is included when available and lagged before use.

## 4. Feature Engineering

Features include momentum, relative strength versus SPY, volatility, drawdown, trend, beta, correlations, market context, and lagged macro variables. Features at month-end `t` are used to predict returns from `t` to `t+1`.

## 5. Target Design

The primary target is next-month excess return versus SPY. This focuses the model on stock selection rather than market-direction timing. Alternative targets are kept as diagnostics, not as a way to optimize the final test period.

## 6. Model Framework

The presentation model set runs Elastic Net, Random Forest, and XGBoost.

## 7. Walk-Forward Validation

Each prediction month trains only on prior months, predicts the next month cross-section, ranks stocks, and forms the next monthly portfolio. Random K-fold cross-validation is not used.

## 8. Portfolio Construction

The dashboard compares Equal Weight Top N, Score Weighted Top N, and Inverse-Vol / Risk-Parity Style Top N. The system tests 10, 15, and 20 stock holding counts. Every name remains at or above 5% of the invested equity sleeve; volatility and drawdown controls can leave the remainder in cash.

## 9. Risk Management

The portfolio targets 10% annualized volatility using trailing data. A 10% drawdown reduces equity exposure to at most 50%, while the existing 15% hard drawdown guard stops risk for the rest of that rebalance period. The hard guard is an idealized research assumption, not guaranteed real-world execution. Monitoring also includes concentration, sector exposure, beta to SPY, rolling volatility, rolling Sharpe, turnover, and regime diagnostics.

## 10. Execution Cost Assumptions

The backtest applies 10 bps fixed transaction costs to monthly two-sided turnover and caps requested monthly L1 turnover at 0.50. Market-impact and TWAP/VWAP models are intentionally excluded.

## 11. Results

Final results should be read from the live dashboard because they update with the latest Yahoo Finance data. Poor ML performance versus SPY or momentum is reported directly rather than hidden.

## 12. Diagnostics

Diagnostics include benchmark comparison, top-minus-bottom ranking spread, monthly Rank IC, IC standard deviation, positive-IC month rate, monthly and annualized ICIR, IC t-statistic, 6- and 12-month rolling IC, rank turnover, gross-versus-net performance, target diagnostics, feature ablation, and portfolio construction comparison.

## 13. Limitations

The default static universe is not survivorship-bias-free. Survivorship-bias-free results require a populated point-in-time membership file from a historical constituent source such as CRSP/Compustat, WRDS, Bloomberg, or another point-in-time database. Yahoo Finance data can still have ticker gaps, missing history, and incomplete fundamentals. GDELT, Google Trends, SEC EDGAR, Black-Litterman, market-impact modeling, and TWAP/VWAP simulation are future work, not current results. Feature importance is not full SHAP unless SHAP is actually used. Selecting the best model, holding count, and weighting method on the same OOS history creates research-selection bias; a separate untouched holdout is still needed.

## 14. Future Work

Future work includes sourcing institutional point-in-time membership data, larger cached universes, optional XGBoost/LightGBM runs, richer alternative data, stricter sector constraints, and execution simulation improvements.
