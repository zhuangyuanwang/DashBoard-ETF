# Working Paper #1: Linear Benchmark Strategies for Equity and Sector Allocation

## Abstract

This paper evaluates whether transparent linear models can forecast one-month forward equity returns using free market data. The study implements OLS, Ridge, LASSO, and Elastic Net benchmarks, then converts forecasts into long-only, inverse-variance, market-neutral, and 130/30 portfolios. Performance is measured net of transaction costs, linear slippage, and a monthly turnover cap.

## 1. Introduction

Stage A1 establishes the baseline research process for the capstone. The goal is not to maximize backtest performance at all costs, but to build a reproducible benchmark that later ML, macro, causality, and derivatives systems must beat.

## 2. Data

The dashboard uses Yahoo Finance daily adjusted prices for a local S&P-style equity universe and sector ETFs. The current implementation uses the local `data/stock_universe.csv` file as the stock source and sector ETFs as liquid factor proxies.

## 3. Feature Engineering

The first feature set includes:

- 12-1 month momentum
- 6-month and 3-month momentum
- 3-month realized volatility, signed so lower volatility receives a higher score
- 6-month beta versus SPY
- 6-month drawdown
- 50-day moving average versus 200-day moving average
- 3-month market momentum

## 4. Models

The benchmark models are:

- OLS
- Ridge
- LASSO
- Elastic Net

A shallow Random Forest is included only as a nonlinear sanity check, not as the Stage A1 primary model.

## 5. Portfolio Construction

Out-of-sample walk-forward forecasts are converted into four portfolios:

- Equal-weight top bucket
- Inverse-variance risk parity top bucket
- Market-neutral long/short
- 130/30 extension

The implementation applies monthly rebalancing, fixed transaction costs, linear slippage, and a turnover cap.

## 6. Backtest Methodology

The target variable is the next 21-trading-day return. Model diagnostics use a 60/20/20 temporal split. The dashboard also includes combinatorial purged cross-validation to reduce leakage from overlapping forward-return labels. Portfolio returns are built with a monthly walk-forward process: at each rebalance, the model is trained only on labels that would have been known before the signal date, then the next month is traded out of sample.

## 7. Results

Results should be filled from the Stage A1 dashboard after selecting the final universe, model, and cost assumptions.

## 8. Robustness Checks

Recommended robustness checks:

- Compare 5 bps, 10 bps, and 20 bps costs.
- Compare 25%, 50%, and 100% turnover caps.
- Compare top 10%, top 20%, and top 30% selection buckets.
- Compare 3-year, 5-year, 7-year, and 10-year windows.
- Compare Ridge and Elastic Net as primary production candidates.

## 9. Limitations

This Stage A1 implementation does not yet include survivorship-bias-free historical constituents, point-in-time fundamentals, full Fama-French factor downloads, or institutional-grade execution simulation. These are planned extensions before Stage A2.

## 10. Conclusion

Stage A1 creates the transparent baseline required for the capstone. Later models in Stage A2 and Stage A3 should be evaluated against these interpretable benchmarks rather than against an artificially weak baseline.
