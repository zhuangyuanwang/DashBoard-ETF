# Working Paper #2: ML-Powered Multi-Asset White-Box Allocation

## Abstract

This paper extends the Stage A1 linear benchmark into an intermediate ML-powered multi-asset framework. The system uses free market and macro data, white-box machine learning models, walk-forward validation, feature importance tracking, regime detection, and institutional-style portfolio construction methods.

## 1. Research Objective

Stage A2 asks whether transparent machine learning models can improve out-of-sample multi-asset allocation after realistic costs, square-root market impact, factor exposure monitoring, and stress testing.

## 2. Data

The current dashboard uses Yahoo Finance prices for a local equity universe, sector ETFs, and global proxies including EFA, EWJ, EEM, IWM, VGK, TLT, IEF, GLD, DBC, HYG, and LQD. It also attempts to load free FRED macro series such as Treasury yields, Fed funds, CPI, unemployment, and the 10Y-2Y yield curve.

## 3. Models

The Stage A2 dashboard includes:

- Decision Tree
- Random Forest
- Gradient Boosting
- Elastic Net

Tree-based feature importance is tracked through time as a white-box interpretability layer. XGBoost, LightGBM, and full SHAP can be added later as optional heavier dependencies.

## 4. Portfolio Construction

The framework converts monthly walk-forward predictions into:

- Hierarchical Risk Parity top ML basket
- Ledoit-Wolf shrinkage mean-variance portfolio
- Fractional Kelly portfolio
- Beta-neutral long/short ML portfolio

## 5. Risk Management

The dashboard includes:

- Bull, bear, and recovery regime visualization
- Factor exposure heatmaps
- Stress windows for 2008, 2020, and 2022 when data history overlaps
- Beta monitoring
- Rolling drawdown

## 6. Execution

Execution cost modeling uses fixed transaction bps plus a square-root market-impact penalty. The dashboard tracks monthly turnover, estimated impact cost, gross exposure, net exposure, and long/short counts.

## 7. Results

Results should be filled from the Stage A2 dashboard after selecting the final universe, model, cost assumptions, and Kelly fraction.

## 8. Limitations

This implementation is an A2 production-oriented MVP, not the final institutional research stack. Current limitations include simplified constituent history, simplified HMM-style regime detection through Gaussian mixture states, no full SHAP dependency by default, and limited alternative data beyond FRED and market proxies.

## 9. Next Steps

Recommended extensions:

- Add optional XGBoost or LightGBM.
- Add full SHAP TreeExplainer output.
- Add GDELT event/sentiment features.
- Add Google Trends topic features.
- Add SEC EDGAR earnings filing metadata.
- Add full HMM with transition matrix if `hmmlearn` is accepted as a dependency.
