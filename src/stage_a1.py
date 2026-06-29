from __future__ import annotations

from itertools import combinations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLI", "XLP", "XLU", "XLV"]
DEFAULT_STAGE_A1_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "JPM",
    "V",
    "XOM",
    "LLY",
    "AVGO",
    "UNH",
    "MA",
    "PG",
    "HD",
    "COST",
    "BAC",
    "KO",
    "PEP",
    "CSCO",
    "SPY",
    *SECTOR_ETFS,
]


@dataclass(frozen=True)
class StageA1Config:
    symbols: tuple[str, ...]
    start_date: str
    end_date: str
    target_horizon_days: int
    transaction_cost_bps: float
    slippage_bps: float
    turnover_cap: float
    long_short_gross: float
    short_weight: float


def _safe_divide(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


@st.cache_data(ttl=3600, show_spinner=False)
def load_stage_a1_prices(symbols: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(
        list(symbols),
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    if data.empty or "Close" not in data:
        return pd.DataFrame()
    close = data["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close = close.dropna(how="all")
    missing_ratio = close.isna().mean()
    close = close.loc[:, missing_ratio <= 0.20].ffill(limit=5).dropna(how="all")
    return close.loc[:, ~close.columns.duplicated()].sort_index()


def build_stage_a1_features(close: pd.DataFrame, benchmark_symbol: str = "SPY") -> tuple[pd.DataFrame, pd.Series]:
    returns = close.pct_change(fill_method=None)
    forward_returns = close.pct_change(21, fill_method=None).shift(-21)
    spy_returns = returns[benchmark_symbol] if benchmark_symbol in returns else returns.mean(axis=1)

    rows = []
    targets = []
    for symbol in close.columns:
        if symbol == benchmark_symbol:
            continue
        symbol_returns = returns[symbol]
        rolling_cov = symbol_returns.rolling(126).cov(spy_returns)
        rolling_var = spy_returns.rolling(126).var()
        beta = rolling_cov / rolling_var.replace(0, np.nan)
        features = pd.DataFrame(
            {
                "symbol": symbol,
                "mom_12_1": close[symbol].pct_change(252, fill_method=None).shift(21)
                - close[symbol].pct_change(21, fill_method=None),
                "mom_6m": close[symbol].pct_change(126, fill_method=None),
                "mom_3m": close[symbol].pct_change(63, fill_method=None),
                "low_vol_3m": -symbol_returns.rolling(63).std() * np.sqrt(252),
                "beta_6m": beta,
                "drawdown_6m": close[symbol] / close[symbol].rolling(126).max() - 1,
                "ma50_vs_ma200": close[symbol].rolling(50).mean() / close[symbol].rolling(200).mean() - 1,
                "market_mom_3m": close[benchmark_symbol].pct_change(63, fill_method=None)
                if benchmark_symbol in close
                else close.mean(axis=1).pct_change(63, fill_method=None),
            }
        )
        rows.append(features)
        targets.append(forward_returns[symbol].rename(symbol))

    feature_panel = pd.concat(rows).reset_index(names="date")
    target_panel = pd.concat(targets, axis=1).stack(future_stack=True).rename("target_1m")
    target_panel.index.names = ["date", "symbol"]
    feature_panel = feature_panel.set_index(["date", "symbol"]).join(target_panel).dropna()
    y = feature_panel.pop("target_1m")
    return feature_panel, y


def temporal_train_validation_test_split(index: pd.Index) -> dict[str, pd.Index]:
    dates = pd.Index(sorted(index.get_level_values("date").unique()))
    n_dates = len(dates)
    train_end = dates[int(n_dates * 0.60)]
    validation_end = dates[int(n_dates * 0.80)]
    date_index = index.get_level_values("date")
    return {
        "train": index[date_index < train_end],
        "validation": index[(date_index >= train_end) & (date_index < validation_end)],
        "test": index[date_index >= validation_end],
    }


def create_stage_a1_model(model_name: str):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model_specs = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=5.0),
        "LASSO": Lasso(alpha=0.0005, max_iter=10000),
        "Elastic Net": ElasticNet(alpha=0.0005, l1_ratio=0.50, max_iter=10000),
        "RF Sanity Check": RandomForestRegressor(n_estimators=150, max_depth=4, min_samples_leaf=30, random_state=7),
    }
    return make_pipeline(StandardScaler(), model_specs[model_name])


def fit_stage_a1_models(x: pd.DataFrame, y: pd.Series, splits: dict[str, pd.Index]):
    model_names = ["OLS", "Ridge", "LASSO", "Elastic Net", "RF Sanity Check"]
    fitted = {}
    predictions = pd.DataFrame(index=x.index)
    metrics = []
    x_train = x.loc[splits["train"]]
    y_train = y.loc[splits["train"]]

    for name in model_names:
        model = create_stage_a1_model(name)
        model.fit(x_train, y_train)
        fitted[name] = model
        predictions[name] = model.predict(x)
        for split_name, split_index in splits.items():
            actual = y.loc[split_index]
            pred = predictions.loc[split_index, name]
            ic = pred.corr(actual, method="spearman")
            hit_rate = (np.sign(pred) == np.sign(actual)).mean()
            mse = ((pred - actual) ** 2).mean()
            metrics.append(
                {
                    "Model": name,
                    "Split": split_name,
                    "Information Coefficient": ic,
                    "Hit Rate": hit_rate,
                    "MSE": mse,
                    "Samples": len(split_index),
                }
            )
    return fitted, predictions, pd.DataFrame(metrics)


def build_walk_forward_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    month_ends: pd.Index,
    min_train_samples: int = 500,
    label_embargo_days: int = 35,
) -> tuple[pd.Series, pd.DataFrame]:
    row_dates = x.index.get_level_values("date")
    prediction_parts = []
    log_rows = []

    for month_end in month_ends:
        available_feature_dates = pd.Index(row_dates[row_dates <= month_end].unique())
        if available_feature_dates.empty:
            continue
        signal_date = available_feature_dates.max()
        label_cutoff = signal_date - pd.Timedelta(days=label_embargo_days)
        train_mask = row_dates <= label_cutoff
        prediction_mask = row_dates == signal_date

        if train_mask.sum() < min_train_samples or prediction_mask.sum() == 0:
            continue

        model = create_stage_a1_model(model_name)
        model.fit(x.loc[train_mask], y.loc[train_mask])
        prediction = pd.Series(model.predict(x.loc[prediction_mask]), index=x.loc[prediction_mask].index)
        prediction_parts.append(prediction)
        log_rows.append(
            {
                "Signal Date": signal_date,
                "Training Cutoff": label_cutoff,
                "Train Samples": int(train_mask.sum()),
                "Predicted Names": int(prediction_mask.sum()),
                "Model": model_name,
            }
        )

    if not prediction_parts:
        return pd.Series(dtype=float, name="score"), pd.DataFrame(log_rows)
    predictions = pd.concat(prediction_parts).sort_index()
    predictions.name = "score"
    return predictions, pd.DataFrame(log_rows)


def build_cpcv_summary(
    x: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 6,
    test_group_size: int = 2,
    purge_days: int = 21,
) -> pd.DataFrame:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dates = pd.Index(sorted(x.index.get_level_values("date").unique()))
    folds = np.array_split(dates, n_splits)
    rows = []
    for path_id, test_group_ids in enumerate(combinations(range(n_splits), test_group_size), start=1):
        test_dates = pd.Index([])
        for group_id in test_group_ids:
            test_dates = test_dates.append(pd.Index(folds[group_id]))
        test_start = test_dates.min()
        test_end = test_dates.max()
        row_dates = x.index.get_level_values("date")
        test_mask = row_dates.isin(test_dates)
        purged_mask = pd.Series(False, index=x.index)
        for group_id in test_group_ids:
            group_dates = pd.Index(folds[group_id])
            purge_start = group_dates.min() - pd.Timedelta(days=purge_days)
            purge_end = group_dates.max() + pd.Timedelta(days=purge_days)
            purged_mask = purged_mask | ((row_dates >= purge_start) & (row_dates <= purge_end))
        train_mask = (~test_mask) & (~purged_mask.to_numpy())
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
        model.fit(x.loc[train_mask], y.loc[train_mask])
        pred = pd.Series(model.predict(x.loc[test_mask]), index=x.loc[test_mask].index)
        actual = y.loc[test_mask]
        rows.append(
            {
                "Path": path_id,
                "Test Groups": ",".join(str(group_id + 1) for group_id in test_group_ids),
                "Test Start": test_start.date(),
                "Test End": test_end.date(),
                "Purged Days": purge_days,
                "Train Samples": int(train_mask.sum()),
                "Test Samples": int(test_mask.sum()),
                "IC": pred.corr(actual, method="spearman"),
                "Hit Rate": (np.sign(pred) == np.sign(actual)).mean(),
            }
        )
    return pd.DataFrame(rows)


def inverse_variance_weights(returns: pd.DataFrame) -> pd.Series:
    vol = returns.std().replace(0, np.nan)
    inv_var = 1 / (vol**2)
    weights = inv_var / inv_var.sum()
    return weights.fillna(0.0)


def build_stage_a1_portfolios(
    close: pd.DataFrame,
    predictions: pd.Series,
    transaction_cost_bps: float,
    slippage_bps: float,
    turnover_cap: float,
    short_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_returns = close.pct_change(fill_method=None).dropna()
    month_ends = close.resample("ME").last().index
    prediction_frame = predictions.rename("score").reset_index().set_index("date")
    transaction_cost = (transaction_cost_bps + slippage_bps) / 10000
    previous_weights = {
        "EW Top Decile": pd.Series(0.0, index=close.columns),
        "Risk Parity Top Decile": pd.Series(0.0, index=close.columns),
        "Market Neutral L/S": pd.Series(0.0, index=close.columns),
        "130/30 Extension": pd.Series(0.0, index=close.columns),
    }
    returns_by_strategy = {name: pd.Series(0.0, index=daily_returns.index) for name in previous_weights}
    turnover_rows = []

    for idx in range(12, len(month_ends) - 1):
        signal_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        available_dates = prediction_frame.index[prediction_frame.index <= signal_date]
        if len(available_dates) == 0:
            continue
        signal_rows = prediction_frame.loc[available_dates[-1]]
        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T
        signal_scores = signal_rows.set_index("symbol")["score"].dropna()
        tradable = [symbol for symbol in signal_scores.index if symbol in daily_returns.columns]
        if len(tradable) < 8:
            continue
        signal_scores = signal_scores.loc[tradable].sort_values(ascending=False)
        bucket = max(3, int(len(signal_scores) * 0.20))
        longs = signal_scores.head(bucket).index.tolist()
        shorts = signal_scores.tail(bucket).index.tolist()
        lookback = daily_returns.loc[:signal_date, longs].tail(63)

        target_weights = {}
        ew = pd.Series(0.0, index=close.columns)
        ew[longs] = 1 / len(longs)
        target_weights["EW Top Decile"] = ew

        rp = pd.Series(0.0, index=close.columns)
        rp_weights = inverse_variance_weights(lookback)
        rp[rp_weights.index] = rp_weights
        target_weights["Risk Parity Top Decile"] = rp

        ls = pd.Series(0.0, index=close.columns)
        ls[longs] = 0.50 / len(longs)
        ls[shorts] = -0.50 / len(shorts)
        target_weights["Market Neutral L/S"] = ls

        extension = pd.Series(0.0, index=close.columns)
        extension[longs] = 1.30 / len(longs)
        extension[shorts] = -short_weight / len(shorts)
        target_weights["130/30 Extension"] = extension

        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_date)
        if not period_mask.any():
            continue

        for strategy_name, weights in target_weights.items():
            turnover = (weights - previous_weights[strategy_name]).abs().sum()
            scale = min(1.0, turnover_cap / turnover) if turnover > turnover_cap else 1.0
            if scale < 1.0:
                weights = previous_weights[strategy_name] + (weights - previous_weights[strategy_name]) * scale
                turnover = turnover_cap
            period_returns = daily_returns.loc[period_mask, weights.index].dot(weights)
            if len(period_returns) > 0:
                period_returns.iloc[0] -= turnover * transaction_cost
            returns_by_strategy[strategy_name].loc[period_mask] = period_returns
            previous_weights[strategy_name] = weights
            turnover_rows.append(
                {
                    "Date": signal_date,
                    "Portfolio": strategy_name,
                    "Turnover": turnover,
                    "Long Count": int((weights > 0).sum()),
                    "Short Count": int((weights < 0).sum()),
                    "Gross Exposure": weights.abs().sum(),
                    "Net Exposure": weights.sum(),
                }
            )

    portfolio_returns = pd.DataFrame(returns_by_strategy).dropna(how="all")
    return portfolio_returns, pd.DataFrame(turnover_rows)


def calculate_stage_a1_metrics(returns: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name in returns.columns:
        series = returns[name].dropna()
        if series.empty:
            continue
        equity = (1 + series).cumprod()
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        cagr = equity.iloc[-1] ** (1 / years) - 1
        ann_vol = series.std() * np.sqrt(252)
        downside = series[series < 0].std() * np.sqrt(252)
        sharpe = series.mean() * 252 / ann_vol if ann_vol > 0 else np.nan
        sortino = series.mean() * 252 / downside if downside > 0 else np.nan
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
        info_ratio = np.nan
        beta = np.nan
        alpha = np.nan
        if benchmark is not None and not benchmark.empty:
            common = series.index.intersection(benchmark.index)
            if len(common) > 20:
                active = series.loc[common] - benchmark.loc[common]
                info_ratio = active.mean() * 252 / (active.std() * np.sqrt(252)) if active.std() > 0 else np.nan
                cov = series.loc[common].cov(benchmark.loc[common])
                beta = cov / benchmark.loc[common].var() if benchmark.loc[common].var() > 0 else np.nan
                alpha = series.loc[common].mean() * 252 - beta * benchmark.loc[common].mean() * 252 if pd.notna(beta) else np.nan
        rows.append(
            {
                "Series": name,
                "CAGR": cagr,
                "Annual Volatility": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max Drawdown": max_dd,
                "Calmar": calmar,
                "Information Ratio": info_ratio,
                "Alpha": alpha,
                "Beta": beta,
                "Hit Rate": (series > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def render_stage_a1_dashboard(stock_universe_file) -> None:
    st.title("Stage A1 Research Lab: Linear Benchmarks")
    st.write(
        "A Stage A1 capstone implementation: free data, linear return models, 60/20/20 validation, "
        "purged CV sanity checks, long-only and long/short portfolios, costs, turnover limits, and research-grade metrics."
    )

    universe_table = pd.read_csv(stock_universe_file) if stock_universe_file.exists() else pd.DataFrame()
    default_stocks = universe_table["Ticker"].dropna().astype(str).str.upper().head(30).tolist() if "Ticker" in universe_table else []
    default_symbols = tuple(dict.fromkeys([*(default_stocks or DEFAULT_STAGE_A1_UNIVERSE), "SPY", *SECTOR_ETFS]))

    with st.sidebar:
        st.header("Stage A1 Controls")
        years = st.slider("Research window", min_value=3, max_value=12, value=7, step=1)
        max_names = st.slider("Max stock names", min_value=10, max_value=min(120, len(default_symbols)), value=min(40, len(default_symbols)), step=5)
        model_name = st.selectbox("Portfolio signal model", ["Ridge", "Elastic Net", "LASSO", "OLS", "RF Sanity Check"], index=0)
        transaction_cost_bps = st.slider("Transaction cost bps", min_value=5, max_value=20, value=10, step=1)
        slippage_bps = st.slider("Linear slippage bps", min_value=0, max_value=20, value=5, step=1)
        turnover_cap = st.slider("Monthly turnover cap", min_value=0.10, max_value=1.00, value=0.50, step=0.05)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    selected_symbols = tuple(dict.fromkeys(list(default_symbols[:max_names]) + ["SPY", *SECTOR_ETFS]))

    with st.spinner("Loading Yahoo Finance prices for Stage A1..."):
        close = load_stage_a1_prices(selected_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    if close.empty or close.shape[1] < 8:
        st.error("Not enough price data loaded for Stage A1. Try a smaller universe or a different date window.")
        return

    with st.spinner("Building features, training models, and running Stage A1 backtests..."):
        x, y = build_stage_a1_features(close)
        splits = temporal_train_validation_test_split(x.index)
        fitted, predictions, model_metrics = fit_stage_a1_models(x, y, splits)
        cpcv = build_cpcv_summary(x, y)
        month_ends = close.resample("ME").last().index
        walk_forward_predictions, walk_forward_log = build_walk_forward_predictions(x, y, model_name, month_ends)
        portfolio_returns, turnover = build_stage_a1_portfolios(
            close,
            walk_forward_predictions,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            turnover_cap=turnover_cap,
            short_weight=0.30,
        )
        if portfolio_returns.empty:
            st.error("Not enough walk-forward predictions to build a true out-of-sample portfolio return.")
            return
        benchmark = close["SPY"].pct_change(fill_method=None).reindex(portfolio_returns.index).fillna(0.0) if "SPY" in close else None
        metrics = calculate_stage_a1_metrics(portfolio_returns, benchmark=benchmark)

    st.caption(
        f"Loaded {close.shape[1]} symbols from {close.index.min().date()} to {close.index.max().date()}. "
        f"Target is next 21-trading-day return. Costs = {transaction_cost_bps}+{slippage_bps} bps per unit turnover."
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric("Feature Samples", f"{len(x):,}")
    summary_cols[1].metric("Tradable Symbols", close.shape[1])
    summary_cols[2].metric("Train / Val / Test", "60 / 20 / 20")
    summary_cols[3].metric("OOS Rebalances", f"{len(walk_forward_log):,}")

    overview_tab, models_tab, portfolio_tab, validation_tab, paper_tab = st.tabs(
        ["Overview", "Linear Models", "Portfolio Backtest", "CPCV", "Working Paper"]
    )

    with overview_tab:
        st.subheader("Stage A1 Completion Map")
        completion = pd.DataFrame(
            [
                ("Universe", "S&P-style local stock universe + sector ETFs", "Implemented"),
                ("Models", "OLS, Ridge, LASSO, Elastic Net plus RF sanity check", "Implemented"),
                ("Features", "12-1 momentum, 3M/6M momentum, low-vol, beta, drawdown, trend", "Implemented"),
                ("Portfolio", "Walk-forward EW, inverse-variance risk parity, market-neutral L/S, 130/30", "Implemented"),
                ("Validation", "60/20/20 split plus combinatorial purged CV", "Implemented"),
                ("Execution", "10-20 bps costs, linear slippage, 50% turnover cap", "Implemented"),
                ("Deliverable", "Dashboard + paper template", "Implemented"),
            ],
            columns=["Element", "Stage A1 Target", "Status"],
        )
        st.dataframe(completion, use_container_width=True, hide_index=True)
        st.plotly_chart(px.line(close[["SPY", *[s for s in SECTOR_ETFS if s in close.columns]]].dropna(), title="Benchmark and Sector ETF Prices"), use_container_width=True)

    with models_tab:
        st.subheader("Out-of-Sample Model Diagnostics")
        st.dataframe(
            model_metrics.style.format(
                {
                    "Information Coefficient": "{:.3f}",
                    "Hit Rate": "{:.1%}",
                    "MSE": "{:.6f}",
                    "Samples": "{:,.0f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        feature_names = x.columns.tolist()
        linear_model = fitted.get(model_name)
        coefficients = None
        if linear_model is not None and hasattr(linear_model[-1], "coef_"):
            coefficients = pd.DataFrame({"Feature": feature_names, "Coefficient": linear_model[-1].coef_}).sort_values("Coefficient")
        if coefficients is not None:
            st.plotly_chart(px.bar(coefficients, x="Coefficient", y="Feature", orientation="h", title=f"{model_name} Feature Coefficients"), use_container_width=True)

    with portfolio_tab:
        st.subheader("True Walk-Forward Net Portfolio Performance")
        st.caption(
            "This return chart is built only from out-of-sample monthly predictions. "
            "Each rebalance trains the selected model using data available before the signal date, then trades the following month."
        )
        equity = (1 + portfolio_returns).cumprod()
        if benchmark is not None:
            equity["SPY Benchmark"] = (1 + benchmark).cumprod()
        st.plotly_chart(px.line(equity, x=equity.index, y=equity.columns, title="Growth of 1.0"), use_container_width=True)
        st.dataframe(
            metrics.style.format(
                {
                    "CAGR": "{:.2%}",
                    "Annual Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Calmar": "{:.2f}",
                    "Information Ratio": "{:.2f}",
                    "Alpha": "{:.2%}",
                    "Beta": "{:.2f}",
                    "Hit Rate": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        if not turnover.empty:
            st.subheader("Execution and Turnover")
            st.plotly_chart(px.line(turnover, x="Date", y="Turnover", color="Portfolio", title="Monthly Turnover After Cap"), use_container_width=True)
            st.dataframe(turnover.tail(30), use_container_width=True, hide_index=True)
        if not walk_forward_log.empty:
            st.subheader("Walk-Forward Training Log")
            st.dataframe(walk_forward_log.tail(24), use_container_width=True, hide_index=True)

    with validation_tab:
        st.subheader("Purged K-Fold Validation Sanity Check")
        st.write("This uses combinations of time groups as test paths and purges nearby observations around each test group.")
        st.dataframe(
            cpcv.style.format({"IC": "{:.3f}", "Hit Rate": "{:.1%}", "Train Samples": "{:,.0f}", "Test Samples": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )

    with paper_tab:
        st.subheader("Working Paper #1 Checklist")
        st.markdown(
            """
The project now has the implementation pieces needed for Working Paper #1:

- Data: Yahoo Finance daily prices, local stock universe, sector ETFs.
- Methods: OLS, Ridge, LASSO, Elastic Net, feature selection proxy through coefficients.
- Validation: 60/20/20 temporal split and purged k-fold diagnostic.
- Portfolios: equal-weight, inverse-variance risk parity, market-neutral long/short, 130/30.
- Execution: fixed costs, linear slippage, monthly turnover cap.
- Metrics: Sharpe, Sortino, max drawdown, Calmar, information ratio, alpha, beta, hit rate.

Use `reports/working_paper_1.md` as the academic-format draft.
"""
        )
