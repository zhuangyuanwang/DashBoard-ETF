from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


STAGE_A2_INITIAL_CAPITAL = 1_000_000
SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLI", "XLP", "XLU", "XLV"]
GLOBAL_PROXIES = ["SPY", "EFA", "EWJ", "EEM", "IWM", "VGK", "TLT", "IEF", "GLD", "DBC", "HYG", "LQD"]
MAX_LONG_ONLY_WEIGHT = 0.25
DEFAULT_A2_STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "AVGO",
    "JPM",
    "LLY",
    "XOM",
    "V",
    "MA",
    "UNH",
    "HD",
    "PG",
    "COST",
    "BAC",
    "KO",
    "PEP",
    "CSCO",
    "WMT",
    "CVX",
    "MRK",
    "ABBV",
    "CRM",
    "AMD",
    "NFLX",
    "ADBE",
    "MCD",
    "TMO",
]


@st.cache_data(ttl=3600, show_spinner=False)
def load_stage_a2_prices(symbols: tuple[str, ...], start_date: str, end_date: str) -> pd.DataFrame:
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
    close = close.loc[:, missing_ratio <= 0.25].ffill(limit=5).dropna(how="all")
    return close.loc[:, ~close.columns.duplicated()].sort_index()


@st.cache_data(ttl=86400, show_spinner=False)
def load_fred_macro(start_date: str, end_date: str) -> pd.DataFrame:
    series_map = {
        "DGS10": "ten_year_yield",
        "DGS2": "two_year_yield",
        "DFF": "fed_funds",
        "CPIAUCSL": "cpi",
        "UNRATE": "unemployment",
    }
    frames = []
    for fred_id, name in series_map.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
            series = pd.read_csv(url)
            series["observation_date"] = pd.to_datetime(series["observation_date"])
            series = series.set_index("observation_date")[fred_id].replace(".", np.nan).astype(float)
            frames.append(series.rename(name))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    macro = pd.concat(frames, axis=1).sort_index().ffill()
    macro = macro.loc[(macro.index >= pd.to_datetime(start_date)) & (macro.index <= pd.to_datetime(end_date))]
    if {"ten_year_yield", "two_year_yield"}.issubset(macro.columns):
        macro["yield_curve_10y2y"] = macro["ten_year_yield"] - macro["two_year_yield"]
    if "cpi" in macro.columns:
        macro["inflation_yoy"] = macro["cpi"].pct_change(12, fill_method=None)
    return macro


def build_stage_a2_features(close: pd.DataFrame, macro: pd.DataFrame, benchmark_symbol: str = "SPY") -> tuple[pd.DataFrame, pd.Series]:
    monthly = close.resample("ME").last()
    monthly_returns = monthly.pct_change(fill_method=None)
    daily_returns = close.pct_change(fill_method=None)
    target = monthly_returns.shift(-1)
    spy = monthly_returns[benchmark_symbol] if benchmark_symbol in monthly_returns else monthly_returns.mean(axis=1)

    macro_monthly = pd.DataFrame(index=monthly.index)
    if not macro.empty:
        # FRED fields are aligned to month-end using observations available in
        # the downloaded time series. This is a research proxy, not a full
        # point-in-time release-calendar model for macro revisions.
        macro_monthly = macro.resample("ME").last().reindex(monthly.index).ffill()
        macro_monthly = macro_monthly.pct_change(fill_method=None).add_prefix("macro_delta_").join(
            macro.resample("ME").last().reindex(monthly.index).ffill().add_prefix("macro_level_")
        )

    rows = []
    targets = []
    for symbol in monthly.columns:
        if symbol == benchmark_symbol:
            continue
        symbol_daily = daily_returns[symbol]
        beta = symbol_daily.rolling(126).cov(daily_returns[benchmark_symbol]) / daily_returns[benchmark_symbol].rolling(126).var() if benchmark_symbol in daily_returns else np.nan
        features = pd.DataFrame(
            {
                "symbol": symbol,
                "ret_1m": monthly[symbol].pct_change(1, fill_method=None),
                "ret_3m": monthly[symbol].pct_change(3, fill_method=None),
                "ret_6m": monthly[symbol].pct_change(6, fill_method=None),
                "ret_12m": monthly[symbol].pct_change(12, fill_method=None),
                "vol_3m": monthly_returns[symbol].rolling(3).std(),
                "vol_12m": monthly_returns[symbol].rolling(12).std(),
                "drawdown_12m": monthly[symbol] / monthly[symbol].rolling(12).max() - 1,
                "beta_6m": beta.resample("ME").last().reindex(monthly.index),
                "market_ret_3m": monthly[benchmark_symbol].pct_change(3, fill_method=None) if benchmark_symbol in monthly else monthly.mean(axis=1).pct_change(3, fill_method=None),
                "global_risk": monthly_returns[[symbol for symbol in ["EFA", "EEM", "IWM", "VGK"] if symbol in monthly_returns]].mean(axis=1),
                "rates_proxy": monthly_returns[[symbol for symbol in ["TLT", "IEF"] if symbol in monthly_returns]].mean(axis=1),
                "credit_proxy": monthly_returns[[symbol for symbol in ["HYG", "LQD"] if symbol in monthly_returns]].mean(axis=1),
                "commodity_proxy": monthly_returns[[symbol for symbol in ["GLD", "DBC"] if symbol in monthly_returns]].mean(axis=1),
                "spy_trend_state": (spy.rolling(3).mean() > 0).astype(float),
            },
            index=monthly.index,
        )
        if not macro_monthly.empty:
            features = features.join(macro_monthly)
        rows.append(features)
        targets.append(target[symbol].rename(symbol))

    feature_panel = pd.concat(rows).reset_index(names="date")
    target_panel = pd.concat(targets, axis=1).stack(future_stack=True).rename("target_1m")
    target_panel.index.names = ["date", "symbol"]
    feature_panel = feature_panel.set_index(["date", "symbol"]).join(target_panel).replace([np.inf, -np.inf], np.nan)
    feature_panel = feature_panel.dropna(subset=["target_1m"])
    y = feature_panel.pop("target_1m")
    feature_panel = feature_panel.dropna(axis=1, how="all")
    feature_panel = feature_panel.apply(lambda column: column.fillna(column.median()), axis=0).fillna(0.0)
    return feature_panel, y


def create_stage_a2_model(model_name: str):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    if model_name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=11)
    if model_name == "Random Forest":
        return RandomForestRegressor(n_estimators=120, max_depth=5, min_samples_leaf=12, random_state=11, n_jobs=-1)
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=120, max_depth=3, learning_rate=0.035, random_state=11)
    if model_name == "Elastic Net":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.35, max_iter=10000))
    raise ValueError(f"Unsupported Stage A2 model: {model_name}")


def build_walk_forward_ml_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    min_train_samples: int = 300,
    min_history_months: int = 18,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    # Walk-forward discipline:
    # - Each signal_date trains only on rows with feature dates strictly before signal_date.
    # - The target is next-month return from monthly_returns.shift(-1), used only in training.
    # - Prediction rows are exactly the cross-section at signal_date; portfolio returns are
    #   earned in the following month inside build_stage_a2_portfolios.
    dates = pd.Index(sorted(x.index.get_level_values("date").unique()))
    row_dates = x.index.get_level_values("date")
    predictions = []
    log_rows = []
    importance_rows = []

    start_idx = min(min_history_months, max(1, len(dates) - 2))
    for signal_date in dates[start_idx:-1]:
        train_mask = row_dates < signal_date
        predict_mask = row_dates == signal_date
        if train_mask.sum() < min_train_samples or predict_mask.sum() == 0:
            continue
        model = create_stage_a2_model(model_name)
        model.fit(x.loc[train_mask], y.loc[train_mask])
        prediction = pd.Series(model.predict(x.loc[predict_mask]), index=x.loc[predict_mask].index)
        predictions.append(prediction)
        log_rows.append(
            {
                "Train Start": row_dates[train_mask].min(),
                "Train End": row_dates[train_mask].max(),
                "Prediction Month": signal_date,
                "Signal Date": signal_date,
                "Train Samples": int(train_mask.sum()),
                "Predicted Assets": int(predict_mask.sum()),
                "Model": model_name,
                "Prediction Source": "ML",
            }
        )
        importance = extract_feature_importance(model, x.columns)
        if not importance.empty:
            importance["Signal Date"] = signal_date
            importance_rows.append(importance)

    if not predictions:
        return pd.Series(dtype=float, name="score"), pd.DataFrame(log_rows), pd.DataFrame()
    signal = pd.concat(predictions).sort_index()
    signal.name = "score"
    importance_history = pd.concat(importance_rows, ignore_index=True) if importance_rows else pd.DataFrame()
    return signal, pd.DataFrame(log_rows), importance_history


def build_white_box_fallback_predictions(x: pd.DataFrame, min_history_months: int = 12) -> tuple[pd.Series, pd.DataFrame]:
    if x.empty:
        return pd.Series(dtype=float, name="score"), pd.DataFrame()
    dates = pd.Index(sorted(x.index.get_level_values("date").unique()))
    row_dates = x.index.get_level_values("date")
    prediction_parts = []
    log_rows = []
    start_idx = min(min_history_months, max(1, len(dates) - 2))
    for signal_date in dates[start_idx:-1]:
        cross_section = x.loc[row_dates == signal_date]
        if cross_section.empty:
            continue
        score = pd.Series(0.0, index=cross_section.index)
        for feature, weight in {
            "ret_12m": 0.30,
            "ret_6m": 0.25,
            "ret_3m": 0.20,
            "market_ret_3m": 0.10,
            "vol_12m": -0.10,
            "drawdown_12m": 0.05,
        }.items():
            if feature in cross_section:
                values = cross_section[feature]
                z_score = (values - values.mean()) / values.std() if values.std() > 0 else values * 0
                score = score + weight * z_score.fillna(0.0)
        prediction_parts.append(score)
        log_rows.append(
            {
                "Train Start": pd.NaT,
                "Train End": pd.NaT,
                "Prediction Month": signal_date,
                "Signal Date": signal_date,
                "Train Samples": 0,
                "Predicted Assets": int(len(score)),
                "Model": "White-Box Fallback",
                "Prediction Source": "Fallback",
            }
        )
    if not prediction_parts:
        return pd.Series(dtype=float, name="score"), pd.DataFrame(log_rows)
    predictions = pd.concat(prediction_parts).sort_index()
    predictions.name = "score"
    return predictions, pd.DataFrame(log_rows)


def extract_feature_importance(model, feature_names: pd.Index) -> pd.DataFrame:
    estimator = model[-1] if hasattr(model, "steps") else model
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    else:
        return pd.DataFrame()
    importance = pd.DataFrame({"Feature": feature_names, "Importance": values})
    total = importance["Importance"].abs().sum()
    if total > 0:
        importance["Importance"] = importance["Importance"].abs() / total
    return importance.sort_values("Importance", ascending=False).head(20)


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    returns = returns.dropna(axis=1, how="all").fillna(0.0)
    if returns.shape[1] == 0:
        return pd.Series(dtype=float)
    if returns.shape[1] == 1:
        return pd.Series(1.0, index=returns.columns)
    cov = returns.cov()
    corr = cov_to_corr(cov).clip(-1, 1)
    distance = np.sqrt((1 - corr) / 2)
    order = leaves_list(linkage(squareform(distance, checks=False), method="single"))
    sorted_assets = list(corr.index[order])
    weights = pd.Series(1.0, index=sorted_assets)

    clusters = [sorted_assets]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        left_var = cluster_variance(cov, left)
        right_var = cluster_variance(cov, right)
        alpha = 1 - left_var / (left_var + right_var) if (left_var + right_var) > 0 else 0.5
        weights[left] *= alpha
        weights[right] *= 1 - alpha
        clusters.extend([left, right])

    weights = weights.reindex(returns.columns).fillna(0.0)
    return weights / weights.sum()


def cap_and_redistribute_long_weights(weights: pd.Series, max_weight: float = MAX_LONG_ONLY_WEIGHT) -> tuple[pd.Series, bool]:
    weights = weights.clip(lower=0).fillna(0.0)
    if weights.sum() <= 0:
        return weights, False
    weights = weights / weights.sum()
    capped = weights.copy()
    was_capped = False
    for _ in range(20):
        over = capped > max_weight
        if not over.any():
            break
        was_capped = True
        excess = (capped[over] - max_weight).sum()
        capped[over] = max_weight
        under = capped < max_weight
        if not under.any() or excess <= 0:
            break
        under_weights = capped[under]
        if under_weights.sum() <= 0:
            capped[under] += excess / under.sum()
        else:
            capped[under] += excess * under_weights / under_weights.sum()
    return capped / capped.sum() if capped.sum() > 0 else capped, was_capped


def cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    sub_cov = cov.loc[assets, assets]
    inv_diag = 1 / np.diag(sub_cov).clip(min=1e-12)
    weights = inv_diag / inv_diag.sum()
    return float(weights.T @ sub_cov.values @ weights)


def ledoit_wolf_weights(returns: pd.DataFrame, scores: pd.Series) -> pd.Series:
    from sklearn.covariance import LedoitWolf

    assets = [asset for asset in scores.index if asset in returns.columns]
    if not assets:
        return pd.Series(dtype=float)
    clean = returns[assets].dropna().tail(252)
    if clean.shape[0] < 40:
        return pd.Series(1 / len(assets), index=assets)
    cov = LedoitWolf().fit(clean.fillna(0.0)).covariance_
    mu = scores.loc[assets].clip(lower=0).values
    if mu.sum() <= 0:
        mu = np.ones(len(assets))
    raw = np.linalg.pinv(cov) @ mu
    raw = np.clip(raw, 0, None)
    weights = pd.Series(raw, index=assets)
    if weights.sum() <= 0:
        weights = pd.Series(1 / len(assets), index=assets)
    weights, _ = cap_and_redistribute_long_weights(weights)
    return weights


def fractional_kelly_weights(returns: pd.DataFrame, scores: pd.Series, fraction: float) -> pd.Series:
    assets = [asset for asset in scores.index if asset in returns.columns]
    if not assets:
        return pd.Series(dtype=float)
    variance = returns[assets].tail(252).var().replace(0, np.nan)
    raw = (scores.loc[assets].clip(lower=0) / variance).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0, index=assets)
    weights, _ = cap_and_redistribute_long_weights(raw)
    return weights * fraction


def beta_neutral_weights(returns: pd.DataFrame, scores: pd.Series, benchmark: str = "SPY") -> pd.Series:
    ranked = scores.dropna().sort_values(ascending=False)
    bucket = max(3, int(len(ranked) * 0.20))
    longs = ranked.head(bucket).index.tolist()
    shorts = ranked.tail(bucket).index.tolist()
    weights = pd.Series(0.0, index=ranked.index)
    weights[longs] = 0.5 / len(longs)
    weights[shorts] = -0.5 / len(shorts)
    if benchmark in returns.columns:
        benchmark_returns = returns[benchmark].tail(252)
        benchmark_var = benchmark_returns.var()
        beta = returns[ranked.index].tail(252).apply(
            lambda series: series.cov(benchmark_returns) / benchmark_var if benchmark_var > 0 else np.nan
        )
        portfolio_beta = float((weights * beta.reindex(weights.index).fillna(1.0)).sum())
        if abs(portfolio_beta) > 0.05 and benchmark in weights.index:
            weights[benchmark] = weights.get(benchmark, 0.0) - portfolio_beta
    return weights


def apply_square_root_market_impact(weights: pd.Series, previous: pd.Series, base_bps: float, impact_bps: float) -> float:
    trade_size = (weights - previous.reindex(weights.index).fillna(0.0)).abs()
    turnover = trade_size.sum()
    return turnover * base_bps / 10000 + np.sqrt(trade_size.clip(lower=0)).sum() * impact_bps / 10000


def classify_risk_overlay_state(lookback: pd.DataFrame, benchmark: str = "SPY") -> tuple[str, float]:
    if benchmark not in lookback or lookback[benchmark].dropna().shape[0] < 126:
        return "Neutral", 1.0
    benchmark_returns = lookback[benchmark].dropna()
    cumulative = (1 + benchmark_returns).cumprod()
    drawdown = cumulative.iloc[-1] / cumulative.cummax().iloc[-1] - 1
    vol_3m = benchmark_returns.tail(63).std() * np.sqrt(252)
    ret_3m = cumulative.iloc[-1] / cumulative.iloc[max(0, len(cumulative) - 64)] - 1 if len(cumulative) > 64 else 0.0
    if drawdown < -0.15 or vol_3m > 0.30:
        return "Bear / High Vol", 0.45
    if drawdown < -0.08 or ret_3m < -0.06:
        return "Defensive", 0.65
    if ret_3m > 0.08 and vol_3m < 0.22:
        return "Risk-On", 1.0
    return "Neutral", 0.85


def apply_a2_risk_overlay(
    weights: pd.Series,
    previous_weights: pd.Series,
    lookback_returns: pd.DataFrame,
    target_volatility: float,
    smoothing: float,
    enable_regime_overlay: bool,
) -> tuple[pd.Series, dict]:
    weights = weights.fillna(0.0)
    previous = previous_weights.reindex(weights.index).fillna(0.0)
    smoothed = previous * smoothing + weights * (1 - smoothing)

    realized_vol = np.nan
    vol_scale = 1.0
    if not lookback_returns.empty and smoothed.abs().sum() > 0:
        aligned = lookback_returns.reindex(columns=smoothed.index).fillna(0.0)
        portfolio_returns = aligned.dot(smoothed).tail(63)
        realized_vol = portfolio_returns.std() * np.sqrt(252)
        if pd.notna(realized_vol) and realized_vol > 0 and target_volatility > 0:
            vol_scale = min(1.0, target_volatility / realized_vol)

    regime_state, regime_scale = classify_risk_overlay_state(lookback_returns)
    if not enable_regime_overlay:
        regime_state, regime_scale = "Disabled", 1.0

    final_scale = min(vol_scale, regime_scale)
    adjusted = smoothed * final_scale
    return adjusted, {
        "Risk Overlay State": regime_state,
        "Realized Volatility": realized_vol,
        "Vol Target Scale": vol_scale,
        "Regime Scale": regime_scale,
        "Final Risk Scale": final_scale,
        "Smoothing": smoothing,
    }


def build_stage_a2_portfolios(
    close: pd.DataFrame,
    predictions: pd.Series,
    base_cost_bps: float,
    impact_bps: float,
    kelly_fraction: float,
    target_volatility: float,
    smoothing: float,
    enable_regime_overlay: bool,
    max_drawdown_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    daily_returns = close.pct_change(fill_method=None).dropna()
    prediction_frame = predictions.rename("score").reset_index().set_index("date")
    month_ends = close.resample("ME").last().index
    strategy_names = ["HRP-style / Risk-Parity Fallback", "Ledoit-Wolf Mean-Variance", "Fractional Kelly", "Beta-Neutral ML"]
    returns_by_strategy = {name: pd.Series(0.0, index=daily_returns.index) for name in strategy_names}
    previous_weights = {name: pd.Series(0.0, index=close.columns) for name in strategy_names}
    equity_state = {name: 1.0 for name in strategy_names}
    peak_state = {name: 1.0 for name in strategy_names}
    execution_rows = []

    for idx in range(18, len(month_ends) - 1):
        signal_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        available_dates = prediction_frame.index[prediction_frame.index <= signal_date]
        if len(available_dates) == 0:
            continue
        signal_rows = prediction_frame.loc[available_dates[-1]]
        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T
        scores = signal_rows.set_index("symbol")["score"].dropna().sort_values(ascending=False)
        tradable = [symbol for symbol in scores.index if symbol in daily_returns.columns]
        if len(tradable) < 8:
            continue
        scores = scores.loc[tradable]
        top_assets = scores.head(max(5, int(len(scores) * 0.30))).index.tolist()
        lookback = daily_returns.loc[:signal_date].tail(252)

        target_weights = {}
        hrp_weight, _ = cap_and_redistribute_long_weights(hrp_weights(lookback[top_assets]))
        target_weights["HRP-style / Risk-Parity Fallback"] = hrp_weight.reindex(close.columns).fillna(0.0)
        target_weights["Ledoit-Wolf Mean-Variance"] = ledoit_wolf_weights(lookback, scores.head(max(5, int(len(scores) * 0.30)))).reindex(close.columns).fillna(0.0)
        target_weights["Fractional Kelly"] = fractional_kelly_weights(lookback, scores.head(max(5, int(len(scores) * 0.30))), kelly_fraction).reindex(close.columns).fillna(0.0)
        target_weights["Beta-Neutral ML"] = beta_neutral_weights(lookback, scores).reindex(close.columns).fillna(0.0)

        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_date)
        if not period_mask.any():
            continue
        for strategy_name, weights in target_weights.items():
            weights, overlay_info = apply_a2_risk_overlay(
                weights,
                previous_weights[strategy_name],
                lookback,
                target_volatility,
                smoothing,
                enable_regime_overlay,
            )
            cost = apply_square_root_market_impact(weights, previous_weights[strategy_name], base_cost_bps, impact_bps)
            raw_period_returns = daily_returns.loc[period_mask, weights.index].dot(weights)
            period_returns = raw_period_returns.copy()
            drawdown_guard_triggered = False
            stopped_for_period = False
            for day_idx, day in enumerate(period_returns.index):
                if stopped_for_period:
                    period_returns.loc[day] = 0.0
                    continue
                net_return = raw_period_returns.loc[day]
                if day_idx == 0:
                    net_return -= cost
                peak = max(peak_state[strategy_name], equity_state[strategy_name])
                tentative_equity = equity_state[strategy_name] * (1 + net_return)
                if tentative_equity / peak - 1 < -max_drawdown_limit:
                    floor_equity = peak * (1 - max_drawdown_limit)
                    net_return = floor_equity / equity_state[strategy_name] - 1
                    tentative_equity = floor_equity
                    drawdown_guard_triggered = True
                    stopped_for_period = True
                period_returns.loc[day] = net_return
                equity_state[strategy_name] = tentative_equity
                peak_state[strategy_name] = max(peak_state[strategy_name], equity_state[strategy_name])
            returns_by_strategy[strategy_name].loc[period_mask] = period_returns
            turnover = (weights - previous_weights[strategy_name]).abs().sum()
            execution_rows.append(
                {
                    "Date": signal_date,
                    "Portfolio": strategy_name,
                    "Turnover": turnover,
                    "Market Impact Cost": cost,
                    "Gross Exposure": weights.abs().sum(),
                    "Net Exposure": weights.sum(),
                    "Max Position Weight": weights.max(),
                    "Concentration Warning": "Yes" if weights.max() > MAX_LONG_ONLY_WEIGHT + 1e-9 else "No",
                    "Long Count": int((weights > 0).sum()),
                    "Short Count": int((weights < 0).sum()),
                    "Max Drawdown Limit": max_drawdown_limit,
                    "Drawdown Guard Triggered": "Yes" if drawdown_guard_triggered else "No",
                    **overlay_info,
                }
            )
            previous_weights[strategy_name] = weights

    return pd.DataFrame(returns_by_strategy).dropna(how="all"), pd.DataFrame(execution_rows)


def detect_regime_states(close: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
    from sklearn.mixture import GaussianMixture

    if benchmark not in close:
        return pd.DataFrame()
    monthly = close[benchmark].resample("ME").last()
    returns = monthly.pct_change(fill_method=None)
    features = pd.DataFrame(
        {
            "return_1m": returns,
            "vol_6m": returns.rolling(6).std(),
            "drawdown_12m": monthly / monthly.rolling(12).max() - 1,
        }
    ).dropna()
    if len(features) < 18:
        return pd.DataFrame()
    model = GaussianMixture(n_components=3, covariance_type="full", random_state=7)
    labels = model.fit_predict(features)
    state_returns = pd.Series(labels, index=features.index).to_frame("State").join(features).groupby("State")["return_1m"].mean()
    ordered = state_returns.sort_values().index.tolist()
    names = {ordered[0]: "Bear", ordered[1]: "Recovery", ordered[2]: "Bull"}
    result = features.copy()
    result["State Id"] = labels
    result["Regime"] = [names[label] for label in labels]
    result["Regime Code"] = result["Regime"].map({"Bear": 0, "Recovery": 1, "Bull": 2})
    return result


def calculate_metrics(returns: pd.DataFrame, benchmark: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for name in returns.columns:
        series = returns[name].dropna()
        if series.empty:
            continue
        equity = (1 + series).cumprod()
        years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
        cagr = equity.iloc[-1] ** (1 / years) - 1
        ann_vol = series.std() * np.sqrt(252)
        sharpe = series.mean() * 252 / ann_vol if ann_vol > 0 else np.nan
        downside = series[series < 0].std() * np.sqrt(252)
        sortino = series.mean() * 252 / downside if downside > 0 else np.nan
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan
        beta = np.nan
        alpha = np.nan
        if benchmark is not None and not benchmark.empty:
            common = series.index.intersection(benchmark.index)
            if len(common) > 20 and benchmark.loc[common].var() > 0:
                beta = series.loc[common].cov(benchmark.loc[common]) / benchmark.loc[common].var()
                alpha = series.loc[common].mean() * 252 - beta * benchmark.loc[common].mean() * 252
        rows.append(
            {
                "Series": name,
                "Ending Value": STAGE_A2_INITIAL_CAPITAL * equity.iloc[-1],
                "Total PnL": STAGE_A2_INITIAL_CAPITAL * (equity.iloc[-1] - 1),
                "Total Return": equity.iloc[-1] - 1,
                "CAGR": cagr,
                "Annual Volatility": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max Drawdown": max_dd,
                "Calmar": calmar,
                "Alpha": alpha,
                "Beta": beta,
                "Hit Rate": (series > 0).mean(),
            }
        )
    return pd.DataFrame(rows)


def stress_test_returns(returns: pd.DataFrame) -> pd.DataFrame:
    windows = {
        "2008 GFC": ("2008-01-01", "2009-03-31"),
        "2020 COVID": ("2020-02-01", "2020-04-30"),
        "2022 Inflation": ("2022-01-01", "2022-12-31"),
    }
    rows = []
    for label, (start, end) in windows.items():
        sample = returns.loc[(returns.index >= start) & (returns.index <= end)]
        if sample.empty:
            rows.append(
                {
                    "Stress Window": label,
                    "Series": "All",
                    "Status": "Not enough data",
                    "Return": np.nan,
                    "Max Drawdown": np.nan,
                    "Volatility": np.nan,
                    "Observations": 0,
                }
            )
            continue
        equity = (1 + sample).cumprod()
        drawdown = equity / equity.cummax() - 1
        for column in sample.columns:
            rows.append(
                {
                    "Stress Window": label,
                    "Series": column,
                    "Status": "Available",
                    "Return": equity[column].iloc[-1] - 1,
                    "Max Drawdown": drawdown[column].min(),
                    "Volatility": sample[column].std() * np.sqrt(252),
                    "Observations": len(sample[column].dropna()),
                }
            )
    return pd.DataFrame(rows)


def factor_exposure_heatmap(portfolio_returns: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    factor_symbols = [symbol for symbol in ["SPY", "IWM", "EFA", "EEM", "TLT", "GLD", "HYG", "DBC"] if symbol in close]
    factor_returns = close[factor_symbols].pct_change(fill_method=None).reindex(portfolio_returns.index).dropna()
    rows = []
    for portfolio in portfolio_returns.columns:
        common = portfolio_returns.index.intersection(factor_returns.index)
        if len(common) < 60:
            continue
        y = portfolio_returns.loc[common, portfolio]
        x = factor_returns.loc[common]
        x_mat = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(x_mat, y.values, rcond=None)[0][1:]
        rows.append({"Portfolio": portfolio, **{factor: beta[idx] for idx, factor in enumerate(x.columns)}})
    return pd.DataFrame(rows).set_index("Portfolio") if rows else pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def run_stage_a2_research(
    close: pd.DataFrame,
    macro: pd.DataFrame,
    model_name: str,
    base_cost_bps: float,
    impact_bps: float,
    kelly_fraction: float,
    target_volatility: float,
    smoothing: float,
    enable_regime_overlay: bool,
    max_drawdown_limit: float,
):
    x, y = build_stage_a2_features(close, macro)
    diagnostics = {
        "Requested Model": model_name,
        "Effective Model": model_name,
        "Prediction Source": "ML",
        "Fallback Used": False,
        "Feature Samples": len(x),
        "Feature Dates": len(x.index.get_level_values("date").unique()) if not x.empty else 0,
        "Feature Assets": len(x.index.get_level_values("symbol").unique()) if not x.empty else 0,
    }
    predictions = pd.Series(dtype=float, name="score")
    walk_log = pd.DataFrame()
    importance_history = pd.DataFrame()
    attempts = [
        (model_name, 300, 18),
        (model_name, 150, 12),
        ("Decision Tree", 100, 9),
        ("Elastic Net", 60, 6),
    ]
    seen = set()
    for candidate_model, min_samples, min_history in attempts:
        key = (candidate_model, min_samples, min_history)
        if key in seen or x.empty:
            continue
        seen.add(key)
        try:
            predictions, walk_log, importance_history = build_walk_forward_ml_predictions(
                x,
                y,
                candidate_model,
                min_train_samples=min_samples,
                min_history_months=min_history,
            )
        except Exception:
            predictions, walk_log, importance_history = pd.Series(dtype=float, name="score"), pd.DataFrame(), pd.DataFrame()
        if not predictions.empty:
            diagnostics["Effective Model"] = candidate_model
            diagnostics["Min Train Samples"] = min_samples
            diagnostics["Min History Months"] = min_history
            break

    if predictions.empty and not x.empty:
        predictions, walk_log = build_white_box_fallback_predictions(x)
        importance_history = pd.DataFrame()
        diagnostics["Effective Model"] = "White-Box Fallback"
        diagnostics["Prediction Source"] = "Fallback"
        diagnostics["Fallback Used"] = True

    portfolio_returns, execution = build_stage_a2_portfolios(
        close,
        predictions,
        base_cost_bps,
        impact_bps,
        kelly_fraction,
        target_volatility,
        smoothing,
        enable_regime_overlay,
        max_drawdown_limit,
    )
    if portfolio_returns.empty:
        return x, predictions, walk_log, importance_history, portfolio_returns, execution, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), diagnostics
    benchmark = close["SPY"].pct_change(fill_method=None).reindex(portfolio_returns.index).fillna(0.0) if "SPY" in close else None
    metrics = calculate_metrics(portfolio_returns, benchmark)
    regimes = detect_regime_states(close)
    stress = stress_test_returns(portfolio_returns)
    exposures = factor_exposure_heatmap(portfolio_returns, close)
    diagnostics["Prediction Count"] = len(predictions)
    diagnostics["Portfolio Return Rows"] = len(portfolio_returns)
    return x, predictions, walk_log, importance_history, portfolio_returns, execution, metrics, regimes, stress, exposures, diagnostics


def render_stage_a2_dashboard(stock_universe_file) -> None:
    st.title("Stage A2 Research Lab: ML-Powered Multi-Asset White-Box")
    st.write(
        "Intermediate capstone stage with white-box ML, walk-forward validation, HRP-style/Ledoit/Kelly portfolios, "
        "rule-based regime proxy visualization, factor exposure monitoring, stress tests, and execution cost tracking."
    )

    universe_table = pd.read_csv(stock_universe_file) if stock_universe_file.exists() else pd.DataFrame()
    local_stocks = universe_table["Ticker"].dropna().astype(str).str.upper().head(80).tolist() if "Ticker" in universe_table else DEFAULT_A2_STOCKS
    default_symbols = tuple(dict.fromkeys([*local_stocks, *SECTOR_ETFS, *GLOBAL_PROXIES]))

    with st.sidebar:
        st.header("Stage A2 Controls")
        st.metric("Initial Capital", f"${STAGE_A2_INITIAL_CAPITAL:,.0f}")
        years = st.slider("A2 research window", min_value=5, max_value=15, value=10, step=1)
        max_names = st.slider("A2 max stock names", min_value=20, max_value=min(120, len(default_symbols)), value=min(40, len(default_symbols)), step=10)
        model_name = st.selectbox("White-box ML model", ["Random Forest", "Gradient Boosting", "Decision Tree", "Elastic Net"], index=2)
        base_cost_bps = st.slider("Txn cost bps", min_value=10, max_value=20, value=12, step=1)
        impact_bps = st.slider("Square-root impact bps", min_value=1, max_value=20, value=6, step=1)
        kelly_fraction = st.slider("Fractional Kelly", min_value=0.25, max_value=0.50, value=0.25, step=0.05)
        target_volatility = st.slider("Target volatility", min_value=0.06, max_value=0.20, value=0.12, step=0.01)
        smoothing = st.slider("Weight smoothing", min_value=0.00, max_value=0.80, value=0.25, step=0.05)
        enable_regime_overlay = st.toggle("Regime risk overlay", value=False)
        max_drawdown_limit = st.slider("Max drawdown guard", min_value=0.05, max_value=0.25, value=0.15, step=0.01)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    selected_symbols = tuple(dict.fromkeys(list(default_symbols[:max_names]) + SECTOR_ETFS + GLOBAL_PROXIES))

    with st.spinner("Loading A2 Yahoo Finance prices and FRED macro data..."):
        close = load_stage_a2_prices(selected_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        macro = load_fred_macro(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if close.empty or close.shape[1] < 12:
        st.error("Not enough data loaded for Stage A2. Try a smaller universe or shorter window.")
        return

    with st.spinner("Training walk-forward ML models and building A2 portfolios..."):
        x, predictions, walk_log, importance_history, portfolio_returns, execution, metrics, regimes, stress, exposures, diagnostics = run_stage_a2_research(
            close,
            macro,
            model_name,
            base_cost_bps,
            impact_bps,
            kelly_fraction,
            target_volatility,
            smoothing,
            enable_regime_overlay,
            max_drawdown_limit,
        )
        if portfolio_returns.empty:
            st.error("Not enough walk-forward predictions to build Stage A2 portfolios.")
            st.dataframe(pd.DataFrame([diagnostics]), use_container_width=True, hide_index=True)
            return
        benchmark = close["SPY"].pct_change(fill_method=None).reindex(portfolio_returns.index).fillna(0.0) if "SPY" in close else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tradable Assets", close.shape[1])
    c2.metric("ML Samples", f"{len(x):,}")
    c3.metric("OOS Rebalances", f"{len(walk_log):,}")
    c4.metric("Prediction Source", diagnostics.get("Prediction Source", "ML"))
    if diagnostics.get("Fallback Used"):
        st.warning("A2 used the white-box fallback signal because the selected ML model did not produce enough walk-forward predictions with the available data.")
    elif diagnostics.get("Effective Model") != diagnostics.get("Requested Model"):
        st.info(f"A2 used {diagnostics.get('Effective Model')} after the requested model produced insufficient predictions.")

    performance_tab, model_tab, regime_tab, risk_tab, execution_tab, paper_tab = st.tabs(
        ["Performance", "White-Box ML", "Regime States", "Risk & Stress", "Execution", "Working Paper #2"]
    )

    with performance_tab:
        st.subheader("Stage A2 Walk-Forward Net Performance")
        st.caption(
            "Performance is net of fixed transaction costs and square-root market-impact estimates. "
            "The default settings use a simpler Decision Tree, volatility targeting, and weight smoothing because this has been more stable in the current walk-forward sample. "
            "The max drawdown guard is applied on daily backtest bars and de-risks a portfolio after the threshold is reached."
        )
        best = metrics.sort_values("Sharpe", ascending=False).iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Best Portfolio", best["Series"])
        m2.metric("Ending Value", f"${best['Ending Value']:,.0f}")
        m3.metric("Total PnL", f"${best['Total PnL']:,.0f}")
        m4.metric("Sharpe", f"{best['Sharpe']:.2f}")
        st.dataframe(
            metrics.style.format(
                {
                    "Ending Value": "${:,.0f}",
                    "Total PnL": "${:,.0f}",
                    "Total Return": "{:.2%}",
                    "CAGR": "{:.2%}",
                    "Annual Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Calmar": "{:.2f}",
                    "Alpha": "{:.2%}",
                    "Beta": "{:.2f}",
                    "Hit Rate": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        equity = (1 + portfolio_returns).cumprod()
        dollar_equity = equity * STAGE_A2_INITIAL_CAPITAL
        if benchmark is not None:
            dollar_equity["SPY Benchmark"] = (1 + benchmark).cumprod() * STAGE_A2_INITIAL_CAPITAL
        st.plotly_chart(px.line(dollar_equity, x=dollar_equity.index, y=dollar_equity.columns, title="$1,000,000 A2 Equity Curve"), use_container_width=True)
        drawdown = equity / equity.cummax() - 1
        st.plotly_chart(px.line(drawdown, x=drawdown.index, y=drawdown.columns, title="Rolling Drawdown"), use_container_width=True)

    with model_tab:
        st.subheader("Feature Importance Tracking")
        if importance_history.empty:
            st.info("Feature importance is unavailable for the selected model.")
        else:
            latest_importance = importance_history.sort_values("Signal Date").groupby("Feature")["Importance"].tail(1)
            latest_importance = importance_history.loc[latest_importance.index].sort_values("Importance", ascending=False).head(15)
            st.plotly_chart(px.bar(latest_importance, x="Importance", y="Feature", orientation="h", title=f"{model_name} Latest Feature Importance"), use_container_width=True)
            history = importance_history.groupby(["Signal Date", "Feature"], as_index=False)["Importance"].mean()
            top_features = latest_importance["Feature"].head(8).tolist()
            st.plotly_chart(px.line(history[history["Feature"].isin(top_features)], x="Signal Date", y="Importance", color="Feature", title="Feature Importance Through Time"), use_container_width=True)
        st.subheader("Walk-Forward Training Log")
        debug_columns = [
            "Train Start",
            "Train End",
            "Prediction Month",
            "Signal Date",
            "Train Samples",
            "Predicted Assets",
            "Model",
            "Prediction Source",
        ]
        st.dataframe(walk_log[[column for column in debug_columns if column in walk_log.columns]].tail(36), use_container_width=True, hide_index=True)
        fallback_months = walk_log[walk_log.get("Prediction Source", pd.Series(dtype=str)).eq("Fallback")] if not walk_log.empty and "Prediction Source" in walk_log else pd.DataFrame()
        st.subheader("Fallback Months")
        if fallback_months.empty:
            st.success("No fallback months. The selected ML path produced the walk-forward predictions.")
        else:
            st.dataframe(fallback_months, use_container_width=True, hide_index=True)

    with regime_tab:
        st.subheader("Rule-Based Regime Proxy Visualization")
        st.write(
            "This is not a true Hidden Markov Model. It uses a Gaussian-mixture clustering proxy on monthly SPY return, volatility, "
            "and drawdown, then labels states as Bear, Recovery, or Bull by average return."
        )
        if regimes.empty:
            st.warning("Not enough data for regime state detection.")
        else:
            regime_chart = regimes.reset_index()
            regime_chart = regime_chart.rename(columns={regime_chart.columns[0]: "Date"})
            regime_chart["vol_6m"] = regime_chart["vol_6m"].abs().fillna(0.0).clip(lower=0.001)
            st.plotly_chart(
                px.scatter(
                    regime_chart,
                    x="Date",
                    y="return_1m",
                    color="Regime",
                    size="vol_6m",
                    title="Rule-Based Bull / Bear / Recovery Regime Proxy",
                ),
                use_container_width=True,
            )
            st.plotly_chart(
                px.area(regime_chart, x="Date", y="Regime Code", color="Regime", title="Rule-Based Regime Proxy Timeline"),
                use_container_width=True,
            )

    with risk_tab:
        st.subheader("Factor Exposure Heatmap")
        if exposures.empty:
            st.warning("Not enough overlapping returns for factor exposure estimation.")
        else:
            st.plotly_chart(px.imshow(exposures, aspect="auto", color_continuous_scale="RdBu", title="Portfolio Factor Betas"), use_container_width=True)
            st.caption("A2 target: monitor exposures and keep unusually large factor loads under review.")
        st.subheader("Stress Testing")
        if stress.empty:
            st.info("Selected history does not overlap 2008, 2020, or 2022 stress windows.")
        else:
            st.dataframe(stress.style.format({"Return": "{:.2%}", "Max Drawdown": "{:.2%}", "Volatility": "{:.2%}"}), use_container_width=True, hide_index=True)

    with execution_tab:
        st.subheader("Execution Quality Tracking")
        st.write("Costs use fixed transaction bps plus a square-root market-impact penalty based on monthly trade size.")
        st.caption("A2 performance tables show net returns after estimated execution costs. Gross-return attribution is a known extension.")
        if not execution.empty:
            cost_summary = execution.groupby("Portfolio", as_index=False).agg(
                Average_Turnover=("Turnover", "mean"),
                Total_Estimated_Cost=("Market Impact Cost", "sum"),
                Average_Estimated_Cost=("Market Impact Cost", "mean"),
                Max_Position_Weight=("Max Position Weight", "max"),
                Average_Risk_Scale=("Final Risk Scale", "mean"),
                Min_Risk_Scale=("Final Risk Scale", "min"),
                Drawdown_Guard_Triggers=("Drawdown Guard Triggered", lambda values: int((values == "Yes").sum())),
            )
            st.dataframe(
                cost_summary.style.format(
                    {
                        "Average_Turnover": "{:.2f}x",
                        "Total_Estimated_Cost": "{:.2%}",
                        "Average_Estimated_Cost": "{:.2%}",
                        "Max_Position_Weight": "{:.2%}",
                        "Average_Risk_Scale": "{:.2%}",
                        "Min_Risk_Scale": "{:.2%}",
                        "Drawdown_Guard_Triggers": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        concentrated = execution[execution["Concentration Warning"].eq("Yes")] if "Concentration Warning" in execution else pd.DataFrame()
        if not concentrated.empty:
            st.warning("Some portfolios exceeded the 25% max-position concentration check. Review the rows marked in the execution table.")
        drawdown_guard_rows = execution[execution["Drawdown Guard Triggered"].eq("Yes")] if "Drawdown Guard Triggered" in execution else pd.DataFrame()
        if not drawdown_guard_rows.empty:
            st.warning("The drawdown guard was triggered for one or more portfolios. Those portfolios were de-risked for the rest of the affected monthly holding period.")
        st.dataframe(
            execution.tail(48).style.format(
                {
                    "Turnover": "{:.2f}x",
                    "Market Impact Cost": "{:.2%}",
                    "Gross Exposure": "{:.2f}x",
                    "Net Exposure": "{:.2f}x",
                    "Max Position Weight": "{:.2%}",
                    "Realized Volatility": "{:.2%}",
                    "Vol Target Scale": "{:.2%}",
                    "Regime Scale": "{:.2%}",
                    "Final Risk Scale": "{:.2%}",
                    "Smoothing": "{:.2%}",
                    "Max Drawdown Limit": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.plotly_chart(px.line(execution, x="Date", y="Market Impact Cost", color="Portfolio", title="Estimated Execution Cost"), use_container_width=True)
        st.plotly_chart(px.line(execution, x="Date", y="Turnover", color="Portfolio", title="Monthly Turnover"), use_container_width=True)

    with paper_tab:
        st.subheader("Working Paper #2 Checklist")
        st.markdown(
            """
- Universe: local equity universe, sector ETFs, and global proxies.
- ML: Decision Tree, Random Forest, Gradient Boosting, Elastic Net.
- White-box: feature importance and importance drift.
- Portfolio: HRP-style / risk-parity fallback, Ledoit-Wolf mean-variance, fractional Kelly, beta-neutral long/short.
- Risk: rule-based bull/bear/recovery regime proxy, factor exposure heatmap, stress tests.
- Execution: fixed costs plus square-root market impact.

Use `reports/working_paper_2.md` as the Stage A2 draft.
"""
        )
