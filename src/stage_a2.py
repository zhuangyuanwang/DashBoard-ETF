from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


STAGE_A2_INITIAL_CAPITAL = 1_000_000
SECTOR_ETFS = ["XLF", "XLE", "XLK", "XLI", "XLP", "XLU", "XLV"]
GLOBAL_PROXIES = ["SPY", "EFA", "EWJ", "EEM", "IWM", "VGK", "TLT", "IEF", "GLD", "DBC", "HYG", "LQD"]
A2_ETF_SYMBOLS = set(SECTOR_ETFS + GLOBAL_PROXIES)
MAX_LONG_ONLY_WEIGHT = 0.25
TARGET_TYPES = [
    "Next-month raw return",
    "Next-month excess return vs SPY",
    "Next-month outperform SPY classification",
    "Next-month cross-sectional rank percentile",
]
STAGE_A2_MODEL_OPTIONS = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Decision Tree", "Elastic Net"]
STAGE_A2_PRESENTATION_MODELS = ["Elastic Net", "Decision Tree", "Random Forest", "Gradient Boosting"]
STAGE_A2_PRESENTATION_CONFIG = {
    "Initial Capital": STAGE_A2_INITIAL_CAPITAL,
    "Rebalance Frequency": "Monthly",
    "Transaction Cost Bps": 10.0,
    "Square-Root Impact Bps": 6.0,
    "Top-N ETFs": 5,
    "Max ETF Weight": MAX_LONG_ONLY_WEIGHT,
    "Benchmark": "SPY",
    "Research Window Years": 10,
    "Target Type": "Next-month raw return",
    "Kelly Fraction": 0.25,
    "Target Volatility": 0.12,
    "Weight Smoothing": 0.0,
    "Regime Overlay": True,
    "Max Drawdown Guard": 0.15,
    "Monthly Turnover Cap": 0.75,
    "Rebalance Threshold": 0.03,
}
ETF_CATEGORY_MAP = {
    **{symbol: "Sector ETF" for symbol in SECTOR_ETFS},
    "SPY": "US Equity Benchmark",
    "EFA": "Developed ex-US Equity",
    "EWJ": "Japan Equity",
    "EEM": "Emerging Markets Equity",
    "IWM": "US Small Cap Equity",
    "VGK": "Europe Equity",
    "TLT": "Long Treasury",
    "IEF": "Intermediate Treasury",
    "GLD": "Gold",
    "DBC": "Broad Commodities",
    "HYG": "High Yield Credit",
    "LQD": "Investment Grade Credit",
}
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
        "INDPRO": "industrial_production",
        "STLFSI4": "financial_stress",
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
    if "industrial_production" in macro.columns:
        macro["industrial_production_yoy"] = macro["industrial_production"].pct_change(12, fill_method=None)
    return macro


def build_stage_a2_features(
    close: pd.DataFrame,
    macro: pd.DataFrame,
    benchmark_symbol: str = "SPY",
    target_type: str = "Next-month raw return",
    return_forward_returns: bool = False,
):
    monthly = close.resample("ME").last()
    monthly_returns = monthly.pct_change(fill_method=None)
    daily_returns = close.pct_change(fill_method=None)
    forward_raw = monthly_returns.shift(-1)
    forward_spy = forward_raw[benchmark_symbol] if benchmark_symbol in forward_raw else forward_raw.mean(axis=1)
    forward_excess = forward_raw.sub(forward_spy, axis=0)
    forward_rank = forward_raw.rank(axis=1, pct=True)
    target = forward_raw
    if target_type == "Next-month excess return vs SPY":
        target = forward_excess
    elif target_type == "Next-month outperform SPY classification":
        target = (forward_excess > 0).astype(float).where(forward_excess.notna())
    elif target_type == "Next-month cross-sectional rank percentile":
        target = forward_rank
    spy = monthly_returns[benchmark_symbol] if benchmark_symbol in monthly_returns else monthly_returns.mean(axis=1)
    spy_1m = monthly[benchmark_symbol].pct_change(1, fill_method=None) if benchmark_symbol in monthly else monthly.mean(axis=1).pct_change(1, fill_method=None)
    spy_3m = monthly[benchmark_symbol].pct_change(3, fill_method=None) if benchmark_symbol in monthly else monthly.mean(axis=1).pct_change(3, fill_method=None)
    spy_6m = monthly[benchmark_symbol].pct_change(6, fill_method=None) if benchmark_symbol in monthly else monthly.mean(axis=1).pct_change(6, fill_method=None)
    spy_12m = monthly[benchmark_symbol].pct_change(12, fill_method=None) if benchmark_symbol in monthly else monthly.mean(axis=1).pct_change(12, fill_method=None)
    ret_3m_rank = monthly.pct_change(3, fill_method=None).rank(axis=1, pct=True)
    vol_12m_rank = monthly_returns.rolling(12).std().rank(axis=1, pct=True)
    drawdown_12m_panel = monthly / monthly.rolling(12).max() - 1
    drawdown_rank = drawdown_12m_panel.rank(axis=1, pct=True)
    spy_daily = close[benchmark_symbol] if benchmark_symbol in close else close.mean(axis=1)
    spy_above_200d = (spy_daily > spy_daily.rolling(200).mean()).astype(float).resample("ME").last().reindex(monthly.index)
    spy_vol_60d = daily_returns[benchmark_symbol].rolling(60).std() * np.sqrt(252) if benchmark_symbol in daily_returns else daily_returns.mean(axis=1).rolling(60).std() * np.sqrt(252)
    market_drawdown_252d = (spy_daily / spy_daily.rolling(252).max() - 1).resample("ME").last().reindex(monthly.index)
    risk_off_dummy = ((spy_above_200d == 0) & (spy_vol_60d.resample("ME").last().reindex(monthly.index) > 0.25)).astype(float)

    macro_monthly = pd.DataFrame(index=monthly.index)
    if not macro.empty:
        # FRED fields are aligned to month-end using observations available in
        # the downloaded time series. This is a research proxy, not a full
        # point-in-time release-calendar model for macro revisions.
        macro_base = macro.resample("ME").last().reindex(monthly.index).ffill().shift(1)
        macro_monthly = macro_base.pct_change(fill_method=None).add_prefix("macro_delta_").join(
            macro_base.add_prefix("macro_level_")
        )
        if "macro_level_ten_year_yield" in macro_monthly:
            macro_monthly["macro_ten_year_yield_3m_change"] = macro_base["ten_year_yield"].diff(3)
        if {"ten_year_yield", "two_year_yield"}.issubset(macro_base.columns):
            macro_monthly["macro_2y10y_spread"] = macro_base["ten_year_yield"] - macro_base["two_year_yield"]

    rows = []
    targets = []
    for symbol in monthly.columns:
        if symbol == benchmark_symbol:
            continue
        symbol_daily = daily_returns[symbol]
        beta = symbol_daily.rolling(126).cov(daily_returns[benchmark_symbol]) / daily_returns[benchmark_symbol].rolling(126).var() if benchmark_symbol in daily_returns else np.nan
        beta_60d = symbol_daily.rolling(60).cov(daily_returns[benchmark_symbol]) / daily_returns[benchmark_symbol].rolling(60).var() if benchmark_symbol in daily_returns else np.nan
        corr_spy_60d = symbol_daily.rolling(60).corr(daily_returns[benchmark_symbol]) if benchmark_symbol in daily_returns else np.nan
        corr_tlt_60d = symbol_daily.rolling(60).corr(daily_returns["TLT"]) if "TLT" in daily_returns else np.nan
        ma50 = close[symbol].rolling(50).mean()
        ma200 = close[symbol].rolling(200).mean()
        features = pd.DataFrame(
            {
                "symbol": symbol,
                "ret_1m": monthly[symbol].pct_change(1, fill_method=None),
                "ret_3m": monthly[symbol].pct_change(3, fill_method=None),
                "ret_6m": monthly[symbol].pct_change(6, fill_method=None),
                "ret_12m": monthly[symbol].pct_change(12, fill_method=None),
                "relative_ret_1m": monthly[symbol].pct_change(1, fill_method=None) - spy_1m,
                "relative_ret_3m": monthly[symbol].pct_change(3, fill_method=None) - spy_3m,
                "relative_ret_6m": monthly[symbol].pct_change(6, fill_method=None) - spy_6m,
                "relative_ret_12m": monthly[symbol].pct_change(12, fill_method=None) - spy_12m,
                "cross_sectional_momentum_rank": ret_3m_rank[symbol],
                "vol_3m": monthly_returns[symbol].rolling(3).std(),
                "vol_12m": monthly_returns[symbol].rolling(12).std(),
                "cross_sectional_volatility_rank": vol_12m_rank[symbol],
                "drawdown_12m": monthly[symbol] / monthly[symbol].rolling(12).max() - 1,
                "cross_sectional_drawdown_rank": drawdown_rank[symbol],
                "beta_6m": beta.resample("ME").last().reindex(monthly.index),
                "beta_60d": beta_60d.resample("ME").last().reindex(monthly.index),
                "corr_spy_60d": corr_spy_60d.resample("ME").last().reindex(monthly.index),
                "corr_tlt_60d": corr_tlt_60d.resample("ME").last().reindex(monthly.index) if not isinstance(corr_tlt_60d, float) else np.nan,
                "etf_above_200d": (close[symbol] > ma200).astype(float).resample("ME").last().reindex(monthly.index),
                "etf_ma50_ma200_ratio": (ma50 / ma200 - 1).resample("ME").last().reindex(monthly.index),
                "spy_above_200d": spy_above_200d,
                "market_drawdown_252d": market_drawdown_252d,
                "risk_off_dummy": risk_off_dummy,
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
    if return_forward_returns:
        forward = forward_raw.stack(future_stack=True).rename("forward_raw_return")
        forward.index.names = ["date", "symbol"]
        return feature_panel, y, forward.reindex(feature_panel.index)
    return feature_panel, y


def create_stage_a2_model(model_name: str, params: dict | None = None, target_type: str = "Next-month raw return"):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    params = params or {}
    is_classification = target_type == "Next-month outperform SPY classification"
    if model_name == "Decision Tree":
        if is_classification:
            return DecisionTreeClassifier(max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 20), random_state=11)
        return DecisionTreeRegressor(max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 20), random_state=11)
    if model_name == "Random Forest":
        model_class = RandomForestClassifier if is_classification else RandomForestRegressor
        return model_class(
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 5),
            min_samples_leaf=params.get("min_samples_leaf", 12),
            max_features=params.get("max_features", "sqrt"),
            random_state=11,
            n_jobs=-1,
        )
    if model_name == "Gradient Boosting":
        model_class = GradientBoostingClassifier if is_classification else GradientBoostingRegressor
        return model_class(
            n_estimators=params.get("n_estimators", 120),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.035),
            subsample=params.get("subsample", 1.0),
            random_state=11,
        )
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier, XGBRegressor

            model_class = XGBClassifier if is_classification else XGBRegressor
            objective = "binary:logistic" if is_classification else "reg:squarederror"
            model = model_class(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 3),
                min_child_weight=params.get("min_samples_leaf", 10),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("max_features_ratio", 0.8),
                objective=objective,
                random_state=11,
                n_jobs=-1,
                verbosity=0,
            )
            model._stage_a2_engine = "XGBoost"
            return model
        except Exception:
            model_class = HistGradientBoostingClassifier if is_classification else HistGradientBoostingRegressor
            model = model_class(
                max_iter=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.03),
                max_leaf_nodes=15,
                l2_regularization=0.01,
                random_state=11,
            )
            model._stage_a2_engine = "XGBoost unavailable; sklearn HistGradientBoosting fallback"
            return model
    if model_name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor

            model_class = LGBMClassifier if is_classification else LGBMRegressor
            model = model_class(
                n_estimators=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 3),
                min_child_samples=params.get("min_samples_leaf", 10),
                subsample=params.get("subsample", 0.9),
                colsample_bytree=params.get("max_features_ratio", 0.8),
                random_state=11,
                n_jobs=-1,
                verbose=-1,
            )
            model._stage_a2_engine = "LightGBM"
            return model
        except Exception:
            model_class = HistGradientBoostingClassifier if is_classification else HistGradientBoostingRegressor
            model = model_class(
                max_iter=params.get("n_estimators", 100),
                learning_rate=params.get("learning_rate", 0.03),
                max_leaf_nodes=15,
                l2_regularization=0.01,
                random_state=11,
            )
            model._stage_a2_engine = "LightGBM unavailable; sklearn HistGradientBoosting fallback"
            return model
    if model_name == "Elastic Net":
        if is_classification:
            from sklearn.linear_model import LogisticRegression

            return make_pipeline(StandardScaler(), LogisticRegression(C=1 / params.get("alpha", 0.001), max_iter=20000))
        return make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=params.get("alpha", 0.0005), l1_ratio=params.get("l1_ratio", 0.35), max_iter=50000),
        )
    raise ValueError(f"Unsupported Stage A2 model: {model_name}")


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.DataFrame({"left": left, "right": right}).dropna()
    if aligned.empty or aligned["left"].nunique() <= 1 or aligned["right"].nunique() <= 1:
        return np.nan
    return aligned["left"].corr(aligned["right"], method="spearman")


def predict_stage_a2_model(model, x: pd.DataFrame, target_type: str) -> np.ndarray:
    if target_type == "Next-month outperform SPY classification" and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x)
        return probabilities[:, -1]
    return model.predict(x)


def describe_model_engine(model) -> str:
    estimator = model[-1] if hasattr(model, "steps") else model
    return getattr(estimator, "_stage_a2_engine", estimator.__class__.__name__)


def build_walk_forward_ml_predictions(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_params: dict | None = None,
    target_type: str = "Next-month raw return",
    min_train_samples: int = 300,
    min_history_months: int = 18,
    collect_importance: bool = True,
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
        if target_type == "Next-month outperform SPY classification" and y.loc[train_mask].nunique() < 2:
            continue
        model = create_stage_a2_model(model_name, model_params, target_type)
        model.fit(x.loc[train_mask], y.loc[train_mask])
        prediction = pd.Series(predict_stage_a2_model(model, x.loc[predict_mask], target_type), index=x.loc[predict_mask].index)
        predictions.append(prediction)
        train_prediction = pd.Series(predict_stage_a2_model(model, x.loc[train_mask], target_type), index=x.loc[train_mask].index)
        train_ic = safe_spearman(train_prediction, y.loc[train_mask])
        log_rows.append(
            {
                "Train Start": row_dates[train_mask].min(),
                "Train End": row_dates[train_mask].max(),
                "Prediction Month": signal_date,
                "Signal Date": signal_date,
                "Train Samples": int(train_mask.sum()),
                "Predicted Assets": int(predict_mask.sum()),
                "Model": model_name,
                "Model Engine": describe_model_engine(model),
                "Model Params": str(model_params or {}),
                "Prediction Source": "ML",
                "Train IC": train_ic,
            }
        )
        if collect_importance and len(log_rows) % 3 == 0:
            importance = extract_feature_importance(model, x.columns, x.loc[predict_mask], target_type)
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


def get_stage_a2_param_grid(model_name: str) -> list[dict]:
    if model_name == "Elastic Net":
        return [{"alpha": alpha, "l1_ratio": l1_ratio} for alpha in [0.0001, 0.001, 0.01, 0.1] for l1_ratio in [0.1, 0.5, 0.9]]
    if model_name == "Random Forest":
        return [
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": max_features}
            for n in [100, 300]
            for depth in [2, 3, 4, 5]
            for leaf in [5, 10, 20]
            for max_features in ["sqrt", "log2"]
        ]
    if model_name == "Gradient Boosting":
        return [
            {"n_estimators": n, "learning_rate": lr, "max_depth": depth, "subsample": subsample}
            for n in [50, 100, 200]
            for lr in [0.01, 0.03, 0.05]
            for depth in [2, 3]
            for subsample in [0.7, 0.9, 1.0]
        ]
    if model_name in ["XGBoost", "LightGBM"]:
        return [
            {
                "n_estimators": n,
                "learning_rate": lr,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "subsample": subsample,
                "max_features_ratio": max_features_ratio,
            }
            for n in [50, 100, 200]
            for lr in [0.01, 0.03, 0.05]
            for depth in [2, 3]
            for leaf in [5, 10, 20]
            for subsample in [0.7, 0.9]
            for max_features_ratio in [0.7, 0.9]
        ]
    return [{}]


def evaluate_prediction_signal(predictions: pd.Series, forward_returns: pd.Series) -> dict:
    if predictions.empty or forward_returns.empty:
        return {"Prediction IC": np.nan, "Top-Bottom Spread": np.nan, "Top Basket Sharpe": np.nan}
    frame = pd.DataFrame({"score": predictions, "forward": forward_returns.reindex(predictions.index)}).dropna()
    if frame.empty:
        return {"Prediction IC": np.nan, "Top-Bottom Spread": np.nan, "Top Basket Sharpe": np.nan}
    monthly_rows = []
    for date, group in frame.groupby(level="date"):
        ranked = group.sort_values("score", ascending=False)
        if len(ranked) < 6:
            continue
        top_n = min(5, max(1, len(ranked) // 5))
        bottom_n = top_n
        top_return = ranked.head(top_n)["forward"].mean()
        bottom_return = ranked.tail(bottom_n)["forward"].mean()
        monthly_rows.append(
            {
                "Date": date,
                "Top Return": top_return,
                "Bottom Return": bottom_return,
                "Spread": top_return - bottom_return,
                "IC": safe_spearman(ranked["score"], ranked["forward"]),
            }
        )
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        return {"Prediction IC": np.nan, "Top-Bottom Spread": np.nan, "Top Basket Sharpe": np.nan}
    top_series = monthly["Top Return"].dropna()
    top_sharpe = top_series.mean() * 12 / (top_series.std() * np.sqrt(12)) if top_series.std() > 0 else np.nan
    return {
        "Prediction IC": monthly["IC"].mean(),
        "Top-Bottom Spread": monthly["Spread"].mean(),
        "Top Basket Sharpe": top_sharpe,
        "Top Basket Net Return Estimate": (1 + top_series - 0.0012).prod() - 1 if not top_series.empty else np.nan,
    }


def tune_stage_a2_hyperparameters(
    x: pd.DataFrame,
    y: pd.Series,
    forward_returns: pd.Series,
    model_name: str,
    target_type: str,
    max_candidates: int = 18,
) -> tuple[dict, pd.DataFrame]:
    grid = get_stage_a2_param_grid(model_name)
    if len(grid) > max_candidates:
        selection = np.linspace(0, len(grid) - 1, max_candidates, dtype=int)
        grid = [grid[index] for index in selection]
    rows = []
    best_params = {}
    best_score = -np.inf
    for params in grid:
        predictions, _, _ = build_walk_forward_ml_predictions(
            x,
            y,
            model_name,
            model_params=params,
            target_type=target_type,
            min_train_samples=180,
            min_history_months=12,
        )
        signal_metrics = evaluate_prediction_signal(predictions, forward_returns)
        # Hyperparameter selection uses only walk-forward out-of-sample diagnostics.
        # The composite keeps spread as the anchor while also rewarding IC, top-basket
        # Sharpe, and a simple net-return estimate after assumed monthly costs.
        spread = signal_metrics.get("Top-Bottom Spread", np.nan)
        ic = signal_metrics.get("Prediction IC", np.nan)
        sharpe = signal_metrics.get("Top Basket Sharpe", np.nan)
        net_estimate = signal_metrics.get("Top Basket Net Return Estimate", np.nan)
        score = (
            (spread if pd.notna(spread) else 0.0)
            + 0.01 * (ic if pd.notna(ic) else 0.0)
            + 0.002 * (sharpe if pd.notna(sharpe) else 0.0)
            + 0.02 * (net_estimate if pd.notna(net_estimate) else 0.0)
        )
        if not np.isfinite(score):
            score = -np.inf
        row = {"Model": model_name, "Params": str(params), **signal_metrics, "Selection Score": score}
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params
    return best_params, pd.DataFrame(rows).sort_values("Selection Score", ascending=False) if rows else pd.DataFrame()


def build_signal_diagnostics(
    predictions: pd.Series,
    forward_returns: pd.Series,
    symbol_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame({"score": predictions, "forward": forward_returns.reindex(predictions.index)}).dropna()
    if symbol_filter is not None and not frame.empty:
        symbols = frame.index.get_level_values("symbol")
        frame = frame.loc[symbols.isin(symbol_filter)]
    rows = []
    for date, group in frame.groupby(level="date"):
        ranked = group.sort_values("score", ascending=False)
        if len(ranked) < 6:
            continue
        rows.append(
            {
                "Date": date,
                "Top 3 Avg Forward Return": ranked.head(min(3, len(ranked)))["forward"].mean(),
                "Top 5 Avg Forward Return": ranked.head(min(5, len(ranked)))["forward"].mean(),
                "Bottom 3 Avg Forward Return": ranked.tail(min(3, len(ranked)))["forward"].mean(),
                "Bottom 5 Avg Forward Return": ranked.tail(min(5, len(ranked)))["forward"].mean(),
                "Top-Bottom 3 Spread": ranked.head(min(3, len(ranked)))["forward"].mean() - ranked.tail(min(3, len(ranked)))["forward"].mean(),
                "Top-Bottom 5 Spread": ranked.head(min(5, len(ranked)))["forward"].mean() - ranked.tail(min(5, len(ranked)))["forward"].mean(),
                "Prediction IC": safe_spearman(ranked["score"], ranked["forward"]),
                "Assets Ranked": len(ranked),
            }
        )
    monthly = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "Average Top 3 Return": monthly["Top 3 Avg Forward Return"].mean() if not monthly.empty else np.nan,
                "Average Top 5 Return": monthly["Top 5 Avg Forward Return"].mean() if not monthly.empty else np.nan,
                "Average Bottom 3 Return": monthly["Bottom 3 Avg Forward Return"].mean() if not monthly.empty else np.nan,
                "Average Bottom 5 Return": monthly["Bottom 5 Avg Forward Return"].mean() if not monthly.empty else np.nan,
                "Average Top-Bottom 5 Spread": monthly["Top-Bottom 5 Spread"].mean() if not monthly.empty else np.nan,
                "Average Prediction IC": monthly["Prediction IC"].mean() if not monthly.empty else np.nan,
                "Months": len(monthly),
            }
        ]
    )
    return summary, monthly


def build_classification_diagnostics(predictions: pd.Series, y: pd.Series, forward_returns: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "score": predictions,
            "actual": y.reindex(predictions.index),
            "forward": forward_returns.reindex(predictions.index),
        }
    ).dropna()
    if frame.empty:
        return pd.DataFrame()
    predicted = frame["score"] >= 0.5
    actual = frame["actual"] >= 0.5
    true_positive = (predicted & actual).sum()
    predicted_positive = predicted.sum()
    return pd.DataFrame(
        [
            {
                "Classification Accuracy": (predicted == actual).mean(),
                "Precision": true_positive / predicted_positive if predicted_positive > 0 else np.nan,
                "Predicted Outperformers": int(predicted_positive),
                "Avg Forward Return of Predicted Outperformers": frame.loc[predicted, "forward"].mean() if predicted_positive > 0 else np.nan,
            }
        ]
    )


def extract_feature_importance(
    model,
    feature_names: pd.Index,
    sample_x: pd.DataFrame | None = None,
    target_type: str = "Next-month raw return",
) -> pd.DataFrame:
    estimator = model[-1] if hasattr(model, "steps") else model
    if sample_x is not None and hasattr(estimator, "predict"):
        try:
            import shap

            shap_sample = sample_x.tail(min(40, len(sample_x)))
            explainer = shap.Explainer(estimator, shap_sample)
            shap_values = explainer(shap_sample)
            values = np.asarray(shap_values.values)
            if values.ndim == 3:
                values = values[:, :, -1]
            values = np.abs(values).mean(axis=0)
            importance = pd.DataFrame({"Feature": feature_names, "Importance": values, "Importance Type": "SHAP mean absolute value"})
            total = importance["Importance"].abs().sum()
            if total > 0:
                importance["Importance"] = importance["Importance"].abs() / total
            return importance.sort_values("Importance", ascending=False).head(20)
        except Exception:
            pass
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_)
    else:
        if sample_x is None or sample_x.empty:
            return pd.DataFrame()
        try:
            baseline = np.asarray(predict_stage_a2_model(model, sample_x, target_type))
            sensitivity_values = []
            for feature in feature_names:
                shocked = sample_x.copy()
                shocked[feature] = shocked[feature].median()
                shocked_prediction = np.asarray(predict_stage_a2_model(model, shocked, target_type))
                sensitivity_values.append(np.nanmean(np.abs(baseline - shocked_prediction)))
            values = np.asarray(sensitivity_values)
            importance = pd.DataFrame({"Feature": feature_names, "Importance": values, "Importance Type": "Prediction sensitivity fallback"})
            total = importance["Importance"].abs().sum()
            if total > 0:
                importance["Importance"] = importance["Importance"].abs() / total
            return importance.sort_values("Importance", ascending=False).head(20)
        except Exception:
            return pd.DataFrame()
    values = np.asarray(values).reshape(-1)
    importance = pd.DataFrame({"Feature": feature_names, "Importance": values, "Importance Type": "Native model importance"})
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
    turnover_cap: float,
    rebalance_threshold: float,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    daily_returns = close.pct_change(fill_method=None).dropna()
    prediction_frame = predictions.rename("score").reset_index().set_index("date")
    month_ends = close.resample("ME").last().index
    strategy_names = ["HRP-style / Risk-Parity Fallback", "Ledoit-Wolf Mean-Variance", "Fractional Kelly", "Beta-Neutral ML"]
    returns_by_strategy = {name: pd.Series(0.0, index=daily_returns.index) for name in strategy_names}
    gross_returns_by_strategy = {name: pd.Series(0.0, index=daily_returns.index) for name in strategy_names}
    previous_weights = {name: pd.Series(0.0, index=close.columns) for name in strategy_names}
    equity_state = {name: 1.0 for name in strategy_names}
    peak_state = {name: 1.0 for name in strategy_names}
    execution_rows = []
    weight_rows = []

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
        top_assets = scores.head(max(1, top_n)).index.tolist()
        lookback = daily_returns.loc[:signal_date].tail(252)

        target_weights = {}
        hrp_weight, _ = cap_and_redistribute_long_weights(hrp_weights(lookback[top_assets]))
        target_weights["HRP-style / Risk-Parity Fallback"] = hrp_weight.reindex(close.columns).fillna(0.0)
        target_weights["Ledoit-Wolf Mean-Variance"] = ledoit_wolf_weights(lookback, scores.head(max(1, top_n))).reindex(close.columns).fillna(0.0)
        target_weights["Fractional Kelly"] = fractional_kelly_weights(lookback, scores.head(max(1, top_n)), kelly_fraction).reindex(close.columns).fillna(0.0)
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
            previous = previous_weights[strategy_name].reindex(weights.index).fillna(0.0)
            requested_turnover = (weights - previous).abs().sum()
            rebalance_skipped = False
            if requested_turnover < rebalance_threshold:
                weights = previous
                rebalance_skipped = True
            elif requested_turnover > turnover_cap:
                weights = previous + (weights - previous) * (turnover_cap / requested_turnover)
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
            gross_returns_by_strategy[strategy_name].loc[period_mask] = raw_period_returns
            turnover = (weights - previous_weights[strategy_name]).abs().sum()
            execution_rows.append(
                {
                    "Date": signal_date,
                    "Portfolio": strategy_name,
                    "Turnover": turnover,
                    "Requested Turnover": requested_turnover,
                    "Turnover Cap": turnover_cap,
                    "Rebalance Skipped": "Yes" if rebalance_skipped else "No",
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
            for symbol, weight in weights[weights.abs() > 1e-8].sort_values(ascending=False).items():
                weight_rows.append(
                    {
                        "Date": signal_date,
                        "Portfolio": strategy_name,
                        "ETF": symbol,
                        "Weight": weight,
                        "Category": ETF_CATEGORY_MAP.get(symbol, "ETF Proxy"),
                    }
                )
            previous_weights[strategy_name] = weights

    return (
        pd.DataFrame(returns_by_strategy).dropna(how="all"),
        pd.DataFrame(gross_returns_by_strategy).dropna(how="all"),
        pd.DataFrame(execution_rows),
        pd.DataFrame(weight_rows),
    )


def detect_regime_states(close: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
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
    method = "HMM"
    try:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(n_components=3, covariance_type="full", n_iter=300, random_state=7)
        labels = model.fit_predict(features)
    except Exception:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=3, covariance_type="full", random_state=7)
        labels = model.fit_predict(features)
        method = "HMM unavailable; Gaussian-mixture regime proxy"
    state_returns = pd.Series(labels, index=features.index).to_frame("State").join(features).groupby("State")["return_1m"].mean()
    ordered = state_returns.sort_values().index.tolist()
    names = {ordered[0]: "Bear", ordered[1]: "Recovery", ordered[2]: "Bull"}
    result = features.copy()
    result["State Id"] = labels
    result["Regime"] = [names[label] for label in labels]
    result["Regime Code"] = result["Regime"].map({"Bear": 0, "Recovery": 1, "Bull": 2})
    result["Regime Method"] = method
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
    turnover_cap: float,
    rebalance_threshold: float,
    target_type: str,
    enable_tuning: bool,
    collect_importance: bool = True,
    top_n: int = 5,
):
    x, y, forward_returns = build_stage_a2_features(close, macro, target_type=target_type, return_forward_returns=True)
    diagnostics = {
        "Requested Model": model_name,
        "Effective Model": model_name,
        "Prediction Source": "ML",
        "Fallback Used": False,
        "Feature Samples": len(x),
        "Feature Dates": len(x.index.get_level_values("date").unique()) if not x.empty else 0,
        "Feature Assets": len(x.index.get_level_values("symbol").unique()) if not x.empty else 0,
    }
    selected_params = {}
    tuning_results = pd.DataFrame()
    if enable_tuning and model_name in ["Elastic Net", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"] and not x.empty:
        selected_params, tuning_results = tune_stage_a2_hyperparameters(x, y, forward_returns, model_name, target_type)
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
                model_params=selected_params if candidate_model == model_name else None,
                target_type=target_type,
                min_train_samples=min_samples,
                min_history_months=min_history,
                collect_importance=collect_importance,
            )
        except Exception:
            predictions, walk_log, importance_history = pd.Series(dtype=float, name="score"), pd.DataFrame(), pd.DataFrame()
        if not predictions.empty:
            diagnostics["Effective Model"] = candidate_model
            diagnostics["Selected Hyperparameters"] = str(selected_params or {})
            diagnostics["Min Train Samples"] = min_samples
            diagnostics["Min History Months"] = min_history
            break

    if predictions.empty and not x.empty:
        predictions, walk_log = build_white_box_fallback_predictions(x)
        importance_history = pd.DataFrame()
        diagnostics["Effective Model"] = "White-Box Fallback"
        diagnostics["Prediction Source"] = "Fallback"
        diagnostics["Fallback Used"] = True

    signal_summary, signal_monthly = build_signal_diagnostics(predictions, forward_returns)
    etf_signal_summary, etf_signal_monthly = build_signal_diagnostics(predictions, forward_returns, A2_ETF_SYMBOLS)
    classification_summary = build_classification_diagnostics(predictions, y, forward_returns) if target_type == "Next-month outperform SPY classification" else pd.DataFrame()
    portfolio_returns, gross_returns, execution, weight_history = build_stage_a2_portfolios(
        close,
        predictions,
        base_cost_bps,
        impact_bps,
        kelly_fraction,
        target_volatility,
        smoothing,
        enable_regime_overlay,
        max_drawdown_limit,
        turnover_cap,
        rebalance_threshold,
        top_n,
    )
    if portfolio_returns.empty:
        return (
            x,
            predictions,
            walk_log,
            importance_history,
            portfolio_returns,
            gross_returns,
            execution,
            weight_history,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            diagnostics,
            signal_summary,
            signal_monthly,
            etf_signal_summary,
            etf_signal_monthly,
            tuning_results,
            classification_summary,
        )
    benchmark = close["SPY"].pct_change(fill_method=None).reindex(portfolio_returns.index).fillna(0.0) if "SPY" in close else None
    metrics = calculate_metrics(portfolio_returns, benchmark)
    regimes = detect_regime_states(close)
    stress = stress_test_returns(portfolio_returns)
    exposures = factor_exposure_heatmap(portfolio_returns, close)
    diagnostics["Prediction Count"] = len(predictions)
    diagnostics["Portfolio Return Rows"] = len(portfolio_returns)
    return (
        x,
        predictions,
        walk_log,
        importance_history,
        portfolio_returns,
        gross_returns,
        execution,
        weight_history,
        metrics,
        regimes,
        stress,
        exposures,
        diagnostics,
        signal_summary,
        signal_monthly,
        etf_signal_summary,
        etf_signal_monthly,
        tuning_results,
        classification_summary,
    )


def select_best_stage_a2_model(model_results: dict[str, tuple], benchmark: pd.Series | None) -> tuple[str, pd.DataFrame, str]:
    rows = []
    for model_name, result in model_results.items():
        metrics = result[8]
        execution = result[6]
        signal_summary = result[13]
        walk_log = result[2]
        if metrics.empty:
            continue
        best_metric = metrics.sort_values(["Sharpe", "Max Drawdown"], ascending=[False, False]).iloc[0]
        turnover = execution["Turnover"].mean() if not execution.empty and "Turnover" in execution else np.nan
        top_bottom = signal_summary["Average Top-Bottom 5 Spread"].iloc[0] if not signal_summary.empty and "Average Top-Bottom 5 Spread" in signal_summary else np.nan
        prediction_ic = signal_summary["Average Prediction IC"].iloc[0] if not signal_summary.empty and "Average Prediction IC" in signal_summary else np.nan
        train_ic = walk_log["Train IC"].mean() if not walk_log.empty and "Train IC" in walk_log else np.nan
        rows.append(
            {
                "Model": model_name,
                "Selected Portfolio During Model Test": best_metric["Series"],
                "OOS Sharpe": best_metric["Sharpe"],
                "OOS Annualized Return": best_metric["CAGR"],
                "OOS Max Drawdown": best_metric["Max Drawdown"],
                "Calmar": best_metric["Calmar"],
                "Net Return After Costs": best_metric["Total Return"],
                "Top-Minus-Bottom Spread": top_bottom,
                "Prediction IC": prediction_ic,
                "Average Turnover": turnover,
                "Train Score": train_ic,
                "OOS Score": prediction_ic,
            }
        )
    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return "", leaderboard, "No model produced enough walk-forward out-of-sample results."
    leaderboard = leaderboard.sort_values(
        ["OOS Sharpe", "OOS Max Drawdown", "Top-Minus-Bottom Spread", "Average Turnover"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = leaderboard.iloc[0]["Model"]
    reason = (
        f"{selected} was selected because it had the strongest walk-forward OOS Sharpe "
        f"({leaderboard.iloc[0]['OOS Sharpe']:.2f}), with drawdown and turnover used as tie-breakers."
    )
    if benchmark is not None and not benchmark.empty:
        spy_metrics = calculate_metrics(benchmark.to_frame("SPY"))
        if not spy_metrics.empty and leaderboard.iloc[0]["OOS Sharpe"] < spy_metrics.iloc[0]["Sharpe"]:
            reason += " Warning: the selected ML model's best portfolio Sharpe is below SPY over the same available window."
    return selected, leaderboard, reason


def recommend_stage_a2_portfolio_method(
    metrics: pd.DataFrame,
    gross_returns: pd.DataFrame,
    execution: pd.DataFrame,
    benchmark: pd.Series | None,
) -> tuple[str, pd.DataFrame, str]:
    if metrics.empty:
        return "", pd.DataFrame(), "No portfolio method produced enough returns."
    gross_metrics = calculate_metrics(gross_returns, benchmark) if not gross_returns.empty else pd.DataFrame()
    comparison = metrics.rename(columns={"Series": "Portfolio Method"}).copy()
    if not gross_metrics.empty:
        gross = gross_metrics.rename(columns={"Series": "Portfolio Method"})[["Portfolio Method", "Total Return"]].rename(columns={"Total Return": "Gross Return"})
        comparison = comparison.merge(gross, on="Portfolio Method", how="left")
        comparison["Transaction Cost Drag"] = comparison["Gross Return"] - comparison["Total Return"]
    else:
        comparison["Gross Return"] = np.nan
        comparison["Transaction Cost Drag"] = np.nan
    if not execution.empty:
        exec_summary = execution.groupby("Portfolio", as_index=False).agg(
            Average_Turnover=("Turnover", "mean"),
            Average_Cost=("Market Impact Cost", "mean"),
            Max_Position_Weight=("Max Position Weight", "max"),
        ).rename(columns={"Portfolio": "Portfolio Method"})
        comparison = comparison.merge(exec_summary, on="Portfolio Method", how="left")
    comparison = comparison.sort_values(
        ["Sharpe", "Max Drawdown", "Average_Turnover", "Transaction Cost Drag"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    recommended = comparison.iloc[0]["Portfolio Method"]
    reason = (
        f"{recommended} is recommended because it had the highest OOS Sharpe "
        f"({comparison.iloc[0]['Sharpe']:.2f}); drawdown, turnover, and cost drag were used as tie-breakers."
    )
    return recommended, comparison, reason


def get_stage_a2_current_holdings(
    weight_history: pd.DataFrame,
    predictions: pd.Series,
    recommended_method: str,
    importance_history: pd.DataFrame,
) -> pd.DataFrame:
    if weight_history.empty or not recommended_method:
        return pd.DataFrame()
    latest_date = weight_history.loc[weight_history["Portfolio"].eq(recommended_method), "Date"].max()
    holdings = weight_history[(weight_history["Portfolio"].eq(recommended_method)) & (weight_history["Date"].eq(latest_date))].copy()
    if holdings.empty:
        return holdings
    latest_signal_date = predictions.index.get_level_values("date").max() if not predictions.empty else None
    if latest_signal_date is not None:
        latest_scores = predictions.loc[predictions.index.get_level_values("date") == latest_signal_date]
        if not latest_scores.empty:
            score_frame = latest_scores.rename("Prediction Score").reset_index()
            score_frame["ML Rank"] = score_frame["Prediction Score"].rank(ascending=False, method="first").astype(int)
            holdings = holdings.merge(score_frame[["symbol", "Prediction Score", "ML Rank"]], left_on="ETF", right_on="symbol", how="left").drop(columns=["symbol"], errors="ignore")
    top_features = []
    if not importance_history.empty:
        top_features = importance_history.groupby("Feature")["Importance"].mean().sort_values(ascending=False).head(3).index.tolist()
    driver_text = "High ML rank; top model drivers: " + ", ".join(top_features) if top_features else "High ML rank based on current model score."
    holdings["Reason / Top Drivers"] = driver_text
    return holdings.sort_values("Weight", ascending=False)


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    return returns.rolling(window).mean() * 252 / (returns.rolling(window).std() * np.sqrt(252))


def monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    monthly = (1 + returns).resample("ME").prod() - 1
    table = monthly.to_frame("Return")
    table["Year"] = table.index.year
    table["Month"] = table.index.strftime("%b")
    return table.pivot(index="Year", columns="Month", values="Return")


def format_diagnostic_metrics(
    returns_map: dict[str, pd.Series],
    benchmark: pd.Series | None = None,
    turnover_map: dict[str, float] | None = None,
    cost_drag_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    frames = []
    for name, series in returns_map.items():
        clean = series.dropna()
        if clean.empty:
            continue
        metrics = calculate_metrics(clean.to_frame(name), benchmark).rename(columns={"Series": "Strategy"})
        metrics["Turnover"] = (turnover_map or {}).get(name, np.nan)
        metrics["Transaction Cost Drag"] = (cost_drag_map or {}).get(name, np.nan)
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).rename(
        columns={"CAGR": "Annualized Return", "Annual Volatility": "Annualized Volatility"}
    )


def backtest_simple_rotation_benchmark(
    close: pd.DataFrame,
    method: str,
    top_n: int = 5,
    base_cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    daily_returns = close.pct_change(fill_method=None).dropna()
    month_ends = close.resample("ME").last().index
    net = pd.Series(0.0, index=daily_returns.index)
    gross = pd.Series(0.0, index=daily_returns.index)
    previous = pd.Series(0.0, index=close.columns)
    rows = []
    for idx in range(12, len(month_ends) - 1):
        signal_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_date)
        if not period_mask.any():
            continue
        weights = pd.Series(0.0, index=close.columns)
        if method == "SPY Buy-and-Hold":
            if "SPY" in weights.index:
                weights["SPY"] = 1.0
        elif method == "Equal-Weight ETF Universe":
            weights[:] = 1.0 / len(weights)
        else:
            lookback_return = close.loc[:signal_date].iloc[-1] / close.loc[:signal_date].iloc[max(0, len(close.loc[:signal_date]) - 252)] - 1
            ranked = lookback_return.dropna().sort_values(ascending=False)
            selected = ranked.head(top_n).index.tolist()
            if method == "Dual Momentum Top 5":
                selected = [symbol for symbol in selected if ranked.get(symbol, 0.0) > 0]
                if not selected:
                    defensive = [symbol for symbol in ["IEF", "TLT", "SPY"] if symbol in close.columns]
                    selected = defensive[:1] if defensive else ranked.head(1).index.tolist()
            if selected:
                weights[selected] = 1.0 / len(selected)
        cost = (weights - previous).abs().sum() * base_cost_bps / 10000
        raw = daily_returns.loc[period_mask, weights.index].dot(weights)
        period = raw.copy()
        if not period.empty:
            period.iloc[0] -= cost
        gross.loc[period_mask] = raw
        net.loc[period_mask] = period
        rows.append({"Date": signal_date, "Strategy": method, "Turnover": (weights - previous).abs().sum(), "Cost Drag": cost})
        previous = weights
    return net, gross, pd.DataFrame(rows)


def build_stage_a2_benchmark_comparison(close: pd.DataFrame, recommended_returns: pd.Series, benchmark: pd.Series | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns_map = {"Stage A2 Recommended ML": recommended_returns}
    turnover_map = {"Stage A2 Recommended ML": np.nan}
    cost_drag_map = {"Stage A2 Recommended ML": np.nan}
    execution_rows = []
    for method in ["SPY Buy-and-Hold", "Equal-Weight ETF Universe", "12M Momentum Top 5", "Dual Momentum Top 5"]:
        net, gross, execution = backtest_simple_rotation_benchmark(
            close,
            method,
            top_n=int(STAGE_A2_PRESENTATION_CONFIG["Top-N ETFs"]),
            base_cost_bps=float(STAGE_A2_PRESENTATION_CONFIG["Transaction Cost Bps"]),
        )
        returns_map[method] = net.reindex(recommended_returns.index).fillna(0.0)
        turnover_map[method] = execution["Turnover"].mean() if not execution.empty else np.nan
        gross_total = (1 + gross.reindex(recommended_returns.index).fillna(0.0)).prod() - 1
        net_total = (1 + returns_map[method]).prod() - 1
        cost_drag_map[method] = gross_total - net_total
        execution_rows.append(execution)
    table = format_diagnostic_metrics(returns_map, benchmark, turnover_map, cost_drag_map)
    execution = pd.concat(execution_rows, ignore_index=True) if execution_rows else pd.DataFrame()
    return table, execution


def build_ranking_stability(predictions: pd.Series, top_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    previous_top = set()
    holding_streaks: dict[str, int] = {}
    completed = []
    for date, group in predictions.groupby(level="date"):
        ranked = group.sort_values(ascending=False)
        top = [idx[1] for idx in ranked.head(top_n).index]
        top_set = set(top)
        for symbol in list(holding_streaks):
            if symbol in top_set:
                holding_streaks[symbol] += 1
            else:
                completed.append(holding_streaks.pop(symbol))
        for symbol in top_set:
            holding_streaks.setdefault(symbol, 1)
        overlap = len(top_set & previous_top) if previous_top else np.nan
        rows.append(
            {
                "Date": date,
                "Top ETFs": ", ".join(top),
                "Top-5 Changed": np.nan if not previous_top else int(top_set != previous_top),
                "Rank Turnover": np.nan if not previous_top else 1 - overlap / top_n,
                "Top-5 Overlap": overlap,
            }
        )
        previous_top = top_set
    completed.extend(holding_streaks.values())
    monthly = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "Current Top-Ranked ETFs": monthly["Top ETFs"].iloc[-1] if not monthly.empty else "",
                "Previous Month Top-Ranked ETFs": monthly["Top ETFs"].iloc[-2] if len(monthly) > 1 else "",
                "Average Rank Turnover": monthly["Rank Turnover"].mean() if not monthly.empty else np.nan,
                "Top-5 Basket Change Rate": monthly["Top-5 Changed"].mean() if not monthly.empty else np.nan,
                "Average Holding Period Months": np.mean(completed) if completed else np.nan,
            }
        ]
    )
    return summary, monthly


def backtest_ml_equal_weight_top_n(
    close: pd.DataFrame,
    predictions: pd.Series,
    top_n: int = 5,
    base_cost_bps: float = 10.0,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    if predictions.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
    daily_returns = close.pct_change(fill_method=None).dropna()
    month_ends = close.resample("ME").last().index
    prediction_frame = predictions.rename("score").reset_index().set_index("date")
    net = pd.Series(0.0, index=daily_returns.index)
    gross = pd.Series(0.0, index=daily_returns.index)
    previous = pd.Series(0.0, index=close.columns)
    rows = []
    for idx in range(12, len(month_ends) - 1):
        signal_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        available_dates = prediction_frame.index[prediction_frame.index <= signal_date]
        if len(available_dates) == 0:
            continue
        signal_rows = prediction_frame.loc[available_dates[-1]]
        if isinstance(signal_rows, pd.Series):
            signal_rows = signal_rows.to_frame().T
        top = signal_rows.set_index("symbol")["score"].sort_values(ascending=False).head(top_n).index
        weights = pd.Series(0.0, index=close.columns)
        tradable = [symbol for symbol in top if symbol in weights.index]
        if not tradable:
            continue
        weights[tradable] = 1.0 / len(tradable)
        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_date)
        if not period_mask.any():
            continue
        cost = (weights - previous).abs().sum() * base_cost_bps / 10000
        raw = daily_returns.loc[period_mask, weights.index].dot(weights)
        period = raw.copy()
        if not period.empty:
            period.iloc[0] -= cost
        gross.loc[period_mask] = raw
        net.loc[period_mask] = period
        rows.append({"Date": signal_date, "Portfolio": "Equal-Weight ML Top 5", "Turnover": (weights - previous).abs().sum(), "Market Impact Cost": cost})
        previous = weights
    return net, gross, pd.DataFrame(rows)


def feature_columns_for_group(columns: pd.Index, group: str) -> list[str]:
    columns = list(columns)
    if group == "momentum only":
        keys = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "relative_ret", "momentum_rank", "market_ret"]
    elif group == "trend only":
        keys = ["above_200d", "ma50", "spy_trend", "risk_off", "market_drawdown"]
    elif group == "volatility/drawdown only":
        keys = ["vol", "drawdown"]
    elif group == "macro only":
        keys = ["macro_"]
    elif group == "momentum + trend":
        keys = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "relative_ret", "momentum_rank", "market_ret", "above_200d", "ma50", "spy_trend", "risk_off", "market_drawdown"]
    else:
        return columns
    selected = [column for column in columns if any(key in column for key in keys)]
    return selected or columns


def run_stage_a2_feature_subset_diagnostic(
    close: pd.DataFrame,
    macro: pd.DataFrame,
    model_name: str,
    feature_group: str,
    benchmark: pd.Series | None,
) -> dict:
    config = STAGE_A2_PRESENTATION_CONFIG
    x, y, forward = build_stage_a2_features(close, macro, target_type=config["Target Type"], return_forward_returns=True)
    columns = feature_columns_for_group(x.columns, feature_group)
    x = x[columns]
    predictions, walk_log, _ = build_walk_forward_ml_predictions(
        x,
        y,
        model_name,
        target_type=config["Target Type"],
        min_train_samples=80,
        min_history_months=9,
        collect_importance=False,
    )
    signal_summary, _ = build_signal_diagnostics(predictions, forward)
    portfolio_returns, _, execution, _ = build_stage_a2_portfolios(
        close,
        predictions,
        config["Transaction Cost Bps"],
        config["Square-Root Impact Bps"],
        config["Kelly Fraction"],
        config["Target Volatility"],
        0.0,
        config["Regime Overlay"],
        config["Max Drawdown Guard"],
        config["Monthly Turnover Cap"],
        config["Rebalance Threshold"],
        int(config["Top-N ETFs"]),
    )
    if portfolio_returns.empty:
        return {"Feature Group": feature_group}
    metrics = calculate_metrics(portfolio_returns, benchmark)
    best = metrics.sort_values("Sharpe", ascending=False).iloc[0]
    return {
        "Feature Group": feature_group,
        "Best Portfolio": best["Series"],
        "OOS Sharpe": best["Sharpe"],
        "OOS Return": best["Total Return"],
        "Max Drawdown": best["Max Drawdown"],
        "Top-Minus-Bottom Spread": signal_summary["Average Top-Bottom 5 Spread"].iloc[0] if not signal_summary.empty else np.nan,
        "Prediction IC": signal_summary["Average Prediction IC"].iloc[0] if not signal_summary.empty else np.nan,
        "Average Turnover": execution["Turnover"].mean() if not execution.empty else np.nan,
        "Train IC": walk_log["Train IC"].mean() if not walk_log.empty and "Train IC" in walk_log else np.nan,
    }


def run_stage_a2_target_diagnostic(close: pd.DataFrame, macro: pd.DataFrame, model_name: str, benchmark: pd.Series | None) -> pd.DataFrame:
    rows = []
    config = STAGE_A2_PRESENTATION_CONFIG
    for target in TARGET_TYPES:
        result = run_stage_a2_research(
            close,
            macro,
            model_name,
            config["Transaction Cost Bps"],
            config["Square-Root Impact Bps"],
            config["Kelly Fraction"],
            config["Target Volatility"],
            0.0,
            config["Regime Overlay"],
            config["Max Drawdown Guard"],
            config["Monthly Turnover Cap"],
            config["Rebalance Threshold"],
            target,
            False,
            collect_importance=False,
            top_n=int(config["Top-N ETFs"]),
        )
        metrics = result[8]
        signal_summary = result[13]
        execution = result[6]
        if metrics.empty:
            rows.append({"Target": target})
            continue
        best = metrics.sort_values("Sharpe", ascending=False).iloc[0]
        rows.append(
            {
                "Target": target,
                "Best Portfolio": best["Series"],
                "OOS Sharpe": best["Sharpe"],
                "OOS Return": best["Total Return"],
                "Max Drawdown": best["Max Drawdown"],
                "Top-Minus-Bottom Spread": signal_summary["Average Top-Bottom 5 Spread"].iloc[0] if not signal_summary.empty else np.nan,
                "Prediction IC": signal_summary["Average Prediction IC"].iloc[0] if not signal_summary.empty else np.nan,
                "Average Turnover": execution["Turnover"].mean() if not execution.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_stage_a2_performance_diagnostics(close: pd.DataFrame, macro: pd.DataFrame, bundle: dict) -> dict:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    benchmark = bundle["benchmark"]
    recommended_returns = result[4][recommended]
    benchmark_table, simple_execution = build_stage_a2_benchmark_comparison(close, recommended_returns, benchmark)
    if not benchmark_table.empty:
        ml_mask = benchmark_table["Strategy"].eq("Stage A2 Recommended ML")
        method_execution = result[6][result[6]["Portfolio"].eq(recommended)] if not result[6].empty else pd.DataFrame()
        benchmark_table.loc[ml_mask, "Turnover"] = method_execution["Turnover"].mean() if not method_execution.empty else np.nan
        if recommended in result[5]:
            gross_total = (1 + result[5][recommended].reindex(recommended_returns.index).fillna(0.0)).prod() - 1
            net_total = (1 + recommended_returns).prod() - 1
            benchmark_table.loc[ml_mask, "Transaction Cost Drag"] = gross_total - net_total
    ranking_summary, ranking_monthly = build_ranking_stability(result[1], int(STAGE_A2_PRESENTATION_CONFIG["Top-N ETFs"]))
    ml_equal_net, ml_equal_gross, ml_equal_execution = backtest_ml_equal_weight_top_n(
        close,
        result[1],
        int(STAGE_A2_PRESENTATION_CONFIG["Top-N ETFs"]),
        float(STAGE_A2_PRESENTATION_CONFIG["Transaction Cost Bps"]),
    )
    ml_equal_metrics = format_diagnostic_metrics(
        {"Equal-Weight ML Top 5": ml_equal_net.reindex(recommended_returns.index).fillna(0.0)},
        benchmark,
        {"Equal-Weight ML Top 5": ml_equal_execution["Turnover"].mean() if not ml_equal_execution.empty else np.nan},
        {"Equal-Weight ML Top 5": ((1 + ml_equal_gross.reindex(recommended_returns.index).fillna(0.0)).prod() - 1) - ((1 + ml_equal_net.reindex(recommended_returns.index).fillna(0.0)).prod() - 1)},
    )
    target_comparison = run_stage_a2_target_diagnostic(close, macro, bundle["selected_model"], benchmark)
    feature_ablation = pd.DataFrame(
        [
            run_stage_a2_feature_subset_diagnostic(close, macro, bundle["selected_model"], group, benchmark)
            for group in ["momentum only", "trend only", "volatility/drawdown only", "macro only", "momentum + trend", "all features"]
        ]
    )
    model_winners = []
    monthly_frames = []
    for model_name, model_result in bundle["model_results"].items():
        monthly = model_result[14].copy()
        if not monthly.empty:
            monthly["Model"] = model_name
            monthly_frames.append(monthly)
    if monthly_frames:
        combined = pd.concat(monthly_frames, ignore_index=True)
        for date, group in combined.groupby("Date"):
            best = group.sort_values("Top-Bottom 5 Spread", ascending=False).iloc[0]
            model_winners.append({"Date": date, "Monthly Best Model": best["Model"]})
    model_winners = pd.DataFrame(model_winners)
    model_change_rate = model_winners["Monthly Best Model"].ne(model_winners["Monthly Best Model"].shift()).mean() if not model_winners.empty else np.nan
    return {
        "benchmark_table": benchmark_table,
        "simple_execution": simple_execution,
        "ranking_summary": ranking_summary,
        "ranking_monthly": ranking_monthly,
        "ml_equal_top5_metrics": ml_equal_metrics,
        "ml_equal_top5_execution": ml_equal_execution,
        "target_comparison": target_comparison,
        "feature_ablation": feature_ablation,
        "model_winners": model_winners,
        "model_change_rate": model_change_rate,
    }


def stage_a2_plain_english_diagnosis(bundle: dict, diagnostics: dict) -> list[str]:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    metrics = result[8]
    signal_summary = result[13]
    execution = result[6]
    benchmark_table = diagnostics["benchmark_table"]
    rows = []
    spread = signal_summary["Average Top-Bottom 5 Spread"].iloc[0] if not signal_summary.empty else np.nan
    rows.append(f"ML signal has {'positive' if pd.notna(spread) and spread > 0 else 'negative or weak'} ranking spread: average top-minus-bottom 5 spread is {spread:.2%}." if pd.notna(spread) else "ML signal spread is unavailable.")
    if not execution.empty:
        avg_turnover = execution.loc[execution["Portfolio"].eq(recommended), "Turnover"].mean()
        rows.append(f"Average monthly turnover for the recommended portfolio is {avg_turnover:.2f}x, so transaction costs can materially reduce returns." if pd.notna(avg_turnover) and avg_turnover > 0.5 else f"Average turnover is {avg_turnover:.2f}x, so cost drag is not the only explanation.")
    if not benchmark_table.empty and "SPY Buy-and-Hold" in benchmark_table["Strategy"].values:
        ml_return = metrics.loc[metrics["Series"].eq(recommended), "Total Return"].iloc[0]
        spy_return = benchmark_table.loc[benchmark_table["Strategy"].eq("SPY Buy-and-Hold"), "Total Return"].iloc[0]
        rows.append("ML underperforms SPY because the period favored buy-and-hold equity beta." if ml_return < spy_return else "ML outperformed SPY on total return in this sample.")
    if not benchmark_table.empty and "12M Momentum Top 5" in benchmark_table["Strategy"].values:
        ml_return = metrics.loc[metrics["Series"].eq(recommended), "Total Return"].iloc[0]
        mom_return = benchmark_table.loc[benchmark_table["Strategy"].eq("12M Momentum Top 5"), "Total Return"].iloc[0]
        rows.append("Current ML model does not add value over the simple 12-month momentum benchmark." if ml_return < mom_return else "Current ML model adds value over the simple 12-month momentum benchmark in this sample.")
    target = diagnostics["target_comparison"].sort_values("OOS Sharpe", ascending=False).head(1)
    if not target.empty:
        rows.append(f"Recommended next improvement: review target choice. Best diagnostic target by OOS Sharpe is '{target.iloc[0]['Target']}', but this is diagnostic only and was not used to retroactively optimize the final model.")
    return rows


def compare_stress_to_spy(strategy_returns: pd.Series, benchmark: pd.Series | None) -> pd.DataFrame:
    windows = {
        "2008 Crisis": ("2008-01-01", "2009-03-31"),
        "2020 COVID Crash": ("2020-02-01", "2020-04-30"),
        "2022 Inflation / Rate-Hike Bear Market": ("2022-01-01", "2022-12-31"),
    }
    rows = []
    for label, (start, end) in windows.items():
        strat = strategy_returns.loc[(strategy_returns.index >= start) & (strategy_returns.index <= end)].dropna()
        spy = benchmark.loc[(benchmark.index >= start) & (benchmark.index <= end)].dropna() if benchmark is not None else pd.Series(dtype=float)
        if strat.empty or spy.empty:
            rows.append({"Stress Period": label, "Status": "Not enough data", "Comment": "Selected research window does not cover this stress period."})
            continue
        strat_equity = (1 + strat).cumprod()
        spy_equity = (1 + spy).cumprod()
        rows.append(
            {
                "Stress Period": label,
                "Status": "Available",
                "Strategy Return": strat_equity.iloc[-1] - 1,
                "SPY Return": spy_equity.iloc[-1] - 1,
                "Strategy Max Drawdown": (strat_equity / strat_equity.cummax() - 1).min(),
                "SPY Max Drawdown": (spy_equity / spy_equity.cummax() - 1).min(),
                "Strategy Volatility": strat.std() * np.sqrt(252),
                "Comment": "Strategy outperformed SPY in this window." if strat_equity.iloc[-1] > spy_equity.iloc[-1] else "SPY outperformed strategy in this window.",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def run_stage_a2_presentation_research(close: pd.DataFrame, macro: pd.DataFrame) -> dict:
    config = STAGE_A2_PRESENTATION_CONFIG
    model_results = {}
    for model_name in STAGE_A2_PRESENTATION_MODELS:
        model_results[model_name] = run_stage_a2_research(
            close,
            macro,
            model_name,
            config["Transaction Cost Bps"],
            config["Square-Root Impact Bps"],
            config["Kelly Fraction"],
            config["Target Volatility"],
            config["Weight Smoothing"],
            config["Regime Overlay"],
            config["Max Drawdown Guard"],
            config["Monthly Turnover Cap"],
            config["Rebalance Threshold"],
            config["Target Type"],
            False,
            collect_importance=True,
            top_n=int(config["Top-N ETFs"]),
        )
    benchmark = close["SPY"].pct_change(fill_method=None) if "SPY" in close else None
    selected_model, model_leaderboard, model_reason = select_best_stage_a2_model(model_results, benchmark)
    selected_result = model_results[selected_model] if selected_model else next(iter(model_results.values()))
    recommended_method, portfolio_comparison, portfolio_reason = recommend_stage_a2_portfolio_method(
        selected_result[8], selected_result[5], selected_result[6], benchmark
    )
    holdings = get_stage_a2_current_holdings(selected_result[7], selected_result[1], recommended_method, selected_result[3])
    bundle = {
        "config": config,
        "model_results": model_results,
        "selected_model": selected_model,
        "selected_result": selected_result,
        "model_leaderboard": model_leaderboard,
        "model_reason": model_reason,
        "recommended_method": recommended_method,
        "portfolio_comparison": portfolio_comparison,
        "portfolio_reason": portfolio_reason,
        "current_holdings": holdings,
        "benchmark": benchmark,
    }
    bundle["performance_diagnostics"] = build_stage_a2_performance_diagnostics(close, macro, bundle)
    return bundle


def render_stage_a2_executive_overview(bundle: dict) -> None:
    result = bundle["selected_result"]
    portfolio_returns = result[4]
    metrics = result[8]
    regimes = result[9]
    recommended = bundle["recommended_method"]
    benchmark = bundle["benchmark"]
    holdings = bundle["current_holdings"]
    selected_metrics = metrics[metrics["Series"].eq(recommended)].iloc[0]
    strategy_returns = portfolio_returns[recommended]
    current_regime = regimes["Regime"].iloc[-1] if not regimes.empty and "Regime" in regimes else "Not available"
    latest_rebalance = holdings["Date"].max().date().isoformat() if not holdings.empty else "Not available"

    cols = st.columns(4)
    cols[0].metric("Current Portfolio Value", f"${selected_metrics['Ending Value']:,.0f}")
    cols[1].metric("Net Return", f"{selected_metrics['Total Return']:.2%}")
    cols[2].metric("OOS Sharpe", f"{selected_metrics['Sharpe']:.2f}")
    cols[3].metric("Max Drawdown", f"{selected_metrics['Max Drawdown']:.2%}")
    cols = st.columns(4)
    cols[0].metric("Current Regime", current_regime)
    cols[1].metric("Latest Rebalance Date", latest_rebalance)
    cols[2].metric("Selected Model", bundle["selected_model"])
    cols[3].metric("Recommended Method", recommended)

    st.info(
        "Stage A2 is a white-box ML multi-asset ETF rotation strategy. The system ranks ETFs using momentum, trend, "
        "volatility, drawdown, beta, regime, and macro features, then builds a portfolio using the best-performing "
        "portfolio construction method based on walk-forward out-of-sample validation."
    )
    equity = (1 + strategy_returns).cumprod().to_frame(recommended)
    if benchmark is not None and not benchmark.empty:
        equity["SPY"] = (1 + benchmark.reindex(equity.index).fillna(0.0)).cumprod()
    st.plotly_chart(px.line(equity * STAGE_A2_INITIAL_CAPITAL, title="Recommended Strategy vs SPY"), use_container_width=True)
    drawdown = equity / equity.cummax() - 1
    st.plotly_chart(px.line(drawdown, title="Drawdown"), use_container_width=True)
    st.subheader("Current Allocation Summary")
    if holdings.empty:
        st.warning("Current holdings are unavailable.")
    else:
        st.plotly_chart(px.bar(holdings, x="ETF", y="Weight", color="Category", title="Current Recommended Allocation"), use_container_width=True)
        st.dataframe(holdings[["ETF", "Weight", "Category", "ML Rank", "Prediction Score"]].style.format({"Weight": "{:.2%}", "Prediction Score": "{:.4f}"}), use_container_width=True, hide_index=True)
    st.success(bundle["model_reason"])
    st.success(bundle["portfolio_reason"])


def render_stage_a2_portfolio_performance(bundle: dict) -> None:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    returns = result[4]
    gross_returns = result[5]
    execution = result[6]
    metrics = result[8]
    benchmark = bundle["benchmark"]
    strategy = returns[recommended]
    equity = (1 + strategy).cumprod().to_frame("Recommended Strategy")
    if benchmark is not None:
        equity["SPY"] = (1 + benchmark.reindex(strategy.index).fillna(0.0)).cumprod()
    st.plotly_chart(px.line(equity * STAGE_A2_INITIAL_CAPITAL, title="Cumulative Return: Recommended Strategy vs SPY"), use_container_width=True)
    gross_net = pd.DataFrame()
    if recommended in gross_returns:
        gross_metrics = calculate_metrics(gross_returns[[recommended]], benchmark)
        net_metrics = metrics[metrics["Series"].eq(recommended)]
        gross_net = net_metrics[["Series", "Total Return", "Sharpe", "Max Drawdown"]].merge(
            gross_metrics[["Series", "Total Return", "Sharpe", "Max Drawdown"]],
            on="Series",
            suffixes=(" Net", " Gross"),
        )
        gross_net["Transaction Cost Drag"] = gross_net["Total Return Gross"] - gross_net["Total Return Net"]
        st.dataframe(gross_net.style.format({c: "{:.2%}" for c in gross_net.columns if "Return" in c or "Drawdown" in c or "Drag" in c}).format({"Sharpe Net": "{:.2f}", "Sharpe Gross": "{:.2f}"}), use_container_width=True, hide_index=True)
    st.plotly_chart(px.line(rolling_sharpe(strategy).to_frame("Rolling Sharpe"), title="Rolling Sharpe"), use_container_width=True)
    drawdown = equity / equity.cummax() - 1
    st.plotly_chart(px.line(drawdown, title="Rolling Drawdown"), use_container_width=True)
    try:
        st.dataframe(monthly_return_table(strategy).style.format("{:.2%}"), use_container_width=True)
    except Exception:
        st.info("Monthly return table is unavailable for this sample.")
    perf = metrics[metrics["Series"].eq(recommended)].rename(
        columns={
            "CAGR": "Annualized Return",
            "Annual Volatility": "Annualized Volatility",
            "Ending Value": "Final Portfolio Value",
        }
    )
    avg_turnover = execution.loc[execution["Portfolio"].eq(recommended), "Turnover"].mean() if not execution.empty else np.nan
    cost_drag = gross_net["Transaction Cost Drag"].iloc[0] if not gross_net.empty else np.nan
    perf["Turnover"] = avg_turnover
    perf["Transaction Cost Drag"] = cost_drag
    st.dataframe(
        perf[["Annualized Return", "Annualized Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown", "Beta", "Turnover", "Transaction Cost Drag", "Final Portfolio Value"]].style.format(
            {
                "Annualized Return": "{:.2%}",
                "Annualized Volatility": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Sortino": "{:.2f}",
                "Calmar": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Beta": "{:.2f}",
                "Turnover": "{:.2f}x",
                "Transaction Cost Drag": "{:.2%}",
                "Final Portfolio Value": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_stage_a2_performance_diagnostics(bundle: dict) -> None:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    diagnostics = bundle["performance_diagnostics"]
    benchmark = bundle["benchmark"]
    returns = result[4]
    gross_returns = result[5]
    execution = result[6]
    signal_monthly = result[14]
    importance = result[3]

    st.subheader("Plain-English Diagnosis")
    for item in stage_a2_plain_english_diagnosis(bundle, diagnostics):
        st.info(item)

    st.subheader("Benchmark Comparison")
    benchmark_table = diagnostics["benchmark_table"].copy()
    if not benchmark_table.empty:
        st.dataframe(
            benchmark_table[["Strategy", "Total Return", "Annualized Return", "Annualized Volatility", "Sharpe", "Sortino", "Calmar", "Max Drawdown", "Turnover", "Transaction Cost Drag"]].style.format(
                {
                    "Total Return": "{:.2%}",
                    "Annualized Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Sortino": "{:.2f}",
                    "Calmar": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Turnover": "{:.2f}x",
                    "Transaction Cost Drag": "{:.2%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    if benchmark is not None:
        st.caption("Benchmark table is calculated over the same available OOS return window as the Stage A2 recommended portfolio.")

    st.subheader("ML Signal Quality")
    if signal_monthly.empty:
        st.warning("Monthly signal quality table is unavailable.")
    else:
        signal = signal_monthly.copy()
        signal["Rolling Top-Bottom 5 Spread"] = signal["Top-Bottom 5 Spread"].rolling(6).mean()
        signal["Rolling Prediction IC"] = signal["Prediction IC"].rolling(6).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg Top-Bottom 5 Spread", f"{signal['Top-Bottom 5 Spread'].mean():.2%}")
        c2.metric("Positive Spread Hit Rate", f"{(signal['Top-Bottom 5 Spread'] > 0).mean():.1%}")
        c3.metric("Avg Prediction IC", f"{signal['Prediction IC'].mean():.3f}")
        st.plotly_chart(px.line(signal, x="Date", y=["Rolling Top-Bottom 5 Spread", "Rolling Prediction IC"], title="Rolling ML Signal Quality"), use_container_width=True)
        st.dataframe(
            signal[["Date", "Top 3 Avg Forward Return", "Top 5 Avg Forward Return", "Bottom 3 Avg Forward Return", "Bottom 5 Avg Forward Return", "Top-Bottom 5 Spread", "Prediction IC"]].tail(36).style.format(
                {
                    "Top 3 Avg Forward Return": "{:.2%}",
                    "Top 5 Avg Forward Return": "{:.2%}",
                    "Bottom 3 Avg Forward Return": "{:.2%}",
                    "Bottom 5 Avg Forward Return": "{:.2%}",
                    "Top-Bottom 5 Spread": "{:.2%}",
                    "Prediction IC": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Gross vs Net Return")
    if recommended in gross_returns:
        gross = (1 + gross_returns[recommended]).cumprod()
        net = (1 + returns[recommended]).cumprod()
        gross_net = pd.DataFrame({"Gross Before Costs": gross, "Net After Costs": net})
        st.plotly_chart(px.line(gross_net * STAGE_A2_INITIAL_CAPITAL, title="Gross vs Net Cumulative Performance"), use_container_width=True)
    if not execution.empty:
        method_execution = execution[execution["Portfolio"].eq(recommended)].copy()
        st.metric("Average Monthly Turnover", f"{method_execution['Turnover'].mean():.2f}x")
        st.metric("Total Cost Drag Estimate", f"{method_execution['Market Impact Cost'].sum():.2%}")
        st.plotly_chart(px.line(method_execution, x="Date", y="Turnover", title="Monthly Turnover"), use_container_width=True)
        st.dataframe(method_execution.sort_values("Turnover", ascending=False).head(10), use_container_width=True, hide_index=True)

    st.subheader("Ranking Stability")
    ranking_summary = diagnostics["ranking_summary"]
    ranking_monthly = diagnostics["ranking_monthly"]
    if not ranking_summary.empty:
        st.dataframe(ranking_summary.style.format({"Average Rank Turnover": "{:.2%}", "Top-5 Basket Change Rate": "{:.2%}", "Average Holding Period Months": "{:.1f}"}), use_container_width=True, hide_index=True)
    if not ranking_monthly.empty:
        st.plotly_chart(px.line(ranking_monthly, x="Date", y="Rank Turnover", title="Top-5 Rank Turnover"), use_container_width=True)
        st.dataframe(ranking_monthly.tail(24), use_container_width=True, hide_index=True)

    st.subheader("Overfitting Check")
    walk_log = result[2]
    oos_ic = result[13]["Average Prediction IC"].iloc[0] if not result[13].empty else np.nan
    train_ic = walk_log["Train IC"].mean() if not walk_log.empty and "Train IC" in walk_log else np.nan
    overfit_rows = [{"Check": "Train IC vs OOS IC", "Train": train_ic, "OOS": oos_ic, "Warning": "Yes" if pd.notna(train_ic) and pd.notna(oos_ic) and train_ic - oos_ic > 0.15 else "No"}]
    if not importance.empty:
        stability = importance.groupby("Feature")["Importance"].agg(["mean", "std"])
        unstable_share = ((stability["std"] / stability["mean"].replace(0, np.nan)) > 1.5).mean()
        overfit_rows.append({"Check": "Feature Importance Instability", "Train": np.nan, "OOS": unstable_share, "Warning": "Yes" if unstable_share > 0.5 else "No"})
    change_rate = diagnostics.get("model_change_rate", np.nan)
    overfit_rows.append({"Check": "Monthly Best Model Change Rate", "Train": np.nan, "OOS": change_rate, "Warning": "Yes" if pd.notna(change_rate) and change_rate > 0.5 else "No"})
    st.dataframe(pd.DataFrame(overfit_rows).style.format({"Train": "{:.3f}", "OOS": "{:.3f}"}), use_container_width=True, hide_index=True)
    if any(row["Warning"] == "Yes" for row in overfit_rows):
        st.warning("One or more overfitting diagnostics are elevated. Treat the ML edge as unstable until validated on more data.")

    st.subheader("Target Comparison")
    target_comparison = diagnostics["target_comparison"]
    if not target_comparison.empty:
        st.dataframe(
            target_comparison.style.format(
                {
                    "OOS Sharpe": "{:.2f}",
                    "OOS Return": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Top-Minus-Bottom Spread": "{:.2%}",
                    "Prediction IC": "{:.3f}",
                    "Average Turnover": "{:.2f}x",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Feature Ablation")
    feature_ablation = diagnostics["feature_ablation"]
    if not feature_ablation.empty:
        st.dataframe(
            feature_ablation.style.format(
                {
                    "OOS Sharpe": "{:.2f}",
                    "OOS Return": "{:.2%}",
                    "Max Drawdown": "{:.2%}",
                    "Top-Minus-Bottom Spread": "{:.2%}",
                    "Prediction IC": "{:.3f}",
                    "Average Turnover": "{:.2f}x",
                    "Train IC": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Portfolio Construction Diagnosis")
    portfolio_comparison = bundle["portfolio_comparison"].copy()
    if not portfolio_comparison.empty:
        extra = portfolio_comparison[["Portfolio Method", "Total Return", "Sharpe", "Max Drawdown", "Average_Turnover", "Transaction Cost Drag", "Beta"]].copy()
        ml_equal = diagnostics.get("ml_equal_top5_metrics", pd.DataFrame())
        if not ml_equal.empty:
            ml_equal_row = ml_equal.rename(
                columns={
                    "Strategy": "Portfolio Method",
                    "Turnover": "Average_Turnover",
                }
            )[["Portfolio Method", "Total Return", "Sharpe", "Max Drawdown", "Average_Turnover", "Transaction Cost Drag", "Beta"]]
            extra = pd.concat([ml_equal_row, extra], ignore_index=True)
        extra["Diagnosis"] = np.where(
            extra["Sharpe"] < 0,
            "Weak signal or poor weighting",
            np.where(extra["Average_Turnover"] > 1.0, "High turnover risk", np.where(extra["Beta"].abs() > 0.8, "High beta exposure", "Relatively stable")),
        )
        st.dataframe(
            extra.style.format(
                {
                    "Total Return": "{:.2%}",
                    "Sharpe": "{:.2f}",
                    "Max Drawdown": "{:.2%}",
                    "Average_Turnover": "{:.2f}x",
                    "Transaction Cost Drag": "{:.2%}",
                    "Beta": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def render_stage_a2_current_portfolio(bundle: dict) -> None:
    holdings = bundle["current_holdings"]
    diagnostics = bundle["selected_result"][12]
    if holdings.empty:
        st.warning("Current holdings are unavailable.")
        return
    st.dataframe(
        holdings[["ETF", "Weight", "Category", "ML Rank", "Prediction Score", "Reason / Top Drivers"]].style.format({"Weight": "{:.2%}", "Prediction Score": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(px.bar(holdings, x="ETF", y="Weight", color="Category", title="Current Holdings"), use_container_width=True)
    if holdings["Weight"].abs().max() > MAX_LONG_ONLY_WEIGHT + 1e-9:
        st.warning("One or more ETF weights exceed the 25% max ETF weight guideline.")
    if holdings["Weight"].abs().head(3).sum() > 0.75:
        st.warning("Portfolio is concentrated: top three absolute weights exceed 75%.")
    if diagnostics.get("Prediction Source") == "Fallback":
        st.warning("Prediction source is fallback instead of ML.")


def render_stage_a2_model_selection(bundle: dict) -> None:
    leaderboard = bundle["model_leaderboard"].copy()
    if leaderboard.empty:
        st.warning("No model leaderboard is available.")
        return
    leaderboard["Selected"] = leaderboard["Model"].eq(bundle["selected_model"])
    st.success(bundle["model_reason"])
    st.plotly_chart(px.bar(leaderboard, x="Model", y="OOS Sharpe", color="Selected", title="Model Leaderboard by Walk-Forward OOS Sharpe"), use_container_width=True)
    st.dataframe(
        leaderboard.style.format(
            {
                "OOS Sharpe": "{:.2f}",
                "OOS Annualized Return": "{:.2%}",
                "OOS Max Drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
                "Net Return After Costs": "{:.2%}",
                "Top-Minus-Bottom Spread": "{:.2%}",
                "Prediction IC": "{:.3f}",
                "Average Turnover": "{:.2f}x",
                "Train Score": "{:.3f}",
                "OOS Score": "{:.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    if (leaderboard["Top-Minus-Bottom Spread"].fillna(0) <= 0).all():
        st.warning("All ML models have non-positive top-minus-bottom spread. The ranking signal may be weak in this sample.")
    overfit = leaderboard[(leaderboard["Train Score"].notna()) & (leaderboard["OOS Score"].notna()) & ((leaderboard["Train Score"] - leaderboard["OOS Score"]) > 0.15)]
    if not overfit.empty:
        st.warning("Some models show much higher train score than OOS score. Treat those results as overfitting risk.")


def render_stage_a2_portfolio_comparison(bundle: dict) -> None:
    comparison = bundle["portfolio_comparison"].copy()
    result = bundle["selected_result"]
    if comparison.empty:
        st.warning("Portfolio method comparison is unavailable.")
        return
    comparison["Recommended"] = comparison["Portfolio Method"].eq(bundle["recommended_method"])
    st.success(bundle["portfolio_reason"])
    st.dataframe(
        comparison[["Portfolio Method", "Recommended", "Total Return", "Sharpe", "Max Drawdown", "Calmar", "Average_Turnover", "Transaction Cost Drag", "Beta", "Ending Value"]].style.format(
            {
                "Total Return": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Max Drawdown": "{:.2%}",
                "Calmar": "{:.2f}",
                "Average_Turnover": "{:.2f}x",
                "Transaction Cost Drag": "{:.2%}",
                "Beta": "{:.2f}",
                "Ending Value": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    equity = (1 + result[4]).cumprod() * STAGE_A2_INITIAL_CAPITAL
    st.plotly_chart(px.line(equity, title="Portfolio Method Cumulative Return Comparison"), use_container_width=True)
    drawdown = (1 + result[4]).cumprod()
    drawdown = drawdown / drawdown.cummax() - 1
    st.plotly_chart(px.line(drawdown, title="Portfolio Method Drawdown Comparison"), use_container_width=True)
    if not result[6].empty:
        cost_turnover = result[6].groupby("Portfolio", as_index=False).agg(Turnover=("Turnover", "mean"), Cost_Drag=("Market Impact Cost", "sum"))
        st.plotly_chart(px.bar(cost_turnover, x="Portfolio", y=["Turnover", "Cost_Drag"], barmode="group", title="Turnover and Cost Drag"), use_container_width=True)


def render_stage_a2_white_box_explanation(bundle: dict) -> None:
    result = bundle["selected_result"]
    importance = result[3]
    holdings = bundle["current_holdings"]
    st.info("This section explains which variables are driving the model's ETF ranking.")
    if importance.empty:
        st.warning("Feature importance is unavailable for the selected model.")
    else:
        latest = importance.sort_values("Signal Date").groupby("Feature")["Importance"].tail(1)
        latest = importance.loc[latest.index].sort_values("Importance", ascending=False).head(10)
        importance_label = latest["Importance Type"].iloc[0] if "Importance Type" in latest else "Feature Importance Fallback, not full SHAP"
        if "SHAP" not in importance_label:
            st.warning("Feature Importance Fallback, not full SHAP.")
        st.plotly_chart(px.bar(latest, x="Importance", y="Feature", orientation="h", title=f"Top 10 Feature Importance: {importance_label}"), use_container_width=True)
        stability = importance.groupby("Feature")["Importance"].agg(["mean", "std"]).reset_index().sort_values("mean", ascending=False).head(15)
        st.dataframe(stability.style.format({"mean": "{:.3f}", "std": "{:.3f}"}), use_container_width=True, hide_index=True)
    if not holdings.empty:
        st.subheader("Current Top-Ranked ETFs")
        st.dataframe(holdings.sort_values("ML Rank").head(10), use_container_width=True, hide_index=True)


def render_stage_a2_risk_dashboard(bundle: dict) -> None:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    returns = result[4][recommended]
    regimes = result[9]
    exposures = result[11]
    execution = result[6]
    close_benchmark = bundle["benchmark"]
    if regimes.empty:
        st.warning("Regime detection is unavailable.")
    else:
        method = regimes["Regime Method"].iloc[-1] if "Regime Method" in regimes else "Rule-Based Regime Proxy"
        label = "HMM Regime Detection" if method == "HMM" else "Rule-Based Regime Proxy"
        st.subheader(label)
        st.plotly_chart(px.area(regimes.reset_index(), x=regimes.reset_index().columns[0], y="Regime Code", color="Regime", title=label), use_container_width=True)
    st.subheader("Factor Exposure Monitoring")
    if exposures.empty:
        st.info("Factor exposure heatmap is unavailable.")
    else:
        st.plotly_chart(px.imshow(exposures, aspect="auto", color_continuous_scale="RdBu", title="Factor Exposure Monitoring"), use_container_width=True)
    if close_benchmark is not None:
        aligned = pd.DataFrame({"Strategy": returns, "SPY": close_benchmark.reindex(returns.index)}).dropna()
        if len(aligned) > 60:
            st.metric("Beta to SPY", f"{aligned['Strategy'].cov(aligned['SPY']) / aligned['SPY'].var():.2f}")
            st.metric("Correlation to SPY", f"{aligned['Strategy'].corr(aligned['SPY']):.2f}")
    st.plotly_chart(px.line((returns.rolling(63).std() * np.sqrt(252)).to_frame("Rolling Volatility"), title="Rolling Volatility"), use_container_width=True)
    st.plotly_chart(px.line(rolling_sharpe(returns).to_frame("Rolling Sharpe"), title="Rolling Sharpe"), use_container_width=True)
    equity = (1 + returns).cumprod()
    st.plotly_chart(px.line((equity / equity.cummax() - 1).to_frame("Rolling Drawdown"), title="Rolling Drawdown"), use_container_width=True)
    if not execution.empty and execution["Max Position Weight"].max() > MAX_LONG_ONLY_WEIGHT:
        st.warning("Concentration warning: at least one historical rebalance exceeded the max ETF weight guideline.")
    if not execution.empty:
        st.metric("Total Transaction Cost Drag Estimate", f"{execution.loc[execution['Portfolio'].eq(recommended), 'Market Impact Cost'].sum():.2%}")


def render_stage_a2_stress_tests(bundle: dict) -> None:
    result = bundle["selected_result"]
    recommended = bundle["recommended_method"]
    stress = compare_stress_to_spy(result[4][recommended], bundle["benchmark"])
    format_map = {
        "Strategy Return": "{:.2%}",
        "SPY Return": "{:.2%}",
        "Strategy Max Drawdown": "{:.2%}",
        "SPY Max Drawdown": "{:.2%}",
        "Strategy Volatility": "{:.2%}",
    }
    st.dataframe(stress.style.format({key: value for key, value in format_map.items() if key in stress.columns}), use_container_width=True, hide_index=True)


def render_stage_a2_methodology(bundle: dict) -> None:
    st.markdown(
        """
### What Stage A2 Is
Stage A2 is a white-box ML multi-asset ETF rotation strategy. It ranks ETFs monthly, validates predictions walk-forward, and converts the ranking into portfolio weights.

### ML ETF Ranking
The model scores each ETF using momentum, relative strength, volatility, drawdown, trend, beta, correlation, regime, and lagged macro features. The default target is next-month ETF return.

### Walk-Forward Validation
Each prediction month trains only on data before that month. Forward returns use `shift(-1)` only to build training targets, not prediction-time features.

### Selection Rules
The selected model is chosen by walk-forward OOS Sharpe, with lower drawdown, positive top-minus-bottom spread, and lower turnover as tie-breakers. The portfolio method is chosen by OOS Sharpe, then drawdown, turnover, and cost drag.

### Portfolio Methods
HRP-style / Risk-Parity Fallback clusters assets when possible and falls back to risk-parity behavior when data is sparse. Ledoit-Wolf Mean-Variance uses shrinkage covariance. Fractional Kelly sizes by score relative to variance. Beta-Neutral ML builds long/short exposure from top and bottom ranks.

### Current Limitations
This is an ETF proxy universe, not full Russell 3000 / MSCI ACWI constituent coverage. FRED macro features may be included, but GDELT, Google Trends, and SEC EDGAR are not fully implemented in this Stage A2 dashboard. Feature importance is not full SHAP unless SHAP is actually used. Regime detection is HMM only when `hmmlearn` is available; otherwise it is a proxy. Black-Litterman and full TWAP/VWAP simulation are not implemented yet. ML may underperform simple strategies, and this dashboard reports that transparently.
"""
    )
    config_rows = [{"Setting": key, "Value": value} for key, value in bundle["config"].items()]
    st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)


def render_stage_a2_dashboard(stock_universe_file) -> None:
    st.title("Stage A2: White-Box ML Multi-Asset ETF Rotation Strategy")
    st.caption("Presentation-focused dashboard: the system compares models and portfolio methods automatically using walk-forward OOS validation.")

    config = STAGE_A2_PRESENTATION_CONFIG
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=int(config["Research Window Years"]))
    selected_symbols = tuple(dict.fromkeys([*SECTOR_ETFS, *GLOBAL_PROXIES]))

    with st.spinner("Loading ETF prices and FRED macro data..."):
        close = load_stage_a2_prices(selected_symbols, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        macro = load_fred_macro(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if close.empty or close.shape[1] < 8:
        st.error("Not enough ETF data loaded for Stage A2.")
        return

    with st.spinner("Running all Stage A2 models, selecting the best model, and comparing portfolio methods..."):
        bundle = run_stage_a2_presentation_research(close, macro)
    if not bundle["recommended_method"]:
        st.error("Stage A2 did not produce enough walk-forward results.")
        return

    tabs = st.tabs(
        [
            "Executive Overview",
            "Performance Diagnostics",
            "Portfolio Performance",
            "Current Portfolio",
            "Model Selection",
            "Portfolio Method Comparison",
            "White-Box Explanation",
            "Risk Dashboard",
            "Stress Tests",
            "Methodology and Limitations",
        ]
    )
    with tabs[0]:
        render_stage_a2_executive_overview(bundle)
    with tabs[1]:
        render_stage_a2_performance_diagnostics(bundle)
    with tabs[2]:
        render_stage_a2_portfolio_performance(bundle)
    with tabs[3]:
        render_stage_a2_current_portfolio(bundle)
    with tabs[4]:
        render_stage_a2_model_selection(bundle)
    with tabs[5]:
        render_stage_a2_portfolio_comparison(bundle)
    with tabs[6]:
        render_stage_a2_white_box_explanation(bundle)
    with tabs[7]:
        render_stage_a2_risk_dashboard(bundle)
    with tabs[8]:
        render_stage_a2_stress_tests(bundle)
    with tabs[9]:
        render_stage_a2_methodology(bundle)

    with st.expander("Advanced Research Details", expanded=False):
        selected = bundle["selected_result"]
        st.write("Fixed assumptions are stored in `STAGE_A2_PRESENTATION_CONFIG`; the final dashboard does not ask the audience to tune models or parameters.")
        st.dataframe(pd.DataFrame([selected[12]]), use_container_width=True, hide_index=True)
        st.subheader("Walk-Forward Debug Log")
        st.dataframe(selected[2].tail(60), use_container_width=True, hide_index=True)
        st.subheader("Execution Details")
        st.dataframe(selected[6].tail(80), use_container_width=True, hide_index=True)
