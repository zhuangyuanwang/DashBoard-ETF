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
STAGE_A2_PRESET = {
    "Preset": "Codex Recommended A2 White-Box Ranking",
    "Research Window Years": 10,
    "Max Stock Names": 50,
    "Model": "LightGBM",
    "Target": "Next-month cross-sectional rank percentile",
    "Walk-Forward Hyperparameter Tuning": False,
    "Transaction Cost Bps": 12,
    "Square-Root Impact Bps": 6,
    "Fractional Kelly": 0.25,
    "Target Volatility": 0.10,
    "Weight Smoothing": 0.35,
    "Regime Risk Overlay": True,
    "Max Drawdown Guard": 0.15,
    "Monthly Turnover Cap": 0.60,
    "Rebalance Threshold": 0.04,
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
        if len(log_rows) % 3 == 0:
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
            previous_weights[strategy_name] = weights

    return pd.DataFrame(returns_by_strategy).dropna(how="all"), pd.DataFrame(gross_returns_by_strategy).dropna(how="all"), pd.DataFrame(execution_rows)


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
    portfolio_returns, gross_returns, execution = build_stage_a2_portfolios(
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


def render_stage_a2_dashboard(stock_universe_file) -> None:
    st.title("Stage A2 Research Lab: ML-Powered Multi-Asset White-Box")
    st.write(
        "Intermediate capstone stage with white-box ML, walk-forward validation, HRP-style/Ledoit/Kelly portfolios, "
        "rule-based regime proxy visualization, factor exposure monitoring, stress tests, and execution cost tracking."
    )

    universe_table = pd.read_csv(stock_universe_file) if stock_universe_file.exists() else pd.DataFrame()
    local_stocks = universe_table["Ticker"].dropna().astype(str).str.upper().head(80).tolist() if "Ticker" in universe_table else DEFAULT_A2_STOCKS
    default_symbols = tuple(dict.fromkeys([*local_stocks, *SECTOR_ETFS, *GLOBAL_PROXIES]))

    years = int(STAGE_A2_PRESET["Research Window Years"])
    max_names = min(int(STAGE_A2_PRESET["Max Stock Names"]), len(default_symbols))
    model_name = str(STAGE_A2_PRESET["Model"])
    base_cost_bps = float(STAGE_A2_PRESET["Transaction Cost Bps"])
    impact_bps = float(STAGE_A2_PRESET["Square-Root Impact Bps"])
    kelly_fraction = float(STAGE_A2_PRESET["Fractional Kelly"])
    target_volatility = float(STAGE_A2_PRESET["Target Volatility"])
    smoothing = float(STAGE_A2_PRESET["Weight Smoothing"])
    enable_regime_overlay = bool(STAGE_A2_PRESET["Regime Risk Overlay"])
    max_drawdown_limit = float(STAGE_A2_PRESET["Max Drawdown Guard"])
    turnover_cap = float(STAGE_A2_PRESET["Monthly Turnover Cap"])
    rebalance_threshold = float(STAGE_A2_PRESET["Rebalance Threshold"])
    target_type = str(STAGE_A2_PRESET["Target"])
    enable_tuning = bool(STAGE_A2_PRESET["Walk-Forward Hyperparameter Tuning"])

    st.caption("Stage A2 is preset-driven: parameters are fixed in code so the dashboard presents one reproducible research pipeline.")
    preset_rows = [
        {"Setting": key, "Value": value if not isinstance(value, float) else f"{value:.2%}" if value < 1 else f"{value:g}"}
        for key, value in STAGE_A2_PRESET.items()
    ]
    preset_rows.insert(1, {"Setting": "Initial Capital", "Value": f"${STAGE_A2_INITIAL_CAPITAL:,.0f}"})
    st.dataframe(pd.DataFrame(preset_rows), use_container_width=True, hide_index=True)

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
        (
            x,
            predictions,
            walk_log,
            importance_history,
            portfolio_returns,
            gross_returns,
            execution,
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
        ) = run_stage_a2_research(
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
            turnover_cap,
            rebalance_threshold,
            target_type,
            enable_tuning,
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

    performance_tab, diagnostics_tab, model_tab, regime_tab, risk_tab, execution_tab, paper_tab = st.tabs(
        ["Performance", "ML Signal Diagnostics", "White-Box ML", "Regime States", "Risk & Stress", "Execution", "Working Paper #2"]
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

    with diagnostics_tab:
        st.subheader("Stage A2 ML Signal Diagnostics")
        gross_metrics = calculate_metrics(gross_returns, benchmark) if not gross_returns.empty else pd.DataFrame()
        if not gross_metrics.empty:
            gross_net = metrics[["Series", "Total Return", "Sharpe", "Max Drawdown"]].merge(
                gross_metrics[["Series", "Total Return", "Sharpe", "Max Drawdown"]],
                on="Series",
                suffixes=(" Net", " Gross"),
            )
            gross_net["Transaction Cost / Guard Drag"] = gross_net["Total Return Gross"] - gross_net["Total Return Net"]
            st.dataframe(
                gross_net.style.format(
                    {
                        "Total Return Net": "{:.2%}",
                        "Sharpe Net": "{:.2f}",
                        "Max Drawdown Net": "{:.2%}",
                        "Total Return Gross": "{:.2%}",
                        "Sharpe Gross": "{:.2f}",
                        "Max Drawdown Gross": "{:.2%}",
                        "Transaction Cost / Guard Drag": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not signal_summary.empty:
            st.subheader("Prediction Spread Quality: Ranked Assets")
            st.dataframe(
                signal_summary.style.format(
                    {
                        "Average Top 3 Return": "{:.2%}",
                        "Average Top 5 Return": "{:.2%}",
                        "Average Bottom 3 Return": "{:.2%}",
                        "Average Bottom 5 Return": "{:.2%}",
                        "Average Top-Bottom 5 Spread": "{:.2%}",
                        "Average Prediction IC": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not etf_signal_summary.empty:
            st.subheader("Prediction Spread Quality: ETF / Global Proxy Subset")
            st.dataframe(
                etf_signal_summary.style.format(
                    {
                        "Average Top 3 Return": "{:.2%}",
                        "Average Top 5 Return": "{:.2%}",
                        "Average Bottom 3 Return": "{:.2%}",
                        "Average Bottom 5 Return": "{:.2%}",
                        "Average Top-Bottom 5 Spread": "{:.2%}",
                        "Average Prediction IC": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not classification_summary.empty:
            st.subheader("Classification Target Diagnostics")
            st.dataframe(
                classification_summary.style.format(
                    {
                        "Classification Accuracy": "{:.2%}",
                        "Precision": "{:.2%}",
                        "Avg Forward Return of Predicted Outperformers": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not signal_monthly.empty:
            st.plotly_chart(px.line(signal_monthly, x="Date", y=["Top-Bottom 5 Spread", "Prediction IC"], title="Monthly Prediction Spread and IC"), use_container_width=True)
            st.dataframe(signal_monthly.tail(24).style.format({column: "{:.2%}" for column in signal_monthly.columns if "Return" in column or "Spread" in column}), use_container_width=True, hide_index=True)
        if not etf_signal_monthly.empty:
            st.plotly_chart(
                px.line(etf_signal_monthly, x="Date", y=["Top-Bottom 5 Spread", "Prediction IC"], title="ETF / Global Proxy Monthly Spread and IC"),
                use_container_width=True,
            )
        st.subheader("Monthly Turnover and Cost Drag")
        if execution.empty:
            st.info("No execution rows are available for turnover diagnostics.")
        else:
            monthly_execution = execution.groupby(["Date", "Portfolio"], as_index=False).agg(
                Turnover=("Turnover", "mean"),
                Requested_Turnover=("Requested Turnover", "mean"),
                Estimated_Cost=("Market Impact Cost", "sum"),
                Rebalance_Skipped=("Rebalance Skipped", lambda values: int((values == "Yes").sum())),
            )
            st.dataframe(
                monthly_execution.tail(48).style.format(
                    {
                        "Turnover": "{:.2f}x",
                        "Requested_Turnover": "{:.2f}x",
                        "Estimated_Cost": "{:.2%}",
                        "Rebalance_Skipped": "{:.0f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Prediction Source and Fallback")
        st.write(f"Prediction source: **{diagnostics.get('Prediction Source', 'ML')}**")
        if not walk_log.empty and "Model Engine" in walk_log:
            engine_summary = walk_log["Model Engine"].value_counts().rename_axis("Model Engine").reset_index(name="Months")
            st.dataframe(engine_summary, use_container_width=True, hide_index=True)
        fallback_count = int((walk_log["Prediction Source"] == "Fallback").sum()) if not walk_log.empty and "Prediction Source" in walk_log else 0
        st.metric("Fallback Months", fallback_count)
        if not walk_log.empty and "Train IC" in walk_log and not signal_summary.empty:
            train_ic = walk_log["Train IC"].mean()
            oos_ic = signal_summary["Average Prediction IC"].iloc[0]
            if pd.notna(train_ic) and pd.notna(oos_ic) and train_ic - oos_ic > 0.15:
                st.warning(f"Overfitting warning: average train IC ({train_ic:.3f}) is much higher than OOS IC ({oos_ic:.3f}).")
        st.subheader("Selected Hyperparameters")
        st.write(diagnostics.get("Selected Hyperparameters", "{}"))
        if not tuning_results.empty:
            st.dataframe(
                tuning_results.head(20).style.format(
                    {
                        "Prediction IC": "{:.3f}",
                        "Top-Bottom Spread": "{:.2%}",
                        "Top Basket Sharpe": "{:.2f}",
                        "Top Basket Net Return Estimate": "{:.2%}",
                        "Selection Score": "{:.4f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        if not importance_history.empty:
            stability = importance_history.groupby("Feature")["Importance"].agg(["mean", "std"]).reset_index()
            stability["Stability Ratio"] = stability["mean"] / stability["std"].replace(0, np.nan)
            st.subheader("Feature Importance Stability")
            st.dataframe(stability.sort_values("mean", ascending=False).head(20).style.format({"mean": "{:.3f}", "std": "{:.3f}", "Stability Ratio": "{:.2f}"}), use_container_width=True, hide_index=True)

    with model_tab:
        st.subheader("SHAP / Feature Importance Tracking")
        if importance_history.empty:
            st.info("Feature importance is unavailable for the selected model.")
        else:
            if "Importance Type" in importance_history:
                st.caption("When `shap` is installed and compatible with the selected estimator, importance uses mean absolute SHAP values. Otherwise it falls back to native model importance or absolute linear coefficients.")
                st.dataframe(importance_history["Importance Type"].value_counts().rename_axis("Importance Type").reset_index(name="Rows"), use_container_width=True, hide_index=True)
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
            "Model Engine",
        ]
        st.dataframe(walk_log[[column for column in debug_columns if column in walk_log.columns]].tail(36), use_container_width=True, hide_index=True)
        fallback_months = walk_log[walk_log.get("Prediction Source", pd.Series(dtype=str)).eq("Fallback")] if not walk_log.empty and "Prediction Source" in walk_log else pd.DataFrame()
        st.subheader("Fallback Months")
        if fallback_months.empty:
            st.success("No fallback months. The selected ML path produced the walk-forward predictions.")
        else:
            st.dataframe(fallback_months, use_container_width=True, hide_index=True)

    with regime_tab:
        if regimes.empty:
            st.warning("Not enough data for regime state detection.")
        else:
            regime_method = regimes["Regime Method"].iloc[-1] if "Regime Method" in regimes else "Unknown"
            st.subheader("HMM / Regime State Visualization")
            if regime_method == "HMM":
                st.write("Regime states use a 3-state Gaussian Hidden Markov Model on monthly SPY return, volatility, and drawdown, then label states as Bear, Recovery, or Bull by average return.")
            else:
                st.write("A true HMM engine is not available in this runtime, so Stage A2 is using the clearly labeled Gaussian-mixture regime proxy.")
            st.metric("Regime Engine", regime_method)
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
