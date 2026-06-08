import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ETF Dashboard", page_icon="📈", layout="wide")

ETF_UNIVERSE = {
    "Broad Market": ["SPY", "QQQ", "IWM", "DIA"],
    "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE", "XLC"],
    "Asset Classes": ["TLT", "SHY", "AGG", "TIP", "HYG", "LQD", "GLD", "SLV", "DBC", "USO", "VNQ"],
    "Geography": ["EFA", "EEM", "FXI", "VGK", "EWJ", "INDA"],
}

PERIOD_LABELS = {
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y": "1 Year",
}

PERFORMANCE_WINDOWS = {
    "1D": 1,
    "5D": 5,
    "1M": 22,
    "3M": 66,
}

STRATEGY_TYPES = [
    "Momentum Rotation",
    "Moving Average Trend Following",
    "Dual Momentum",
    "Low Volatility Rotation",
    "Defensive Rotation Strategy",
    "Risk-On / Risk-Off Regime Strategy",
    "Breakout Strategy",
    "Mean Reversion Strategy",
    "Volatility Target Strategy",
    "Equal Weight Multi-ETF Strategy",
]

SUMMARY_CHOICES = ["Combined Strategy"] + STRATEGY_TYPES

DEFENSIVE_ETFS = ["TLT", "IEF", "GLD", "SHY", "XLV", "XLU"]
RISK_ON_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLY", "XLI", "XLC", "EFA", "EEM"]
PORTFOLIO_INCEPTION_DATE = pd.Timestamp("2026-06-04")
BUY_TRANSACTION_COST = 0.0005
SELL_TRANSACTION_COST = 0.0005

ALPHA_STRATEGIES = [
    {"name": "Overnight Mean Reversion", "category": "Mean Reversion"},
    {"name": "Short-Term Momentum", "category": "Momentum"},
    {"name": "Intraday Reversal", "category": "Mean Reversion"},
    {"name": "Typical Price Reversion", "category": "Price-Volume"},
    {"name": "Volume Spike Reversal", "category": "Price-Volume"},
    {"name": "High-Low Range Breakout", "category": "Breakout"},
    {"name": "Low Volatility Rotation Alpha", "category": "Risk Control"},
    {"name": "Rank-Based Momentum", "category": "Cross-Sectional"},
    {"name": "Price-Volume Confirmation", "category": "Price-Volume"},
    {"name": "Correlation Diversification Alpha", "category": "Diversification"},
]

BACKTEST_PERIODS = {
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
}

REBALANCE_FREQUENCIES = ["Daily", "Weekly", "Monthly"]
SIGNAL_AGGREGATION_WINDOWS = [5, 10, 20, 30]


def get_all_etf_symbols():
    symbols = []
    for tickers in ETF_UNIVERSE.values():
        symbols.extend(tickers)
    return list(dict.fromkeys(symbols))


def extract_close_prices(df, symbols):
    close = df["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    close = close.dropna(how="all")
    return close


@st.cache_data(ttl=3600)
def load_data(symbols, period):
    symbols = tuple(sorted(symbols))
    df = yf.download(symbols, period=period, interval="1d", auto_adjust=True, threads=True, progress=False)
    if df.empty:
        return pd.DataFrame()

    return extract_close_prices(df, symbols)


def compute_returns(close):
    return close.pct_change().dropna()


def calculate_indicators(close):
    return {
        "returns_1m": close.pct_change(21),
        "returns_3m": close.pct_change(63),
        "returns_6m": close.pct_change(126),
        "returns_12m": close.pct_change(252),
        "ma50": close.rolling(50).mean(),
        "ma200": close.rolling(200).mean(),
        "rolling_high_55": close.rolling(55).max().shift(1),
        "rolling_low_20": close.rolling(20).min().shift(1),
        "rsi14": calculate_rsi(close),
        "vol20": close.pct_change().rolling(20).std() * np.sqrt(252),
        "vol60": close.pct_change().rolling(60).std() * np.sqrt(252),
    }


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_momentum_scores(indicators):
    scores = (
        indicators["returns_1m"] * 0.15
        + indicators["returns_3m"] * 0.35
        + indicators["returns_6m"] * 0.30
        + indicators["returns_12m"] * 0.20
    )
    return scores.dropna(how="all")


def calculate_volatility(close, window=20):
    return close.pct_change().rolling(window).std() * np.sqrt(252)


def available_symbols(close, symbols):
    return [symbol for symbol in symbols if symbol in close.columns]


def latest_valid_date(close, signal_date):
    valid_dates = close.index[close.index <= signal_date]
    if len(valid_dates) == 0:
        return None
    return valid_dates[-1]


def build_line_chart(df, y, title, y_title):
    fig = px.line(df, x=df.index, y=y, title=title)
    fig.update_traces(mode="lines+markers", marker=dict(size=4, opacity=0.8))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=y_title,
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=30),
        legend_title_text="Ticker",
    )
    return fig


@st.cache_data(ttl=3600)
def load_backtest_data(symbols, start_date, end_date):
    symbols = tuple(sorted(set(symbols)))
    df = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()
    return extract_close_prices(df, symbols)


def extract_price_field(df, field, symbols):
    if field not in df:
        return pd.DataFrame()
    values = df[field].copy()
    if isinstance(values, pd.Series):
        values = values.to_frame(name=symbols[0])
    values = values.dropna(how="all")
    return values


@st.cache_data(ttl=3600)
def load_ohlcv_data(symbols, start_date, end_date):
    symbols = tuple(sorted(set(symbols)))
    df = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )
    if df.empty:
        return {}
    return {
        "open": extract_price_field(df, "Open", symbols),
        "high": extract_price_field(df, "High", symbols),
        "low": extract_price_field(df, "Low", symbols),
        "close": extract_price_field(df, "Close", symbols),
        "volume": extract_price_field(df, "Volume", symbols),
    }


def calculate_performance_metrics(strategy_equity, strategy_returns, number_rebalances):
    total_return = strategy_equity.iloc[-1] - 1
    years = (strategy_equity.index[-1] - strategy_equity.index[0]).days / 365.25
    cagr = strategy_equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = strategy_returns.std() * np.sqrt(252)
    sharpe = (strategy_returns.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
    drawdown = strategy_equity / strategy_equity.cummax() - 1
    max_dd = drawdown.min()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "number_rebalances": number_rebalances,
    }


def calculate_backtest_metrics(strategy_equity, strategy_returns, number_rebalances):
    return calculate_performance_metrics(strategy_equity, strategy_returns, number_rebalances)


def build_backtest_result(
    close,
    strategy_returns,
    strategy_name,
    benchmark_symbol="SPY",
    transaction_cost=0.0,
    rebalances=None,
    turnover_list=None,
    monthly_holdings=None,
    trade_log=None,
    latest_scores=None,
    summary_points=None,
):
    rebalances = rebalances or []
    turnover_list = turnover_list or []
    monthly_holdings = monthly_holdings or []
    trade_log = trade_log or []

    strategy_returns = strategy_returns.dropna()
    if strategy_returns.empty or benchmark_symbol not in close.columns:
        return None

    daily_returns = close.pct_change().dropna()
    benchmark_returns = daily_returns[benchmark_symbol].reindex(strategy_returns.index).dropna()
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    if common_index.empty:
        return None

    strategy_returns = strategy_returns.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    strategy_equity = (1 + strategy_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()

    monthly_strategy = strategy_equity.resample("ME").last().pct_change().dropna()
    monthly_benchmark = benchmark_equity.resample("ME").last().pct_change().dropna()
    monthly_returns_df = pd.DataFrame(
        {strategy_name: monthly_strategy, f"{benchmark_symbol} Buy & Hold": monthly_benchmark}
    ).dropna() * 100

    yearly_strategy = strategy_equity.resample("YE").last().pct_change().dropna()
    yearly_benchmark = benchmark_equity.resample("YE").last().pct_change().dropna()
    yearly_returns_df = pd.DataFrame(
        {strategy_name: yearly_strategy, f"{benchmark_symbol} Buy & Hold": yearly_benchmark}
    ).dropna() * 100

    metrics = calculate_backtest_metrics(strategy_equity, strategy_returns, len(rebalances))
    metrics["avg_turnover"] = np.mean(turnover_list) if turnover_list else 0.0

    current_holdings = monthly_holdings[-1]["Holdings"] if monthly_holdings else []

    return {
        "strategy_name": strategy_name,
        "benchmark_symbol": benchmark_symbol,
        "strategy_returns": strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_equity": strategy_equity,
        "benchmark_equity": benchmark_equity,
        "monthly_returns": monthly_returns_df,
        "yearly_returns": yearly_returns_df,
        "metrics": metrics,
        "rebalances": rebalances,
        "transaction_cost": transaction_cost,
        "monthly_holdings": monthly_holdings,
        "trade_log": trade_log,
        "current_holdings": current_holdings,
        "latest_scores": latest_scores,
        "summary_points": summary_points or [],
    }


def get_next_trade_date(daily_returns, signal_date):
    next_days = daily_returns.index[daily_returns.index > signal_date]
    if len(next_days) == 0:
        return signal_date
    return next_days[0]


def format_holdings(weights):
    holdings = weights[weights > 0].sort_values(ascending=False)
    if holdings.empty:
        return "Cash"
    return ", ".join([f"{ticker} ({weight * 100:.2f}%)" for ticker, weight in holdings.items()])


def build_trade_log_entry(signal_date, trade_date, previous_weights, new_weights, next_rebalance_date):
    previous_holdings = set(previous_weights[previous_weights > 0].index)
    current_holdings = set(new_weights[new_weights > 0].index)
    buy_etfs = sorted(list(current_holdings - previous_holdings))
    sell_etfs = sorted(list(previous_holdings - current_holdings))

    return {
        "Signal Date": signal_date.strftime("%Y-%m-%d"),
        "Trade Date": trade_date.strftime("%Y-%m-%d"),
        "New Holdings": format_holdings(new_weights),
        "Previous Holdings": format_holdings(previous_weights),
        "Buy ETFs": ", ".join(buy_etfs) if buy_etfs else "None",
        "Sell ETFs": ", ".join(sell_etfs) if sell_etfs else "None",
        "Next Rebalance Date": next_rebalance_date.strftime("%Y-%m-%d") if next_rebalance_date else "N/A",
    }


def make_signal(strategy, signal_date, selected_asset, final_signal, strength, reason, vote):
    rule_match_score = max(0, min(100, round(strength, 1)))
    return {
        "Strategy": strategy,
        "Latest Signal Date": signal_date.strftime("%Y-%m-%d") if hasattr(signal_date, "strftime") else "N/A",
        "Selected Asset": selected_asset,
        "Signal": final_signal,
        "Rule Match Score": rule_match_score,
        "Signal Strength": rule_match_score,
        "Reason": reason,
        "Vote": vote,
    }


def generate_strategy_signal(strategy_type, close, ma_symbol="SPY"):
    close = close.dropna(how="all")
    if close.empty:
        return make_signal(strategy_type, "N/A", "N/A", "CASH", 0, "No price data is available.", -1)

    indicators = calculate_indicators(close)
    signal_date = close.index[-1]

    if strategy_type == "Momentum Rotation":
        momentum_scores = calculate_momentum_scores(indicators)
        if momentum_scores.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough history to calculate weighted momentum.", -1)
        latest_scores = momentum_scores.iloc[-1].dropna().sort_values(ascending=False)
        if latest_scores.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "No ETF has a valid momentum score.", -1)
        selected = latest_scores.index[0]
        second_score = latest_scores.iloc[1] if len(latest_scores) > 1 else 0
        top_score = latest_scores.iloc[0]
        rank_advantage = top_score - second_score
        strength = 50 + min(50, max(0, rank_advantage * 500))
        final_signal = "BUY" if top_score > 0 else "HOLD"
        vote = 1 if final_signal == "BUY" else 0
        reason = f"Selected {selected} because it has the highest weighted momentum score among the ETF universe."
        return make_signal(strategy_type, signal_date, selected, final_signal, strength, reason, vote)

    if strategy_type == "Moving Average Trend Following":
        symbol = ma_symbol if ma_symbol in close.columns else close.columns[0]
        latest_close = close[symbol].dropna()
        if latest_close.empty:
            return make_signal(strategy_type, signal_date, symbol, "CASH", 0, f"No price data is available for {symbol}.", -1)
        symbol_date = latest_close.index[-1]
        price = latest_close.iloc[-1]
        ma50 = indicators["ma50"][symbol].dropna()
        ma200 = indicators["ma200"][symbol].dropna()
        if ma50.empty or ma200.empty:
            return make_signal(strategy_type, symbol_date, symbol, "CASH", 0, f"Not enough history to calculate MA50 and MA200 for {symbol}.", -1)
        latest_ma50 = ma50.iloc[-1]
        latest_ma200 = ma200.iloc[-1]
        conditions = [price > latest_ma50, price > latest_ma200, latest_ma50 > latest_ma200]
        passed = sum(conditions)
        strength = passed / 3 * 100
        if passed >= 2:
            final_signal = "BUY" if passed == 3 else "HOLD"
            vote = 1 if final_signal == "BUY" else 0
        else:
            final_signal = "CASH"
            vote = -1
        reason = (
            f"{symbol} satisfies {passed}/3 bullish trend conditions: "
            "Close > MA50, Close > MA200, and MA50 > MA200."
        )
        return make_signal(strategy_type, symbol_date, symbol, final_signal, strength, reason, vote)

    if strategy_type == "Dual Momentum":
        returns_12m = indicators["returns_12m"].dropna(how="all")
        if returns_12m.empty:
            return make_signal(strategy_type, signal_date, "Cash", "RISK-OFF", 0, "Not enough history to calculate 12-month momentum.", -1)
        latest_returns = returns_12m.iloc[-1].dropna().sort_values(ascending=False)
        if latest_returns.empty:
            return make_signal(strategy_type, signal_date, "Cash", "RISK-OFF", 0, "No ETF has a valid 12-month momentum value.", -1)
        selected = latest_returns.index[0]
        top_return = latest_returns.iloc[0]
        second_return = latest_returns.iloc[1] if len(latest_returns) > 1 else 0
        relative_score = 50 + min(25, max(0, (top_return - second_return) * 250))
        absolute_score = min(25, max(0, top_return * 100))
        strength = relative_score + absolute_score
        if top_return > 0:
            reason = f"Selected {selected} because it ranked highest among ETFs and had positive 12-month momentum."
            return make_signal(strategy_type, signal_date, selected, "BUY", strength, reason, 1)
        reason = f"Moved risk-off because the top-ranked ETF, {selected}, did not have positive 12-month momentum."
        defensive_asset = "SHY" if "SHY" in close.columns else "Cash"
        return make_signal(strategy_type, signal_date, defensive_asset, "RISK-OFF", max(0, 100 - strength), reason, -1)

    if strategy_type == "Low Volatility Rotation":
        vol60 = indicators["vol60"].dropna(how="all")
        if vol60.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough history to calculate 60-day volatility.", -1)
        latest_vol = vol60.iloc[-1].dropna().sort_values()
        if latest_vol.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "No ETF has a valid recent volatility value.", -1)
        returns_3m = indicators["returns_3m"].reindex(latest_vol.index, axis=1).iloc[-1]
        eligible = returns_3m[returns_3m > 0].dropna().index.tolist()
        ranked_vol = latest_vol.loc[eligible].sort_values() if eligible else latest_vol
        selected = ranked_vol.index[0]
        rank_position = latest_vol.index.get_loc(selected) + 1
        rank_score = (len(latest_vol) - rank_position + 1) / len(latest_vol) * 70
        momentum_score = 30 if selected in eligible else 0
        strength = rank_score + momentum_score
        final_signal = "BUY" if selected in eligible else "HOLD"
        vote = 1 if final_signal == "BUY" else 0
        reason = f"Selected {selected} because it had one of the lowest recent volatility levels among eligible ETFs."
        return make_signal(strategy_type, signal_date, selected, final_signal, strength, reason, vote)

    if strategy_type == "Defensive Rotation Strategy":
        if "SPY" not in close.columns:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "SPY data is required to classify the market regime.", -1)
        spy_close = close["SPY"].dropna()
        ma50 = indicators["ma50"]["SPY"].dropna()
        ma200 = indicators["ma200"]["SPY"].dropna()
        if spy_close.empty or ma50.empty or ma200.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough SPY history to calculate defensive regime conditions.", -1)
        spy_price = spy_close.iloc[-1]
        spy_ma50 = ma50.iloc[-1]
        spy_ma200 = ma200.iloc[-1]
        risk_off_conditions = [spy_price < spy_ma50, spy_price < spy_ma200, spy_ma50 < spy_ma200]
        risk_off_count = sum(risk_off_conditions)
        strength = risk_off_count / 3 * 100
        if risk_off_count >= 2:
            defensive_symbols = available_symbols(close, DEFENSIVE_ETFS)
            vol60 = indicators["vol60"][defensive_symbols].iloc[-1].dropna() if defensive_symbols else pd.Series(dtype=float)
            selected = vol60.sort_values().index[0] if not vol60.empty else "SHY"
            reason = f"Selected {selected} because SPY is below key moving averages, indicating a defensive market regime."
            return make_signal(strategy_type, signal_date, selected, "RISK-OFF", strength, reason, -1)
        risk_symbols = available_symbols(close, RISK_ON_ETFS)
        scores = calculate_momentum_scores(indicators)[risk_symbols].iloc[-1].dropna() if risk_symbols else pd.Series(dtype=float)
        selected = scores.sort_values(ascending=False).index[0] if not scores.empty else "SPY"
        final_signal = "BUY" if risk_off_count == 0 else "HOLD"
        vote = 1 if final_signal == "BUY" else 0
        reason = f"Selected {selected} because SPY trend is healthy enough to allow risk-on exposure."
        return make_signal(strategy_type, signal_date, selected, final_signal, max(0, 100 - strength), reason, vote)

    if strategy_type == "Risk-On / Risk-Off Regime Strategy":
        if "SPY" not in close.columns:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "SPY data is required to classify the regime.", -1)
        spy_close = close["SPY"].dropna()
        ma50 = indicators["ma50"]["SPY"].dropna()
        ma200 = indicators["ma200"]["SPY"].dropna()
        vol20 = indicators["vol20"]["SPY"].dropna()
        if spy_close.empty or ma50.empty or ma200.empty or vol20.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough SPY history for trend and volatility regime checks.", -1)
        price = spy_close.iloc[-1]
        above_ma50 = price > ma50.iloc[-1]
        above_ma200 = price > ma200.iloc[-1]
        calm_vol = vol20.iloc[-1] < 0.25
        healthy_count = sum([above_ma50, above_ma200, calm_vol])
        strength = healthy_count / 3 * 100
        if above_ma50 and above_ma200:
            reason = "Risk-on signal because SPY is trading above both MA50 and MA200."
            return make_signal(strategy_type, signal_date, "SPY", "BUY", strength, reason, 1)
        if (not above_ma50) and (not above_ma200):
            reason = "Risk-off signal because SPY is trading below both MA50 and MA200."
            return make_signal(strategy_type, signal_date, "SHY", "RISK-OFF", 100 - strength, reason, -1)
        reason = "Mixed regime because SPY is above one key moving average but below the other."
        return make_signal(strategy_type, signal_date, "SPY", "HOLD", 50, reason, 0)

    if strategy_type == "Breakout Strategy":
        rolling_high = indicators["rolling_high_55"]
        rolling_low = indicators["rolling_low_20"]
        latest_close = close.iloc[-1].dropna()
        latest_high = rolling_high.iloc[-1].dropna()
        latest_low = rolling_low.iloc[-1].dropna()
        breakout_symbols = latest_close.index.intersection(latest_high.index)
        if len(breakout_symbols) == 0:
            return make_signal(strategy_type, signal_date, "Cash", "HOLD", 0, "Not enough history to calculate breakout levels.", 0)
        breakout_ratio = (latest_close[breakout_symbols] / latest_high[breakout_symbols] - 1).dropna()
        selected = breakout_ratio.sort_values(ascending=False).index[0]
        best_ratio = breakout_ratio.loc[selected]
        if best_ratio > 0:
            strength = min(100, 50 + best_ratio * 1000)
            reason = f"BUY signal because {selected} closed above its 55-day high."
            return make_signal(strategy_type, signal_date, selected, "BUY", strength, reason, 1)
        low_symbols = latest_close.index.intersection(latest_low.index)
        breakdown_ratio = (latest_close[low_symbols] / latest_low[low_symbols] - 1).dropna() if len(low_symbols) else pd.Series(dtype=float)
        if not breakdown_ratio.empty and breakdown_ratio.min() < 0:
            selected = breakdown_ratio.sort_values().index[0]
            strength = min(100, abs(breakdown_ratio.min()) * 1000)
            reason = f"CASH signal because {selected} broke below its recent 20-day low."
            return make_signal(strategy_type, signal_date, "Cash", "CASH", strength, reason, -1)
        reason = "No ETF closed above its 55-day high or below its 20-day low."
        return make_signal(strategy_type, signal_date, selected, "HOLD", max(0, 50 + best_ratio * 1000), reason, 0)

    if strategy_type == "Mean Reversion Strategy":
        rsi = indicators["rsi14"].iloc[-1].dropna()
        ma200 = indicators["ma200"].iloc[-1].dropna()
        latest_close = close.iloc[-1].dropna()
        valid = rsi.index.intersection(ma200.index).intersection(latest_close.index)
        if len(valid) == 0:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough history to calculate RSI and MA200.", -1)
        healthy = latest_close[valid] > ma200[valid]
        oversold = rsi[valid] < 30
        candidates = rsi[valid][healthy & oversold].sort_values()
        if not candidates.empty:
            selected = candidates.index[0]
            strength = min(100, (30 - candidates.iloc[0]) / 30 * 100)
            reason = f"BUY signal because {selected} is oversold with RSI below 30 while still above MA200."
            return make_signal(strategy_type, signal_date, selected, "BUY", strength, reason, 1)
        overbought = rsi[valid] > 70
        weak = ~healthy
        if overbought.any() or weak.mean() > 0.5:
            selected = rsi[valid].sort_values(ascending=False).index[0]
            strength = min(100, max(0, rsi[valid].max() - 50) * 2)
            reason = f"CASH signal because {selected} is overbought or the ETF universe has weak long-term trend conditions."
            return make_signal(strategy_type, signal_date, "Cash", "CASH", strength, reason, -1)
        selected = rsi[valid].sub(50).abs().sort_values().index[0]
        reason = "HOLD signal because RSI is neutral and no oversold long-term uptrend setup is present."
        return make_signal(strategy_type, signal_date, selected, "HOLD", 50, reason, 0)

    if strategy_type == "Volatility Target Strategy":
        if "SPY" not in close.columns:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "SPY data is required for volatility targeting.", -1)
        vol20 = indicators["vol20"]["SPY"].dropna()
        ma200 = indicators["ma200"]["SPY"].dropna()
        spy_close = close["SPY"].dropna()
        if vol20.empty or ma200.empty or spy_close.empty:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough SPY history to calculate volatility target conditions.", -1)
        target_vol = 0.15
        high_vol = 0.25
        current_vol = vol20.iloc[-1]
        trend_positive = spy_close.iloc[-1] > ma200.iloc[-1]
        distance = abs(current_vol - target_vol) / target_vol
        strength = min(100, distance * 100)
        if current_vol <= target_vol and trend_positive:
            reason = "BUY signal because 20-day annualized volatility is below target and SPY trend is positive."
            return make_signal(strategy_type, signal_date, "SPY", "BUY", max(50, strength), reason, 1)
        if current_vol >= high_vol or not trend_positive:
            reason = "RISK-OFF signal because 20-day annualized volatility is above the target threshold or trend is weak."
            return make_signal(strategy_type, signal_date, "SHY", "RISK-OFF", max(50, strength), reason, -1)
        reason = "HOLD signal because volatility is near the target range."
        return make_signal(strategy_type, signal_date, "SPY", "HOLD", max(25, 100 - strength), reason, 0)

    if strategy_type == "Equal Weight Multi-ETF Strategy":
        ma50 = indicators["ma50"].iloc[-1].dropna()
        ma200 = indicators["ma200"].iloc[-1].dropna()
        latest_close = close.iloc[-1].dropna()
        valid = latest_close.index.intersection(ma50.index).intersection(ma200.index)
        if len(valid) == 0:
            return make_signal(strategy_type, signal_date, "Cash", "CASH", 0, "Not enough history to calculate MA50 and MA200 filters.", -1)
        passing = latest_close[valid][(latest_close[valid] > ma50[valid]) & (latest_close[valid] > ma200[valid])].index.tolist()
        strength = len(passing) / len(valid) * 100
        if len(passing) >= max(3, len(valid) * 0.4):
            reason = f"BUY signal because {len(passing)} out of {len(valid)} ETFs are above both MA50 and MA200."
            return make_signal(strategy_type, signal_date, ", ".join(passing[:6]), "BUY", strength, reason, 1)
        if passing:
            reason = f"HOLD signal because only {len(passing)} out of {len(valid)} ETFs pass the trend filter."
            return make_signal(strategy_type, signal_date, ", ".join(passing[:6]), "HOLD", strength, reason, 0)
        reason = f"RISK-OFF signal because 0 out of {len(valid)} ETFs are above both MA50 and MA200."
        return make_signal(strategy_type, signal_date, "SHY", "RISK-OFF", 100, reason, -1)

    return make_signal(strategy_type, signal_date, "N/A", "HOLD", 0, "Unknown strategy type.", 0)


def generate_all_strategy_signals(close, ma_symbol="SPY"):
    return {strategy_type: generate_strategy_signal(strategy_type, close, ma_symbol=ma_symbol) for strategy_type in STRATEGY_TYPES}


def combine_strategy_signals(signals):
    valid_signals = [signal for signal in signals.values() if signal is not None]
    if not valid_signals:
        return make_signal("Combined Strategy", "N/A", "Cash", "HOLD", 0, "No strategy signals are available.", 0)

    average_vote = np.mean([signal["Vote"] for signal in valid_signals])
    if average_vote >= 0.3:
        final_signal = "BUY"
    elif average_vote <= -0.3:
        final_signal = "RISK-OFF"
    else:
        final_signal = "HOLD"

    buy_assets = [signal["Selected Asset"] for signal in valid_signals if signal["Vote"] == 1]
    selected_asset = buy_assets[0] if buy_assets else "Cash"
    strength = abs(average_vote) * 100
    reason = f"Combined vote score is {average_vote:.2f}. BUY/HOLD/RISK-OFF votes are averaged across the four strategies."
    latest_dates = [signal["Latest Signal Date"] for signal in valid_signals if signal["Latest Signal Date"] != "N/A"]
    latest_date = max(latest_dates) if latest_dates else "N/A"
    return make_signal("Combined Strategy", latest_date, selected_asset, final_signal, strength, reason, average_vote)


def run_momentum_backtest(close, top_n=3, lookback_months=3, transaction_cost=0.0005, benchmark_symbol="SPY"):
    monthly_prices = close.resample("ME").last()
    trailing_returns = monthly_prices.pct_change(lookback_months)
    daily_returns = close.pct_change().dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    previous_weights = pd.Series(0.0, index=close.columns)
    turnover_list = []
    rebalances = []
    latest_top3 = []
    latest_top3_returns = []
    monthly_holdings = []
    monthly_rankings = []
    trade_log = []

    month_ends = monthly_prices.index
    for idx in range(lookback_months, len(month_ends) - 1):
        rebalance_date = month_ends[idx]
        return_scores = trailing_returns.loc[rebalance_date].dropna()
        if len(return_scores) < top_n:
            continue
        selected = return_scores.nlargest(top_n).index.tolist()
        latest_top3 = selected
        latest_top3_returns = (return_scores.nlargest(top_n) * 100).values.tolist()
        
        ranking_df = (return_scores.sort_values(ascending=False) * 100).to_frame(name="3M Return (%)")
        ranking_df.index.name = "Ticker"
        ranking_df = ranking_df.round(2)
        monthly_rankings.append({"date": rebalance_date, "ranking": ranking_df})
        
        new_weights = pd.Series(0.0, index=close.columns)
        new_weights[selected] = 1.0 / top_n
        turnover = (new_weights - previous_weights).abs().sum() / 2
        turnover_list.append(turnover)
        rebalances.append(rebalance_date)
        
        holdings_dict = {"Date": rebalance_date, "Holdings": selected, "Holdings_str": format_holdings(new_weights)}
        monthly_holdings.append(holdings_dict)
        
        trade_date = get_next_trade_date(daily_returns, rebalance_date)
        next_rebalance_date = month_ends[idx + 1] if idx + 1 < len(month_ends) else None
        trade_log.append(build_trade_log_entry(rebalance_date, trade_date, previous_weights, new_weights, next_rebalance_date))

        next_rebalance_date = month_ends[idx + 1]
        period_mask = (daily_returns.index > rebalance_date) & (daily_returns.index <= next_rebalance_date)
        if not period_mask.any():
            previous_weights = new_weights
            continue

        tranche_returns = daily_returns.loc[period_mask, selected].dot(new_weights[selected])
        if len(tranche_returns) > 0:
            first_day = tranche_returns.index[0]
            tranche_returns.iloc[0] -= turnover * transaction_cost
        strategy_returns.loc[period_mask] = tranche_returns
        previous_weights = new_weights

    result = build_backtest_result(
        close,
        strategy_returns,
        "Momentum Rotation",
        benchmark_symbol=benchmark_symbol,
        transaction_cost=transaction_cost,
        rebalances=rebalances,
        turnover_list=turnover_list,
        monthly_holdings=monthly_holdings,
        trade_log=trade_log,
        latest_scores=monthly_rankings[-1]["ranking"] if monthly_rankings else None,
        summary_points=[
            "Ranks ETFs by trailing 3-month return at each month-end.",
            "Holds the top 3 ETFs equally during the following month.",
            "Uses next-month returns only, which avoids look-ahead bias.",
        ],
    )
    if result:
        result["latest_top3"] = latest_top3
        result["latest_top3_returns"] = latest_top3_returns
        result["monthly_rankings"] = monthly_rankings
    return result


def run_moving_average_trend_backtest(close, symbol="SPY", ma_window=200, benchmark_symbol=None, transaction_cost=0.0005):
    benchmark_symbol = benchmark_symbol or symbol
    if symbol not in close.columns:
        return None

    symbol_close = close[[symbol]].dropna()
    daily_returns = symbol_close.pct_change().dropna()
    ma200 = symbol_close[symbol].rolling(ma_window).mean()
    signal = (symbol_close[symbol] > ma200).astype(float)
    strategy_exposure = signal.shift(1).reindex(daily_returns.index).fillna(0.0)
    strategy_returns = daily_returns[symbol] * strategy_exposure
    exposure_turnover = strategy_exposure.diff().abs().fillna(strategy_exposure.abs())
    strategy_returns = strategy_returns - (exposure_turnover * transaction_cost)

    monthly_prices = symbol_close.resample("ME").last()
    rebalances = monthly_prices.index[monthly_prices.index >= strategy_returns.index.min()].tolist()
    monthly_holdings = []
    trade_log = []
    previous_weights = pd.Series(0.0, index=close.columns)

    for idx, rebalance_date in enumerate(rebalances):
        if rebalance_date not in signal.index:
            signal_date = signal.index[signal.index <= rebalance_date]
            if len(signal_date) == 0:
                continue
            rebalance_date = signal_date[-1]

        new_weights = pd.Series(0.0, index=close.columns)
        if signal.loc[rebalance_date] == 1:
            new_weights[symbol] = 1.0

        trade_date = get_next_trade_date(daily_returns, rebalance_date)
        next_rebalance_date = rebalances[idx + 1] if idx + 1 < len(rebalances) else None
        monthly_holdings.append(
            {
                "Date": rebalance_date,
                "Holdings": [symbol] if new_weights[symbol] > 0 else ["Cash"],
                "Holdings_str": format_holdings(new_weights),
            }
        )
        trade_log.append(build_trade_log_entry(rebalance_date, trade_date, previous_weights, new_weights, next_rebalance_date))
        previous_weights = new_weights

    latest_price = symbol_close[symbol].dropna().iloc[-1]
    latest_ma = ma200.dropna().iloc[-1] if not ma200.dropna().empty else np.nan
    latest_status = "Invested" if latest_price > latest_ma else "Cash"

    return build_backtest_result(
        close,
        strategy_returns,
        "Moving Average Trend Following",
        benchmark_symbol=benchmark_symbol,
        transaction_cost=transaction_cost,
        rebalances=rebalances,
        monthly_holdings=monthly_holdings,
        trade_log=trade_log,
        summary_points=[
            f"Tracks {symbol} against its 200-day moving average.",
            f"Latest status: {latest_status}. Latest price is {latest_price:.2f}; 200-day MA is {latest_ma:.2f}.",
            "Uses the next trading day after the signal, so today's signal is not traded with yesterday's knowledge.",
        ],
    )


def run_dual_momentum_backtest(close, top_n=3, lookback_months=3, transaction_cost=0.0005, benchmark_symbol="SPY"):
    if "SHY" not in close.columns:
        return None

    monthly_prices = close.resample("ME").last()
    trailing_returns = monthly_prices.pct_change(lookback_months)
    daily_returns = close.pct_change().dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    previous_weights = pd.Series(0.0, index=close.columns)
    turnover_list = []
    rebalances = []
    monthly_holdings = []
    monthly_rankings = []
    trade_log = []

    month_ends = monthly_prices.index
    for idx in range(lookback_months, len(month_ends) - 1):
        rebalance_date = month_ends[idx]
        return_scores = trailing_returns.loc[rebalance_date].dropna()
        if "SHY" not in return_scores:
            continue

        shy_return = return_scores["SHY"]
        risk_scores = return_scores.drop(index="SHY", errors="ignore")
        eligible = risk_scores[risk_scores > shy_return].nlargest(top_n)
        selected = eligible.index.tolist()

        new_weights = pd.Series(0.0, index=close.columns)
        if selected:
            new_weights[selected] = 1.0 / top_n
        if len(selected) < top_n:
            new_weights["SHY"] += (top_n - len(selected)) / top_n

        ranking_df = (return_scores.sort_values(ascending=False) * 100).to_frame(name="3M Return (%)")
        ranking_df["Above SHY"] = ranking_df["3M Return (%)"] > (shy_return * 100)
        ranking_df.index.name = "Ticker"
        monthly_rankings.append({"date": rebalance_date, "ranking": ranking_df.round(2)})

        turnover = (new_weights - previous_weights).abs().sum() / 2
        turnover_list.append(turnover)
        rebalances.append(rebalance_date)
        monthly_holdings.append(
            {
                "Date": rebalance_date,
                "Holdings": new_weights[new_weights > 0].index.tolist(),
                "Holdings_str": format_holdings(new_weights),
            }
        )

        trade_date = get_next_trade_date(daily_returns, rebalance_date)
        next_rebalance_date = month_ends[idx + 1]
        trade_log.append(build_trade_log_entry(rebalance_date, trade_date, previous_weights, new_weights, next_rebalance_date))

        period_mask = (daily_returns.index > rebalance_date) & (daily_returns.index <= next_rebalance_date)
        if period_mask.any():
            active_weights = new_weights[new_weights > 0]
            tranche_returns = daily_returns.loc[period_mask, active_weights.index].dot(active_weights)
            if len(tranche_returns) > 0:
                tranche_returns.iloc[0] -= turnover * transaction_cost
            strategy_returns.loc[period_mask] = tranche_returns
        previous_weights = new_weights

    result = build_backtest_result(
        close,
        strategy_returns,
        "Dual Momentum",
        benchmark_symbol=benchmark_symbol,
        transaction_cost=transaction_cost,
        rebalances=rebalances,
        turnover_list=turnover_list,
        monthly_holdings=monthly_holdings,
        trade_log=trade_log,
        latest_scores=monthly_rankings[-1]["ranking"] if monthly_rankings else None,
        summary_points=[
            "Ranks ETFs by trailing 3-month return and compares them with SHY.",
            "Only ETFs beating SHY qualify; unused slots are allocated to SHY.",
            "Trades during the following month to avoid look-ahead bias.",
        ],
    )
    if result:
        result["monthly_rankings"] = monthly_rankings
    return result


def run_low_volatility_backtest(close, top_n=3, lookback_days=66, transaction_cost=0.0005, benchmark_symbol="SPY"):
    monthly_prices = close.resample("ME").last()
    daily_returns = close.pct_change().dropna()
    trailing_volatility = daily_returns.rolling(lookback_days).std() * np.sqrt(252)
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    previous_weights = pd.Series(0.0, index=close.columns)
    turnover_list = []
    rebalances = []
    monthly_holdings = []
    monthly_rankings = []
    trade_log = []

    month_ends = monthly_prices.index
    for idx in range(3, len(month_ends) - 1):
        rebalance_date = month_ends[idx]
        available_dates = trailing_volatility.index[trailing_volatility.index <= rebalance_date]
        if len(available_dates) == 0:
            continue

        signal_date = available_dates[-1]
        vol_scores = trailing_volatility.loc[signal_date].dropna()
        if len(vol_scores) < top_n:
            continue

        selected = vol_scores.nsmallest(top_n).index.tolist()
        ranking_df = (vol_scores.sort_values() * 100).to_frame(name="3M Annualized Volatility (%)")
        ranking_df.index.name = "Ticker"
        monthly_rankings.append({"date": signal_date, "ranking": ranking_df.round(2)})

        new_weights = pd.Series(0.0, index=close.columns)
        new_weights[selected] = 1.0 / top_n
        turnover = (new_weights - previous_weights).abs().sum() / 2
        turnover_list.append(turnover)
        rebalances.append(signal_date)
        monthly_holdings.append({"Date": signal_date, "Holdings": selected, "Holdings_str": format_holdings(new_weights)})

        trade_date = get_next_trade_date(daily_returns, signal_date)
        next_rebalance_date = month_ends[idx + 1]
        trade_log.append(build_trade_log_entry(signal_date, trade_date, previous_weights, new_weights, next_rebalance_date))

        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_rebalance_date)
        if period_mask.any():
            tranche_returns = daily_returns.loc[period_mask, selected].dot(new_weights[selected])
            if len(tranche_returns) > 0:
                tranche_returns.iloc[0] -= turnover * transaction_cost
            strategy_returns.loc[period_mask] = tranche_returns
        previous_weights = new_weights

    result = build_backtest_result(
        close,
        strategy_returns,
        "Low Volatility Rotation",
        benchmark_symbol=benchmark_symbol,
        transaction_cost=transaction_cost,
        rebalances=rebalances,
        turnover_list=turnover_list,
        monthly_holdings=monthly_holdings,
        trade_log=trade_log,
        latest_scores=monthly_rankings[-1]["ranking"] if monthly_rankings else None,
        summary_points=[
            "Ranks ETFs by trailing 3-month daily return volatility.",
            "Holds the 3 lowest-volatility ETFs equally during the following month.",
            "Uses only information available at the month-end signal date.",
        ],
    )
    if result:
        result["monthly_rankings"] = monthly_rankings
    return result


def weights_from_signal(signal, close_columns):
    weights = pd.Series(0.0, index=close_columns)
    selected_assets = [asset.strip() for asset in str(signal["Selected Asset"]).split(",")]
    selected_assets = [asset for asset in selected_assets if asset in weights.index]

    if signal["Signal"] in ["CASH", "SELL"] and "SHY" not in selected_assets:
        return weights

    if not selected_assets:
        if signal["Signal"] == "RISK-OFF" and "SHY" in weights.index:
            weights["SHY"] = 1.0
        return weights

    for asset in selected_assets:
        weights[asset] = 1.0 / len(selected_assets)
    return weights


def run_rule_based_monthly_backtest(strategy_type, close, transaction_cost=0.0005, ma_symbol="SPY", benchmark_symbol="SPY"):
    monthly_prices = close.resample("ME").last()
    daily_returns = close.pct_change().dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    previous_weights = pd.Series(0.0, index=close.columns)
    turnover_list = []
    rebalances = []
    monthly_holdings = []
    trade_log = []
    signal_rows = []

    month_ends = monthly_prices.index
    for idx in range(12, len(month_ends) - 1):
        month_end = month_ends[idx]
        signal_date = latest_valid_date(close, month_end)
        if signal_date is None:
            continue

        history = close.loc[:signal_date]
        signal = generate_strategy_signal(strategy_type, history, ma_symbol=ma_symbol)
        new_weights = weights_from_signal(signal, close.columns)
        turnover = (new_weights - previous_weights).abs().sum() / 2
        turnover_list.append(turnover)
        rebalances.append(signal_date)
        monthly_holdings.append(
            {
                "Date": signal_date,
                "Holdings": new_weights[new_weights > 0].index.tolist() or ["Cash"],
                "Holdings_str": format_holdings(new_weights),
            }
        )
        signal_rows.append(signal)

        trade_date = get_next_trade_date(daily_returns, signal_date)
        next_rebalance_date = month_ends[idx + 1]
        trade_log.append(build_trade_log_entry(signal_date, trade_date, previous_weights, new_weights, next_rebalance_date))

        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_rebalance_date)
        if period_mask.any():
            active_weights = new_weights[new_weights > 0]
            if active_weights.empty:
                tranche_returns = pd.Series(0.0, index=daily_returns.loc[period_mask].index)
            else:
                tranche_returns = daily_returns.loc[period_mask, active_weights.index].dot(active_weights)
            if len(tranche_returns) > 0:
                tranche_returns.iloc[0] -= turnover * transaction_cost
            strategy_returns.loc[period_mask] = tranche_returns
        previous_weights = new_weights

    latest_signal = generate_strategy_signal(strategy_type, close, ma_symbol=ma_symbol)
    signal_table = pd.DataFrame(signal_rows).drop(columns=["Vote"], errors="ignore") if signal_rows else None
    result = build_backtest_result(
        close,
        strategy_returns,
        strategy_type,
        benchmark_symbol=benchmark_symbol,
        transaction_cost=transaction_cost,
        rebalances=rebalances,
        turnover_list=turnover_list,
        monthly_holdings=monthly_holdings,
        trade_log=trade_log,
        latest_scores=signal_table.tail(12) if signal_table is not None else None,
        summary_points=[
            "Historical signals use data available up to each month-end signal date.",
            "Trades are applied during the following month after the signal date.",
            "The latest daily signal is shown separately at the top of this strategy tab.",
        ],
    )
    if result is not None:
        result["signal"] = latest_signal
    return result


def backtest_strategy(strategy_type, close, transaction_cost, ma_symbol="SPY", benchmark_symbol="SPY"):
    if strategy_type == "Momentum Rotation":
        result = run_momentum_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    elif strategy_type == "Moving Average Trend Following":
        result = run_moving_average_trend_backtest(close, symbol=ma_symbol, benchmark_symbol=benchmark_symbol, transaction_cost=transaction_cost)
    elif strategy_type == "Dual Momentum":
        result = run_dual_momentum_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    elif strategy_type == "Low Volatility Rotation":
        result = run_low_volatility_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    elif strategy_type in STRATEGY_TYPES:
        return run_rule_based_monthly_backtest(
            strategy_type,
            close,
            transaction_cost=transaction_cost,
            ma_symbol=ma_symbol,
            benchmark_symbol=benchmark_symbol,
        )
    else:
        return None
    if result is not None:
        result["signal"] = generate_strategy_signal(strategy_type, close, ma_symbol=ma_symbol)
    return result


def run_strategy_backtest(strategy_type, close, transaction_cost, ma_symbol="SPY", benchmark_symbol="SPY"):
    return backtest_strategy(strategy_type, close, transaction_cost, ma_symbol=ma_symbol, benchmark_symbol=benchmark_symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def run_all_strategy_backtests(close, transaction_cost, ma_symbol="SPY"):
    results = {}
    for strategy_type in STRATEGY_TYPES:
        results[strategy_type] = backtest_strategy(
            strategy_type,
            close,
            transaction_cost,
            ma_symbol=ma_symbol,
        )
    return results


def build_strategy_summary(result):
    metrics = result["metrics"]
    holdings = ", ".join(result["current_holdings"]) if result["current_holdings"] else "Cash"
    lines = [
        f"**{result['strategy_name']}** returned {metrics['total_return'] * 100:.2f}% with a CAGR of {metrics['cagr'] * 100:.2f}%.",
        f"Annualized volatility was {metrics['volatility'] * 100:.2f}%, Sharpe ratio was {metrics['sharpe']:.2f}, and max drawdown was {metrics['max_drawdown'] * 100:.2f}%.",
        f"Current holdings: **{holdings}**.",
    ]
    lines.extend(result["summary_points"])
    return "\n\n".join(lines)


def benchmark_total_return(result):
    if result["benchmark_equity"].empty:
        return np.nan
    return result["benchmark_equity"].iloc[-1] - 1


def render_signal_card(signal):
    st.subheader("Latest Daily Signal")
    st.write(
        "Rule Match Score is a rule-based score showing how strongly the latest data satisfies the strategy conditions. "
        "It is not a predicted probability of profit."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Latest Signal Date", signal["Latest Signal Date"])
    c2.metric("Selected ETF / Asset", signal["Selected Asset"])
    c3.metric("Signal", signal["Signal"])
    c4.metric("Rule Match Score", f"{signal['Rule Match Score']:.1f}%")
    c5.metric("Vote", f"{signal['Vote']:+.1f}")
    st.write(f"**Reason:** {signal['Reason']}")


def render_top_strategy_summary(result):
    signal = result["signal"]
    metrics = result["metrics"]
    strategy_return = metrics["total_return"] * 100
    spy_return = benchmark_total_return(result) * 100
    st.subheader("Selected Strategy Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Strategy", result["strategy_name"])
    c2.metric("Latest Data Date", signal["Latest Signal Date"])
    c3.metric("Selected ETF", signal["Selected Asset"])
    c4.metric("Final Signal", signal["Signal"])
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Rule Match Score", f"{signal['Rule Match Score']:.1f}%")
    c6.metric("Benchmark", result["benchmark_symbol"])
    c7.metric("Strategy vs SPY", f"{strategy_return:.2f}% / {spy_return:.2f}%")
    c8.metric("Sharpe / Max DD", f"{metrics['sharpe']:.2f} / {metrics['max_drawdown'] * 100:.2f}%")


def render_combined_strategy_section(combined_signal, strategy_signals):
    st.subheader("Combined Strategy Signal")
    render_signal_card(combined_signal)
    if not strategy_signals:
        st.warning("No individual strategy signals are available.")
        return
    vote_df = pd.DataFrame(strategy_signals).T[
        ["Latest Signal Date", "Selected Asset", "Signal", "Rule Match Score", "Vote", "Reason"]
    ]
    st.dataframe(vote_df, use_container_width=True)


def build_all_strategy_signals_table(results):
    rows = []
    for strategy_name, result in results.items():
        if result is None or "signal" not in result:
            continue
        signal = result["signal"]
        rows.append(
            {
                "Strategy Name": strategy_name,
                "Latest Signal": signal["Signal"],
                "Selected ETF / Asset": signal["Selected Asset"],
                "Rule Match Score": signal["Rule Match Score"],
                "Reason": signal["Reason"],
                "Strategy Return": result["metrics"]["total_return"] * 100,
                "SPY Return": benchmark_total_return(result) * 100,
                "Sharpe Ratio": result["metrics"]["sharpe"],
                "Max Drawdown": result["metrics"]["max_drawdown"] * 100,
            }
        )
    return pd.DataFrame(rows)


def render_combined_signal_section(combined_signal, results):
    strategy_signals = {
        strategy_name: result["signal"]
        for strategy_name, result in results.items()
        if result is not None and "signal" in result
    }
    render_combined_strategy_section(combined_signal, strategy_signals)
    st.subheader("All Strategy Signals")
    signal_table = build_all_strategy_signals_table(results)
    if signal_table.empty:
        st.warning("No strategy signal table is available.")
    else:
        st.dataframe(
            signal_table.style.format(
                {
                    "Rule Match Score": "{:.1f}",
                    "Strategy Return": "{:.2f}%",
                    "SPY Return": "{:.2f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max Drawdown": "{:.2f}%",
                }
            ),
            use_container_width=True,
        )


def select_asset_for_combined_signal(signal, close_columns):
    if signal["Signal"] == "BUY":
        candidates = [asset.strip() for asset in str(signal["Selected Asset"]).split(",")]
        for asset in candidates:
            if asset in close_columns:
                return asset
        if "SPY" in close_columns:
            return "SPY"
    if signal["Signal"] in ["RISK-OFF", "CASH", "SELL"]:
        for asset in ["SHY", "IEF", "TLT"]:
            if asset in close_columns:
                return asset
    return "Cash"


def build_hypothetical_portfolio_tracker(close, transaction_cost=0.0005, ma_symbol="SPY", starting_value=1_000_000):
    daily_returns = close.pct_change().dropna()
    if daily_returns.empty or "SPY" not in daily_returns.columns:
        return pd.DataFrame(), {}

    month_ends = close.resample("ME").last().index
    portfolio_value = starting_value
    previous_asset = "Cash"
    rows = []

    for idx in range(12, len(month_ends) - 1):
        signal_date = latest_valid_date(close, month_ends[idx])
        if signal_date is None:
            continue

        history = close.loc[:signal_date]
        strategy_signals = generate_all_strategy_signals(history, ma_symbol=ma_symbol)
        combined_signal = combine_strategy_signals(strategy_signals)
        target_asset = select_asset_for_combined_signal(combined_signal, close.columns)

        if combined_signal["Signal"] == "HOLD":
            held_asset = previous_asset
        else:
            held_asset = target_asset

        next_rebalance_date = month_ends[idx + 1]
        period_mask = (daily_returns.index > signal_date) & (daily_returns.index <= next_rebalance_date)
        period_returns = daily_returns.loc[period_mask]
        if period_returns.empty:
            continue

        turnover = 0 if held_asset == previous_asset else 1
        for day_index, (date, return_row) in enumerate(period_returns.iterrows()):
            if held_asset in return_row.index:
                daily_return = return_row[held_asset]
                cash_balance = 0
                portfolio_weight = 1.0
            else:
                daily_return = 0.0
                cash_balance = portfolio_value
                portfolio_weight = 0.0

            if day_index == 0 and turnover > 0:
                daily_return -= turnover * transaction_cost

            portfolio_value *= 1 + daily_return
            rows.append(
                {
                    "Date": date,
                    "Combined Signal": combined_signal["Signal"],
                    "Held Asset": held_asset,
                    "Portfolio Value": portfolio_value,
                    "Daily Return": daily_return,
                    "Cumulative Return": portfolio_value / starting_value - 1,
                    "Cash Balance": cash_balance if held_asset == "Cash" else 0,
                    "Portfolio Weight": portfolio_weight,
                    "Cash / Allocation": "Cash" if held_asset == "Cash" else f"100% {held_asset}",
                    "Signal Date": combined_signal["Latest Signal Date"],
                }
            )

        previous_asset = held_asset

    portfolio_df = pd.DataFrame(rows)
    if portfolio_df.empty:
        return portfolio_df, {}

    portfolio_df = portfolio_df.set_index("Date")
    spy_returns = daily_returns["SPY"].reindex(portfolio_df.index).fillna(0)
    portfolio_df["Benchmark SPY Value"] = starting_value * (1 + spy_returns).cumprod()
    portfolio_df["Excess Return vs SPY"] = (
        portfolio_df["Portfolio Value"] / starting_value - 1
    ) - (portfolio_df["Benchmark SPY Value"] / starting_value - 1)

    tracker_returns = portfolio_df["Portfolio Value"].pct_change().dropna()
    tracker_equity = portfolio_df["Portfolio Value"] / starting_value
    tracker_metrics = calculate_performance_metrics(tracker_equity, tracker_returns, portfolio_df["Signal Date"].nunique())
    tracker_metrics["spy_total_return"] = portfolio_df["Benchmark SPY Value"].iloc[-1] / starting_value - 1
    tracker_metrics["ending_value"] = portfolio_df["Portfolio Value"].iloc[-1]
    tracker_metrics["current_holding"] = portfolio_df["Held Asset"].iloc[-1]
    return portfolio_df, tracker_metrics


def render_workflow_diagram():
    st.subheader("Workflow Diagram")
    st.write(
        "Process Overview: The dashboard starts with market data inputs, calculates indicators and strategy features, "
        "runs 10 strategy models, generates daily strategy signals, aggregates them into a combined signal, applies "
        "the selected rebalance schedule, constructs a hypothetical $1MM portfolio, and tracks performance over time "
        "versus the benchmark."
    )
    steps = [
        ("Market Data Inputs", "ETF prices, benchmark, defensive assets"),
        ("Data Processing", "Returns, moving averages, RSI, volatility"),
        ("Strategy Models", "10 rule-based strategy engines"),
        ("Signal Generation", "BUY / HOLD / RISK-OFF per strategy"),
        ("Combined Signal", "Vote score across all strategies"),
        ("Portfolio Allocation", "Risk-on ETF, defensive ETF, or cash"),
        ("Performance Tracking", "Portfolio value, returns, drawdown"),
        ("Dashboard Output", "Signals, charts, tables, summaries"),
    ]
    cards = ""
    for index, (title, detail) in enumerate(steps, start=1):
        arrow = "<div class='workflow-arrow'>-></div>" if index < len(steps) else ""
        cards += (
            "<div class='workflow-step'>"
            f"<div class='workflow-number'>{index}</div>"
            f"<div class='workflow-title'>{title}</div>"
            f"<div class='workflow-detail'>{detail}</div>"
            "</div>"
            f"{arrow}"
        )
    st.markdown(
        f"""
        <style>
        .workflow-wrap {{
            display: flex;
            align-items: stretch;
            gap: 10px;
            overflow-x: auto;
            padding: 8px 2px 12px 2px;
        }}
        .workflow-step {{
            min-width: 150px;
            max-width: 170px;
            border: 1px solid #d7dde8;
            border-radius: 8px;
            padding: 12px;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .workflow-number {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #1f77b4;
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .workflow-title {{
            font-size: 14px;
            font-weight: 700;
            color: #111827;
            line-height: 1.25;
            margin-bottom: 6px;
        }}
        .workflow-detail {{
            font-size: 12px;
            color: #4b5563;
            line-height: 1.35;
        }}
        .workflow-arrow {{
            display: flex;
            align-items: center;
            color: #6b7280;
            font-weight: 700;
            font-size: 18px;
        }}
        </style>
        <div class="workflow-wrap">{cards}</div>
        """,
        unsafe_allow_html=True,
    )


def render_hypothetical_portfolio_tracker(portfolio_df, tracker_metrics):
    st.subheader("Hypothetical $1MM Portfolio Tracker")
    st.write(
        "This is a hypothetical model portfolio for tracking strategy behavior over time. It is not a live investment account."
    )
    if portfolio_df.empty:
        st.warning("Hypothetical portfolio tracker is unavailable for the current data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${tracker_metrics['ending_value']:,.0f}")
    c2.metric("Cumulative Return", f"{tracker_metrics['total_return'] * 100:.2f}%")
    c3.metric("SPY Return", f"{tracker_metrics['spy_total_return'] * 100:.2f}%")
    c4.metric("Current Holding", tracker_metrics["current_holding"])
    c5, c6, c7 = st.columns(3)
    c5.metric("Sharpe Ratio", f"{tracker_metrics['sharpe']:.2f}")
    c6.metric("Max Drawdown", f"{tracker_metrics['max_drawdown'] * 100:.2f}%")
    c7.metric("Annualized Volatility", f"{tracker_metrics['volatility'] * 100:.2f}%")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["Portfolio Value"],
            name="Hypothetical $1MM Portfolio",
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df["Benchmark SPY Value"],
            name="SPY Benchmark",
            line=dict(width=2, dash="dash"),
        )
    )
    fig.update_layout(
        title="Hypothetical Portfolio Value vs SPY",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    display_df = portfolio_df.reset_index()[
        [
            "Date",
            "Combined Signal",
            "Held Asset",
            "Portfolio Value",
            "Daily Return",
            "Cumulative Return",
            "Cash / Allocation",
        ]
    ].tail(60)
    st.dataframe(
        display_df.style.format(
            {
                "Portfolio Value": "${:,.2f}",
                "Daily Return": "{:.2%}",
                "Cumulative Return": "{:.2%}",
            }
        ),
        use_container_width=True,
    )


def pick_rebalance_dates(signal_history, frequency):
    if signal_history.empty:
        return []
    if frequency == "Daily":
        return signal_history.index.tolist()
    if frequency == "Weekly":
        return signal_history.resample("W-FRI").last().dropna(how="all").index.tolist()
    return signal_history.resample("ME").last().dropna(how="all").index.tolist()


def get_signal_counts(window_df):
    return {
        "BUY": int((window_df["Combined Signal"] == "BUY").sum()),
        "HOLD": int((window_df["Combined Signal"] == "HOLD").sum()),
        "RISK-OFF": int(window_df["Combined Signal"].isin(["RISK-OFF", "CASH", "SELL"]).sum()),
    }


def get_most_frequent_selected_asset(window_df, close_columns):
    candidates = window_df["Selected ETF"].dropna().astype(str)
    expanded = []
    for value in candidates:
        for asset in value.split(","):
            asset = asset.strip()
            if asset in close_columns and asset not in ["Cash", "SHY", "IEF", "TLT"]:
                expanded.append(asset)
    if not expanded:
        return "SPY" if "SPY" in close_columns else close_columns[0]
    return pd.Series(expanded).value_counts().index[0]


def get_defensive_allocation(close_columns):
    for asset in ["SHY", "IEF", "TLT"]:
        if asset in close_columns:
            return asset
    return "Cash"


def build_daily_signal_history(close, ma_symbol="SPY"):
    rows = []
    indicators = calculate_indicators(close)
    momentum_scores = calculate_momentum_scores(indicators)
    start_index = min(252, max(0, len(close) - 1))
    for date in close.index[start_index:]:
        strategy_signals = {}
        signal_date = date

        score_row = momentum_scores.loc[date].dropna() if date in momentum_scores.index else pd.Series(dtype=float)
        if not score_row.empty:
            selected = score_row.sort_values(ascending=False).index[0]
            vote = 1 if score_row[selected] > 0 else 0
            strategy_signals["Momentum Rotation"] = make_signal("Momentum Rotation", signal_date, selected, "BUY" if vote == 1 else "HOLD", 50, "Daily weighted momentum score.", vote)

        symbol = ma_symbol if ma_symbol in close.columns else close.columns[0]
        if symbol in close.columns:
            price = close.at[date, symbol]
            ma50 = indicators["ma50"].at[date, symbol] if symbol in indicators["ma50"].columns else np.nan
            ma200 = indicators["ma200"].at[date, symbol] if symbol in indicators["ma200"].columns else np.nan
            if pd.notna(price) and pd.notna(ma50) and pd.notna(ma200):
                passed = sum([price > ma50, price > ma200, ma50 > ma200])
                signal = "BUY" if passed == 3 else "HOLD" if passed >= 2 else "CASH"
                vote = 1 if signal == "BUY" else 0 if signal == "HOLD" else -1
                strategy_signals["Moving Average Trend Following"] = make_signal("Moving Average Trend Following", signal_date, symbol, signal, passed / 3 * 100, "Daily MA50/MA200 trend conditions.", vote)

        ret12 = indicators["returns_12m"].loc[date].dropna()
        if not ret12.empty:
            selected = ret12.sort_values(ascending=False).index[0]
            signal = "BUY" if ret12[selected] > 0 else "RISK-OFF"
            strategy_signals["Dual Momentum"] = make_signal("Dual Momentum", signal_date, selected if signal == "BUY" else get_defensive_allocation(close.columns), signal, 50, "Daily relative and absolute momentum check.", 1 if signal == "BUY" else -1)

        vol60 = indicators["vol60"].loc[date].dropna()
        ret3 = indicators["returns_3m"].loc[date].dropna()
        if not vol60.empty:
            eligible = ret3[ret3 > 0].index.intersection(vol60.index)
            ranked = vol60.loc[eligible].sort_values() if len(eligible) else vol60.sort_values()
            selected = ranked.index[0]
            signal = "BUY" if selected in eligible else "HOLD"
            strategy_signals["Low Volatility Rotation"] = make_signal("Low Volatility Rotation", signal_date, selected, signal, 50, "Daily low-volatility ranking.", 1 if signal == "BUY" else 0)

        if "SPY" in close.columns:
            spy_price = close.at[date, "SPY"]
            spy_ma50 = indicators["ma50"].at[date, "SPY"] if "SPY" in indicators["ma50"].columns else np.nan
            spy_ma200 = indicators["ma200"].at[date, "SPY"] if "SPY" in indicators["ma200"].columns else np.nan
            spy_vol20 = indicators["vol20"].at[date, "SPY"] if "SPY" in indicators["vol20"].columns else np.nan
            if pd.notna(spy_price) and pd.notna(spy_ma50) and pd.notna(spy_ma200):
                risk_off_count = sum([spy_price < spy_ma50, spy_price < spy_ma200, spy_ma50 < spy_ma200])
                defensive_asset = get_defensive_allocation(close.columns)
                defensive_signal = "RISK-OFF" if risk_off_count >= 2 else "BUY" if risk_off_count == 0 else "HOLD"
                strategy_signals["Defensive Rotation Strategy"] = make_signal("Defensive Rotation Strategy", signal_date, defensive_asset if risk_off_count >= 2 else "SPY", defensive_signal, risk_off_count / 3 * 100, "Daily SPY defensive regime check.", -1 if defensive_signal == "RISK-OFF" else 1 if defensive_signal == "BUY" else 0)

                regime_signal = "BUY" if spy_price > spy_ma50 and spy_price > spy_ma200 else "RISK-OFF" if spy_price < spy_ma50 and spy_price < spy_ma200 else "HOLD"
                strategy_signals["Risk-On / Risk-Off Regime Strategy"] = make_signal("Risk-On / Risk-Off Regime Strategy", signal_date, "SPY" if regime_signal != "RISK-OFF" else defensive_asset, regime_signal, 50, "Daily SPY trend regime check.", 1 if regime_signal == "BUY" else -1 if regime_signal == "RISK-OFF" else 0)

                if pd.notna(spy_vol20):
                    vol_signal = "BUY" if spy_vol20 <= 0.15 and spy_price > spy_ma200 else "RISK-OFF" if spy_vol20 >= 0.25 or spy_price <= spy_ma200 else "HOLD"
                    strategy_signals["Volatility Target Strategy"] = make_signal("Volatility Target Strategy", signal_date, "SPY" if vol_signal != "RISK-OFF" else defensive_asset, vol_signal, 50, "Daily volatility target check.", 1 if vol_signal == "BUY" else -1 if vol_signal == "RISK-OFF" else 0)

        latest_close = close.loc[date].dropna()
        high55 = indicators["rolling_high_55"].loc[date].dropna()
        low20 = indicators["rolling_low_20"].loc[date].dropna()
        common_high = latest_close.index.intersection(high55.index)
        if len(common_high):
            breakout = (latest_close[common_high] / high55[common_high] - 1).dropna()
            selected = breakout.sort_values(ascending=False).index[0]
            if breakout[selected] > 0:
                strategy_signals["Breakout Strategy"] = make_signal("Breakout Strategy", signal_date, selected, "BUY", 50, "Daily 55-day breakout check.", 1)
            else:
                common_low = latest_close.index.intersection(low20.index)
                breakdown = (latest_close[common_low] / low20[common_low] - 1).dropna() if len(common_low) else pd.Series(dtype=float)
                signal = "CASH" if not breakdown.empty and breakdown.min() < 0 else "HOLD"
                strategy_signals["Breakout Strategy"] = make_signal("Breakout Strategy", signal_date, "Cash" if signal == "CASH" else selected, signal, 50, "Daily breakout/breakdown check.", -1 if signal == "CASH" else 0)

        rsi = indicators["rsi14"].loc[date].dropna()
        ma200_row = indicators["ma200"].loc[date].dropna()
        common = latest_close.index.intersection(rsi.index).intersection(ma200_row.index)
        if len(common):
            healthy = latest_close[common] > ma200_row[common]
            oversold = rsi[common] < 30
            if (healthy & oversold).any():
                selected = rsi[common][healthy & oversold].sort_values().index[0]
                strategy_signals["Mean Reversion Strategy"] = make_signal("Mean Reversion Strategy", signal_date, selected, "BUY", 50, "Daily oversold RSI with healthy trend.", 1)
            else:
                cash_signal = (rsi[common] > 70).any() or (~healthy).mean() > 0.5
                strategy_signals["Mean Reversion Strategy"] = make_signal("Mean Reversion Strategy", signal_date, "Cash" if cash_signal else common[0], "CASH" if cash_signal else "HOLD", 50, "Daily RSI mean-reversion check.", -1 if cash_signal else 0)

        ma50_row = indicators["ma50"].loc[date].dropna()
        common = latest_close.index.intersection(ma50_row.index).intersection(ma200_row.index)
        if len(common):
            passing = latest_close[common][(latest_close[common] > ma50_row[common]) & (latest_close[common] > ma200_row[common])].index.tolist()
            strength = len(passing) / len(common) * 100
            signal = "BUY" if len(passing) >= max(3, len(common) * 0.4) else "HOLD" if passing else "RISK-OFF"
            strategy_signals["Equal Weight Multi-ETF Strategy"] = make_signal("Equal Weight Multi-ETF Strategy", signal_date, ", ".join(passing[:6]) if passing else get_defensive_allocation(close.columns), signal, strength, "Daily equal-weight trend filter breadth.", 1 if signal == "BUY" else -1 if signal == "RISK-OFF" else 0)

        for strategy_type in STRATEGY_TYPES:
            if strategy_type not in strategy_signals:
                strategy_signals[strategy_type] = make_signal(strategy_type, signal_date, "Cash", "HOLD", 0, "Not enough data for this daily signal.", 0)

        combined_signal = combine_strategy_signals(strategy_signals)
        votes = [signal["Vote"] for signal in strategy_signals.values()]
        rows.append(
            {
                "Date": date,
                "Combined Signal Score": np.mean(votes) if votes else np.nan,
                "Combined Signal": combined_signal["Signal"],
                "Selected ETF": combined_signal["Selected Asset"],
                "BUY Votes": sum(1 for vote in votes if vote == 1),
                "HOLD Votes": sum(1 for vote in votes if vote == 0),
                "RISK-OFF Votes": sum(1 for vote in votes if vote == -1),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Date")


def calculate_win_rate(returns):
    if returns.empty:
        return np.nan
    return (returns > 0).mean()


def normalize_alpha_weights(score_row, close_columns, allow_defensive=True):
    weights = pd.Series(0.0, index=close_columns)
    scores = score_row.dropna().sort_values(ascending=False)
    if scores.empty:
        return weights
    top_asset = scores.index[0]
    top_score = scores.iloc[0]
    if top_score > 0:
        weights[top_asset] = 1.0
    elif allow_defensive and "SHY" in weights.index:
        weights["SHY"] = 1.0
    return weights


def calculate_alpha_scores(alpha_name, ohlcv):
    open_price = ohlcv["open"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    returns = close.pct_change()
    typical_price = (high + low + close) / 3

    if alpha_name == "Overnight Mean Reversion":
        gap = open_price / close.shift(1) - 1
        return -gap
    if alpha_name == "Short-Term Momentum":
        return close.pct_change(5)
    if alpha_name == "Intraday Reversal":
        intraday_move = close / open_price - 1
        return -intraday_move
    if alpha_name == "Typical Price Reversion":
        distance = close / typical_price - 1
        return -distance
    if alpha_name == "Volume Spike Reversal":
        volume_z = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
        extreme_move = returns
        return -(volume_z.clip(lower=0) * extreme_move)
    if alpha_name == "High-Low Range Breakout":
        breakout = close / close.rolling(20).max().shift(1) - 1
        breakdown = close / close.rolling(20).min().shift(1) - 1
        return breakout.where(breakout > 0, breakdown)
    if alpha_name == "Low Volatility Rotation Alpha":
        vol = returns.rolling(20).std() * np.sqrt(252)
        return -vol
    if alpha_name == "Rank-Based Momentum":
        return close.pct_change(10).rank(axis=1, pct=True) - 0.5
    if alpha_name == "Price-Volume Confirmation":
        momentum = close.pct_change(5)
        volume_trend = volume / volume.rolling(20).mean() - 1
        return momentum * volume_trend
    if alpha_name == "Correlation Diversification Alpha":
        scores = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        for date in close.index:
            window = returns.loc[:date].tail(60)
            if len(window.dropna(how="all")) < 20:
                continue
            corr = window.corr().abs()
            corr = corr.mask(np.eye(len(corr), dtype=bool))
            scores.loc[date] = -corr.mean()
        return scores
    return pd.DataFrame(index=close.index, columns=close.columns)


def calculate_alpha_weights(alpha_name, ohlcv):
    close = ohlcv["close"]
    scores = calculate_alpha_scores(alpha_name, ohlcv)
    weight_rows = [normalize_alpha_weights(scores.loc[date], close.columns) for date in scores.index]
    weights = pd.DataFrame(weight_rows, index=scores.index).reindex(columns=close.columns)
    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scores, weights


def calculate_return_metrics(returns):
    returns = returns.dropna()
    if returns.empty:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "current_drawdown": np.nan,
        }
    equity = (1 + returns).cumprod()
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    total_return = equity.iloc[-1] - 1
    annualized_return = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std() * np.sqrt(252)
    sharpe = returns.mean() * 252 / volatility if volatility and volatility > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "current_drawdown": drawdown.iloc[-1],
    }


def classify_alpha_status(metrics, turnover, cost_drag, correlation):
    if pd.notna(metrics["max_drawdown"]) and metrics["max_drawdown"] < -0.25:
        return "Pause"
    if pd.notna(metrics["max_drawdown"]) and metrics["max_drawdown"] < -0.15:
        return "Reduce"
    if pd.notna(metrics["sharpe"]) and metrics["sharpe"] < 0:
        return "Watch"
    if turnover > 1.5 or cost_drag > 0.03 or abs(correlation) > 0.85:
        return "Watch"
    return "Healthy"


def build_alpha_allocations(alpha_results, net_returns_df, max_alpha_allocation, allocation_method):
    eligible = [name for name, result in alpha_results.items() if result["status"] != "Pause"]
    allocations = pd.Series(0.0, index=alpha_results.keys(), dtype=float)
    if not eligible:
        return allocations

    if allocation_method == "Volatility Scaled":
        vols = net_returns_df[eligible].std() * np.sqrt(252)
        raw = 1 / vols.replace(0, np.nan)
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            raw = pd.Series(1.0, index=eligible)
        weights = raw / raw.sum()
    else:
        weights = pd.Series(1 / len(eligible), index=eligible)

    for name, weight in weights.items():
        adjusted_weight = min(weight, max_alpha_allocation)
        if alpha_results[name]["status"] == "Reduce":
            adjusted_weight *= 0.5
        allocations[name] = adjusted_weight
    return allocations


@st.cache_data(ttl=3600, show_spinner=False)
def run_multi_alpha_backtest(ohlcv, initial_capital=1_000_000, max_alpha_allocation=0.10, allocation_method="Equal Weight"):
    close = ohlcv["close"].dropna(how="all")
    asset_returns = close.pct_change().dropna()
    if asset_returns.empty:
        return {}, pd.DataFrame(), pd.DataFrame(), {}

    alpha_results = {}
    net_returns = {}
    gross_returns = {}
    latest_allocations = {}

    for alpha in ALPHA_STRATEGIES:
        name = alpha["name"]
        scores, target_weights = calculate_alpha_weights(name, ohlcv)
        target_weights = target_weights.reindex(asset_returns.index).fillna(0.0)
        execution_weights = target_weights.shift(1).fillna(0.0)
        gross = (execution_weights * asset_returns).sum(axis=1)
        weight_change = execution_weights.diff().fillna(execution_weights)
        buy_turnover = weight_change.clip(lower=0).sum(axis=1)
        sell_turnover = (-weight_change.clip(upper=0)).sum(axis=1)
        trading_cost = buy_turnover * BUY_TRANSACTION_COST + sell_turnover * SELL_TRANSACTION_COST
        net = gross - trading_cost
        gross_returns[name] = gross
        net_returns[name] = net

        metrics = calculate_return_metrics(net)
        gross_metrics = calculate_return_metrics(gross)
        turnover = (buy_turnover + sell_turnover).mean() * 252
        cost_drag = trading_cost.sum()
        latest_weights = target_weights.iloc[-1]
        selected_asset = latest_weights[latest_weights > 0].index[0] if (latest_weights > 0).any() else "Cash"
        signal_value = scores.reindex(target_weights.index).iloc[-1].dropna()
        current_signal = "BUY" if selected_asset != "Cash" and signal_value.max() > 0 else "CASH"
        latest_allocations[name] = latest_weights

        alpha_results[name] = {
            "category": alpha["category"],
            "scores": scores,
            "target_weights": target_weights,
            "execution_weights": execution_weights,
            "gross_returns": gross,
            "net_returns": net,
            "trading_cost": trading_cost,
            "turnover": buy_turnover + sell_turnover,
            "metrics": metrics,
            "gross_metrics": gross_metrics,
            "cost_drag": cost_drag,
            "selected_asset": selected_asset,
            "current_signal": current_signal,
            "signal_history": signal_value,
        }

    net_returns_df = pd.DataFrame(net_returns).dropna(how="all")
    gross_returns_df = pd.DataFrame(gross_returns).dropna(how="all")
    initial_portfolio_net_returns = net_returns_df.mean(axis=1)
    for name, result in alpha_results.items():
        corr = net_returns_df[name].corr(initial_portfolio_net_returns) if name in net_returns_df else np.nan
        result["correlation_with_portfolio"] = corr
        result["status"] = classify_alpha_status(result["metrics"], result["turnover"].mean() * 252, result["cost_drag"], corr)

    alpha_allocations = build_alpha_allocations(alpha_results, net_returns_df, max_alpha_allocation, allocation_method)
    portfolio_net_returns = net_returns_df.mul(alpha_allocations, axis=1).sum(axis=1)
    portfolio_gross_returns = gross_returns_df.mul(alpha_allocations, axis=1).sum(axis=1)
    portfolio_equity = initial_capital * (1 + portfolio_net_returns).cumprod()
    portfolio_gross_equity = initial_capital * (1 + portfolio_gross_returns).cumprod()
    portfolio_metrics = calculate_return_metrics(portfolio_net_returns)
    portfolio_metrics["gross_return"] = portfolio_gross_equity.iloc[-1] / initial_capital - 1 if not portfolio_gross_equity.empty else np.nan
    portfolio_metrics["net_return"] = portfolio_equity.iloc[-1] / initial_capital - 1 if not portfolio_equity.empty else np.nan
    portfolio_metrics["transaction_cost_drag"] = portfolio_metrics["gross_return"] - portfolio_metrics["net_return"]
    portfolio_metrics["portfolio_value"] = portfolio_equity.iloc[-1] if not portfolio_equity.empty else initial_capital
    portfolio_metrics["total_pnl"] = portfolio_metrics["portfolio_value"] - initial_capital
    portfolio_metrics["total_turnover"] = sum(result["turnover"].sum() * alpha_allocations[name] for name, result in alpha_results.items())
    portfolio_metrics["allocation_method"] = allocation_method
    portfolio_metrics["cash_weight"] = max(0.0, 1 - alpha_allocations.sum())

    portfolio_returns = pd.DataFrame(
        {
            "Portfolio Gross Return": portfolio_gross_returns,
            "Portfolio Net Return": portfolio_net_returns,
            "Portfolio Value": portfolio_equity,
            "Portfolio Gross Value": portfolio_gross_equity,
        }
    )

    correlations = net_returns_df.corr()
    for name, result in alpha_results.items():
        corr = net_returns_df[name].corr(portfolio_net_returns) if name in net_returns_df else np.nan
        result["correlation_with_portfolio"] = corr
        result["allocation"] = alpha_allocations[name]

    return alpha_results, portfolio_returns, correlations, portfolio_metrics


def build_alpha_monitoring_table(alpha_results, portfolio_returns):
    portfolio_net = portfolio_returns["Portfolio Net Return"] if "Portfolio Net Return" in portfolio_returns else pd.Series(dtype=float)
    rows = []
    for name, result in alpha_results.items():
        metrics = result["metrics"]
        gross_metrics = result["gross_metrics"]
        latest_net = result["net_returns"].dropna()
        daily_pnl = latest_net.iloc[-1] * result["allocation"] * 1_000_000 if not latest_net.empty else 0
        cumulative_pnl = metrics["total_return"] * result["allocation"] * 1_000_000 if pd.notna(metrics["total_return"]) else np.nan
        rows.append(
            {
                "Alpha name": name,
                "Category": result["category"],
                "Current signal": result["current_signal"],
                "Current allocation": result["allocation"],
                "Position": result["selected_asset"],
                "Daily PnL": daily_pnl,
                "Cumulative PnL": cumulative_pnl,
                "Gross return": gross_metrics["total_return"],
                "Net return": metrics["total_return"],
                "Sharpe ratio": metrics["sharpe"],
                "Max drawdown": metrics["max_drawdown"],
                "Turnover": result["turnover"].sum(),
                "Transaction cost drag": result["cost_drag"],
                "Correlation with portfolio": result["correlation_with_portfolio"],
                "Status": result["status"],
            }
        )
    return pd.DataFrame(rows)


def render_portfolio_overview(alpha_results, portfolio_returns, portfolio_metrics):
    st.header("Portfolio Overview")
    st.write(
        "Multi-alpha portfolio starting from 2026-06-04 with $1,000,000 initial capital. "
        "Gross returns are shown before transaction costs; net returns include 5 bps buy cost and 5 bps sell cost."
    )
    if portfolio_returns.empty:
        st.warning("Multi-alpha portfolio results are unavailable for the current data.")
        return

    active_count = sum(1 for result in alpha_results.values() if result["status"] == "Healthy")
    watch_count = sum(1 for result in alpha_results.values() if result["status"] in ["Watch", "Reduce", "Pause"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Value", f"${portfolio_metrics['portfolio_value']:,.0f}")
    c2.metric("Total PnL", f"${portfolio_metrics['total_pnl']:,.0f}")
    c3.metric("Gross Return", f"{portfolio_metrics['gross_return'] * 100:.2f}%")
    c4.metric("Net Return", f"{portfolio_metrics['net_return'] * 100:.2f}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Portfolio Sharpe", f"{portfolio_metrics['sharpe']:.2f}")
    c6.metric("Volatility", f"{portfolio_metrics['volatility'] * 100:.2f}%")
    c7.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown'] * 100:.2f}%")
    c8.metric("Current Drawdown", f"{portfolio_metrics['current_drawdown'] * 100:.2f}%")
    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Total Turnover", f"{portfolio_metrics['total_turnover']:.2f}x")
    c10.metric("Cost Drag", f"{portfolio_metrics['transaction_cost_drag'] * 100:.2f}%")
    c11.metric("Active Alphas", str(active_count))
    c12.metric("Alphas on Watch", str(watch_count))
    st.caption(
        f"Portfolio construction: {portfolio_metrics.get('allocation_method', 'Equal Weight')} with "
        f"{portfolio_metrics.get('cash_weight', 0) * 100:.1f}% unallocated cash after alpha caps and risk status rules."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns["Portfolio Value"], name="Net Portfolio Value", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns["Portfolio Gross Value"], name="Gross Portfolio Value", line=dict(width=2, dash="dash")))
    fig.update_layout(title="Portfolio Equity Curve", xaxis_title="Date", yaxis_title="Value ($)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    drawdown = portfolio_returns["Portfolio Value"] / portfolio_returns["Portfolio Value"].cummax() - 1
    dd_fig = px.area(x=drawdown.index, y=drawdown, title="Portfolio Drawdown")
    dd_fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", template="plotly_white")
    st.plotly_chart(dd_fig, use_container_width=True)

    contribution_rows = []
    for name, result in alpha_results.items():
        contribution_rows.append({"Alpha": name, "PnL Contribution": result["metrics"]["total_return"] * result["allocation"] * 1_000_000})
    contribution_df = pd.DataFrame(contribution_rows).sort_values("PnL Contribution", ascending=False)
    st.plotly_chart(px.bar(contribution_df, x="Alpha", y="PnL Contribution", title="Alpha Contribution to PnL"), use_container_width=True)


def render_alpha_live_monitoring(alpha_results, portfolio_returns):
    st.header("Alpha Live Monitoring")
    table = build_alpha_monitoring_table(alpha_results, portfolio_returns)
    if table.empty:
        st.warning("Alpha monitoring data is unavailable.")
        return
    st.dataframe(
        table.style.format(
            {
                "Current allocation": "{:.2%}",
                "Daily PnL": "${:,.0f}",
                "Cumulative PnL": "${:,.0f}",
                "Gross return": "{:.2%}",
                "Net return": "{:.2%}",
                "Sharpe ratio": "{:.2f}",
                "Max drawdown": "{:.2%}",
                "Turnover": "{:.2f}",
                "Transaction cost drag": "{:.2%}",
                "Correlation with portfolio": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    selected_alpha = st.selectbox("Select Alpha Detail", table["Alpha name"].tolist())
    result = alpha_results[selected_alpha]
    net_equity = (1 + result["net_returns"].dropna()).cumprod()
    gross_equity = (1 + result["gross_returns"].dropna()).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=net_equity.index, y=net_equity, name="Net"))
    fig.add_trace(go.Scatter(x=gross_equity.index, y=gross_equity, name="Gross", line=dict(dash="dash")))
    fig.update_layout(title=f"{selected_alpha} Gross vs Net Performance", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    drawdown = net_equity / net_equity.cummax() - 1
    st.plotly_chart(px.area(x=drawdown.index, y=drawdown, title=f"{selected_alpha} Drawdown"), use_container_width=True)
    rolling_sharpe = result["net_returns"].rolling(63).mean() * 252 / (result["net_returns"].rolling(63).std() * np.sqrt(252))
    st.plotly_chart(px.line(x=rolling_sharpe.index, y=rolling_sharpe, title=f"{selected_alpha} Rolling Sharpe"), use_container_width=True)
    st.plotly_chart(px.line(x=result["turnover"].index, y=result["turnover"], title=f"{selected_alpha} Daily Turnover"), use_container_width=True)
    st.write(f"**Current explanation:** {selected_alpha} currently selects **{result['selected_asset']}** with signal **{result['current_signal']}**.")
    st.dataframe(result["target_weights"].tail(30), use_container_width=True)


def render_alpha_correlation(alpha_results, correlations, portfolio_returns):
    st.header("Alpha Correlation")
    if correlations.empty:
        st.warning("Alpha correlation data is unavailable.")
        return
    st.plotly_chart(px.imshow(correlations, text_auto=".2f", title="Alpha Correlation Matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), use_container_width=True)
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
    high_corr = upper[upper.abs() > 0.8]
    if not high_corr.empty:
        st.warning("Concentration warning: several alpha pairs have absolute correlation above 0.80. High Sharpe alphas should not be automatically overweighted when correlation is high.")

    net_returns_df = pd.DataFrame({name: result["net_returns"] for name, result in alpha_results.items()}).dropna(how="all")
    rolling_avg_corr = net_returns_df.rolling(63).corr().groupby(level=0).mean().mean(axis=1)
    st.plotly_chart(px.line(x=rolling_avg_corr.index, y=rolling_avg_corr, title="Rolling Average Alpha Correlation"), use_container_width=True)

    contribution = []
    portfolio_net = portfolio_returns["Portfolio Net Return"]
    for name, result in alpha_results.items():
        contribution.append(
            {
                "Alpha": name,
                "Return Contribution": result["metrics"]["total_return"] * result["allocation"],
                "Risk Contribution Proxy": result["net_returns"].std() * result["allocation"],
            }
        )
    contribution_df = pd.DataFrame(contribution)
    if not contribution_df.empty and contribution_df["Risk Contribution Proxy"].max() > contribution_df["Risk Contribution Proxy"].sum() * 0.35:
        st.warning("Risk concentration warning: one alpha contributes a large share of total risk.")
    st.dataframe(contribution_df, use_container_width=True)


def render_alpha_backtesting(alpha_results, portfolio_returns, portfolio_metrics):
    st.header("Backtesting")
    render_portfolio_overview(alpha_results, portfolio_returns, portfolio_metrics)
    table = build_alpha_monitoring_table(alpha_results, portfolio_returns)
    st.subheader("Strategy-Level Backtest Summary")
    st.dataframe(table, use_container_width=True)


@st.cache_data(ttl=3600, show_spinner=False)
def backtest_recommendation_framework(
    close,
    rebalance_frequency,
    signal_window,
    initial_value,
    benchmark_symbol="SPY",
    transaction_cost=0.0005,
    ma_symbol="SPY",
):
    signal_history = build_daily_signal_history(close, ma_symbol=ma_symbol)
    daily_returns = close.pct_change().dropna()
    if signal_history.empty or daily_returns.empty or benchmark_symbol not in daily_returns.columns:
        return pd.DataFrame(), signal_history, pd.DataFrame(), {}

    rebalance_dates = pick_rebalance_dates(signal_history, rebalance_frequency)
    portfolio_value = initial_value
    current_allocation = "Cash"
    portfolio_rows = []
    rebalance_rows = []

    for idx, rebalance_date in enumerate(rebalance_dates):
        actual_rebalance_date = latest_valid_date(signal_history, rebalance_date)
        if actual_rebalance_date is None:
            continue
        window_df = signal_history.loc[:actual_rebalance_date].tail(signal_window)
        if window_df.empty:
            continue

        average_score = window_df["Combined Signal Score"].mean()
        counts = get_signal_counts(window_df)
        if average_score >= 0.3:
            recommended_allocation = get_most_frequent_selected_asset(window_df, close.columns)
        elif average_score <= -0.3:
            recommended_allocation = get_defensive_allocation(close.columns)
        else:
            recommended_allocation = current_allocation

        trade_date = get_next_trade_date(daily_returns, actual_rebalance_date)
        next_rebalance_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else daily_returns.index[-1]
        period_mask = (daily_returns.index >= trade_date) & (daily_returns.index <= next_rebalance_date)
        period_returns = daily_returns.loc[period_mask]
        if period_returns.empty:
            continue

        turnover = 0 if recommended_allocation == current_allocation else 1
        for day_index, (date, return_row) in enumerate(period_returns.iterrows()):
            if recommended_allocation in return_row.index:
                day_return = return_row[recommended_allocation]
                portfolio_weight = 1.0
                cash_balance = 0.0
            else:
                day_return = 0.0
                portfolio_weight = 0.0
                cash_balance = portfolio_value

            if day_index == 0 and turnover > 0:
                day_return -= transaction_cost

            portfolio_value *= 1 + day_return
            signal_row = signal_history.loc[actual_rebalance_date]
            portfolio_rows.append(
                {
                    "Date": date,
                    "Combined Signal": signal_row["Combined Signal"],
                    "Held Asset": recommended_allocation,
                    "Portfolio Value": portfolio_value,
                    "Daily Return": day_return,
                    "Cumulative Return": portfolio_value / initial_value - 1,
                    "Cash Balance": cash_balance,
                    "Portfolio Weight": portfolio_weight,
                    "Cash / Allocation": "Cash" if recommended_allocation == "Cash" else f"100% {recommended_allocation}",
                }
            )

        current_allocation = recommended_allocation
        rebalance_rows.append(
            {
                "Rebalance Date": actual_rebalance_date,
                "Lookback Window": signal_window,
                "Average Signal Score": average_score,
                "BUY Count": counts["BUY"],
                "HOLD Count": counts["HOLD"],
                "RISK-OFF Count": counts["RISK-OFF"],
                "Recommended Allocation": recommended_allocation,
                "Executed Trade Date": trade_date,
                "Portfolio Value": portfolio_value,
            }
        )

    portfolio_df = pd.DataFrame(portfolio_rows)
    rebalance_history = pd.DataFrame(rebalance_rows)
    if portfolio_df.empty:
        return portfolio_df, signal_history, rebalance_history, {}

    portfolio_df = portfolio_df.drop_duplicates(subset=["Date"], keep="last").set_index("Date")
    benchmark_returns = daily_returns[benchmark_symbol].reindex(portfolio_df.index).fillna(0)
    portfolio_df["Benchmark Value"] = initial_value * (1 + benchmark_returns).cumprod()
    portfolio_df["Excess Return vs Benchmark"] = (
        portfolio_df["Portfolio Value"] / initial_value - 1
    ) - (portfolio_df["Benchmark Value"] / initial_value - 1)

    portfolio_returns = portfolio_df["Portfolio Value"].pct_change().dropna()
    portfolio_equity = portfolio_df["Portfolio Value"] / initial_value
    metrics = calculate_performance_metrics(portfolio_equity, portfolio_returns, len(rebalance_history))
    metrics["annualized_return"] = metrics["cagr"]
    metrics["win_rate"] = calculate_win_rate(portfolio_returns)
    metrics["benchmark_return"] = portfolio_df["Benchmark Value"].iloc[-1] / initial_value - 1
    metrics["current_allocation"] = portfolio_df["Held Asset"].iloc[-1]
    latest_signal_row = signal_history.iloc[-1]
    metrics["latest_recommendation"] = latest_signal_row["Combined Signal"]
    recent_window = signal_history.tail(signal_window)
    metrics["recent_counts"] = get_signal_counts(recent_window)
    return portfolio_df, signal_history, rebalance_history, metrics


def render_recommendation_framework_backtest(portfolio_df, signal_history, rebalance_history, metrics, benchmark_symbol):
    st.subheader("Actual Recommended Strategy")
    st.markdown(
        "**Actual Recommended Strategy: 10-Strategy Signal Aggregation with Scheduled Rebalancing**\n\n"
        "This section backtests the actual recommendation process, not just the individual strategies. The recommended "
        "strategy is the Recommendation Framework, which aggregates daily signals from all 10 strategy models into a "
        "combined signal score. Daily signals are generated and stored for monitoring, but portfolio trades only occur "
        "on the selected rebalance schedule, such as daily, weekly, or monthly. On each rebalance date, the framework "
        "looks back over the selected signal aggregation window and uses the average combined signal score to decide "
        "whether to allocate to a risk-on ETF, hold the current allocation, or move to cash / defensive assets.\n\n"
        "The recommended strategy is used only on the selected rebalance dates. It uses the previous N trading days of "
        "combined signal history to make the allocation decision.\n\n"
        "To avoid look-ahead bias, signals are generated using only data available up to that date. Rebalance decisions "
        "use only prior signal history, and trades are assumed to execute on the next trading day."
    )
    st.subheader("Recommendation Framework Backtest")
    st.write(
        "Daily signals are generated for monitoring and stored as signal history. Portfolio trades only occur on the selected rebalance schedule. "
        "The recommendation framework uses recent signal history, not just a single day's signal."
    )
    if portfolio_df.empty:
        st.warning("Recommendation framework backtest is unavailable for the current data.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Annualized Return", f"{metrics['annualized_return'] * 100:.2f}%")
    c2.metric("Cumulative Return", f"{metrics['total_return'] * 100:.2f}%")
    c3.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Win Rate", f"{metrics['win_rate'] * 100:.2f}%")
    c6.metric("Number of Rebalances", str(metrics["number_rebalances"]))
    c7.metric("Current Allocation", metrics["current_allocation"])
    c8.metric("Latest Recommendation", metrics["latest_recommendation"])

    recent_counts = metrics["recent_counts"]
    st.write(
        f"**Recent signal breakdown:** BUY {recent_counts['BUY']} | HOLD {recent_counts['HOLD']} | RISK-OFF {recent_counts['RISK-OFF']}"
    )
    st.write(
        "The hypothetical $1MM portfolio is driven by the Recommendation Framework. It does not trade every daily signal. "
        "Instead, it updates allocation only on the selected rebalance schedule using the recent combined signal history."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["Portfolio Value"], name="Recommendation Portfolio", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=portfolio_df.index, y=portfolio_df["Benchmark Value"], name=f"{benchmark_symbol} Benchmark", line=dict(width=2, dash="dash")))
    fig.update_layout(
        title="Recommendation Framework Portfolio vs Benchmark",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Signal History")
    st.write(
        "Daily signals are generated for monitoring and stored as signal history. They do not necessarily trigger daily trades. "
        "Actual portfolio trades only occur based on the selected rebalance frequency. The rebalance decision uses the recent "
        "signal history, not just a single day’s signal."
    )
    st.dataframe(
        signal_history.reset_index().tail(120).style.format({"Combined Signal Score": "{:.2f}"}),
        use_container_width=True,
    )

    st.subheader("Rebalance History")
    st.write(
        "Daily signals are generated for monitoring and stored as signal history. They do not necessarily trigger daily trades. "
        "Actual portfolio trades only occur based on the selected rebalance frequency. The rebalance decision uses the recent "
        "signal history, not just a single day’s signal."
    )
    if rebalance_history.empty:
        st.warning("No rebalance history is available.")
    else:
        st.dataframe(
            rebalance_history.tail(60).style.format(
                {
                    "Average Signal Score": "{:.2f}",
                    "Portfolio Value": "${:,.2f}",
                }
            ),
            use_container_width=True,
        )


def render_strategy_results(result):
    if "signal" in result:
        render_signal_card(result["signal"])

    strategy_return = result["metrics"]["total_return"] * 100
    spy_return = benchmark_total_return(result) * 100
    st.write(f"**Backtest performance vs SPY:** {strategy_return:.2f}% vs {spy_return:.2f}%")

    eq_fig = go.Figure()
    eq_fig.add_trace(
        go.Scatter(
            x=result["strategy_equity"].index,
            y=result["strategy_equity"],
            name=result["strategy_name"],
            line=dict(width=2),
        )
    )
    eq_fig.add_trace(
        go.Scatter(
            x=result["benchmark_equity"].index,
            y=result["benchmark_equity"],
            name=f"{result['benchmark_symbol']} Buy & Hold",
            line=dict(width=2, dash="dash"),
        )
    )
    eq_fig.update_layout(
        title=f"{result['strategy_name']} Equity Curve",
        xaxis_title="Date",
        yaxis_title="Growth of 1.0",
        template="plotly_white",
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    returns_chart_df = result["monthly_returns"].reset_index()
    returns_chart_df = returns_chart_df.rename(columns={returns_chart_df.columns[0]: "Date"})
    returns_fig = px.bar(
        returns_chart_df,
        x="Date",
        y=result["monthly_returns"].columns,
        barmode="group",
        title=f"{result['strategy_name']} Monthly Returns",
    )
    returns_fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Monthly Return (%)",
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=30),
        legend_title_text="Series",
    )
    st.plotly_chart(returns_fig, use_container_width=True)

    metrics = result["metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{metrics['total_return'] * 100:.2f}%")
    m2.metric("CAGR", f"{metrics['cagr'] * 100:.2f}%")
    m3.metric("Annualized Volatility", f"{metrics['volatility'] * 100:.2f}%")
    m4.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    m5, m6 = st.columns(2)
    m5.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    m6.metric("Number of Monthly Rebalances", str(metrics["number_rebalances"]))

    if result["transaction_cost"] > 0:
        st.write(f"Average monthly turnover: {metrics['avg_turnover'] * 100:.2f}%")
        st.write(f"Transaction cost per trade: {result['transaction_cost'] * 10000:.2f} bps")

    st.divider()
    st.subheader("Current Holdings")
    if result["current_holdings"]:
        current_weights = result["monthly_holdings"][-1]["Holdings_str"]
        st.write(current_weights)
    else:
        st.write("Cash")

    if result["latest_scores"] is not None:
        st.subheader("Latest Ranking / Signal Table")
        st.dataframe(result["latest_scores"].style.format(precision=2), use_container_width=True)

    st.subheader("Monthly Holdings Table")
    if result["monthly_holdings"]:
        holdings_df = pd.DataFrame(
            [
                {"Signal Date": h["Date"].strftime("%Y-%m-%d"), "Holdings": h["Holdings_str"]}
                for h in result["monthly_holdings"]
            ]
        )
        st.dataframe(holdings_df.tail(24), use_container_width=True)

    st.subheader("Trade Log")
    st.write(
        "Signal Date is when the strategy generates the signal. Trade Date is the next trading day when the position becomes active."
    )
    if result["trade_log"]:
        st.dataframe(pd.DataFrame(result["trade_log"]).tail(24), use_container_width=True)

    st.subheader("Monthly Returns Table")
    st.dataframe(result["monthly_returns"].style.format("{:.2f}%"), use_container_width=True)

    st.subheader("Yearly Returns Table")
    st.dataframe(result["yearly_returns"].style.format("{:.2f}%"), use_container_width=True)

    st.subheader("Rule-Based Summary")
    st.markdown(build_strategy_summary(result))


def render_strategy_tab(result):
    render_strategy_results(result)


def build_strategy_comparison(results):
    rows = []
    for name, result in results.items():
        if result is None:
            continue
        metrics = result["metrics"]
        rows.append(
            {
                "Strategy": name,
                "Total Return": metrics["total_return"] * 100,
                "CAGR": metrics["cagr"] * 100,
                "Volatility": metrics["volatility"] * 100,
                "Sharpe": metrics["sharpe"],
                "Max Drawdown": metrics["max_drawdown"] * 100,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("Strategy")


def compute_performance(close):
    performance = {}
    for label, days in PERFORMANCE_WINDOWS.items():
        if len(close) < days + 1:
            performance[label] = np.nan
            continue
        performance[label] = (close.iloc[-1] / close.shift(days).iloc[-1] - 1) * 100
    return pd.DataFrame(performance, index=close.columns)


def compute_volatility(returns):
    vol = returns.rolling(window=22).std() * np.sqrt(252) * 100
    return vol.iloc[-1]


def moving_average_signal(close):
    signal = {}
    for symbol in close.columns:
        prices = close[symbol].dropna()
        if len(prices) < 20:
            signal[symbol] = "Not enough data"
            continue
        ma20 = prices.rolling(window=20).mean()
        ma50 = prices.rolling(window=50).mean()
        if len(ma50.dropna()) == 0:
            signal[symbol] = "Waiting"
        elif ma20.iloc[-1] > ma50.iloc[-1]:
            signal[symbol] = "Bullish"
        elif ma20.iloc[-1] < ma50.iloc[-1]:
            signal[symbol] = "Bearish"
        else:
            signal[symbol] = "Neutral"
    return pd.Series(signal)


def format_percent(df):
    return df.round(2)


def ai_style_summary(performance, volatility, signal, window):
    window_returns = performance[window]
    best = window_returns.idxmax()
    worst = window_returns.idxmin()
    best_return = window_returns.loc[best]
    worst_return = window_returns.loc[worst]
    high_vol = volatility.sort_values(ascending=False).head(3).index.tolist()
    bullish = signal[signal == "Bullish"].index.tolist()
    bearish = signal[signal == "Bearish"].index.tolist()

    summary = [
        f"Over the {window} window, the top performer is **{best}** with a return of {best_return:.2f}%.",
        f"The weakest performer is **{worst}** with a return of {worst_return:.2f}%.",
        f"Highest recent volatility appears in: {', '.join(high_vol)}.",
    ]

    if bullish:
        summary.append(f"The moving average setup favors bullish momentum for: {', '.join(bullish)}.")
    if bearish:
        summary.append(f"Caution is suggested for: {', '.join(bearish)} due to bearish MA signals.")
    if not bullish and not bearish:
        summary.append("The market is mixed and no strong MA signals are present.")

    return "\n\n".join(summary)


def main():
    st.title("ETF Dashboard")
    st.write("A simple ETF performance dashboard using Streamlit and yfinance.")

    st.sidebar.header("Settings")
    selected_tickers = []
    for category, tickers in ETF_UNIVERSE.items():
        selected = st.sidebar.multiselect(f"Select {category}", tickers, default=tickers[:2])
        selected_tickers.extend(selected)

    st.sidebar.markdown("---")
    st.sidebar.subheader("How selection works")
    st.sidebar.write(
        "Select tickers from each category. The dashboard will show charts and metrics only for the tickers you choose. "
        "If you select none in a category, that category tab will not appear."
    )

    selected_tickers = list(dict.fromkeys(selected_tickers))
    if not selected_tickers:
        st.warning("Please select at least one ticker from the sidebar.")
        return

    history_period = st.sidebar.selectbox("History period", list(PERIOD_LABELS.keys()), index=1, format_func=lambda x: PERIOD_LABELS[x])
    summary_window = st.sidebar.selectbox("Summary Performance Window", list(PERFORMANCE_WINDOWS.keys()), index=2)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtest Settings")
    all_etf_symbols = get_all_etf_symbols()
    ma_default_index = all_etf_symbols.index("SPY") if "SPY" in all_etf_symbols else 0
    ma_symbol = st.sidebar.selectbox("Trend Following ETF", all_etf_symbols, index=ma_default_index)
    selected_summary_strategy = st.sidebar.selectbox("Selected Strategy Summary", STRATEGY_TYPES, index=0)
    backtest_period_label = st.sidebar.selectbox("Backtest Date Range", list(BACKTEST_PERIODS.keys()), index=2)
    rebalance_frequency = st.sidebar.selectbox("Rebalance Frequency", REBALANCE_FREQUENCIES, index=2)
    signal_window = st.sidebar.selectbox("Signal Aggregation Window", SIGNAL_AGGREGATION_WINDOWS, index=2)
    initial_portfolio_value = st.sidebar.number_input("Initial Portfolio Value", min_value=10000, value=1000000, step=50000)
    benchmark_symbol = st.sidebar.selectbox("Benchmark", all_etf_symbols, index=ma_default_index)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=5, step=1)
    transaction_cost = transaction_cost_bps / 10000
    allocation_method = st.sidebar.selectbox("Alpha Allocation Method", ["Equal Weight", "Volatility Scaled"], index=0)
    max_alpha_allocation = st.sidebar.slider("Max Allocation per Alpha", min_value=0.05, max_value=0.25, value=0.10, step=0.01)
    st.sidebar.markdown("---")
    st.sidebar.write("**ETF universe includes:**")
    for category, tickers in ETF_UNIVERSE.items():
        st.sidebar.write(f"- **{category}**: {', '.join(tickers)}")

    backtest_end = pd.Timestamp.today().normalize()
    backtest_start = backtest_end - pd.DateOffset(years=BACKTEST_PERIODS[backtest_period_label])
    required_strategy_symbols = {"SPY", "SHY", ma_symbol, benchmark_symbol} | set(DEFENSIVE_ETFS)
    alpha_symbols = tuple(sorted(set(selected_tickers) | required_strategy_symbols))

    close = load_data(selected_tickers, history_period)
    if close.empty:
        st.error("Failed to load data. Please try again or select fewer tickers.")
        return

    with st.spinner("Loading multi-alpha market data..."):
        alpha_ohlcv = load_ohlcv_data(
            alpha_symbols,
            backtest_start.strftime("%Y-%m-%d"),
            backtest_end.strftime("%Y-%m-%d"),
        )
    if not alpha_ohlcv or alpha_ohlcv["close"].empty:
        st.error("Failed to load multi-alpha OHLCV data.")
        return

    with st.spinner("Running multi-alpha portfolio backtest..."):
        alpha_results, alpha_portfolio_returns, alpha_correlations, alpha_portfolio_metrics = run_multi_alpha_backtest(
            alpha_ohlcv,
            initial_capital=initial_portfolio_value,
            max_alpha_allocation=max_alpha_allocation,
            allocation_method=allocation_method,
        )

    returns = compute_returns(close)
    performance = compute_performance(close)
    volatility = compute_volatility(returns)
    signal = moving_average_signal(close)

    top_perf = performance[summary_window].idxmax()
    worst_perf = performance[summary_window].idxmin()
    highest_vol = volatility.sort_values(ascending=False).index[0]
    bullish_count = len(signal[signal == "Bullish"])

    main_tabs = st.tabs(
        [
            "Portfolio Overview",
            "Alpha Live Monitoring",
            "Alpha Correlation",
            "Backtesting",
            "Market Overview",
            "Category Charts",
            "Strategy Testing",
        ]
    )
    portfolio_tab, alpha_monitoring_tab, alpha_correlation_tab, alpha_backtesting_tab, overview_tab, category_tab, strategy_tab = main_tabs

    with portfolio_tab:
        render_portfolio_overview(alpha_results, alpha_portfolio_returns, alpha_portfolio_metrics)

    with alpha_monitoring_tab:
        render_alpha_live_monitoring(alpha_results, alpha_portfolio_returns)

    with alpha_correlation_tab:
        render_alpha_correlation(alpha_results, alpha_correlations, alpha_portfolio_returns)

    with alpha_backtesting_tab:
        render_alpha_backtesting(alpha_results, alpha_portfolio_returns, alpha_portfolio_metrics)

    with overview_tab:
        st.subheader("Overview Metrics")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Top Performer", top_perf)
        stat_col2.metric("Weakest Performer", worst_perf)
        stat_col3.metric("Highest Volatility", highest_vol)
        stat_col4.metric("Bullish MA Signals", str(bullish_count))

        st.subheader("Performance Table")
        performance_display = format_percent(performance)
        st.dataframe(performance_display.style.format("{:.2f}%"), use_container_width=True)

        st.subheader("Volatility Table")
        vol_df = volatility.to_frame(name="Annualized Volatility (%)")
        st.dataframe(vol_df.round(2), use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = returns.corr()
        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="RdYlGn",
                zmin=-1,
                zmax=1,
            )
        )
        heatmap.update_layout(title="Correlation Heatmap", xaxis_title="Ticker", yaxis_title="Ticker")
        st.plotly_chart(heatmap, use_container_width=True)

        st.subheader("Moving Average Signal")
        signal_df = signal.to_frame(name="MA Signal")
        st.dataframe(signal_df, use_container_width=True)

        st.subheader("Market Summary")
        summary_text = ai_style_summary(performance, volatility, signal, summary_window)
        st.markdown(summary_text)

    with category_tab:
        st.subheader("Price Charts by Category")
        category_names = [category for category, tickers in ETF_UNIVERSE.items() if any(t in selected_tickers for t in tickers)]
        category_tabs = st.tabs(category_names)
        for category, tab in zip(category_names, category_tabs):
            with tab:
                category_symbols = [symbol for symbol in ETF_UNIVERSE[category] if symbol in selected_tickers]
                if not category_symbols:
                    st.write("No tickers selected for this category.")
                    continue
                st.write(f"### {category}")
                st.write(f"Selected tickers: {', '.join(category_symbols)}")
                st.divider()
                chart_cols = st.columns(2)
                for index, symbol in enumerate(category_symbols):
                    symbol_close = close[[symbol]].dropna()
                    if symbol_close.empty:
                        chart_cols[index % 2].write(f"Failed to load data for {symbol}.")
                        continue
                    fig = build_line_chart(symbol_close, symbol, f"{symbol} Price Trend ({history_period})", "Adjusted Close")
                    chart_cols[index % 2].plotly_chart(fig, use_container_width=True)

                cat_returns = returns[category_symbols]
                if not cat_returns.empty:
                    ret_fig = build_line_chart(cat_returns, cat_returns.columns, f"{category} Daily Returns", "Daily Return")
                    st.plotly_chart(ret_fig, use_container_width=True)

                cat_perf = format_percent(performance.loc[category_symbols])
                st.write(f"**{category} Performance**")
                st.dataframe(cat_perf.style.format("{:.2f}%"), use_container_width=True)
                st.caption("Returns are daily percentage changes and performance is shown as percent change over each selected window.")

    with strategy_tab:
        st.header("Strategy Backtest")
        st.write("Each strategy uses the same cached backtest price data and avoids look-ahead bias by trading after the signal date.")
        st.divider()

        backtest_close = alpha_ohlcv["close"].copy()

        if backtest_close.empty:
            st.error("Unable to load backtest data. Please try again later.")
        else:
            st.caption(
                f"Backtest data reused from the shared cached OHLCV download for {len(backtest_close.columns)} ETFs from "
                f"{backtest_close.index.min().date()} to {backtest_close.index.max().date()}."
            )
            render_workflow_diagram()

            with st.spinner("Calculating 10 strategy backtests..."):
                comparison_results = run_all_strategy_backtests(
                    backtest_close,
                    transaction_cost,
                    ma_symbol=ma_symbol,
                )

            available_results = {name: result for name, result in comparison_results.items() if result is not None}
            strategy_signals = {name: result["signal"] for name, result in available_results.items()}
            combined_signal = combine_strategy_signals(strategy_signals)

            if selected_summary_strategy in available_results:
                render_top_strategy_summary(available_results[selected_summary_strategy])

            render_combined_signal_section(combined_signal, comparison_results)
            with st.spinner("Backtesting recommendation framework..."):
                recommendation_portfolio, signal_history, rebalance_history, recommendation_metrics = backtest_recommendation_framework(
                    backtest_close,
                    rebalance_frequency,
                    signal_window,
                    initial_portfolio_value,
                    benchmark_symbol=benchmark_symbol,
                    transaction_cost=transaction_cost,
                    ma_symbol=ma_symbol,
                )
            render_recommendation_framework_backtest(
                recommendation_portfolio,
                signal_history,
                rebalance_history,
                recommendation_metrics,
                benchmark_symbol,
            )
            st.divider()

            strategy_tabs = st.tabs(STRATEGY_TYPES)
            for comparison_strategy, tab in zip(STRATEGY_TYPES, strategy_tabs):
                with tab:
                    result = comparison_results[comparison_strategy]
                    if result is None:
                        st.error("Backtest failed due to insufficient historical data for this strategy.")
                    elif comparison_strategy != selected_summary_strategy:
                        render_signal_card(result["signal"])
                        strategy_return = result["metrics"]["total_return"] * 100
                        spy_return = benchmark_total_return(result) * 100
                        st.write(f"**Backtest performance vs SPY:** {strategy_return:.2f}% vs {spy_return:.2f}%")
                        st.info("Select this strategy in the sidebar Selected Strategy Summary control to render its full charts and tables.")
                    else:
                        render_strategy_tab(result)

            st.divider()
            st.subheader("Strategy Comparison")
            comparison_df = build_strategy_comparison(comparison_results)
            if comparison_df.empty:
                st.warning("Strategy comparison is unavailable for the current data.")
            else:
                st.dataframe(comparison_df.style.format("{:.2f}"), use_container_width=True)

    st.write("---")
    st.write("Built with Streamlit, yfinance, pandas, numpy, and plotly.")
    st.caption("Data source: Yahoo Finance via yfinance. No API key is required.")


if __name__ == "__main__":
    main()
