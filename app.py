import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ETF Dashboard", page_icon="📈", layout="wide")

ETF_UNIVERSE = {
    "Broad Market": ["SPY", "QQQ", "IWM", "DIA"],
    "Sectors": ["XLK", "XLF", "XLE", "XLV", "XLP", "XLY", "XLI", "XLU"],
    "Asset Classes": ["TLT", "IEF", "SHY", "BIL", "AGG", "TIP", "HYG", "LQD", "GLD", "SLV", "DBC", "USO"],
    "Geography": ["EFA", "EEM"],
    "Factor ETFs": ["USMV", "SPLV"],
}

STOCK_UNIVERSE = {
    "Technology / Growth": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "AMD"],
    "Financials": ["JPM", "BAC", "GS", "MS", "V", "MA", "BLK"],
    "Energy / Inflation": ["XOM", "CVX", "COP", "SLB"],
    "Defensive": ["JNJ", "PG", "KO", "PEP", "WMT", "COST", "MCD"],
    "Industrials": ["CAT", "GE", "HON", "BA", "LMT"],
    "Consumer / Communication": ["NFLX", "DIS", "TSLA", "HD", "NKE"],
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
]

DEFENSIVE_ETFS = ["TLT", "IEF", "GLD", "SHY", "XLV", "XLU"]
RISK_ON_ETFS = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLY", "XLI", "XLC", "EFA", "EEM"]
INITIAL_CAPITAL = 1_000_000

FACTOR_PROXIES = {
    "Equity / Growth": ["SPY", "QQQ"],
    "Small Cap": ["IWM"],
    "Rates / Duration": ["TLT", "IEF"],
    "Defensive / Cash": ["SHY"],
    "Credit": ["HYG", "LQD"],
    "Commodities / Inflation": ["DBC", "GLD"],
    "International Developed": ["EFA"],
    "Emerging Markets": ["EEM"],
}

BACKTEST_PERIODS = {
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
}

PORTFOLIO_MODES = ["ETF-only", "Stock-only", "ETF + Stock"]

REGIME_NAMES = [
    "Risk-On Bull Market",
    "Risk-Off Bear Market",
    "High Volatility / Crisis",
    "Inflation / Commodity-Led Market",
    "Falling Rates / Bond Rally",
    "Sideways / Choppy Market",
    "Growth-Led Market",
    "Value / Financials-Led Market",
]

SCORE_LABELS = {"high": 1.0, "medium": 0.6, "low": 0.25, "not suitable": 0.0}

STRATEGY_REGIME_SUITABILITY = {
    "Momentum Rotation": {
        "Risk-On Bull Market": "high",
        "Growth-Led Market": "high",
        "Value / Financials-Led Market": "medium",
        "Sideways / Choppy Market": "medium",
        "Risk-Off Bear Market": "low",
        "High Volatility / Crisis": "low",
    },
    "Moving Average Trend Following": {
        "Risk-On Bull Market": "medium",
        "Risk-Off Bear Market": "high",
        "High Volatility / Crisis": "high",
        "Sideways / Choppy Market": "low",
        "Falling Rates / Bond Rally": "medium",
    },
    "Dual Momentum": {
        "Risk-On Bull Market": "high",
        "Growth-Led Market": "high",
        "Risk-Off Bear Market": "medium",
        "Falling Rates / Bond Rally": "medium",
        "Sideways / Choppy Market": "medium",
    },
    "Low Volatility Rotation": {
        "Risk-Off Bear Market": "high",
        "High Volatility / Crisis": "high",
        "Sideways / Choppy Market": "medium",
        "Risk-On Bull Market": "medium",
    },
    "Defensive Rotation Strategy": {
        "Risk-Off Bear Market": "high",
        "High Volatility / Crisis": "high",
        "Falling Rates / Bond Rally": "medium",
        "Risk-On Bull Market": "low",
    },
    "Risk-On / Risk-Off Regime Strategy": {
        "Risk-Off Bear Market": "high",
        "High Volatility / Crisis": "high",
        "Risk-On Bull Market": "high",
        "Sideways / Choppy Market": "medium",
    },
    "Breakout Strategy": {
        "Risk-On Bull Market": "high",
        "Growth-Led Market": "high",
        "High Volatility / Crisis": "medium",
        "Sideways / Choppy Market": "low",
    },
    "Mean Reversion Strategy": {
        "Sideways / Choppy Market": "high",
        "Risk-On Bull Market": "medium",
        "High Volatility / Crisis": "low",
        "Risk-Off Bear Market": "low",
    },
    "Volatility Target Strategy": {
        "High Volatility / Crisis": "high",
        "Risk-Off Bear Market": "high",
        "Sideways / Choppy Market": "medium",
        "Risk-On Bull Market": "medium",
    },
}

REGIME_STOCK_PREFERENCES = {
    "Growth-Led Market": ["Technology / Growth", "Consumer / Communication"],
    "Risk-On Bull Market": ["Technology / Growth", "Industrials", "Consumer / Communication", "Financials"],
    "Risk-Off Bear Market": ["Defensive"],
    "High Volatility / Crisis": ["Defensive"],
    "Inflation / Commodity-Led Market": ["Energy / Inflation"],
    "Falling Rates / Bond Rally": ["Technology / Growth", "Defensive"],
    "Value / Financials-Led Market": ["Financials", "Industrials"],
    "Sideways / Choppy Market": ["Defensive", "Consumer / Communication"],
}

RISK_LIMITS = {
    "max_position_weight": 0.10,
    "max_strategy_weight": 0.25,
    "max_sector_exposure": 0.35,
    "max_drawdown_warning": -0.10,
    "severe_drawdown_warning": -0.20,
    "risk_off_cash_minimum": 0.15,
}


def get_all_etf_symbols():
    symbols = []
    for tickers in ETF_UNIVERSE.values():
        symbols.extend(tickers)
    return list(dict.fromkeys(symbols))


def get_all_stock_symbols():
    symbols = []
    for tickers in STOCK_UNIVERSE.values():
        symbols.extend(tickers)
    return list(dict.fromkeys(symbols))


def get_stock_category_map():
    return {ticker: category for category, tickers in STOCK_UNIVERSE.items() for ticker in tickers}


def get_etf_category_map():
    return {ticker: category for category, tickers in ETF_UNIVERSE.items() for ticker in tickers}


def clean_price_data(close, max_missing_ratio=0.15):
    if close.empty:
        return close
    missing_ratio = close.isna().mean()
    keep_columns = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()
    close = close[keep_columns].ffill(limit=5).dropna(how="all")
    return close


def extract_close_prices(df, symbols):
    close = df["Close"].copy()
    if isinstance(close, pd.Series):
        close = close.to_frame(name=symbols[0])
    return clean_price_data(close.dropna(how="all"))


def extract_price_field(df, field, symbols):
    if field not in df:
        return pd.DataFrame()
    values = df[field].copy()
    if isinstance(values, pd.Series):
        values = values.to_frame(name=symbols[0])
    return values.dropna(how="all")


def calculate_indicators(close):
    return {
        "returns_1m": close.pct_change(21, fill_method=None),
        "returns_3m": close.pct_change(63, fill_method=None),
        "returns_6m": close.pct_change(126, fill_method=None),
        "returns_12m": close.pct_change(252, fill_method=None),
        "ma50": close.rolling(50).mean(),
        "ma200": close.rolling(200).mean(),
        "rolling_high_55": close.rolling(55).max().shift(1),
        "rolling_low_20": close.rolling(20).min().shift(1),
        "rsi14": calculate_rsi(close),
        "vol20": close.pct_change(fill_method=None).rolling(20).std() * np.sqrt(252),
        "vol60": close.pct_change(fill_method=None).rolling(60).std() * np.sqrt(252),
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


def available_symbols(close, symbols):
    return [symbol for symbol in symbols if symbol in close.columns]


def latest_valid_date(close, signal_date):
    valid_dates = close.index[close.index <= signal_date]
    if len(valid_dates) == 0:
        return None
    return valid_dates[-1]


@st.cache_data(ttl=3600)
def load_backtest_ohlcv_data(symbols, start_date, end_date):
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
        "open": clean_price_data(extract_price_field(df, "Open", symbols)),
        "high": clean_price_data(extract_price_field(df, "High", symbols)),
        "low": clean_price_data(extract_price_field(df, "Low", symbols)),
        "close": clean_price_data(extract_price_field(df, "Close", symbols)),
        "volume": clean_price_data(extract_price_field(df, "Volume", symbols)),
    }


@st.cache_data(ttl=3600)
def load_close_data(symbols, start_date, end_date):
    symbols = tuple(sorted(set(symbols)))
    if not symbols:
        return pd.DataFrame()
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


def align_market_data(etf_close, stock_close, benchmark_symbol="SPY"):
    common_index = etf_close.index
    if not stock_close.empty:
        common_index = common_index.intersection(stock_close.index)
    etf_close = etf_close.reindex(common_index).ffill()
    stock_close = stock_close.reindex(common_index).ffill() if not stock_close.empty else stock_close
    benchmark = etf_close[[benchmark_symbol]].dropna() if benchmark_symbol in etf_close.columns else pd.DataFrame()
    return etf_close, stock_close, benchmark



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
    trading_costs=None,
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

    trading_costs = trading_costs if trading_costs is not None else pd.Series(0.0, index=strategy_returns.index)
    trading_costs = trading_costs.reindex(strategy_returns.index).fillna(0.0)
    gross_returns = strategy_returns + trading_costs

    daily_returns = close.pct_change(fill_method=None).dropna()
    benchmark_returns = daily_returns[benchmark_symbol].reindex(strategy_returns.index).dropna()
    common_index = strategy_returns.index.intersection(benchmark_returns.index)
    if common_index.empty:
        return None

    strategy_returns = strategy_returns.loc[common_index]
    gross_returns = gross_returns.loc[common_index]
    trading_costs = trading_costs.loc[common_index]
    benchmark_returns = benchmark_returns.loc[common_index]
    strategy_equity = (1 + strategy_returns).cumprod()
    gross_equity = (1 + gross_returns).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()

    monthly_strategy = strategy_equity.resample("ME").last().pct_change(fill_method=None).dropna()
    monthly_benchmark = benchmark_equity.resample("ME").last().pct_change(fill_method=None).dropna()
    monthly_returns_df = pd.DataFrame(
        {strategy_name: monthly_strategy, f"{benchmark_symbol} Buy & Hold": monthly_benchmark}
    ).dropna() * 100

    yearly_strategy = strategy_equity.resample("YE").last().pct_change(fill_method=None).dropna()
    yearly_benchmark = benchmark_equity.resample("YE").last().pct_change(fill_method=None).dropna()
    yearly_returns_df = pd.DataFrame(
        {strategy_name: yearly_strategy, f"{benchmark_symbol} Buy & Hold": yearly_benchmark}
    ).dropna() * 100

    metrics = calculate_backtest_metrics(strategy_equity, strategy_returns, len(rebalances))
    gross_metrics = calculate_backtest_metrics(gross_equity, gross_returns, len(rebalances))
    metrics["avg_turnover"] = np.mean(turnover_list) if turnover_list else 0.0
    metrics["total_turnover"] = np.sum(turnover_list) if turnover_list else 0.0
    metrics["transaction_cost_drag"] = gross_equity.iloc[-1] - strategy_equity.iloc[-1]
    metrics["net_return"] = metrics["total_return"]
    metrics["gross_return"] = gross_metrics["total_return"]

    current_holdings = monthly_holdings[-1]["Holdings"] if monthly_holdings else []

    return {
        "strategy_name": strategy_name,
        "benchmark_symbol": benchmark_symbol,
        "strategy_returns": strategy_returns,
        "gross_returns": gross_returns,
        "trading_costs": trading_costs,
        "benchmark_returns": benchmark_returns,
        "strategy_equity": strategy_equity,
        "gross_equity": gross_equity,
        "benchmark_equity": benchmark_equity,
        "monthly_returns": monthly_returns_df,
        "yearly_returns": yearly_returns_df,
        "metrics": metrics,
        "gross_metrics": gross_metrics,
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

    return make_signal(strategy_type, signal_date, "N/A", "HOLD", 0, "Unknown strategy type.", 0)


def run_momentum_backtest(close, top_n=3, lookback_months=3, transaction_cost=0.0005, benchmark_symbol="SPY"):
    monthly_prices = close.resample("ME").last()
    trailing_returns = monthly_prices.pct_change(lookback_months, fill_method=None)
    daily_returns = close.pct_change(fill_method=None).dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    trading_costs = pd.Series(0.0, index=daily_returns.index)
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
        turnover = (new_weights - previous_weights).abs().sum()
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
            trade_cost = turnover * transaction_cost
            tranche_returns.iloc[0] -= trade_cost
            trading_costs.loc[first_day] = trade_cost
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
        trading_costs=trading_costs,
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
    daily_returns = symbol_close.pct_change(fill_method=None).dropna()
    ma200 = symbol_close[symbol].rolling(ma_window).mean()
    signal = (symbol_close[symbol] > ma200).astype(float)
    strategy_exposure = signal.shift(1).reindex(daily_returns.index).fillna(0.0)
    exposure_turnover = strategy_exposure.diff().abs().fillna(strategy_exposure.abs())
    trading_costs = exposure_turnover * transaction_cost
    strategy_returns = daily_returns[symbol] * strategy_exposure - trading_costs

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
        turnover_list=exposure_turnover.resample("ME").sum().dropna().tolist(),
        trading_costs=trading_costs,
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
    trailing_returns = monthly_prices.pct_change(lookback_months, fill_method=None)
    daily_returns = close.pct_change(fill_method=None).dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    trading_costs = pd.Series(0.0, index=daily_returns.index)
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

        turnover = (new_weights - previous_weights).abs().sum()
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
                trade_cost = turnover * transaction_cost
                tranche_returns.iloc[0] -= trade_cost
                trading_costs.loc[tranche_returns.index[0]] = trade_cost
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
        trading_costs=trading_costs,
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
    daily_returns = close.pct_change(fill_method=None).dropna()
    trailing_volatility = daily_returns.rolling(lookback_days).std() * np.sqrt(252)
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    trading_costs = pd.Series(0.0, index=daily_returns.index)
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
        turnover = (new_weights - previous_weights).abs().sum()
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
                trade_cost = turnover * transaction_cost
                tranche_returns.iloc[0] -= trade_cost
                trading_costs.loc[tranche_returns.index[0]] = trade_cost
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
        trading_costs=trading_costs,
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
    daily_returns = close.pct_change(fill_method=None).dropna()
    strategy_returns = pd.Series(index=daily_returns.index, dtype=float)
    trading_costs = pd.Series(0.0, index=daily_returns.index)
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
        turnover = (new_weights - previous_weights).abs().sum()
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
                trade_cost = turnover * transaction_cost
                tranche_returns.iloc[0] -= trade_cost
                trading_costs.loc[tranche_returns.index[0]] = trade_cost
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
        trading_costs=trading_costs,
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
def run_all_strategy_backtests(close, transaction_cost, ma_symbol="SPY", benchmark_symbol="SPY"):
    results = {}
    for strategy_type in STRATEGY_TYPES:
        results[strategy_type] = backtest_strategy(
            strategy_type,
            close,
            transaction_cost,
            ma_symbol=ma_symbol,
            benchmark_symbol=benchmark_symbol,
        )
    return results


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


def calculate_current_drawdown(equity):
    if equity.empty:
        return np.nan
    return (equity / equity.cummax() - 1).iloc[-1]


def get_strategy_returns_frame(results, return_key="strategy_returns"):
    return pd.DataFrame(
        {strategy_name: result[return_key] for strategy_name, result in results.items() if result is not None}
    ).dropna(how="all").fillna(0.0)


def calculate_var_es(returns, confidence=0.95):
    returns = returns.dropna()
    if returns.empty:
        return np.nan, np.nan
    var_level = np.percentile(returns, (1 - confidence) * 100)
    tail = returns[returns <= var_level]
    expected_shortfall = tail.mean() if not tail.empty else var_level
    return var_level, expected_shortfall


def calculate_strategy_correlations(results):
    returns_df = get_strategy_returns_frame(results)
    if returns_df.shape[1] < 2:
        return pd.DataFrame()
    return returns_df.corr()


def calculate_strategy_risk_contributions(results):
    returns_df = get_strategy_returns_frame(results)
    if returns_df.empty:
        return pd.Series(dtype=float)
    weights = pd.Series(1 / len(returns_df.columns), index=returns_df.columns)
    cov = returns_df.cov() * 252
    portfolio_variance = float(weights.T @ cov @ weights)
    if portfolio_variance <= 0:
        return (returns_df.std() * np.sqrt(252) * weights).fillna(0.0)
    portfolio_volatility = np.sqrt(portfolio_variance)
    marginal_contribution = cov @ weights
    component_contribution = weights * marginal_contribution / portfolio_volatility
    return component_contribution.fillna(0.0)


def get_max_strategy_correlation(strategy_name, correlations):
    if correlations.empty or strategy_name not in correlations.columns:
        return np.nan
    corr_row = correlations[strategy_name].drop(index=strategy_name, errors="ignore").abs()
    return corr_row.max() if not corr_row.empty else np.nan


def classify_strategy_status(result, max_correlation=np.nan, risk_contribution=np.nan):
    metrics = result["metrics"]
    current_drawdown = calculate_current_drawdown(result["strategy_equity"])
    if pd.notna(metrics["max_drawdown"]) and metrics["max_drawdown"] < -0.15:
        return "Pause"
    if pd.notna(metrics["max_drawdown"]) and metrics["max_drawdown"] < -0.10:
        return "Reduce"
    if pd.notna(current_drawdown) and current_drawdown < -0.08:
        return "Reduce"
    if pd.notna(metrics["sharpe"]) and metrics["sharpe"] < 0:
        return "Watch"
    if metrics.get("transaction_cost_drag", 0) > 0.03 and metrics.get("total_turnover", 0) > 5:
        return "Watch"
    if pd.notna(max_correlation) and max_correlation > 0.80:
        return "Watch"
    if pd.notna(risk_contribution) and risk_contribution > 0.35:
        return "Reduce"
    return "Healthy"


def recommended_action_from_status(status, max_correlation=np.nan, cost_drag=0.0):
    if status == "Pause":
        return "Pause"
    if status == "Reduce":
        return "Reduce"
    if pd.notna(max_correlation) and max_correlation > 0.80:
        return "Rebalance"
    if status == "Watch":
        return "Review"
    if cost_drag > 0.05:
        return "Review"
    return "Keep"


def build_risk_explanation(result, status, max_correlation=np.nan, risk_contribution=np.nan):
    metrics = result["metrics"]
    if status == "Pause":
        return "Max drawdown breached the pause limit."
    if status == "Reduce":
        if pd.notna(risk_contribution) and risk_contribution > 0.35:
            return "Risk contribution is concentrated relative to other sleeves."
        return "Drawdown is above the reduce threshold."
    if status == "Watch":
        if pd.notna(metrics["sharpe"]) and metrics["sharpe"] < 0:
            return "Sharpe ratio is negative."
        if pd.notna(max_correlation) and max_correlation > 0.80:
            return "Correlation with another strategy is above 0.80."
        return "Turnover or transaction cost drag is elevated."
    return "Risk limits are within the basic MVP thresholds."


def build_strategy_allocations(results):
    correlations = calculate_strategy_correlations(results)
    risk_contributions = calculate_strategy_risk_contributions(results)
    statuses = {}
    for strategy_name, result in results.items():
        statuses[strategy_name] = classify_strategy_status(
            result,
            get_max_strategy_correlation(strategy_name, correlations),
            risk_contributions.get(strategy_name, np.nan),
        )
    allocations = pd.Series(1 / max(1, len(results)), index=results.keys(), dtype=float)
    return allocations, statuses


def normalize_score(value, low=-0.10, high=0.20):
    if pd.isna(value):
        return 0.0
    return float(np.clip((value - low) / (high - low), 0, 1))


def get_strategy_regime_fit(strategy_name, regime_name):
    mapping = STRATEGY_REGIME_SUITABILITY.get(strategy_name, {})
    label = mapping.get(regime_name, "medium")
    return SCORE_LABELS.get(label, 0.6)


def calculate_dynamic_strategy_weights(
    strategy_results,
    current_regime,
    regime_fit_mapping=None,
    max_strategy_weight=0.25,
    min_strategy_weight=0.0,
):
    rows = []
    equal_weight = 1 / max(1, len(strategy_results))
    for strategy_name, result in strategy_results.items():
        metrics = result["metrics"]
        recent_returns = result["strategy_returns"].tail(63)
        recent_performance = (1 + recent_returns).prod() - 1 if not recent_returns.empty else np.nan
        recent_score = normalize_score(recent_performance, low=-0.08, high=0.12)
        sharpe = metrics.get("sharpe", np.nan)
        risk_adjusted_score = normalize_score(sharpe, low=-1.0, high=2.0)
        current_drawdown = calculate_current_drawdown(result["strategy_equity"])
        drawdown_penalty = min(1.0, abs(current_drawdown) / 0.20) if pd.notna(current_drawdown) and current_drawdown < 0 else 0.0
        risk_status = classify_strategy_status(result)
        risk_limit_penalty = 0.40 if risk_status == "Pause" else 0.25 if risk_status == "Reduce" else 0.10 if risk_status == "Watch" else 0.0
        signal_info = result.get("signal", {})
        signal = signal_info.get("Signal", "HOLD")
        selected_asset = signal_info.get("Selected Asset", "Cash")
        signal_validity = 0.0 if signal in ["CASH", "RISK-OFF", "SELL"] else 1.0 if signal == "BUY" else 0.6
        signal_penalty = 0.10 if signal in ["CASH", "RISK-OFF", "SELL"] else 0.0
        regime_fit = get_strategy_regime_fit(strategy_name, current_regime["regime_name"])
        final_score = (
            0.40 * regime_fit
            + 0.30 * recent_score
            + 0.20 * risk_adjusted_score
            + 0.10 * signal_validity
            - 0.10 * drawdown_penalty
            - risk_limit_penalty
            - signal_penalty
        )
        final_score = max(0.0, final_score)
        explanation = (
            f"{strategy_name} weight reflects {current_regime['regime_name']} regime fit ({regime_fit:.2f}), "
            f"63-day performance score ({recent_score:.2f}), Sharpe score ({risk_adjusted_score:.2f}), "
            f"drawdown penalty ({drawdown_penalty:.2f}), risk status ({risk_status}), and current signal ({signal})."
        )
        rows.append(
            {
                "Strategy": strategy_name,
                "Selected Asset": selected_asset,
                "Signal": signal,
                "Base Weight": equal_weight,
                "Regime Fit Score": regime_fit,
                "Recent Performance Score": recent_score,
                "Risk Adjusted Score": risk_adjusted_score,
                "Drawdown Penalty": drawdown_penalty,
                "Risk Limit Penalty": risk_limit_penalty + signal_penalty,
                "Final Score": final_score,
                "Risk Status": risk_status,
                "Explanation": explanation,
                "Reason": explanation,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty or table["Final Score"].sum() <= 0:
        table["Final Strategy Weight"] = 1 / max(1, len(table))
        table["Final Allocation"] = table["Final Strategy Weight"]
        table["Weight Change vs Equal Weight"] = 0.0
        table["Allocation Change"] = "Neutral"
        return table
    raw_weights = table["Final Score"] / table["Final Score"].sum()
    constrained = raw_weights.clip(lower=min_strategy_weight, upper=max_strategy_weight)
    if constrained.sum() <= 0:
        constrained = pd.Series(1 / len(table), index=table.index)
    table["Final Strategy Weight"] = constrained / constrained.sum()
    table["Final Allocation"] = table["Final Strategy Weight"]
    table["Weight Change vs Equal Weight"] = table["Final Strategy Weight"] - table["Base Weight"]
    table["Allocation Change"] = np.where(
        table["Weight Change vs Equal Weight"] > 0.002,
        "Increased",
        np.where(table["Weight Change vs Equal Weight"] < -0.002, "Decreased", "Neutral"),
    )
    return table


def build_regime_aware_strategy_allocation(results, regime, max_strategy_weight=0.25):
    return calculate_dynamic_strategy_weights(
        results,
        regime,
        STRATEGY_REGIME_SUITABILITY,
        max_strategy_weight=max_strategy_weight,
        min_strategy_weight=0.0,
    )


def adjust_asset_class_targets(regime_name, etf_weight, stock_weight, cash_weight, enable_cash=True, risk_off_cash_min=0.15):
    total = etf_weight + stock_weight + cash_weight
    etf_weight, stock_weight, cash_weight = etf_weight / total, stock_weight / total, cash_weight / total
    if not enable_cash:
        cash_weight = 0.0
    if regime_name in ["Risk-Off Bear Market", "High Volatility / Crisis"]:
        cash_weight = max(cash_weight, risk_off_cash_min)
        stock_weight *= 0.65
        etf_weight = max(0.0, 1 - stock_weight - cash_weight)
    elif regime_name in ["Growth-Led Market", "Risk-On Bull Market"]:
        cash_weight = min(cash_weight, 0.05 if enable_cash else 0.0)
        stock_weight = min(0.45, stock_weight * 1.15)
        etf_weight = max(0.0, 1 - stock_weight - cash_weight)
    elif regime_name in ["Inflation / Commodity-Led Market", "Falling Rates / Bond Rally"]:
        cash_weight = max(cash_weight, 0.05 if enable_cash else 0.0)
        etf_weight = min(0.75, etf_weight * 1.10)
        stock_weight = max(0.0, 1 - etf_weight - cash_weight)
    final_total = etf_weight + stock_weight + cash_weight
    return {
        "ETFs": etf_weight / final_total,
        "Stocks": stock_weight / final_total,
        "Cash / SHY / BIL": cash_weight / final_total,
    }


def get_current_holdings_text(result):
    if result.get("monthly_holdings"):
        return result["monthly_holdings"][-1]["Holdings_str"]
    if result.get("signal"):
        return result["signal"].get("Selected Asset", "N/A")
    return "N/A"


def build_strategy_monitoring_table(results, initial_capital=INITIAL_CAPITAL, strategy_allocation=None, etf_portfolio_weight=1.0):
    correlations = calculate_strategy_correlations(results)
    risk_contributions = calculate_strategy_risk_contributions(results)
    allocations, statuses = build_strategy_allocations(results)
    if strategy_allocation is not None and not strategy_allocation.empty:
        allocations = strategy_allocation.set_index("Strategy")["Final Strategy Weight"]
    rows = []
    for strategy_name, result in results.items():
        if result is None:
            continue
        metrics = result["metrics"]
        signal = result.get("signal", {})
        max_corr = get_max_strategy_correlation(strategy_name, correlations)
        risk_contribution = risk_contributions.get(strategy_name, np.nan)
        status = statuses.get(strategy_name, classify_strategy_status(result, max_corr, risk_contribution))
        action = recommended_action_from_status(status, max_corr, metrics.get("transaction_cost_drag", 0.0))
        model_allocation = allocations.get(strategy_name, 0.0)
        applied_allocation = model_allocation * etf_portfolio_weight
        daily_return = result["strategy_returns"].iloc[-1] if not result["strategy_returns"].empty else np.nan
        cumulative_return = result["strategy_equity"].iloc[-1] - 1 if not result["strategy_equity"].empty else np.nan
        rows.append(
            {
                "Strategy": strategy_name,
                "Strategy Model Weight": model_allocation,
                "Portfolio Applied Weight": applied_allocation,
                "Allocation $": applied_allocation * initial_capital,
                "Current Holdings": get_current_holdings_text(result),
                "Latest Signal": signal.get("Signal", "N/A"),
                "Daily PnL": daily_return * applied_allocation * initial_capital if pd.notna(daily_return) else np.nan,
                "Cumulative PnL": cumulative_return * applied_allocation * initial_capital if pd.notna(cumulative_return) else np.nan,
                "Gross Return": metrics["gross_return"],
                "Net Return": metrics["net_return"],
                "Sharpe": metrics["sharpe"],
                "Volatility": metrics["volatility"],
                "Max Drawdown": metrics["max_drawdown"],
                "Current Drawdown": calculate_current_drawdown(result["strategy_equity"]),
                "Turnover": metrics["total_turnover"],
                "Cost Drag": metrics["transaction_cost_drag"],
                "Risk Status": status,
                "Recommended Action": action,
                "Reason": build_risk_explanation(result, status, max_corr, risk_contribution),
            }
        )
    return pd.DataFrame(rows)


def calculate_stock_selection(stock_close, benchmark_close, regime, top_n=10):
    if stock_close.empty or benchmark_close.empty:
        return pd.DataFrame()
    category_map = get_stock_category_map()
    if isinstance(benchmark_close, pd.DataFrame):
        spy = benchmark_close.iloc[:, 0].dropna()
    else:
        spy = benchmark_close.dropna()
    common_index = stock_close.index.intersection(spy.index)
    stock_close = stock_close.reindex(common_index).ffill()
    spy = spy.reindex(common_index).ffill()
    if len(common_index) < 200:
        return pd.DataFrame()

    returns = stock_close.pct_change(fill_method=None)
    ret20 = stock_close.pct_change(20, fill_method=None).iloc[-1]
    ret60 = stock_close.pct_change(60, fill_method=None).iloc[-1]
    ret120 = stock_close.pct_change(120, fill_method=None).iloc[-1]
    ma50 = stock_close.rolling(50).mean().iloc[-1]
    ma200 = stock_close.rolling(200).mean().iloc[-1]
    dist_ma50 = stock_close.iloc[-1] / ma50 - 1
    dist_ma200 = stock_close.iloc[-1] / ma200 - 1
    vol20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
    drawdown = stock_close.tail(120) / stock_close.tail(120).cummax() - 1
    recent_max_drawdown = drawdown.min()
    relative_strength = ret60 - spy.pct_change(60, fill_method=None).iloc[-1]

    raw = pd.DataFrame(
        {
            "Ticker": stock_close.columns,
            "Category": [category_map.get(ticker, "Other") for ticker in stock_close.columns],
            "Current Price": stock_close.iloc[-1].values,
            "20D Return": ret20.values,
            "60D Return": ret60.values,
            "120D Return": ret120.values,
            "Distance From MA50": dist_ma50.values,
            "Distance From MA200": dist_ma200.values,
            "Volatility": vol20.values,
            "Drawdown": recent_max_drawdown.values,
            "Relative Strength vs SPY": relative_strength.values,
        }
    ).dropna()
    if raw.empty:
        return raw

    raw["Trend Status"] = np.where(
        (raw["Distance From MA50"] > 0) & (raw["Distance From MA200"] > 0),
        "Uptrend",
        np.where((raw["Distance From MA50"] < 0) & (raw["Distance From MA200"] < 0), "Downtrend", "Mixed"),
    )
    raw["momentum_score"] = raw["60D Return"].rank(pct=True)
    raw["relative_strength_score"] = raw["Relative Strength vs SPY"].rank(pct=True)
    raw["trend_score"] = ((raw["Distance From MA50"] > 0).astype(float) + (raw["Distance From MA200"] > 0).astype(float)) / 2
    raw["volatility_score"] = (-raw["Volatility"]).rank(pct=True)
    raw["drawdown_score"] = raw["Drawdown"].rank(pct=True)
    raw["Stock Score"] = (
        0.30 * raw["momentum_score"]
        + 0.25 * raw["relative_strength_score"]
        + 0.20 * raw["trend_score"]
        + 0.15 * raw["volatility_score"]
        + 0.10 * raw["drawdown_score"]
    )
    preferred = REGIME_STOCK_PREFERENCES.get(regime["regime_name"], [])
    raw["Regime Preference Boost"] = raw["Category"].isin(preferred).astype(float) * 0.15
    if regime["regime_name"] in ["Risk-Off Bear Market", "High Volatility / Crisis"]:
        raw["Stock Score"] += np.where(raw["Category"] == "Defensive", 0.20, -0.10)
        raw["Stock Score"] += raw["volatility_score"] * 0.10
    elif regime["regime_name"] == "Inflation / Commodity-Led Market":
        raw["Stock Score"] += np.where(raw["Category"] == "Energy / Inflation", 0.25, 0)
    elif regime["regime_name"] == "Growth-Led Market":
        raw["Stock Score"] += np.where(raw["Category"] == "Technology / Growth", 0.20, 0)
    elif regime["regime_name"] == "Value / Financials-Led Market":
        raw["Stock Score"] += np.where(raw["Category"] == "Financials", 0.20, 0)
    raw["Stock Score"] += raw["Regime Preference Boost"]
    raw["Reason"] = raw.apply(
        lambda row: f"{row['Ticker']} ranks well for {regime['regime_name']} because of {row['Category']} exposure, relative strength, trend, volatility, and drawdown profile.",
        axis=1,
    )
    return raw.sort_values("Stock Score", ascending=False).head(top_n)


def parse_selected_assets(selected_asset):
    assets = [asset.strip() for asset in str(selected_asset).split(",")]
    return [asset for asset in assets if asset and asset not in ["Cash", "N/A", "None"]]


def get_asset_type(ticker):
    if ticker == "Cash":
        return "Cash"
    if ticker in get_all_stock_symbols():
        return "Stock"
    return "ETF"


def build_position_allocation(
    strategy_results,
    strategy_allocation_table,
    stock_selection,
    asset_targets,
    initial_capital,
    max_position_weight=0.10,
    max_sector_exposure=0.35,
):
    etf_category_map = get_etf_category_map()
    stock_category_map = get_stock_category_map()
    position_rows = []
    etf_budget = asset_targets.get("ETFs", 0.0)
    stock_budget = asset_targets.get("Stocks", 0.0)
    cash_budget = asset_targets.get("Cash / SHY / BIL", 0.0)

    for _, row in strategy_allocation_table.iterrows():
        strategy = row["Strategy"]
        if strategy not in strategy_results:
            continue
        sleeve_weight = etf_budget * row["Final Allocation"]
        signal = strategy_results[strategy].get("signal", {})
        selected = parse_selected_assets(signal.get("Selected Asset", ""))
        if not selected:
            selected = ["Cash"] if signal.get("Signal") in ["CASH", "SELL"] else ["SHY"]
        for asset in selected:
            asset_type = get_asset_type(asset)
            category = "Cash" if asset == "Cash" else etf_category_map.get(asset, stock_category_map.get(asset, "Other"))
            position_rows.append(
                {
                    "Ticker": asset,
                    "Asset Type": asset_type,
                    "Source": strategy,
                    "Sector/Category": category,
                    "Portfolio Weight": sleeve_weight / len(selected),
                    "Signal": signal.get("Signal", "N/A"),
                    "Reason": signal.get("Reason", row.get("Reason", "")),
                }
            )

    if not stock_selection.empty and stock_budget > 0:
        positive_scores = stock_selection["Stock Score"].clip(lower=0)
        if positive_scores.sum() <= 0:
            score_weights = pd.Series(1 / len(stock_selection), index=stock_selection.index)
        else:
            score_weights = positive_scores / positive_scores.sum()
        for idx, row in stock_selection.iterrows():
            ticker = row["Ticker"]
            position_rows.append(
                {
                    "Ticker": ticker,
                    "Asset Type": "Stock",
                    "Source": "Stock Selector",
                    "Sector/Category": row.get("Category", "Other"),
                    "Portfolio Weight": stock_budget * score_weights.loc[idx],
                    "Signal": "BUY",
                    "Reason": row.get("Reason", "Selected by stock ranking score."),
                }
            )

    if cash_budget > 0:
        position_rows.append(
            {
                "Ticker": "Cash",
                "Asset Type": "Cash",
                "Source": "Risk Manager",
                "Sector/Category": "Cash",
                "Portfolio Weight": cash_budget,
                "Signal": "CASH",
                "Reason": "Required cash buffer from asset allocation and risk controls.",
            }
        )

    if not position_rows:
        return pd.DataFrame(columns=["Ticker", "Asset Type", "Source", "Sector/Category", "Portfolio Weight", "Dollar Allocation", "Signal", "Reason"])

    raw = pd.DataFrame(position_rows)
    grouped = (
        raw.groupby("Ticker", as_index=False)
        .agg(
            {
                "Asset Type": "first",
                "Source": lambda values: ", ".join(dict.fromkeys(values)),
                "Sector/Category": "first",
                "Portfolio Weight": "sum",
                "Signal": lambda values: ", ".join(dict.fromkeys(values)),
                "Reason": lambda values: " | ".join(dict.fromkeys(values)),
            }
        )
    )

    capped_weights = {}
    overflow = 0.0
    for _, row in grouped.iterrows():
        asset = row["Ticker"]
        weight = row["Portfolio Weight"]
        if row["Asset Type"] != "Cash" and weight > max_position_weight:
            capped_weights[asset] = max_position_weight
            overflow += weight - max_position_weight
        else:
            capped_weights[asset] = weight
    grouped["Portfolio Weight"] = grouped["Ticker"].map(capped_weights)

    sector_excess = 0.0
    sector_weights = grouped[grouped["Asset Type"] != "Cash"].groupby("Sector/Category")["Portfolio Weight"].sum()
    for category, weight in sector_weights.items():
        if weight <= max_sector_exposure:
            continue
        scale = max_sector_exposure / weight
        mask = (grouped["Sector/Category"] == category) & (grouped["Asset Type"] != "Cash")
        before = grouped.loc[mask, "Portfolio Weight"].sum()
        grouped.loc[mask, "Portfolio Weight"] *= scale
        sector_excess += before - grouped.loc[mask, "Portfolio Weight"].sum()
    overflow += sector_excess

    if overflow > 0:
        cash_mask = grouped["Ticker"] == "Cash"
        if cash_mask.any():
            grouped.loc[cash_mask, "Portfolio Weight"] += overflow
            grouped.loc[cash_mask, "Reason"] = grouped.loc[cash_mask, "Reason"] + " Excess allocation moved to cash after position/sector caps."
        else:
            grouped = pd.concat(
                [
                    grouped,
                    pd.DataFrame(
                        [
                            {
                                "Ticker": "Cash",
                                "Asset Type": "Cash",
                                "Source": "Risk Manager",
                                "Sector/Category": "Cash",
                                "Portfolio Weight": overflow,
                                "Signal": "CASH",
                                "Reason": "Excess allocation moved to cash after position/sector caps.",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    total = grouped["Portfolio Weight"].sum()
    if total > 0:
        grouped["Portfolio Weight"] = grouped["Portfolio Weight"] / total
    grouped["Dollar Allocation"] = grouped["Portfolio Weight"] * initial_capital
    return grouped.sort_values("Portfolio Weight", ascending=False)[
        ["Ticker", "Asset Type", "Source", "Sector/Category", "Portfolio Weight", "Dollar Allocation", "Signal", "Reason"]
    ].reset_index(drop=True)


def build_portfolio_backtest(results, initial_capital=INITIAL_CAPITAL, strategy_weights=None):
    net_returns = {}
    gross_returns = {}
    for strategy_name, result in results.items():
        if result is None:
            continue
        net_returns[strategy_name] = result["strategy_returns"]
        gross_returns[strategy_name] = result["gross_returns"]

    net_df = pd.DataFrame(net_returns).dropna(how="all")
    gross_df = pd.DataFrame(gross_returns).dropna(how="all")
    common_index = net_df.index.intersection(gross_df.index)
    net_df = net_df.loc[common_index].fillna(0.0)
    gross_df = gross_df.loc[common_index].fillna(0.0)
    if net_df.empty:
        return pd.DataFrame(), {}

    _, statuses = build_strategy_allocations({name: results[name] for name in net_df.columns})
    if strategy_weights is None:
        weights = pd.Series(1 / len(net_df.columns), index=net_df.columns)
    else:
        weights = strategy_weights.reindex(net_df.columns).fillna(0.0)
    if weights.sum() <= 0:
        weights = pd.Series(1 / len(net_df.columns), index=net_df.columns)
    else:
        weights = weights / weights.sum()
    portfolio_net_returns = net_df.mul(weights, axis=1).sum(axis=1)
    portfolio_gross_returns = gross_df.mul(weights, axis=1).sum(axis=1)
    portfolio_net_value = initial_capital * (1 + portfolio_net_returns).cumprod()
    portfolio_gross_value = initial_capital * (1 + portfolio_gross_returns).cumprod()
    portfolio_returns = pd.DataFrame(
        {
            "Gross Return": portfolio_gross_returns,
            "Net Return": portfolio_net_returns,
            "Gross Value": portfolio_gross_value,
            "Net Value": portfolio_net_value,
        }
    )
    net_equity = portfolio_net_value / initial_capital
    gross_equity = portfolio_gross_value / initial_capital
    metrics = calculate_performance_metrics(net_equity, portfolio_net_returns, 0)
    var95, es95 = calculate_var_es(portfolio_net_returns, confidence=0.95)
    correlations = calculate_strategy_correlations(results)
    risk_contributions = calculate_strategy_risk_contributions(results)
    metrics["current_drawdown"] = calculate_current_drawdown(net_equity)
    metrics["portfolio_value"] = portfolio_net_value.iloc[-1]
    metrics["total_pnl"] = portfolio_net_value.iloc[-1] - initial_capital
    metrics["gross_return"] = gross_equity.iloc[-1] - 1
    metrics["net_return"] = net_equity.iloc[-1] - 1
    metrics["var_95"] = var95
    metrics["expected_shortfall_95"] = es95
    metrics["transaction_cost_drag"] = gross_equity.iloc[-1] - net_equity.iloc[-1]
    metrics["total_turnover"] = sum(
        results[strategy_name]["metrics"]["total_turnover"] * weights.get(strategy_name, 0.0)
        for strategy_name in net_df.columns
    )
    metrics["active_strategies"] = int((weights > 0).sum())
    metrics["warning_strategies"] = sum(
        statuses.get(strategy_name, "Healthy") in ["Watch", "Reduce", "Pause"]
        for strategy_name in net_df.columns
    )
    metrics["allocations"] = weights
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack() if not correlations.empty else pd.Series(dtype=float)
    metrics["average_pairwise_correlation"] = upper.mean() if not upper.empty else np.nan
    metrics["recommendation"] = build_portfolio_recommendation(metrics)
    return portfolio_returns, metrics


def build_holdings_portfolio_backtest(position_allocation, etf_close, stock_close, initial_capital=INITIAL_CAPITAL, strategy_results=None):
    if position_allocation.empty:
        return pd.DataFrame(), {}

    price_sources = [etf_close]
    if stock_close is not None and not stock_close.empty:
        price_sources.append(stock_close)
    prices = pd.concat(price_sources, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].dropna(how="all")
    returns = prices.pct_change(fill_method=None).fillna(0.0)

    weights = position_allocation.set_index("Ticker")["Portfolio Weight"].copy()
    cash_weight = weights.pop("Cash") if "Cash" in weights.index else 0.0
    tradable_weights = weights[weights.index.isin(returns.columns)]
    missing_weight = weights[~weights.index.isin(returns.columns)].sum()
    total_cash_weight = cash_weight + missing_weight

    if tradable_weights.empty:
        common_index = returns.index
        portfolio_net_returns = pd.Series(0.0, index=common_index)
    else:
        tradable_weights = tradable_weights / max(1.0, tradable_weights.sum() + total_cash_weight)
        common_index = returns.index
        portfolio_net_returns = returns.loc[common_index, tradable_weights.index].mul(tradable_weights, axis=1).sum(axis=1)

    portfolio_gross_returns = portfolio_net_returns.copy()
    portfolio_net_value = initial_capital * (1 + portfolio_net_returns).cumprod()
    portfolio_gross_value = initial_capital * (1 + portfolio_gross_returns).cumprod()
    portfolio_returns = pd.DataFrame(
        {
            "Gross Return": portfolio_gross_returns,
            "Net Return": portfolio_net_returns,
            "Gross Value": portfolio_gross_value,
            "Net Value": portfolio_net_value,
        }
    )
    if portfolio_returns.empty:
        return pd.DataFrame(), {}

    net_equity = portfolio_net_value / initial_capital
    gross_equity = portfolio_gross_value / initial_capital
    metrics = calculate_performance_metrics(net_equity, portfolio_net_returns, 0)
    var95, es95 = calculate_var_es(portfolio_net_returns, confidence=0.95)
    metrics["current_drawdown"] = calculate_current_drawdown(net_equity)
    metrics["portfolio_value"] = portfolio_net_value.iloc[-1]
    metrics["total_pnl"] = portfolio_net_value.iloc[-1] - initial_capital
    metrics["gross_return"] = gross_equity.iloc[-1] - 1
    metrics["net_return"] = net_equity.iloc[-1] - 1
    metrics["var_95"] = var95
    metrics["expected_shortfall_95"] = es95
    metrics["transaction_cost_drag"] = gross_equity.iloc[-1] - net_equity.iloc[-1]
    metrics["total_turnover"] = 0.0
    metrics["active_strategies"] = len(strategy_results) if strategy_results else 0
    metrics["warning_strategies"] = 0
    metrics["allocations"] = position_allocation.set_index("Ticker")["Portfolio Weight"]
    if strategy_results:
        correlations = calculate_strategy_correlations(strategy_results)
        upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack() if not correlations.empty else pd.Series(dtype=float)
        metrics["average_pairwise_correlation"] = upper.mean() if not upper.empty else np.nan
    else:
        metrics["average_pairwise_correlation"] = np.nan
    metrics["recommendation"] = build_portfolio_recommendation(metrics)
    return portfolio_returns, metrics


def build_portfolio_recommendation(metrics):
    if metrics.get("current_drawdown", 0) < -0.10 or metrics.get("max_drawdown", 0) < -0.15:
        return "Reduce risk: portfolio drawdown is above the MVP risk limit."
    if metrics.get("var_95", 0) < -0.03 or metrics.get("expected_shortfall_95", 0) < -0.05:
        return "Hedge or reduce: VaR / Expected Shortfall is elevated."
    if metrics.get("average_pairwise_correlation", 0) > 0.70:
        return "Rebalance: strategy correlations are high."
    if metrics.get("transaction_cost_drag", 0) > 0.05:
        return "Review costs: turnover and cost drag are high."
    return "Keep current allocation: portfolio risk is within MVP limits."


def render_regime_aware_portfolio_overview(
    results,
    portfolio_returns,
    portfolio_metrics,
    regime,
    strategy_allocation,
    asset_targets,
    position_allocation,
    initial_capital=INITIAL_CAPITAL,
):
    st.header("Portfolio Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Capital", f"${initial_capital:,.0f}")
    c2.metric("Portfolio Value", f"${portfolio_metrics.get('portfolio_value', initial_capital):,.0f}")
    c3.metric("Current Regime", regime["regime_name"])
    c4.metric("Regime Confidence", f"{regime['confidence']:.0%}")
    c5, c6, c7, c8 = st.columns(4)
    final_cash_weight = position_allocation.loc[position_allocation["Asset Type"] == "Cash", "Portfolio Weight"].sum() if not position_allocation.empty else asset_targets.get("Cash / SHY / BIL", 0)
    c5.metric("Portfolio Sharpe", f"{portfolio_metrics.get('sharpe', np.nan):.2f}")
    c6.metric("Max Drawdown", f"{portfolio_metrics.get('max_drawdown', np.nan) * 100:.2f}%")
    c7.metric("Final Cash Allocation", f"{final_cash_weight * 100:.1f}%")
    c8.metric("Risk Status", build_portfolio_risk_status(portfolio_metrics, position_allocation, asset_targets)["Status"])
    st.info(regime["regime_explanation"])
    st.success(f"Portfolio recommendation: {portfolio_metrics.get('recommendation', 'Keep current allocation')}")

    left, right = st.columns(2)
    with left:
        st.subheader("Asset-Class Allocation")
        asset_df = pd.DataFrame(
            [{"Sleeve": key, "Weight": value, "Dollar Allocation": value * initial_capital} for key, value in asset_targets.items()]
        )
        st.dataframe(asset_df.style.format({"Weight": "{:.2%}", "Dollar Allocation": "${:,.0f}"}), use_container_width=True)
        st.plotly_chart(px.pie(asset_df, names="Sleeve", values="Weight", title="Aggregate Portfolio Allocation"), use_container_width=True)
    with right:
        st.subheader("Top 10 Holdings")
        st.dataframe(
            position_allocation.head(10).style.format({"Portfolio Weight": "{:.2%}", "Dollar Allocation": "${:,.0f}"}),
            use_container_width=True,
        )

    st.subheader("Strategy Allocation")
    st.dataframe(
        strategy_allocation[["Strategy", "Selected Asset", "Signal", "Final Allocation", "Final Score", "Regime Fit Score", "Allocation Change", "Reason"]]
        .style.format({"Final Allocation": "{:.2%}", "Final Score": "{:.2f}", "Regime Fit Score": "{:.2f}"}),
        use_container_width=True,
    )

    st.subheader("Final Portfolio Holdings")
    st.caption("These are final position-level holdings after aggregating strategy signals, stock selector weights, cash allocation, and position/sector caps.")
    st.dataframe(
        position_allocation.style.format({"Portfolio Weight": "{:.2%}", "Dollar Allocation": "${:,.0f}"}),
        use_container_width=True,
    )
    st.metric("Total Dollar Allocation", f"${position_allocation['Dollar Allocation'].sum():,.0f}" if not position_allocation.empty else "$0")

    if not portfolio_returns.empty:
        st.plotly_chart(px.line(portfolio_returns, x=portfolio_returns.index, y="Net Value", title="Portfolio Net Equity Curve"), use_container_width=True)


def render_market_regime_tab(close, regime):
    st.header("Market Regime")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected Regime", regime["regime_name"])
    c2.metric("Confidence", f"{regime['confidence']:.0%}")
    c3.metric("Detection Date", regime["date"])
    st.write(regime["regime_explanation"])
    indicator_df = pd.DataFrame(
        [{"Indicator": key, "Value": value} for key, value in regime["key_indicators"].items()]
    )
    st.subheader("Key Indicators")
    st.dataframe(indicator_df.style.format({"Value": "{:.2%}"}), use_container_width=True)
    st.subheader("ETF Proxy Performance")
    proxy_symbols = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "GLD", "TLT", "SHY", "BIL", "USMV", "SPLV"]
    proxy_table = build_market_monitoring_table(close, proxy_symbols)
    st.dataframe(
        proxy_table.style.format(
            {
                "Latest Price": "${:,.2f}",
                "1D Return": "{:.2%}",
                "5D Return": "{:.2%}",
                "20D Return": "{:.2%}",
                "60D Return": "{:.2%}",
                "20D Realized Volatility": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    history = build_regime_history(close)
    if not history.empty:
        st.subheader("Regime History")
        fig = px.scatter(history.reset_index(), x="Date", y="Regime", color="Regime", size="Confidence", title="Historical Regime Timeline")
        st.plotly_chart(fig, use_container_width=True)


def render_strategy_allocation_tab(strategy_allocation):
    st.header("Strategy Allocation")
    st.write("Strategy weights are dynamic. They are based on regime fit, recent performance, Sharpe, drawdown, risk status, and signal validity. Equal weight is shown only as a comparison baseline.")
    st.dataframe(
        strategy_allocation.style.format(
            {
                "Base Weight": "{:.2%}",
                "Regime Fit Score": "{:.2f}",
                "Recent Performance Score": "{:.2f}",
                "Risk Adjusted Score": "{:.2f}",
                "Drawdown Penalty": "{:.2f}",
                "Risk Limit Penalty": "{:.2f}",
                "Final Score": "{:.2f}",
                "Final Allocation": "{:.2%}",
                "Final Strategy Weight": "{:.2%}",
                "Weight Change vs Equal Weight": "{:+.2%}",
            }
        ),
        use_container_width=True,
    )
    st.plotly_chart(px.bar(strategy_allocation, x="Strategy", y="Final Allocation", title="Regime-Aware Strategy Weights"), use_container_width=True)


def render_stock_selection_tab(stock_selection):
    st.header("Stock Selection")
    if stock_selection.empty:
        st.warning("Stock selection is unavailable. Check stock data or reduce lookback requirements.")
        return
    st.subheader("Ranked Stock Candidates")
    display_cols = [
        "Ticker",
        "Category",
        "Stock Score",
        "Assigned Portfolio Weight",
        "Dollar Allocation",
        "Current Price",
        "20D Return",
        "60D Return",
        "120D Return",
        "Volatility",
        "Drawdown",
        "Relative Strength vs SPY",
        "Trend Status",
        "Reason",
    ]
    display_cols = [col for col in display_cols if col in stock_selection.columns]
    st.dataframe(
        stock_selection[display_cols].style.format(
            {
                "Stock Score": "{:.2f}",
                "Assigned Portfolio Weight": "{:.2%}",
                "Dollar Allocation": "${:,.0f}",
                "Current Price": "${:,.2f}",
                "20D Return": "{:.2%}",
                "60D Return": "{:.2%}",
                "120D Return": "{:.2%}",
                "Volatility": "{:.2%}",
                "Drawdown": "{:.2%}",
                "Relative Strength vs SPY": "{:.2%}",
            }
        ),
        use_container_width=True,
    )
    st.plotly_chart(px.bar(stock_selection, x="Ticker", y="Stock Score", color="Category", title="Top Regime-Aware Stock Scores"), use_container_width=True)


def build_portfolio_risk_status(portfolio_metrics, position_allocation, asset_targets, max_position_weight=0.10, max_sector_exposure=0.35):
    warnings = []
    if portfolio_metrics.get("max_drawdown", 0) < RISK_LIMITS["severe_drawdown_warning"]:
        warnings.append("Severe portfolio drawdown breach.")
    elif portfolio_metrics.get("max_drawdown", 0) < RISK_LIMITS["max_drawdown_warning"]:
        warnings.append("Portfolio drawdown warning.")
    non_cash_positions = position_allocation[position_allocation["Asset Type"] != "Cash"] if not position_allocation.empty else pd.DataFrame()
    if not non_cash_positions.empty and non_cash_positions["Portfolio Weight"].max() > max_position_weight:
        warnings.append("Top holding exceeds max position weight.")
    if not position_allocation.empty:
        sector_exposure = (
            position_allocation[position_allocation["Asset Type"] != "Cash"]
            .groupby("Sector/Category")["Portfolio Weight"]
            .sum()
        )
        if not sector_exposure.empty and sector_exposure.max() > max_sector_exposure:
            warnings.append("Sector/category exposure exceeds the configured maximum.")
    if asset_targets.get("Cash / SHY / BIL", 0) < 0.15 and portfolio_metrics.get("recommendation", "").lower().startswith("hedge"):
        warnings.append("Cash allocation may be too low for the detected risk level.")
    status = "Breach" if any("breach" in warning.lower() for warning in warnings) else "Watch" if warnings else "OK"
    return {"Status": status, "Warnings": warnings or ["No major risk limit warnings."]}


def render_risk_dashboard(results, close, portfolio_metrics, position_allocation, asset_targets, max_position_weight, max_sector_exposure):
    st.header("Risk Dashboard")
    final_cash_weight = position_allocation.loc[position_allocation["Asset Type"] == "Cash", "Portfolio Weight"].sum() if not position_allocation.empty else asset_targets.get("Cash / SHY / BIL", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Portfolio Volatility", f"{portfolio_metrics.get('volatility', np.nan) * 100:.2f}%")
    c2.metric("Portfolio Sharpe", f"{portfolio_metrics.get('sharpe', np.nan):.2f}")
    c3.metric("Max Drawdown", f"{portfolio_metrics.get('max_drawdown', np.nan) * 100:.2f}%")
    c4.metric("Final Cash Allocation", f"{final_cash_weight * 100:.1f}%")
    risk_status = build_portfolio_risk_status(portfolio_metrics, position_allocation, asset_targets, max_position_weight, max_sector_exposure)
    if risk_status["Status"] == "OK":
        st.success("Risk limit status: OK")
    elif risk_status["Status"] == "Watch":
        st.warning("Risk limit status: Watch")
    else:
        st.error("Risk limit status: Breach")
    for warning in risk_status["Warnings"]:
        st.write(f"- {warning}")

    correlations = calculate_strategy_correlations(results)
    if not correlations.empty:
        st.plotly_chart(
            px.imshow(correlations, text_auto=".2f", title="Strategy Correlation Matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
            use_container_width=True,
            key="risk_dashboard_strategy_correlation_matrix",
        )
    exposures = estimate_factor_exposures(results, close)
    if not exposures.empty:
        portfolio_exposure = exposures.mean().sort_values()
        st.plotly_chart(
            px.bar(x=portfolio_exposure.index, y=portfolio_exposure.values, title="Portfolio Factor Exposure"),
            use_container_width=True,
            key="risk_dashboard_portfolio_factor_exposure",
        )
        st.dataframe(exposures.style.format("{:.2f}"), use_container_width=True)
    risk_contributions = calculate_strategy_risk_contributions(results)
    if not risk_contributions.empty:
        st.plotly_chart(
            px.bar(x=risk_contributions.index, y=risk_contributions.values, title="Risk Contribution by Strategy"),
            use_container_width=True,
            key="risk_dashboard_strategy_risk_contribution",
        )
    st.subheader("Concentration Risk")
    if not position_allocation.empty:
        top_position = position_allocation.iloc[0]
        st.metric("Largest Position", f"{top_position['Ticker']} / {top_position['Portfolio Weight']:.2%}")
        sector_exposure = (
            position_allocation[position_allocation["Asset Type"] != "Cash"]
            .groupby("Sector/Category")["Portfolio Weight"]
            .sum()
            .sort_values(ascending=False)
        )
        st.plotly_chart(px.bar(x=sector_exposure.index, y=sector_exposure.values, title="Sector / Category Exposure"), use_container_width=True)
        st.dataframe(
            position_allocation.style.format({"Portfolio Weight": "{:.2%}", "Dollar Allocation": "${:,.0f}"}),
            use_container_width=True,
        )


def render_clean_strategy_monitoring(results, initial_capital=INITIAL_CAPITAL, strategy_allocation=None, etf_portfolio_weight=1.0):
    st.header("Strategies Live Monitoring")
    st.caption(
        "Strategy Model Weight is the relative weight among ETF strategy sleeves. "
        "Portfolio Applied Weight is the actual portfolio weight after the selected Portfolio Mode and ETF allocation are applied."
    )
    table = build_strategy_monitoring_table(
        results,
        initial_capital=initial_capital,
        strategy_allocation=strategy_allocation,
        etf_portfolio_weight=etf_portfolio_weight,
    )
    if table.empty:
        st.warning("No strategy results are available.")
        return
    st.dataframe(
        table.style.format(
            {
                "Strategy Model Weight": "{:.2%}",
                "Portfolio Applied Weight": "{:.2%}",
                "Allocation $": "${:,.0f}",
                "Daily PnL": "${:,.0f}",
                "Cumulative PnL": "${:,.0f}",
                "Gross Return": "{:.2%}",
                "Net Return": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Volatility": "{:.2%}",
                "Max Drawdown": "{:.2%}",
                "Current Drawdown": "{:.2%}",
                "Turnover": "{:.2f}",
                "Cost Drag": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    selected_strategy = st.selectbox("Select Strategy Detail", table["Strategy"].tolist())
    result = results[selected_strategy]
    render_signal_card(result["signal"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["gross_equity"].index, y=result["gross_equity"], name="Gross", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=result["strategy_equity"].index, y=result["strategy_equity"], name="Net"))
    fig.update_layout(title=f"{selected_strategy} Equity Curve", xaxis_title="Date", yaxis_title="Growth of 1.0", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    drawdown = result["strategy_equity"] / result["strategy_equity"].cummax() - 1
    dd_fig = px.area(x=drawdown.index, y=drawdown, title=f"{selected_strategy} Drawdown")
    dd_fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown", template="plotly_white")
    st.plotly_chart(dd_fig, use_container_width=True)

    rolling_sharpe = result["strategy_returns"].rolling(63).mean() * 252 / (result["strategy_returns"].rolling(63).std() * np.sqrt(252))
    sharpe_fig = px.line(x=rolling_sharpe.index, y=rolling_sharpe, title=f"{selected_strategy} Rolling Sharpe")
    sharpe_fig.update_layout(xaxis_title="Date", yaxis_title="Rolling Sharpe", template="plotly_white")
    st.plotly_chart(sharpe_fig, use_container_width=True)

    rolling_volatility = result["strategy_returns"].rolling(63).std() * np.sqrt(252)
    vol_fig = px.line(x=rolling_volatility.index, y=rolling_volatility, title=f"{selected_strategy} Rolling Volatility")
    vol_fig.update_layout(xaxis_title="Date", yaxis_title="Rolling Volatility", template="plotly_white")
    st.plotly_chart(vol_fig, use_container_width=True)

    st.subheader("Monthly Holdings")
    if result["monthly_holdings"]:
        holdings_df = pd.DataFrame(
            [{"Signal Date": h["Date"].strftime("%Y-%m-%d"), "Holdings": h["Holdings_str"]} for h in result["monthly_holdings"]]
        )
        st.dataframe(holdings_df.tail(24), use_container_width=True)
    else:
        st.write("No monthly holdings are available for this strategy.")

    st.subheader("Latest Signal Explanation")
    st.write(result["signal"]["Reason"])
    st.subheader("Transaction Cost Impact")
    st.write(f"Cost drag: {result['metrics']['transaction_cost_drag'] * 100:.2f}% of growth-of-one performance.")
    st.plotly_chart(px.area(x=result["trading_costs"].index, y=result["trading_costs"].cumsum(), title="Cumulative Transaction Cost Drag"), use_container_width=True)
    st.subheader("Risk Limit Explanation")
    selected_row = table[table["Strategy"] == selected_strategy].iloc[0]
    st.write(f"Risk status: **{selected_row['Risk Status']}**. Recommended action: **{selected_row['Recommended Action']}**.")
    st.write(selected_row["Reason"])


def build_market_monitoring_table(close, symbols):
    rows = []
    returns = close.pct_change(fill_method=None)
    for symbol in symbols:
        if symbol not in close.columns:
            continue
        series = close[symbol].dropna()
        if series.empty:
            continue
        latest_price = series.iloc[-1]
        ma50 = series.rolling(50).mean().iloc[-1] if len(series) >= 50 else np.nan
        ma200 = series.rolling(200).mean().iloc[-1] if len(series) >= 200 else np.nan
        vol20 = returns[symbol].rolling(20).std().iloc[-1] * np.sqrt(252) if symbol in returns else np.nan
        trend_status = "Bullish" if pd.notna(ma50) and pd.notna(ma200) and latest_price > ma50 > ma200 else "Bearish" if pd.notna(ma50) and pd.notna(ma200) and latest_price < ma50 < ma200 else "Neutral"
        rows.append(
            {
                "Ticker": symbol,
                "Latest Price": latest_price,
                "1D Return": series.pct_change(1, fill_method=None).iloc[-1],
                "5D Return": series.pct_change(5, fill_method=None).iloc[-1] if len(series) > 5 else np.nan,
                "20D Return": series.pct_change(20, fill_method=None).iloc[-1] if len(series) > 20 else np.nan,
                "60D Return": series.pct_change(60, fill_method=None).iloc[-1] if len(series) > 60 else np.nan,
                "20D Realized Volatility": vol20,
                "Trend Status": trend_status,
            }
        )
    return pd.DataFrame(rows)


def get_relative_return(close, symbol, benchmark="SPY", window=60):
    if symbol not in close.columns or benchmark not in close.columns or len(close) <= window:
        return np.nan
    return close[symbol].pct_change(window, fill_method=None).iloc[-1] - close[benchmark].pct_change(window, fill_method=None).iloc[-1]


def detect_market_regime(close):
    if close.empty or "SPY" not in close.columns:
        return {
            "regime_name": "Sideways / Choppy Market",
            "regime_score": 0.0,
            "confidence": 0.0,
            "regime_explanation": "SPY data is unavailable, so the detector defaults to Sideways / Choppy Market.",
            "key_indicators": {},
            "date": "N/A",
        }
    spy = close["SPY"].dropna()
    if len(spy) < 200:
        return {
            "regime_name": "Sideways / Choppy Market",
            "regime_score": 0.0,
            "confidence": 0.25,
            "regime_explanation": "Not enough history for full MA200 and regime checks.",
            "key_indicators": {"available_days": len(spy)},
            "date": spy.index[-1].strftime("%Y-%m-%d") if not spy.empty else "N/A",
        }

    returns = close.pct_change(fill_method=None)
    spy_20d = spy.pct_change(20, fill_method=None).iloc[-1]
    spy_60d = spy.pct_change(60, fill_method=None).iloc[-1]
    spy_120d = spy.pct_change(120, fill_method=None).iloc[-1]
    spy_ma200 = spy.rolling(200).mean().iloc[-1]
    spy_dist_ma200 = spy.iloc[-1] / spy_ma200 - 1
    spy_vol20 = returns["SPY"].rolling(20).std().iloc[-1] * np.sqrt(252)
    spy_60d_drawdown = (spy.tail(60) / spy.tail(60).cummax() - 1).min()
    qqq_rel = get_relative_return(close, "QQQ", "SPY", 60)
    xlk_rel = get_relative_return(close, "XLK", "SPY", 60)
    xlf_rel = get_relative_return(close, "XLF", "SPY", 60)
    xle_rel = get_relative_return(close, "XLE", "SPY", 60)
    gld_60d = close["GLD"].pct_change(60, fill_method=None).iloc[-1] if "GLD" in close.columns else np.nan
    tlt_60d = close["TLT"].pct_change(60, fill_method=None).iloc[-1] if "TLT" in close.columns else np.nan
    spy_tlt_corr = returns[["SPY", "TLT"]].dropna().tail(60).corr().iloc[0, 1] if "TLT" in returns.columns else np.nan

    scores = {regime: 0.0 for regime in REGIME_NAMES}
    if spy_dist_ma200 < 0 and spy_60d < -0.08 and spy_vol20 > 0.25:
        scores["High Volatility / Crisis"] += 1.0
    if spy_dist_ma200 < 0 and spy_60d < -0.05:
        scores["Risk-Off Bear Market"] += 0.9
    if spy_dist_ma200 > 0 and spy_60d > 0.03 and spy_120d > 0.05:
        scores["Risk-On Bull Market"] += 0.8
    if spy_dist_ma200 > 0 and qqq_rel > 0.02 and xlk_rel > 0.02:
        scores["Growth-Led Market"] += 1.0
    if xle_rel > 0.03 and pd.notna(gld_60d) and gld_60d > 0.03 and spy_60d < 0.04:
        scores["Inflation / Commodity-Led Market"] += 0.9
    if pd.notna(tlt_60d) and tlt_60d > spy_60d + 0.05 and spy_60d < 0.02:
        scores["Falling Rates / Bond Rally"] += 0.8
    if xlf_rel > 0.02 and spy_dist_ma200 > 0:
        scores["Value / Financials-Led Market"] += 0.7
    if abs(spy_60d) < 0.04 and spy_vol20 > 0.14:
        scores["Sideways / Choppy Market"] += 0.7
    if spy_vol20 > 0.30:
        scores["High Volatility / Crisis"] += 0.5
    if spy_60d_drawdown < -0.10:
        scores["Risk-Off Bear Market"] += 0.3

    regime_name = max(scores, key=scores.get)
    regime_score = scores[regime_name]
    confidence = min(1.0, max(0.20, regime_score))
    if regime_score == 0:
        regime_name = "Sideways / Choppy Market"
        confidence = 0.35

    indicators = {
        "SPY 20D Return": spy_20d,
        "SPY 60D Return": spy_60d,
        "SPY 120D Return": spy_120d,
        "SPY Distance From MA200": spy_dist_ma200,
        "SPY 20D Realized Volatility": spy_vol20,
        "SPY 60D Max Drawdown": spy_60d_drawdown,
        "QQQ vs SPY 60D Relative Strength": qqq_rel,
        "XLK vs SPY 60D Relative Strength": xlk_rel,
        "XLF vs SPY 60D Relative Strength": xlf_rel,
        "XLE vs SPY 60D Relative Strength": xle_rel,
        "GLD 60D Return": gld_60d,
        "TLT 60D Return": tlt_60d,
        "SPY / TLT 60D Correlation": spy_tlt_corr,
    }
    explanation = (
        f"Detected {regime_name} using SPY trend, realized volatility, ETF relative strength, "
        "commodity proxies, and bond proxy behavior."
    )
    return {
        "regime_name": regime_name,
        "regime_score": regime_score,
        "confidence": confidence,
        "regime_explanation": explanation,
        "key_indicators": indicators,
        "date": spy.index[-1].strftime("%Y-%m-%d"),
    }


def build_regime_history(close):
    if close.empty:
        return pd.DataFrame()
    rows = []
    month_ends = close.resample("ME").last().index
    for date in month_ends:
        history = close.loc[:date]
        if len(history) < 200:
            continue
        regime = detect_market_regime(history)
        rows.append(
            {
                "Date": date,
                "Regime": regime["regime_name"],
                "Confidence": regime["confidence"],
                "Score": regime["regime_score"],
            }
        )
    return pd.DataFrame(rows).set_index("Date") if rows else pd.DataFrame()


def build_factor_returns(close):
    factor_returns = {}
    returns = close.pct_change(fill_method=None).dropna()
    for factor, symbols in FACTOR_PROXIES.items():
        available = [symbol for symbol in symbols if symbol in returns.columns]
        if available:
            factor_returns[factor] = returns[available].mean(axis=1)
    return pd.DataFrame(factor_returns).dropna(how="all")


def estimate_factor_exposures(results, close):
    strategy_returns = get_strategy_returns_frame(results)
    factor_returns = build_factor_returns(close)
    common_index = strategy_returns.index.intersection(factor_returns.index)
    if len(common_index) < 60 or factor_returns.empty:
        return pd.DataFrame()
    strategy_returns = strategy_returns.loc[common_index].fillna(0.0)
    factor_returns = factor_returns.loc[common_index].fillna(0.0)
    rows = []
    x = factor_returns.values
    x = np.column_stack([np.ones(len(x)), x])
    for strategy in strategy_returns.columns:
        y = strategy_returns[strategy].values
        try:
            beta = np.linalg.lstsq(x, y, rcond=None)[0][1:]
        except np.linalg.LinAlgError:
            beta = np.full(factor_returns.shape[1], np.nan)
        row = {"Strategy": strategy}
        row.update({factor: beta[idx] for idx, factor in enumerate(factor_returns.columns)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Strategy")


def render_strategy_results(result):
    if "signal" in result:
        render_signal_card(result["signal"])

    strategy_return = result["metrics"]["net_return"] * 100
    benchmark_return = result["benchmark_equity"].iloc[-1] * 100 - 100 if not result["benchmark_equity"].empty else np.nan
    st.write(f"**Backtest performance vs benchmark:** {strategy_return:.2f}% vs {benchmark_return:.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["gross_equity"].index, y=result["gross_equity"], name="Gross", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=result["strategy_equity"].index, y=result["strategy_equity"], name="Net"))
    if not result["benchmark_equity"].empty:
        fig.add_trace(go.Scatter(x=result["benchmark_equity"].index, y=result["benchmark_equity"], name=f"{result['benchmark_symbol']} Benchmark"))
    fig.update_layout(title=f"{result['strategy_name']} Gross vs Net Backtest", xaxis_title="Date", yaxis_title="Growth of 1.0", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Return", f"{metrics['gross_return'] * 100:.2f}%")
    c2.metric("Net Return", f"{metrics['net_return'] * 100:.2f}%")
    c3.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{metrics['max_drawdown'] * 100:.2f}%")
    c5, c6, c7 = st.columns(3)
    c5.metric("Volatility", f"{metrics['volatility'] * 100:.2f}%")
    c6.metric("Turnover", f"{metrics['total_turnover']:.2f}x")
    c7.metric("Cost Drag", f"{metrics['transaction_cost_drag'] * 100:.2f}%")

    if result["monthly_holdings"]:
        holdings_df = pd.DataFrame(
            [{"Signal Date": h["Date"].strftime("%Y-%m-%d"), "Holdings": h["Holdings_str"]} for h in result["monthly_holdings"]]
        )
        st.subheader("Recent Monthly Holdings")
        st.dataframe(holdings_df.tail(24), use_container_width=True)


def render_clean_backtesting(results, portfolio_returns, portfolio_metrics, selected_strategy, benchmark_symbol, strategy_allocation=None, etf_portfolio_weight=1.0):
    st.header("Backtesting")
    result = results.get(selected_strategy)
    if result is None:
        st.warning("Selected strategy backtest is unavailable.")
        return

    st.subheader("Selected Strategy Backtest")
    render_strategy_results(result)

    st.subheader("Portfolio-Level Combined Backtest")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Return", f"{portfolio_metrics['gross_return'] * 100:.2f}%")
    c2.metric("Net Return", f"{portfolio_metrics['net_return'] * 100:.2f}%")
    c3.metric("Turnover", f"{portfolio_metrics['total_turnover']:.2f}x")
    c4.metric("Cost Drag", f"{portfolio_metrics['transaction_cost_drag'] * 100:.2f}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Sharpe", f"{portfolio_metrics['sharpe']:.2f}")
    c6.metric("Volatility", f"{portfolio_metrics['volatility'] * 100:.2f}%")
    c7.metric("Max Drawdown", f"{portfolio_metrics['max_drawdown'] * 100:.2f}%")
    c8.metric("Benchmark", benchmark_symbol)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns["Gross Value"], name="Portfolio Gross", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=portfolio_returns.index, y=portfolio_returns["Net Value"], name="Portfolio Net"))
    benchmark_equity = result["benchmark_equity"].reindex(portfolio_returns.index).ffill()
    if not benchmark_equity.empty:
        fig.add_trace(go.Scatter(x=benchmark_equity.index, y=INITIAL_CAPITAL * benchmark_equity, name=f"{benchmark_symbol} Benchmark"))
    fig.update_layout(title="Portfolio Combined Backtest vs Benchmark", xaxis_title="Date", yaxis_title="Value ($)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    drawdown = portfolio_returns["Net Value"] / portfolio_returns["Net Value"].cummax() - 1
    st.plotly_chart(px.area(x=drawdown.index, y=drawdown, title="Portfolio Drawdown"), use_container_width=True)

    monthly_returns = portfolio_returns["Net Value"].resample("ME").last().pct_change(fill_method=None).dropna().to_frame("Portfolio Monthly Return")
    if not monthly_returns.empty:
        st.subheader("Portfolio Monthly Returns")
        st.dataframe(monthly_returns.style.format("{:.2%}"), use_container_width=True)

    st.subheader("Strategy Returns Table")
    table = build_strategy_monitoring_table(results, strategy_allocation=strategy_allocation, etf_portfolio_weight=etf_portfolio_weight)
    st.dataframe(
        table[["Strategy", "Strategy Model Weight", "Portfolio Applied Weight", "Gross Return", "Net Return", "Sharpe", "Volatility", "Max Drawdown", "Current Drawdown", "Turnover", "Cost Drag"]]
        .style.format(
            {
                "Strategy Model Weight": "{:.2%}",
                "Portfolio Applied Weight": "{:.2%}",
                "Gross Return": "{:.2%}",
                "Net Return": "{:.2%}",
                "Sharpe": "{:.2f}",
                "Volatility": "{:.2%}",
                "Max Drawdown": "{:.2%}",
                "Current Drawdown": "{:.2%}",
                "Turnover": "{:.2f}",
                "Cost Drag": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

def build_dynamic_allocation_backtest(results, close, max_strategy_weight=0.25, benchmark_symbol="SPY"):
    strategy_returns = get_strategy_returns_frame(results)
    if strategy_returns.empty:
        return pd.DataFrame(), pd.DataFrame()
    month_ends = strategy_returns.resample("ME").last().index
    dynamic_returns = pd.Series(0.0, index=strategy_returns.index)
    allocation_rows = []
    for idx in range(6, len(month_ends) - 1):
        signal_date = month_ends[idx]
        next_date = month_ends[idx + 1]
        history_close = close.loc[:signal_date]
        regime = detect_market_regime(history_close)
        allocation_table = build_regime_aware_strategy_allocation(results, regime, max_strategy_weight=max_strategy_weight)
        weights = allocation_table.set_index("Strategy")["Final Allocation"].reindex(strategy_returns.columns).fillna(0.0)
        period_mask = (strategy_returns.index > signal_date) & (strategy_returns.index <= next_date)
        dynamic_returns.loc[period_mask] = strategy_returns.loc[period_mask].mul(weights, axis=1).sum(axis=1)
        allocation_rows.append(
            {
                "Date": signal_date,
                "Regime": regime["regime_name"],
                "Confidence": regime["confidence"],
                **{strategy: weights.get(strategy, 0.0) for strategy in strategy_returns.columns},
            }
        )
    dynamic_returns = dynamic_returns.loc[dynamic_returns != 0]
    common_index = strategy_returns.index.intersection(dynamic_returns.index)
    equal_weight_returns = strategy_returns.loc[common_index].mean(axis=1)
    spy_returns = close[benchmark_symbol].pct_change(fill_method=None).reindex(common_index).fillna(0.0) if benchmark_symbol in close.columns else pd.Series(0.0, index=common_index)
    backtest = pd.DataFrame(
        {
            "Dynamic Regime-Aware Allocation": dynamic_returns.reindex(common_index).fillna(0.0),
            "Equal-Weight Strategy Allocation": equal_weight_returns,
            f"{benchmark_symbol} Benchmark": spy_returns,
        }
    )
    return backtest, pd.DataFrame(allocation_rows).set_index("Date") if allocation_rows else pd.DataFrame()


def render_dynamic_backtest(results, close, max_strategy_weight, benchmark_symbol):
    st.subheader("Dynamic Regime-Aware Walk-Forward Backtest")
    backtest, allocation_history = build_dynamic_allocation_backtest(results, close, max_strategy_weight=max_strategy_weight, benchmark_symbol=benchmark_symbol)
    if backtest.empty:
        st.warning("Not enough data for dynamic allocation backtest.")
        return
    equity = (1 + backtest).cumprod()
    st.plotly_chart(px.line(equity, x=equity.index, y=equity.columns, title="Dynamic Allocation vs Equal-Weight vs Benchmark"), use_container_width=True)
    rows = []
    for column in backtest.columns:
        returns = backtest[column].dropna()
        eq = (1 + returns).cumprod()
        rows.append(
            {
                "Series": column,
                "Cumulative Return": eq.iloc[-1] - 1,
                "Annualized Volatility": returns.std() * np.sqrt(252),
                "Sharpe Ratio": returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else np.nan,
                "Max Drawdown": (eq / eq.cummax() - 1).min(),
            }
        )
    st.dataframe(pd.DataFrame(rows).style.format({"Cumulative Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Max Drawdown": "{:.2%}"}), use_container_width=True)
    rolling_sharpe = backtest["Dynamic Regime-Aware Allocation"].rolling(63).mean() * 252 / (backtest["Dynamic Regime-Aware Allocation"].rolling(63).std() * np.sqrt(252))
    st.plotly_chart(px.line(x=rolling_sharpe.index, y=rolling_sharpe, title="Dynamic Allocation Rolling Sharpe"), use_container_width=True)
    dynamic_equity = equity["Dynamic Regime-Aware Allocation"]
    rolling_drawdown = dynamic_equity / dynamic_equity.cummax() - 1
    st.plotly_chart(px.area(x=rolling_drawdown.index, y=rolling_drawdown, title="Dynamic Allocation Rolling Drawdown"), use_container_width=True)
    monthly_returns = equity["Dynamic Regime-Aware Allocation"].resample("ME").last().pct_change(fill_method=None).dropna().to_frame("Dynamic Monthly Return")
    annual_returns = equity["Dynamic Regime-Aware Allocation"].resample("YE").last().pct_change(fill_method=None).dropna().to_frame("Dynamic Annual Return")
    st.dataframe(monthly_returns.style.format("{:.2%}"), use_container_width=True)
    st.dataframe(annual_returns.style.format("{:.2%}"), use_container_width=True)
    if not allocation_history.empty:
        st.subheader("Regime Timeline")
        st.plotly_chart(px.scatter(allocation_history.reset_index(), x="Date", y="Regime", color="Regime", size="Confidence"), use_container_width=True)


def main():
    st.title("Regime-Aware Multi-Strategy Portfolio Dashboard")
    st.write(
        "A first-version regime-aware allocation, risk management, ETF strategy, and stock selection dashboard. "
        "Signals and backtests use daily bars; transaction costs use 5 bps for buys and 5 bps for sells."
    )

    st.sidebar.header("Settings")
    selected_tickers = []
    for category, tickers in ETF_UNIVERSE.items():
        selected = st.sidebar.multiselect(f"Select {category}", tickers, default=tickers[:2])
        selected_tickers.extend(selected)

    selected_tickers = list(dict.fromkeys(selected_tickers))
    if not selected_tickers:
        st.warning("Please select at least one ticker from the sidebar.")
        return

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtest Settings")
    all_etf_symbols = get_all_etf_symbols()
    ma_default_index = all_etf_symbols.index("SPY") if "SPY" in all_etf_symbols else 0
    ma_symbol = st.sidebar.selectbox("Trend Following ETF", all_etf_symbols, index=ma_default_index)
    selected_strategy = st.sidebar.selectbox("Strategy Detail", STRATEGY_TYPES, index=0)
    backtest_period_label = st.sidebar.selectbox("Backtest Date Range", list(BACKTEST_PERIODS.keys()), index=2)
    portfolio_mode = st.sidebar.selectbox("Portfolio Mode", PORTFOLIO_MODES, index=2)
    top_n_stocks = st.sidebar.slider("Top N Stocks", min_value=3, max_value=25, value=10, step=1)
    benchmark_symbol = st.sidebar.selectbox("Benchmark", all_etf_symbols, index=ma_default_index)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps per buy/sell)", min_value=0, max_value=100, value=5, step=1)
    transaction_cost = transaction_cost_bps / 10000
    initial_portfolio_value = st.sidebar.number_input("Initial Capital", min_value=10000, value=INITIAL_CAPITAL, step=50000)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Allocation Settings")
    default_etf_weight = 0.70 if portfolio_mode == "ETF-only" else 0.0 if portfolio_mode == "Stock-only" else 0.60
    default_stock_weight = 0.0 if portfolio_mode == "ETF-only" else 0.90 if portfolio_mode == "Stock-only" else 0.30
    default_cash_weight = 1 - default_etf_weight - default_stock_weight
    etf_allocation_pct = st.sidebar.slider("ETF Allocation %", 0, 100, int(default_etf_weight * 100), step=5) / 100
    stock_allocation_pct = st.sidebar.slider("Stock Allocation %", 0, 100, int(default_stock_weight * 100), step=5) / 100
    cash_allocation_pct = st.sidebar.slider("Cash Allocation %", 0, 100, int(default_cash_weight * 100), step=5) / 100
    max_strategy_weight = st.sidebar.slider("Max Strategy Weight", 0.05, 0.50, RISK_LIMITS["max_strategy_weight"], step=0.01)
    max_position_weight = st.sidebar.slider("Max Position Weight", 0.02, 0.25, RISK_LIMITS["max_position_weight"], step=0.01)
    max_sector_exposure = st.sidebar.slider("Max Sector Exposure", 0.10, 0.60, RISK_LIMITS["max_sector_exposure"], step=0.05)
    risk_off_cash_min = st.sidebar.slider("Risk-Off Cash Minimum", 0.0, 0.50, RISK_LIMITS["risk_off_cash_minimum"], step=0.05)
    enable_cash_allocation = st.sidebar.checkbox("Enable Cash Allocation", value=True)

    st.sidebar.markdown("---")
    st.sidebar.write("**ETF universe includes:**")
    for category, tickers in ETF_UNIVERSE.items():
        st.sidebar.write(f"- **{category}**: {', '.join(tickers)}")

    backtest_end = pd.Timestamp.today().normalize()
    backtest_start = backtest_end - pd.DateOffset(years=BACKTEST_PERIODS[backtest_period_label])
    required_strategy_symbols = {"SPY", "SHY", ma_symbol, benchmark_symbol} | set(DEFENSIVE_ETFS)
    backtest_symbols = tuple(sorted(set(get_all_etf_symbols()) | set(selected_tickers) | required_strategy_symbols))
    stock_symbols = tuple(sorted(get_all_stock_symbols()))

    with st.spinner("Downloading cached daily OHLCV ETF data..."):
        backtest_ohlcv = load_backtest_ohlcv_data(
            backtest_symbols,
            backtest_start.strftime("%Y-%m-%d"),
            backtest_end.strftime("%Y-%m-%d"),
        )

    if not backtest_ohlcv or backtest_ohlcv["close"].empty:
        st.error("Unable to load backtest data. Please try again later or select fewer ETFs.")
        return

    backtest_close = backtest_ohlcv["close"].copy()
    with st.spinner("Downloading cached daily stock data..."):
        stock_close = load_close_data(
            stock_symbols,
            backtest_start.strftime("%Y-%m-%d"),
            backtest_end.strftime("%Y-%m-%d"),
        )
    backtest_close, stock_close, benchmark_close = align_market_data(backtest_close, stock_close, benchmark_symbol=benchmark_symbol)
    refresh_time = pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(
        f"Using one cached daily OHLCV download for {len(backtest_close.columns)} ETFs from "
        f"{backtest_close.index.min().date()} to {backtest_close.index.max().date()}. "
        f"Last refresh: {refresh_time}. Latest available data date: {backtest_close.index.max().date()}."
    )
    st.caption("Hourly refresh does not mean intraday trading. Official signals and backtests use daily bars.")

    with st.spinner("Running ETF strategy backtests..."):
        strategy_results = run_all_strategy_backtests(
            backtest_close,
            transaction_cost,
            ma_symbol=ma_symbol,
            benchmark_symbol=benchmark_symbol,
        )

    strategy_results = {name: result for name, result in strategy_results.items() if result is not None}
    if not strategy_results:
        st.error("No strategy backtests could be calculated with the selected data.")
        return

    regime = detect_market_regime(backtest_close)
    strategy_allocation = build_regime_aware_strategy_allocation(strategy_results, regime, max_strategy_weight=max_strategy_weight)
    asset_targets = adjust_asset_class_targets(
        regime["regime_name"],
        etf_allocation_pct,
        stock_allocation_pct,
        cash_allocation_pct,
        enable_cash=enable_cash_allocation,
        risk_off_cash_min=risk_off_cash_min,
    )
    if portfolio_mode == "ETF-only":
        cash_target = max(0.10 if enable_cash_allocation else 0.0, risk_off_cash_min if regime["regime_name"] in ["Risk-Off Bear Market", "High Volatility / Crisis"] else 0.0)
        asset_targets = {"ETFs": 1 - cash_target, "Stocks": 0.0, "Cash / SHY / BIL": cash_target}
    elif portfolio_mode == "Stock-only":
        cash_target = max(0.10 if enable_cash_allocation else 0.0, risk_off_cash_min if regime["regime_name"] in ["Risk-Off Bear Market", "High Volatility / Crisis"] else 0.0)
        asset_targets = {"ETFs": 0.0, "Stocks": 1 - cash_target, "Cash / SHY / BIL": cash_target}
    stock_selection = calculate_stock_selection(stock_close, benchmark_close, regime, top_n=top_n_stocks)
    position_allocation = build_position_allocation(
        strategy_results,
        strategy_allocation,
        stock_selection,
        asset_targets,
        initial_portfolio_value,
        max_position_weight=max_position_weight,
        max_sector_exposure=max_sector_exposure,
    )
    if not stock_selection.empty:
        stock_weight_map = position_allocation[position_allocation["Asset Type"] == "Stock"].set_index("Ticker")["Portfolio Weight"]
        stock_selection = stock_selection.copy()
        stock_selection["Assigned Portfolio Weight"] = stock_selection["Ticker"].map(stock_weight_map).fillna(0.0)
        stock_selection["Dollar Allocation"] = stock_selection["Assigned Portfolio Weight"] * initial_portfolio_value
    portfolio_returns, portfolio_metrics = build_holdings_portfolio_backtest(
        position_allocation,
        backtest_close,
        stock_close,
        initial_capital=initial_portfolio_value,
        strategy_results=strategy_results,
    )

    portfolio_tab, regime_tab, strategy_allocation_tab, etf_signals_tab, stock_selection_tab, risk_tab, backtesting_tab = st.tabs(
        [
            "Portfolio Overview",
            "Market Regime",
            "Strategy Allocation",
            "ETF Signals",
            "Stock Selection",
            "Risk Dashboard",
            "Backtest / Walk-Forward",
        ]
    )

    with portfolio_tab:
        render_regime_aware_portfolio_overview(
            strategy_results,
            portfolio_returns,
            portfolio_metrics,
            regime,
            strategy_allocation,
            asset_targets,
            position_allocation,
            initial_capital=initial_portfolio_value,
        )

    with regime_tab:
        render_market_regime_tab(backtest_close, regime)

    with strategy_allocation_tab:
        render_strategy_allocation_tab(strategy_allocation)

    with etf_signals_tab:
        render_clean_strategy_monitoring(
            strategy_results,
            initial_capital=initial_portfolio_value,
            strategy_allocation=strategy_allocation,
            etf_portfolio_weight=asset_targets.get("ETFs", 0.0),
        )

    with stock_selection_tab:
        render_stock_selection_tab(stock_selection)

    with risk_tab:
        render_risk_dashboard(
            strategy_results,
            backtest_close,
            portfolio_metrics,
            position_allocation,
            asset_targets,
            max_position_weight,
            max_sector_exposure,
        )

    with backtesting_tab:
        render_clean_backtesting(
            strategy_results,
            portfolio_returns,
            portfolio_metrics,
            selected_strategy,
            benchmark_symbol,
            strategy_allocation=strategy_allocation,
            etf_portfolio_weight=asset_targets.get("ETFs", 0.0),
        )
        render_dynamic_backtest(strategy_results, backtest_close, max_strategy_weight, benchmark_symbol)

    st.write("---")
    st.write("Built with Streamlit, yfinance, pandas, numpy, and plotly.")
    st.caption("Data source: Yahoo Finance via yfinance. No API key is required.")


if __name__ == "__main__":
    main()
