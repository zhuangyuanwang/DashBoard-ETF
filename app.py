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
]

BACKTEST_PERIODS = {
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
}


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


def calculate_backtest_metrics(strategy_equity, strategy_returns, number_rebalances):
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


def run_strategy_backtest(strategy_type, close, transaction_cost, ma_symbol="SPY", benchmark_symbol="SPY"):
    if strategy_type == "Momentum Rotation":
        return run_momentum_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    if strategy_type == "Moving Average Trend Following":
        return run_moving_average_trend_backtest(close, symbol=ma_symbol, benchmark_symbol=ma_symbol, transaction_cost=transaction_cost)
    if strategy_type == "Dual Momentum":
        return run_dual_momentum_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    if strategy_type == "Low Volatility Rotation":
        return run_low_volatility_backtest(close, transaction_cost=transaction_cost, benchmark_symbol=benchmark_symbol)
    return None


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


def render_strategy_results(result):
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
    backtest_period_label = st.sidebar.selectbox("Backtest Date Range", list(BACKTEST_PERIODS.keys()), index=2)
    transaction_cost_bps = st.sidebar.slider("Transaction Cost (bps)", min_value=0, max_value=100, value=5, step=1)
    transaction_cost = transaction_cost_bps / 10000
    st.sidebar.markdown("---")
    st.sidebar.write("**ETF universe includes:**")
    for category, tickers in ETF_UNIVERSE.items():
        st.sidebar.write(f"- **{category}**: {', '.join(tickers)}")

    close = load_data(selected_tickers, history_period)
    if close.empty:
        st.error("Failed to load data. Please try again or select fewer tickers.")
        return

    returns = compute_returns(close)
    performance = compute_performance(close)
    volatility = compute_volatility(returns)
    signal = moving_average_signal(close)

    top_perf = performance[summary_window].idxmax()
    worst_perf = performance[summary_window].idxmin()
    highest_vol = volatility.sort_values(ascending=False).index[0]
    bullish_count = len(signal[signal == "Bullish"])

    main_tabs = st.tabs(["Overview", "Category Charts", "Strategy Testing"])
    overview_tab, category_tab, strategy_tab = main_tabs

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

        backtest_end = pd.Timestamp.today().normalize()
        backtest_start = backtest_end - pd.DateOffset(years=BACKTEST_PERIODS[backtest_period_label])
        backtest_symbols = tuple(sorted(set(selected_tickers) | {"SPY", "SHY", ma_symbol}))

        with st.spinner("Downloading backtest price data..."):
            backtest_close = load_backtest_data(
                backtest_symbols,
                backtest_start.strftime("%Y-%m-%d"),
                backtest_end.strftime("%Y-%m-%d"),
            )

        if backtest_close.empty:
            st.error("Unable to load backtest data. Please try again later.")
        else:
            st.caption(
                f"Backtest data loaded once for {len(backtest_close.columns)} ETFs from "
                f"{backtest_close.index.min().date()} to {backtest_close.index.max().date()}."
            )

            comparison_results = {}
            for comparison_strategy in STRATEGY_TYPES:
                comparison_results[comparison_strategy] = run_strategy_backtest(
                    comparison_strategy,
                    backtest_close,
                    transaction_cost,
                    ma_symbol=ma_symbol,
                )

            strategy_tabs = st.tabs(STRATEGY_TYPES)
            for comparison_strategy, tab in zip(STRATEGY_TYPES, strategy_tabs):
                with tab:
                    result = comparison_results[comparison_strategy]
                    if result is None:
                        st.error("Backtest failed due to insufficient historical data for this strategy.")
                    else:
                        render_strategy_results(result)

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
