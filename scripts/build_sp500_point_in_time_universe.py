"""Build a point-in-time S&P 500 membership file from a reproducible public source.

This script records membership intervals only. It does not provide delisting
returns; the Stage A2 strict coverage gate rejects a backtest if price history
for any required historical constituent is unavailable.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


SOURCE_URL = "https://raw.githubusercontent.com/leandroloi/historical_sp500_constituents/master/sp_500_historical_components.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_universe_point_in_time.csv"


def yahoo_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def main() -> None:
    with urlopen(SOURCE_URL, timeout=60) as response:
        snapshots = pd.read_csv(io.BytesIO(response.read()))
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="raise")
    snapshots = snapshots.sort_values("date")
    active: dict[str, pd.Timestamp] = {}
    intervals: list[dict[str, str]] = []
    coverage_end = snapshots["date"].max().date().isoformat()

    for row in snapshots.itertuples(index=False):
        as_of = pd.Timestamp(row.date)
        members = {yahoo_ticker(ticker) for ticker in str(row.tickers).split(",") if ticker.strip()}
        for ticker in set(active) - members:
            intervals.append(
                {
                    "Ticker": ticker,
                    "Sector": "Other",
                    "Industry": "Other",
                    "Category": "S&P 500 Historical Constituent",
                    "StartDate": active.pop(ticker).date().isoformat(),
                    "EndDate": as_of.date().isoformat(),
                    "CoverageEnd": coverage_end,
                    "MembershipSource": SOURCE_URL,
                }
            )
        for ticker in members - set(active):
            active[ticker] = as_of

    for ticker, start_date in active.items():
        intervals.append(
            {
                "Ticker": ticker,
                "Sector": "Other",
                "Industry": "Other",
                "Category": "S&P 500 Historical Constituent",
                "StartDate": start_date.date().isoformat(),
                "EndDate": "",
                "CoverageEnd": coverage_end,
                "MembershipSource": SOURCE_URL,
            }
        )

    intervals.sort(key=lambda row: (row["StartDate"], row["Ticker"]))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervals[0]))
        writer.writeheader()
        writer.writerows(intervals)
    print(f"Wrote {len(intervals)} membership intervals through {coverage_end} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
