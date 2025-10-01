"""Backtesting utilities using Backtrader with pre-generated TradingAgents decisions.

This module loads BUY/SELL/HOLD decisions exported by ``cli.main`` and
replays them against historical price data via the Backtrader engine.
Price candles are fetched dynamically through :class:`tradingagents.dataflows.yfin_utils.YFinanceUtils`
instead of relying on static CSV snapshots. Configuration details (decision
file path, broker parameters, history padding, etc.) are specified in a YAML
file to keep the backtesting logic isolated from the interactive CLI workflow.

Example
-------
Run from the project root:

.. code-block:: bash

    python -m cli.backtester --config config/backtester.yaml

The accompanying ``config/backtester.yaml`` contains a sample configuration
showing the expected shape.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import backtrader as bt
import pandas as pd
import yaml

from tradingagents.dataflows.yfin_utils import YFinanceUtils

class DecisionStrategy(bt.Strategy):
    """Simple long-only strategy driven by externally computed decisions."""

    params = dict(
        decisions=None,  # Dict mapping date -> decision
        hold_value="HOLD",
        buy_value="BUY",
        sell_value="SELL",
    )

    def __init__(self) -> None:
        if self.params.decisions is None:
            raise ValueError("Strategy requires a 'decisions' mapping")

        # Normalize decision keywords once to avoid repeated uppercasing.
        self._decisions = {
            key: str(value).strip().upper() if value is not None else ""
            for key, value in self.params.decisions.items()
        }
        self._hold = str(self.params.hold_value).strip().upper()
        self._buy = str(self.params.buy_value).strip().upper()
        self._sell = str(self.params.sell_value).strip().upper()

    def next(self) -> None:  # pragma: no cover - executed within Backtrader engine
        current_date = bt.num2date(self.datas[0].datetime[0]).date()
        decision = self._decisions.get(current_date)
        if not decision or decision == self._hold:
            return

        if decision == self._buy and not self.position:
            self.buy()
        elif decision == self._sell and self.position:
            self.close()


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise ValueError("Backtester configuration must be a mapping of keys to values.")

    config = dict(data)
    required = ["decisions_csv"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Missing required configuration keys: " + ", ".join(missing))

    return config


def _fetch_price_history(
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    *,
    padding_days: int = 0,
) -> pd.DataFrame:
    padded_start = start_date - dt.timedelta(days=padding_days)
    padded_end = end_date + dt.timedelta(days=padding_days)

    start_str = padded_start.strftime("%Y-%m-%d")
    end_str = padded_end.strftime("%Y-%m-%d")

    frame = YFinanceUtils.get_stock_data(ticker, start_str, end_str)
    if frame is None or frame.empty:
        raise ValueError(
            f"No price data returned for {ticker} between {start_str} and {end_str}."
        )

    # Ensure deterministic order and naive timestamps for Backtrader.
    frame = frame.sort_index()
    frame.index = pd.to_datetime(frame.index)
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_localize(None)

    return frame


def _dataframe_to_feed(
    frame: pd.DataFrame,
    column_config: Mapping[str, Any],
) -> bt.feeds.PandasData:
    column_config = dict(column_config)
    column_map = {
        column_config.get("open_col", "Open"): "open",
        column_config.get("high_col", "High"): "high",
        column_config.get("low_col", "Low"): "low",
        column_config.get("close_col", "Close"): "close",
        column_config.get("volume_col", "Volume"): "volume",
    }

    openinterest_source = column_config.get("openinterest_col")
    if openinterest_source:
        column_map[openinterest_source] = "openinterest"

    missing_columns = [source for source in column_map if source not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Price data missing required columns: " + ", ".join(missing_columns)
        )

    renamed = frame.rename(columns=column_map)

    for required in ("open", "high", "low", "close", "volume"):
        if required not in renamed.columns:
            raise ValueError(f"Prepared data missing column '{required}'.")

    if "openinterest" not in renamed.columns:
        renamed["openinterest"] = 0.0

    # Backtrader expects datetime index.
    renamed.index.name = "datetime"

    return bt.feeds.PandasData(dataname=renamed)


def _normalize_decisions(
    decisions: Iterable[Mapping[str, Any]],
    *,
    date_key: str,
    decision_key: str,
) -> Dict[str, Dict[dt.date, str]]:
    grouped: Dict[str, Dict[dt.date, str]] = {}
    for row in decisions:
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        raw_date = row.get(date_key)
        if not raw_date:
            continue
        try:
            date_value = dt.datetime.strptime(str(raw_date), "%Y-%m-%d").date()
        except ValueError:
            # Attempt pandas-compatible parsing fallback
            date_value = pd.to_datetime(raw_date).date()

        decision = row.get(decision_key, "")
        grouped.setdefault(ticker, {})[date_value] = decision
    return grouped


def run_backtests(config: Mapping[str, Any], *, base_path: Path) -> List[Dict[str, Any]]:
    decisions_path = (base_path / config["decisions_csv"]).expanduser().resolve()
    if not decisions_path.is_file():
        raise FileNotFoundError(f"Decisions CSV not found: {decisions_path}")

    date_key = config.get("analysis_date_column", "analysis_date")
    decision_key = config.get("decision_column", "final_decision")

    decisions_frame = pd.read_csv(decisions_path)
    decision_records = decisions_frame.to_dict(orient="records")
    grouped_decisions = _normalize_decisions(
        decision_records, date_key=date_key, decision_key=decision_key
    )

    initial_cash = float(config.get("cash", 100_000))
    commission = float(config.get("commission", 0.001))
    stake = int(config.get("stake", 1))
    padding_days = int(config.get("price_padding_days", 0))
    column_config = config.get("price_data_columns") or {}

    results: List[Dict[str, Any]] = []
    for ticker, ticker_decisions in grouped_decisions.items():
        if not ticker_decisions:
            continue

        decision_dates = sorted(ticker_decisions.keys())
        start_date = decision_dates[0]
        end_date = decision_dates[-1]

        price_frame = _fetch_price_history(
            ticker,
            start_date,
            end_date,
            padding_days=padding_days,
        )
        data_feed = _dataframe_to_feed(price_frame, column_config)

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission)
        cerebro.addsizer(bt.sizers.SizerFix, stake=stake)
        cerebro.adddata(data_feed, name=ticker)
        cerebro.addstrategy(
            DecisionStrategy,
            decisions=ticker_decisions,
            hold_value=config.get("hold_value", "HOLD"),
            buy_value=config.get("buy_value", "BUY"),
            sell_value=config.get("sell_value", "SELL"),
        )

        cerebro.run()
        final_value = cerebro.broker.getvalue()

        results.append(
            {
                "ticker": ticker,
                "starting_cash": initial_cash,
                "ending_value": final_value,
                "return_pct": ((final_value / initial_cash) - 1.0) * 100.0,
                "trades_executed": len(ticker_decisions),
            }
        )

    return results


def write_results(results: List[Dict[str, Any]], config: Mapping[str, Any], base_path: Path) -> None:
    if not results:
        return

    output_path_cfg = config.get("output_csv")
    if not output_path_cfg:
        return

    output_path = (base_path / output_path_cfg).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(results)
    frame.to_csv(output_path, index=False)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Backtrader backtests using TradingAgents decisions.")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to the YAML configuration file controlling the backtester.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    base_path = Path.cwd()
    config = load_config(base_path / args.config)
    results = run_backtests(config, base_path=base_path)
    write_results(results, config, base_path)

    for result in results:
        print(
            f"Ticker {result['ticker']}: start {result['starting_cash']:.2f} -> end {result['ending_value']:.2f}"
            f" ({result['return_pct']:.2f}% return)"
        )


if __name__ == "__main__":
    main()
