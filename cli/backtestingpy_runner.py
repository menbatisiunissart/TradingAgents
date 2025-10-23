"""Backtesting CLI that replays TradingAgents batch decisions with Backtesting.py.

This module ingests the CSV output produced by ``cli.batch_runner`` (or the
interactive CLI) and evaluates the decisions with the `backtesting`
package. Orders are executed on the close of the same session the decision
was made by enabling ``trade_on_close``.

Usage
-----
The runner expects a YAML configuration file located in ``./config`` that
specifies the input CSV, broker parameters, and optional output locations.

Example invocation from the project root::

    python -m cli.backtestingpy_runner config/backtestingpy.yaml

The provided ``config/backtestingpy.yaml`` acts as a template covering the
available options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import yaml
import yfinance as yf
from backtesting import Backtest, Strategy

logger = logging.getLogger(__name__)


class DecisionDrivenStrategy(Strategy):
    """Simple long-only strategy that acts on pre-computed decisions."""

    decisions: Optional[Mapping[pd.Timestamp, str]] = None
    buy_value: str = "BUY"
    sell_value: str = "SELL"
    hold_value: str = "HOLD"
    stake: float = 1.0

    def init(self) -> None:  # noqa: D401 - Backtesting.py lifecycle hook
        """Store normalized decisions for quick access during ``next``."""

        self._decision_map: Dict[pd.Timestamp, str] = {}
        for date, value in dict(getattr(self, "decisions", {}) or {}).items():
            if value is None:
                continue
            timestamp = pd.Timestamp(date).normalize()
            self._decision_map[timestamp] = str(value).strip().upper()

        self._buy = str(getattr(self, "buy_value", "BUY")).strip().upper()
        self._sell = str(getattr(self, "sell_value", "SELL")).strip().upper()
        self._hold = str(getattr(self, "hold_value", "HOLD")).strip().upper()
        self._stake = float(getattr(self, "stake", 1.0))

    def next(self) -> None:  # pragma: no cover - invoked by Backtesting engine
        current_time = pd.Timestamp(self.data.index[-1]).normalize()
        decision = self._decision_map.get(current_time, self._hold)

        if decision == self._buy:
            if not self.position.is_long and self._stake > 0:
                self.buy(size=self._stake)
        elif decision == self._sell:
            if self.position.is_long:
                self.position.close()


def _normalize_stat_value(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive conversion
            pass

    return str(value)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise TypeError(f"Configuration at {path} must be a mapping")
    return dict(config)


def _resolve_path(base: Path, value: str, *, prefer_project_root: bool = True) -> Path:
    """Resolve ``value`` relative to the config directory or project root.

    By default relative paths prefer the project root (the parent of the config
    directory) when the resolved location does not already exist. This keeps
    inputs and generated reports aligned with the repo-level ``results`` folder.
    """

    path = Path(value)
    if path.is_absolute():
        return path

    config_candidate = (base / path).resolve()
    if config_candidate.exists():
        return config_candidate

    if not prefer_project_root:
        return config_candidate

    project_candidate = (base.parent / path).resolve()
    if project_candidate.exists():
        return project_candidate

    return project_candidate


def _resolve_existing_path(base: Path, value: str, description: str) -> Path:
    path = _resolve_path(base, value)
    if path.exists():
        return path

    attempted = []
    candidate = Path(value)
    if not candidate.is_absolute():
        attempted.append((base / candidate).resolve())
        attempted.append((base.parent / candidate).resolve())

    message = f"{description} not found at expected location: {path}"
    if attempted:
        message += f" (checked: {', '.join(str(p) for p in attempted)})"
    raise FileNotFoundError(message)


def _prepare_output_path(path: Optional[str], base_path: Path) -> Optional[Path]:
    if not path:
        return None
    resolved = _resolve_path(base_path, path)
    if resolved.suffix:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _ticker_column_candidates(source_name: str, ticker: str) -> List[str]:
    cleaned = str(source_name).strip()
    lower = cleaned.lower()
    ticker_lower = str(ticker).strip().lower()
    ticker_clean = ticker_lower.replace(".", "_").replace("-", "_")

    candidates = [lower]
    for suffix in {ticker_lower, ticker_clean}:
        if not suffix:
            continue
        candidates.append(f"{lower}_{suffix}")
        candidates.append(f"{suffix}_{lower}")

    seen: Dict[str, None] = {}
    for candidate in candidates:
        if candidate not in seen:
            seen[candidate] = None
    return list(seen.keys())


def _flatten_columns(columns: pd.Index) -> List[str]:
    if not isinstance(columns, pd.MultiIndex):
        return [str(col) for col in columns]

    flattened: List[str] = []
    for col in columns:
        parts = [str(part).strip() for part in col if str(part).strip()]
        flattened.append("_".join(parts) if parts else "")
    return flattened


def _fetch_price_history(
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    interval: str,
    column_mapping: Mapping[str, str],
) -> pd.DataFrame:
    download_end = end_date + dt.timedelta(days=1)
    history = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=download_end.isoformat(),
        interval=interval,
        group_by="column",
        progress=False,
    )

    if history.empty:
        raise ValueError(f"No price data returned for {ticker} between {start_date} and {end_date}")

    history = history.copy()
    history.columns = _flatten_columns(history.columns)

    normalized_columns = {str(col).strip().lower(): str(col) for col in history.columns}
    rename_map: Dict[str, str] = {}
    missing_required: List[str] = []

    for desired_name, source_name in column_mapping.items():
        if not source_name:
            continue

        actual_column = None
        for candidate in _ticker_column_candidates(source_name, ticker):
            actual_column = normalized_columns.get(candidate)
            if actual_column:
                break

        if actual_column:
            rename_map[actual_column] = desired_name
        elif desired_name in {"Open", "High", "Low", "Close"}:
            missing_required.append(str(source_name))

    history.rename(columns=rename_map, inplace=True)

    required_cols = ["Open", "High", "Low", "Close"]
    for column in required_cols:
        if column not in history.columns:
            available = ", ".join(history.columns)
            expected = (
                ", ".join(dict.fromkeys(missing_required))
                if missing_required
                else column
            )
            raise KeyError(
                f"Column '{column}' missing from price data for {ticker}; "
                f"expected source column(s): {expected}; available columns: {available}"
            )

    history = history[required_cols + [col for col in ("Volume",) if col in history.columns]].copy()
    history.index = pd.to_datetime(history.index).tz_localize(None)
    history.sort_index(inplace=True)
    history.dropna(subset=["Open", "High", "Low", "Close"], inplace=True)
    return history


def _build_decision_map(df: pd.DataFrame, decision_column: str) -> Dict[pd.Timestamp, str]:
    decisions = {}
    for row in df.itertuples():
        decision_value = getattr(row, decision_column)
        if not decision_value:
            continue
        decision_date = pd.Timestamp(row.analysis_date).normalize()
        decisions[decision_date] = str(decision_value).strip().upper()
    return decisions


def run_backtests(config_path: Path) -> None:
    base_path = config_path.parent.resolve()
    config = _load_config(config_path)

    decisions_csv = config.get("decisions_csv")
    if not decisions_csv:
        raise KeyError("Configuration must include 'decisions_csv'")

    decisions_path = _resolve_existing_path(base_path, str(decisions_csv), "Decisions CSV")
    decisions_df = pd.read_csv(decisions_path, parse_dates=["analysis_date"])

    if "ticker" not in decisions_df.columns or "analysis_date" not in decisions_df.columns:
        raise KeyError("Decisions CSV must include 'ticker' and 'analysis_date' columns")

    decision_column = config.get("decision_column", "final_decision")
    if decision_column not in decisions_df.columns:
        raise KeyError(f"Decision column '{decision_column}' not present in CSV")

    padding_days = int(config.get("price_padding_days", 0))
    interval = config.get("price_interval", "1d")

    price_columns = config.get("price_data_columns", {})
    column_mapping = {
        "Open": price_columns.get("open_col", "Open"),
        "High": price_columns.get("high_col", "High"),
        "Low": price_columns.get("low_col", "Low"),
        "Close": price_columns.get("close_col", "Close"),
        "Volume": price_columns.get("volume_col", "Volume"),
    }

    buy_value = config.get("buy_value", "BUY")
    sell_value = config.get("sell_value", "SELL")
    hold_value = config.get("hold_value", "HOLD")
    stake = float(config.get("stake", 1))

    backtest_options = config.get("backtest", {})
    cash = float(backtest_options.get("cash", 100_000))
    commission = float(backtest_options.get("commission", 0.0))
    exclusive_orders = bool(backtest_options.get("exclusive_orders", True))
    hedging = bool(backtest_options.get("hedging", False))

    outputs = config.get("outputs", {})
    stats_output = _prepare_output_path(outputs.get("stats_csv"), base_path)
    trades_dir = _prepare_output_path(outputs.get("trades_dir"), base_path)
    equity_dir = _prepare_output_path(outputs.get("equity_curve_dir"), base_path)

    summaries: List[Dict[str, Any]] = []

    for ticker, group in decisions_df.groupby("ticker"):
        group = group.sort_values("analysis_date").reset_index(drop=True)
        if group.empty:
            continue

        start_date = (group["analysis_date"].min() - pd.Timedelta(days=padding_days)).date()
        end_date = (group["analysis_date"].max() + pd.Timedelta(days=padding_days)).date()

        logger.info("Fetching price data for %s from %s to %s", ticker, start_date, end_date)
        price_history = _fetch_price_history(
            ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            column_mapping=column_mapping,
        )

        decision_map = _build_decision_map(group, decision_column)
        if not decision_map:
            logger.warning("No actionable decisions for %s; skipping", ticker)
            continue

        backtest = Backtest(
            price_history,
            DecisionDrivenStrategy,
            cash=cash,
            commission=commission,
            trade_on_close=True,
            exclusive_orders=exclusive_orders,
            hedging=hedging,
        )

        logger.info("Running backtest for %s", ticker)
        stats = backtest.run(
            decisions=decision_map,
            buy_value=buy_value,
            sell_value=sell_value,
            hold_value=hold_value,
            stake=stake,
        )

        summary = {"ticker": ticker}
        for key, value in stats.items():
            if key.startswith("_"):
                continue
            summary[key] = _normalize_stat_value(value)
        summaries.append(summary)

        if trades_dir and "_trades" in stats:
            trades_path = trades_dir / f"{ticker}_trades.csv"
            stats["_trades"].to_csv(trades_path, index=False)

        if equity_dir and "_equity_curve" in stats:
            equity_path = equity_dir / f"{ticker}_equity_curve.csv"
            stats["_equity_curve"].to_csv(equity_path, index=False)

    if summaries and stats_output:
        summaries_df = pd.DataFrame(summaries)
        summaries_df.to_csv(stats_output, index=False)
        logger.info("Wrote summary statistics to %s", stats_output)
    elif stats_output:
        logger.warning("No summaries generated; %s not written", stats_output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay TradingAgents decisions with Backtesting.py using a YAML config.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config/backtestingpy.yaml",
        help="Path to the YAML configuration file located under ./config.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_file():
        parser.error(f"Configuration file not found: {config_path}")
    run_backtests(config_path.resolve())


if __name__ == "__main__":
    main()
