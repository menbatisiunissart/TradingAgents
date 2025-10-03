"""Backtesting utilities using Backtrader with pre-generated TradingAgents decisions.

This module loads BUY/SELL/HOLD decisions exported by ``cli.main`` and
replays them against historical price data via the Backtrader engine.
Price candles are fetched dynamically through :class:`tradingagents.dataflows.yfin_utils.YFinanceUtils`
instead of relying on static CSV snapshots. Configuration details (decision
file path, broker parameters, history padding, evaluation metric knobs, etc.)
are specified in a YAML file to keep the backtesting logic isolated from the
interactive CLI workflow. The exported CSV includes a ``model`` column that
identifies either the TradingAgents run or one of the baseline strategies
evaluated for comparison.

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
import statistics
from collections import OrderedDict, defaultdict
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

        # Capture what is shown on the Backtrader plot so it can be exported.
        self.trading_log: List[Dict[str, Any]] = []
        self._executed_orders = defaultdict(list)

    def notify_order(self, order: bt.Order) -> None:  # pragma: no cover - Backtrader callback
        if order.status not in {order.Completed, order.Partial}:
            return

        executed_dt = order.executed.dt or order.created.dt
        if executed_dt is None:
            return

        timestamp = bt.num2date(executed_dt).replace(tzinfo=None)
        action = "BUY" if order.isbuy() else "SELL"
        self._executed_orders[timestamp.isoformat()].append(
            {
                "action": action,
                "size": float(order.executed.size),
                "price": float(order.executed.price or 0.0),
                "value": float(order.executed.value or 0.0),
            }
        )

    def next(self) -> None:  # pragma: no cover - executed within Backtrader engine
        data = self.datas[0]
        current_dt = bt.num2date(data.datetime[0]).replace(tzinfo=None)
        current_date = current_dt.date()
        decision = self._decisions.get(current_date)

        order_action = ""
        if decision == self._buy and not self.position:
            self.buy(exectype=bt.Order.Close)
            order_action = "BUY"
        elif decision == self._sell and self.position:
            self.close(exectype=bt.Order.Close)
            order_action = "SELL"

        key = current_dt.isoformat()
        executed = self._executed_orders.get(key, [])
        if executed:
            del self._executed_orders[key]
        executed_summary = ";".join(
            f"{event['action']}:{event['size']:.4f}@{event['price']:.4f}"
            for event in executed
        )

        volume_value = float("nan")
        if hasattr(data, "volume"):
            try:
                volume_value = float(data.volume[0])
            except (TypeError, IndexError):
                pass

        self.trading_log.append(
            {
                "datetime": current_dt.isoformat(),
                "open": float(data.open[0]),
                "high": float(data.high[0]),
                "low": float(data.low[0]),
                "close": float(data.close[0]),
                "volume": volume_value,
                "decision": decision or self._hold,
                "order_submitted": order_action,
                "orders_executed": executed_summary,
                "position_size": float(self.position.size),
                "cash": float(self.broker.getcash()),
                "portfolio_value": float(self.broker.getvalue()),
            }
        )


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


def _compute_performance_metrics(
    *,
    initial_cash: float,
    final_value: float,
    start_date: dt.date,
    end_date: dt.date,
    daily_returns: Mapping[Any, float],
    risk_free_rate: float,
    periods_per_year: int,
    drawdown_analysis: Mapping[str, Any],
) -> Dict[str, Optional[float]]:
    cumulative_return = ((final_value / initial_cash) - 1.0) * 100.0

    duration_days = max((end_date - start_date).days + 1, 1)
    duration_years = duration_days / 365.25
    if duration_years > 0:
        annualized_return = ((final_value / initial_cash) ** (1.0 / duration_years) - 1.0) * 100.0
    else:
        annualized_return = None

    ordered_returns = [value for _, value in sorted(daily_returns.items())]
    if ordered_returns and abs(ordered_returns[0]) < 1e-12:
        ordered_returns = ordered_returns[1:]

    sharpe_ratio: Optional[float] = None
    if ordered_returns:
        average_return = sum(ordered_returns) / len(ordered_returns)
        per_period_risk_free = risk_free_rate / max(periods_per_year, 1)
        if len(ordered_returns) > 1:
            volatility = statistics.pstdev(ordered_returns)
        else:
            volatility = 0.0
        if volatility:
            sharpe_ratio = (average_return - per_period_risk_free) / volatility

    max_drawdown = None
    if drawdown_analysis:
        max_drawdown = drawdown_analysis.get("maxdrawdown")

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }


def _build_signal_series(price_frame: pd.DataFrame, decisions: Mapping[dt.date, str]) -> pd.Series:
    signals = []
    for index, _ in price_frame.iterrows():
        signals.append(decisions.get(index.date(), "HOLD"))
    return pd.Series(signals, index=price_frame.index)


def _simulate_decision_series(
    price_frame: pd.DataFrame,
    decisions: Mapping[dt.date, str],
    *,
    initial_cash: float,
    commission: float,
) -> Dict[str, Any]:
    if price_frame.empty:
        return {
            "final_value": initial_cash,
            "trades": 0,
            "returns": OrderedDict(),
            "max_drawdown": None,
        }

    cash = initial_cash
    position = 0.0
    trades = 0
    portfolio_values: List[float] = []

    for index, row in price_frame.iterrows():
        decision = decisions.get(index.date(), "HOLD")
        close_price = float(row["Close"])

        if decision == "BUY" and position == 0 and close_price > 0:
            investable_cash = cash * (1 - commission)
            position = investable_cash / close_price
            cash = 0.0
            trades += 1
        elif decision == "SELL" and position > 0 and close_price > 0:
            proceeds = position * close_price * (1 - commission)
            cash = proceeds
            position = 0.0
            trades += 1

        portfolio_values.append(cash + position * close_price)

    if position > 0:
        final_price = float(price_frame["Close"].iloc[-1])
        portfolio_values[-1] = cash + position * final_price

    portfolio_series = pd.Series(portfolio_values, index=price_frame.index)
    returns_series = portfolio_series.pct_change().dropna()
    returns_mapping = OrderedDict((ts.to_pydatetime(), value) for ts, value in returns_series.items())

    cumulative_max = portfolio_series.cummax()
    drawdowns = (1.0 - (portfolio_series / cumulative_max)) * 100.0
    if drawdowns.empty:
        max_drawdown = None
    else:
        max_drawdown_value = drawdowns.max()
        max_drawdown = (
            float(max_drawdown_value) if pd.notna(max_drawdown_value) else None
        )

    return {
        "final_value": float(portfolio_series.iloc[-1]),
        "trades": trades,
        "returns": returns_mapping,
        "max_drawdown": max_drawdown,
    }


def _generate_backtrader_plot(
    cerebro: bt.Cerebro,
    *,
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    base_path: Path,
    config: Mapping[str, Any],
) -> None:
    plot_dir_cfg = config.get("plot_output_dir")
    if not plot_dir_cfg:
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import dates as mdates
        from matplotlib import pyplot as plt
        from matplotlib.figure import Figure
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Skipping plot generation for {ticker}: {exc}")
        return

    output_dir = (base_path / plot_dir_cfg).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    style = str(config.get("plot_style", "candlestick"))
    volume = bool(config.get("plot_volume", True))
    dpi = int(config.get("plot_dpi", 120))

    try:
        plots = cerebro.plot(style=style, volume=volume, iplot=False)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"Failed to generate Backtrader plot for {ticker}: {exc}")
        return

    figures: List[Figure] = []
    stack: List[Any] = list(plots if isinstance(plots, (list, tuple)) else [plots])
    while stack:
        item = stack.pop()
        if isinstance(item, Figure):
            figures.append(item)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)

    if not figures:
        return

    filename = f"{ticker}_{start_date:%Y%m%d}_{end_date:%Y%m%d}"
    day_formatter = mdates.DateFormatter("%d")
    for index, figure in enumerate(figures, start=1):
        for axis in figure.get_axes():
            axis.xaxis.set_major_formatter(day_formatter)
        suffix = "" if len(figures) == 1 else f"_{index}"
        output_path = output_dir / f"{filename}{suffix}.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)


def _export_plot_data_csv(
    trading_log: Iterable[Mapping[str, Any]],
    *,
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    base_path: Path,
    config: Mapping[str, Any],
) -> None:
    rows = list(trading_log)
    if not rows:
        return

    output_dir_cfg = config.get("plot_data_output_dir") or config.get("plot_output_dir")
    if not output_dir_cfg:
        return

    output_dir = (base_path / output_dir_cfg).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{ticker}_{start_date:%Y%m%d}_{end_date:%Y%m%d}_plotdata.csv"
    frame = pd.DataFrame(rows)
    column_order = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "decision",
        "order_submitted",
        "orders_executed",
        "position_size",
        "cash",
        "portfolio_value",
    ]
    existing_columns = [col for col in column_order if col in frame.columns]
    frame = frame[existing_columns]

    output_path = output_dir / filename
    frame.to_csv(output_path, index=False)


def _baseline_buy_and_hold(
    price_frame: pd.DataFrame,
) -> Dict[dt.date, str]:
    if price_frame.empty:
        return {}
    signals: Dict[dt.date, str] = {}
    first_date = price_frame.index[0].date()
    last_date = price_frame.index[-1].date()
    signals[first_date] = "BUY"
    signals[last_date] = "SELL"
    return signals


def _baseline_macd(price_frame: pd.DataFrame) -> Dict[dt.date, str]:
    signals: Dict[dt.date, str] = {}
    if price_frame.empty:
        return signals

    close = price_frame["Close"].astype(float)
    ema_short = close.ewm(span=12, adjust=False).mean()
    ema_long = close.ewm(span=26, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=9, adjust=False).mean()

    prev_diff = None
    for timestamp, diff in (macd - signal_line).items():
        current_date = timestamp.date()
        if pd.isna(diff) or (prev_diff is None):
            prev_diff = diff
            continue
        if diff > 0 and prev_diff <= 0:
            signals[current_date] = "BUY"
        elif diff < 0 and prev_diff >= 0:
            signals[current_date] = "SELL"
        prev_diff = diff
    return signals


def _baseline_kdj_rsi(price_frame: pd.DataFrame) -> Dict[dt.date, str]:
    signals: Dict[dt.date, str] = {}
    if price_frame.empty:
        return signals

    high = price_frame["High"].astype(float)
    low = price_frame["Low"].astype(float)
    close = price_frame["Close"].astype(float)

    period = 9
    low_min = low.rolling(window=period, min_periods=period).min()
    high_max = high.rolling(window=period, min_periods=period).max()
    rsv = (close - low_min) / (high_max - low_min) * 100
    rsv = rsv.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0).fillna(0)

    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    d = k.ewm(alpha=1 / 3, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    prev_cross = None
    for timestamp, (k_val, d_val, rsi_val) in zip(k.index, zip(k, d, rsi)):
        if pd.isna(k_val) or pd.isna(d_val) or pd.isna(rsi_val):
            continue
        diff = k_val - d_val
        current_date = timestamp.date()
        if prev_cross is None:
            prev_cross = diff
            continue
        if diff > 0 and prev_cross <= 0 and rsi_val < 30:
            signals[current_date] = "BUY"
        elif diff < 0 and prev_cross >= 0 and rsi_val > 70:
            signals[current_date] = "SELL"
        prev_cross = diff
    return signals


def _baseline_zmr(price_frame: pd.DataFrame) -> Dict[dt.date, str]:
    signals: Dict[dt.date, str] = {}
    if price_frame.empty:
        return signals

    close = price_frame["Close"].astype(float)
    rolling = close.rolling(window=20, min_periods=20)
    mean = rolling.mean()
    std = rolling.std()
    zscore = (close - mean) / std

    prev_state = 0
    for timestamp, z_val in zscore.items():
        if pd.isna(z_val):
            continue
        current_date = timestamp.date()
        if z_val < -1 and prev_state >= 0:
            signals[current_date] = "BUY"
            prev_state = -1
        elif z_val > 0.5 and prev_state <= 0:
            signals[current_date] = "SELL"
            prev_state = 1
    return signals


def _baseline_sma(price_frame: pd.DataFrame) -> Dict[dt.date, str]:
    signals: Dict[dt.date, str] = {}
    if price_frame.empty:
        return signals

    close = price_frame["Close"].astype(float)
    short = close.rolling(window=20, min_periods=20).mean()
    long = close.rolling(window=50, min_periods=50).mean()
    prev_diff = None
    for timestamp, (short_val, long_val) in zip(short.index, zip(short, long)):
        if pd.isna(short_val) or pd.isna(long_val):
            continue
        diff = short_val - long_val
        current_date = timestamp.date()
        if prev_diff is None:
            prev_diff = diff
            continue
        if diff > 0 and prev_diff <= 0:
            signals[current_date] = "BUY"
        elif diff < 0 and prev_diff >= 0:
            signals[current_date] = "SELL"
        prev_diff = diff
    return signals


def _evaluate_baselines(
    *,
    price_frame: pd.DataFrame,
    initial_cash: float,
    commission: float,
    start_date: dt.date,
    end_date: dt.date,
    risk_free_rate: float,
    periods_per_year: int,
) -> List[Dict[str, Any]]:
    baseline_builders = {
        "Buy and Hold": _baseline_buy_and_hold,
        "MACD": _baseline_macd,
        "KDJ+RSI": _baseline_kdj_rsi,
        "ZMR": _baseline_zmr,
        "SMA": _baseline_sma,
    }

    results: List[Dict[str, Any]] = []
    for model_name, builder in baseline_builders.items():
        decisions = builder(price_frame)
        simulation = _simulate_decision_series(
            price_frame,
            decisions,
            initial_cash=initial_cash,
            commission=commission,
        )
        metrics = _compute_performance_metrics(
            initial_cash=initial_cash,
            final_value=simulation["final_value"],
            start_date=start_date,
            end_date=end_date,
            daily_returns=simulation["returns"],
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            drawdown_analysis={"maxdrawdown": simulation.get("max_drawdown")},
        )

        results.append(
            {
                "model": model_name,
                "starting_cash": initial_cash,
                "ending_value": simulation["final_value"],
                "return_pct": ((simulation["final_value"] / initial_cash) - 1.0) * 100.0,
                "trades_executed": simulation["trades"],
                **metrics,
            }
        )
    return results


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
    risk_free_rate = float(config.get("risk_free_rate", 0.0))
    periods_per_year = int(config.get("trading_days_per_year", 252))

    results: List[Dict[str, Any]] = []
    for ticker, ticker_decisions in grouped_decisions.items():
        if not ticker_decisions:
            continue

        # Order the decision dates explicitly so we know the first and last
        # trading day covered by the generated signals.
        sorted_decision_dates = sorted(
            ticker_decisions.keys(),
            key=lambda decision_date: decision_date,
        )
        start_date = sorted_decision_dates[0]
        end_date = sorted_decision_dates[-1]

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
        cerebro.broker.set_coc(True)
        cerebro.addsizer(bt.sizers.SizerFix, stake=stake)
        cerebro.adddata(data_feed, name=ticker)
        cerebro.addstrategy(
            DecisionStrategy,
            decisions=ticker_decisions,
            hold_value=config.get("hold_value", "HOLD"),
            buy_value=config.get("buy_value", "BUY"),
            sell_value=config.get("sell_value", "SELL"),
        )
        cerebro.addanalyzer(
            bt.analyzers.TimeReturn,
            _name="returns",
            timeframe=bt.TimeFrame.Days,
        )
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        strategies = cerebro.run()
        strategy = strategies[0]
        final_value = cerebro.broker.getvalue()

        returns_analysis = strategy.analyzers.returns.get_analysis()
        drawdown_analysis = strategy.analyzers.drawdown.get_analysis()

        metrics = _compute_performance_metrics(
            initial_cash=initial_cash,
            final_value=final_value,
            start_date=start_date,
            end_date=end_date,
            daily_returns=returns_analysis,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            drawdown_analysis=drawdown_analysis,
        )

        _export_plot_data_csv(
            getattr(strategy, "trading_log", []),
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            base_path=base_path,
            config=config,
        )
        
        _generate_backtrader_plot(
            cerebro,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            base_path=base_path,
            config=config,
        )

        results.append(
            {
                "model": "TradingAgents",
                "ticker": ticker,
                "starting_cash": initial_cash,
                "ending_value": final_value,
                "return_pct": ((final_value / initial_cash) - 1.0) * 100.0,
                "trades_executed": len(ticker_decisions),
                **metrics,
            }
        )

        baseline_rows = _evaluate_baselines(
            price_frame=price_frame,
            initial_cash=initial_cash,
            commission=commission,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

        for row in baseline_rows:
            row.update({"ticker": ticker})
            results.append(row)

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
    desired_order = ["ticker", "model"]
    remaining_columns = [col for col in frame.columns if col not in desired_order]
    frame = frame[[*desired_order, *remaining_columns]]
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
