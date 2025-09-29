"""Utilities to run the TradingAgents CLI programmatically in batches.

This module provides a thin wrapper around :mod:`cli.main` that allows
running the CLI multiple times with different configurations without
manually answering the interactive prompts each time.
"""

from __future__ import annotations

import datetime as _dt
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union
from unittest.mock import patch

from cli import main as cli_main
from cli.models import AnalystType

AnalystInput = Union[str, AnalystType]


def _normalize_date(value: Union[str, _dt.date]) -> str:
    """Normalize a provided date into the ``YYYY-MM-DD`` string expected by the CLI."""
    if isinstance(value, _dt.date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _coerce_to_date(value: Union[str, _dt.date]) -> _dt.date:
    """Convert a string or date value into a :class:`datetime.date` instance."""
    if isinstance(value, _dt.date):
        return value
    try:
        return _dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Dates must be provided as datetime.date instances or ISO formatted strings"
        ) from exc


def _iterate_dates(start_date: Union[str, _dt.date], end_date: Union[str, _dt.date]) -> Iterator[_dt.date]:
    """Yield every date between ``start_date`` and ``end_date`` (inclusive)."""
    start = _coerce_to_date(start_date)
    end = _coerce_to_date(end_date)
    if end < start:
        raise ValueError("'end_date' must be on or after 'start_date'.")

    current = start
    while current <= end:
        yield current
        current += _dt.timedelta(days=1)


def _normalize_analysts(analysts: Optional[Sequence[AnalystInput]]) -> List[AnalystType]:
    """Convert analyst inputs into ``AnalystType`` instances."""
    if not analysts:
        return [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]

    normalized: List[AnalystType] = []
    for analyst in analysts:
        if isinstance(analyst, AnalystType):
            normalized.append(analyst)
            continue
        try:
            normalized.append(AnalystType(analyst.lower()))
        except ValueError as exc:
            valid = ", ".join(a.value for a in AnalystType)
            raise ValueError(f"Unknown analyst '{analyst}'. Valid options: {valid}.") from exc
    return normalized


def _build_selections(
    *,
    ticker: str,
    analysis_date: Union[str, _dt.date],
    analysts: Optional[Sequence[AnalystInput]] = None,
    research_depth: Optional[int] = None,
    llm_provider: Optional[str] = None,
    backend_url: Optional[str] = None,
    shallow_thinker: Optional[str] = None,
    deep_thinker: Optional[str] = None,
    base_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create the selection dictionary normally gathered by interactive prompts."""
    selections: Dict[str, Any] = {
        "ticker": ticker.upper(),
        "analysis_date": _normalize_date(analysis_date),
        "analysts": _normalize_analysts(analysts),
        "research_depth": research_depth or base_config.get("max_debate_rounds", 1),
        "llm_provider": (llm_provider or base_config.get("llm_provider", "openai")).lower(),
        "backend_url": backend_url or base_config.get("backend_url"),
        "shallow_thinker": shallow_thinker or base_config.get("quick_think_llm"),
        "deep_thinker": deep_thinker or base_config.get("deep_think_llm"),
    }

    missing = [key for key in ("backend_url", "shallow_thinker", "deep_thinker") if not selections.get(key)]
    if missing:
        raise ValueError(
            "Missing required configuration values: " + ", ".join(missing)
        )

    return selections


@contextmanager
def _patched_defaults(new_defaults: Mapping[str, Any]) -> Iterator[None]:
    """Temporarily replace ``cli.main.DEFAULT_CONFIG`` with ``new_defaults``."""
    with patch("cli.main.DEFAULT_CONFIG", new_defaults):
        yield


def _reset_message_buffer() -> None:
    """Reset the global message buffer used by ``cli.main`` before each run."""
    cli_main.message_buffer = cli_main.MessageBuffer()


def run_batch(
    *,
    tickers: Sequence[str],
    start_date: Union[str, _dt.date],
    end_date: Union[str, _dt.date],
    research_depth: Optional[int] = None,
    analysts: Optional[Sequence[AnalystInput]] = None,
    llm_provider: Optional[str] = None,
    backend_url: Optional[str] = None,
    shallow_thinker: Optional[str] = None,
    deep_thinker: Optional[str] = None,
    pause_seconds: float = 0.0,
) -> None:
    """Run the CLI sequentially for each ticker across a shared date range.

    Parameters
    ----------
    tickers
        Ordered collection of ticker symbols to process.
    start_date
        Inclusive beginning of the analysis window shared by all tickers.
    end_date
        Inclusive end of the analysis window shared by all tickers.
    research_depth
        Optional debate depth to reuse for each run; falls back to config when ``None``.
    analysts
        Optional list of analysts to include. Defaults to the CLI's standard set when omitted.
    llm_provider
        Override for the LLM provider; defaults to the CLI configuration.
    backend_url
        Override for the backend service endpoint required by the CLI.
    shallow_thinker
        Name of the quick-think LLM to use for every run.
    deep_thinker
        Name of the deep-think LLM to use for every run.
    pause_seconds
        Delay inserted between runs to avoid hammering downstream services.

    The provided ``start_date`` and ``end_date`` are applied to every ticker in ``tickers``.
    ``research_depth`` (when provided) is likewise reused for each run.
    """

    if not tickers:
        raise ValueError("'tickers' must be a non-empty sequence of symbols.")

    dates = list(_iterate_dates(start_date, end_date))

    defaults: Mapping[str, Any] = dict(cli_main.DEFAULT_CONFIG)

    total_runs = len(tickers) * len(dates)
    run_index = 0

    for ticker in tickers:
        for analysis_date in dates:
            run_index += 1

            selections = _build_selections(
                ticker=str(ticker),
                analysis_date=analysis_date,
                analysts=analysts,
                research_depth=research_depth,
                llm_provider=llm_provider,
                backend_url=backend_url,
                shallow_thinker=shallow_thinker,
                deep_thinker=deep_thinker,
                base_config=defaults,
            )

            _reset_message_buffer()

            with _patched_defaults(dict(defaults)), patch(
                "cli.main.get_user_selections", return_value=selections
            ):
                cli_main.run_analysis()

            if pause_seconds and run_index < total_runs:
                time.sleep(pause_seconds)


__all__ = ["run_batch"]


if __name__ == "__main__":
    run_batch(
        tickers=["SPY", "AAPL"],
        start_date="2024-01-05",
        end_date="2024-01-06",
        research_depth=2,
        pause_seconds=1.0,
    )
