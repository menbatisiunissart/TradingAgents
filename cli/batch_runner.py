"""Utilities to run the TradingAgents CLI programmatically in batches.

This module provides a thin wrapper around :mod:`cli.main` that allows
running the CLI multiple times with different configurations without
manually answering the interactive prompts each time.
"""

from __future__ import annotations

import datetime as _dt
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Union
from unittest.mock import patch

from cli import main as cli_main
from cli.models import AnalystType

AnalystInput = Union[str, AnalystType]


def _normalize_date(value: Union[str, _dt.date]) -> str:
    """Normalize a provided date into the ``YYYY-MM-DD`` string expected by the CLI."""
    if isinstance(value, _dt.date):
        return value.strftime("%Y-%m-%d")
    return str(value)


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
    jobs: Iterable[Mapping[str, Any]],
    *,
    pause_seconds: float = 0.0,
) -> None:
    """Run the CLI sequentially for each provided job configuration.

    Parameters
    ----------
    jobs:
        Iterable of configuration mappings. Each mapping can contain the following keys:
        ``ticker`` (required), ``analysis_date`` (required), ``analysts``, ``research_depth``,
        ``llm_provider``, ``backend_url``, ``shallow_thinker``, ``deep_thinker``, and
        ``config_overrides``. The ``config_overrides`` value, when provided, should be a mapping
        that updates the default configuration used by the CLI before applying selections.
    pause_seconds:
        Optional pause between runs. This can be useful to respect API rate limits.
    """
    job_list = list(jobs)

    for index, job in enumerate(job_list, start=1):
        if "ticker" not in job or "analysis_date" not in job:
            raise ValueError("Each job must include 'ticker' and 'analysis_date'.")

        overrides: MutableMapping[str, Any] = {}
        overrides.update(cli_main.DEFAULT_CONFIG)
        overrides.update(job.get("config_overrides", {}))

        selections = _build_selections(
            ticker=str(job["ticker"]),
            analysis_date=job["analysis_date"],
            analysts=job.get("analysts"),
            research_depth=job.get("research_depth"),
            llm_provider=job.get("llm_provider"),
            backend_url=job.get("backend_url"),
            shallow_thinker=job.get("shallow_thinker"),
            deep_thinker=job.get("deep_thinker"),
            base_config=overrides,
        )

        _reset_message_buffer()

        with _patched_defaults(dict(overrides)), patch(
            "cli.main.get_user_selections", return_value=selections
        ):
            cli_main.run_analysis()

        if pause_seconds and index < len(job_list):
            time.sleep(pause_seconds)


__all__ = ["run_batch"]


if __name__ == "__main__":
    example_jobs = [
        {"ticker": "SPY", "analysis_date": "2024-01-05", "research_depth": 1},
        {"ticker": "AAPL", "analysis_date": "2024-02-01", "research_depth": 3},
    ]
    run_batch(example_jobs)
