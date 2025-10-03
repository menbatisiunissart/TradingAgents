"""Utilities to run the TradingAgents CLI programmatically in batches.

This module provides a thin wrapper around :mod:`cli.main` that allows
running the CLI multiple times with different configurations without
manually answering the interactive prompts each time.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import sys
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union
from unittest.mock import patch

import yaml

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


def load_batch_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Read a YAML batch configuration and return a dictionary of run options."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, Mapping):
        raise ValueError("Batch configuration must be a mapping of option names to values.")

    options = dict(data)
    allowed_keys = {
        "tickers",
        "start_date",
        "end_date",
        "research_depth",
        "analysts",
        "llm_provider",
        "backend_url",
        "shallow_thinker",
        "deep_thinker",
        "pause_seconds",
        "project",
    }

    unexpected = sorted(set(options) - allowed_keys)
    if unexpected:
        raise ValueError(
            "Unexpected configuration keys: " + ", ".join(unexpected)
        )

    missing = [key for key in ("tickers", "start_date", "end_date") if key not in options]
    if missing:
        raise ValueError(
            "Missing required configuration keys: " + ", ".join(missing)
        )

    tickers = options.get("tickers")
    if isinstance(tickers, str):
        options["tickers"] = [tickers]

    analysts = options.get("analysts")
    if isinstance(analysts, str):
        options["analysts"] = [analysts]

    project = options.get("project")
    if project is not None:
        project_str = str(project).strip()
        if not project_str:
            raise ValueError("'project' cannot be an empty string.")
        options["project"] = project_str

    options["config_path"] = path

    return options


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
    project: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
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
    project
        Optional project label used to scope outputs under ``results/{project}``.
    config_path
        Optional path to the YAML batch configuration being executed.

    The provided ``start_date`` and ``end_date`` are applied to every ticker in ``tickers``.
    ``research_depth`` (when provided) is likewise reused for each run.
    """

    if not tickers:
        raise ValueError("'tickers' must be a non-empty sequence of symbols.")

    dates = list(_iterate_dates(start_date, end_date))

    defaults: Dict[str, Any] = dict(cli_main.DEFAULT_CONFIG)

    project_dir: Optional[Path] = None
    config_source = Path(config_path) if config_path else None

    if project:
        base_results_dir = Path(defaults.get("results_dir", "results"))
        project_dir = base_results_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        defaults["results_dir"] = str(project_dir)

        if config_source and config_source.is_file():
            destination = project_dir / config_source.name
            shutil.copy2(config_source, destination)

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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the TradingAgents CLI across a batch of tickers using a YAML config."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML file containing the batch runner parameters.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entrypoint for running batch jobs via ``python -m cli.batch_runner``."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    options = load_batch_config(args.config)
    run_batch(**options)


__all__ = ["load_batch_config", "main", "run_batch"]


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        parser = _build_arg_parser()
        parser.error(str(exc))
