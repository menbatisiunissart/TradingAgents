import datetime as dt
from pathlib import Path

import pytest

import cli.batch_runner as batch_runner
from cli.models import AnalystType


def test_normalize_date_accepts_date_instance():
    sample_date = dt.date(2024, 5, 17)

    normalized = batch_runner._normalize_date(sample_date)

    assert normalized == "2024-05-17"


def test_iterate_dates_filters_to_trading_days():
    dates = list(
        batch_runner._iterate_dates(dt.date(2024, 1, 1), dt.date(2024, 1, 8))
    )

    assert dates == [
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
        dt.date(2024, 1, 4),
        dt.date(2024, 1, 5),
        dt.date(2024, 1, 8),
    ]


def test_iterate_dates_raises_when_no_trading_days():
    with pytest.raises(ValueError) as exc:
        list(batch_runner._iterate_dates(dt.date(2024, 7, 4), dt.date(2024, 7, 4)))

    assert "trading days" in str(exc.value)


def test_normalize_analysts_defaults_to_all_when_missing():
    analysts = batch_runner._normalize_analysts(None)

    assert analysts == [
        AnalystType.MARKET,
        AnalystType.SOCIAL,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]


def test_normalize_analysts_accepts_mixed_inputs():
    analysts = batch_runner._normalize_analysts([
        AnalystType.MARKET,
        "News",
        "FUNDAMENTALS",
    ])

    assert analysts == [
        AnalystType.MARKET,
        AnalystType.NEWS,
        AnalystType.FUNDAMENTALS,
    ]


def test_normalize_analysts_rejects_unknown_value():
    with pytest.raises(ValueError) as exc:
        batch_runner._normalize_analysts(["unknown"])

    assert "Unknown analyst" in str(exc.value)


def test_build_selections_requires_backend_and_llms():
    base_config = {"max_debate_rounds": 2}

    with pytest.raises(ValueError) as exc:
        batch_runner._build_selections(
            ticker="spy",
            analysis_date="2024-01-01",
            base_config=base_config,
        )

    message = str(exc.value)
    assert "backend_url" in message
    assert "shallow_thinker" in message
    assert "deep_thinker" in message


def test_build_selections_merges_expected_defaults():
    base_config = {
        "max_debate_rounds": 3,
        "llm_provider": "OpenAI",
        "backend_url": "https://example",
        "quick_think_llm": "quick",
        "deep_think_llm": "deep",
    }

    selections = batch_runner._build_selections(
        ticker="spy",
        analysis_date=dt.date(2024, 3, 9),
        analysts=[AnalystType.SOCIAL, "news"],
        research_depth=None,
        llm_provider=None,
        backend_url=None,
        shallow_thinker=None,
        deep_thinker=None,
        base_config=base_config,
    )

    assert selections["ticker"] == "SPY"
    assert selections["analysis_date"] == "2024-03-09"
    assert selections["research_depth"] == 3
    assert selections["llm_provider"] == "openai"
    assert selections["backend_url"] == "https://example"
    assert selections["shallow_thinker"] == "quick"
    assert selections["deep_thinker"] == "deep"
    assert selections["analysts"] == [AnalystType.SOCIAL, AnalystType.NEWS]


def test_run_batch_validates_mandatory_fields():
    with pytest.raises(ValueError) as exc:
        batch_runner.run_batch(
            tickers=[],
            start_date="2024-01-01",
            end_date="2024-01-02",
        )

    assert "tickers" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        batch_runner.run_batch(
            tickers=["SPY"],
            start_date="2024-01-05",
            end_date="2024-01-01",
        )

    assert "end_date" in str(exc.value)


def test_run_batch_applies_overrides_and_resets(monkeypatch):
    cli_main = batch_runner.cli_main

    base_config = {
        "backend_url": "base-url",
        "quick_think_llm": "quick",
        "deep_think_llm": "deep",
        "llm_provider": "openai",
        "max_debate_rounds": 2,
        "max_risk_discuss_rounds": 2,
    }
    monkeypatch.setattr(cli_main, "DEFAULT_CONFIG", base_config.copy())

    calls = []

    def fake_run_analysis():
        selection_snapshot = cli_main.get_user_selections()
        calls.append(
            {
                "defaults": dict(cli_main.DEFAULT_CONFIG),
                "selections": selection_snapshot,
                "buffer_id": id(cli_main.message_buffer),
                "buffer_type": type(cli_main.message_buffer),
            }
        )

    monkeypatch.setattr(cli_main, "run_analysis", fake_run_analysis)

    sleep_calls = []

    def fake_sleep(duration):
        sleep_calls.append(duration)

    monkeypatch.setattr(batch_runner.time, "sleep", fake_sleep)

    sentinel_buffer = object()
    cli_main.message_buffer = sentinel_buffer

    batch_runner.run_batch(
        tickers=["spy", "QQQ"],
        start_date=dt.date(2024, 1, 3),
        end_date="2024-01-04",
        research_depth=5,
        analysts=["market", AnalystType.NEWS],
        llm_provider="Anthropic",
        backend_url="override-url",
        shallow_thinker="fast",
        pause_seconds=0.25,
    )

    assert len(calls) == 4
    assert sleep_calls == [0.25, 0.25, 0.25]

    for call in calls:
        defaults = call["defaults"]
        assert defaults["backend_url"] == "base-url"
        assert defaults["quick_think_llm"] == "quick"

        selections = call["selections"]
        assert selections["research_depth"] == 5
        assert selections["llm_provider"] == "anthropic"
        assert selections["backend_url"] == "override-url"
        assert selections["shallow_thinker"] == "fast"
        assert selections["analysts"] == [
            AnalystType.MARKET,
            AnalystType.NEWS,
        ]

    seen_runs = [
        (call["selections"]["ticker"], call["selections"]["analysis_date"])
        for call in calls
    ]

    assert seen_runs == [
        ("SPY", "2024-01-03"),
        ("SPY", "2024-01-04"),
        ("QQQ", "2024-01-03"),
        ("QQQ", "2024-01-04"),
    ]

    buffer_ids = [call["buffer_id"] for call in calls]
    assert all(buf_id != id(sentinel_buffer) for buf_id in buffer_ids)
    assert all(
        buffer_ids[index] != buffer_ids[index - 1]
        for index in range(1, len(buffer_ids))
    )
    assert all(call["buffer_type"] is cli_main.MessageBuffer for call in calls)

    # DEFAULT_CONFIG should be restored after the run
    assert cli_main.DEFAULT_CONFIG == base_config.copy()


def test_load_batch_config_reads_yaml(tmp_path: Path):
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
tickers: SPY
start_date: 2024-01-05
end_date: 2024-01-07
analysts: market
pause_seconds: 0.5
project: Example
""".strip()
    )

    options = batch_runner.load_batch_config(config_path)

    assert options["tickers"] == ["SPY"]
    assert options["start_date"] == dt.date(2024, 1, 5)
    assert options["end_date"] == dt.date(2024, 1, 7)
    assert options["analysts"] == ["market"]
    assert options["pause_seconds"] == 0.5
    assert options["project"] == "Example"
    assert options["config_path"] == config_path


def test_load_batch_config_rejects_empty_project(tmp_path: Path):
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
tickers:
  - SPY
start_date: 2024-01-05
end_date: 2024-01-07
project: "  "
""".strip()
    )

    with pytest.raises(ValueError) as exc:
        batch_runner.load_batch_config(config_path)

    assert "project" in str(exc.value)


def test_load_batch_config_validates_keys(tmp_path: Path):
    config_path = tmp_path / "missing.yaml"
    config_path.write_text("tickers: []\n")

    with pytest.raises(ValueError) as exc:
        batch_runner.load_batch_config(config_path)

    assert "start_date" in str(exc.value)

    config_path.write_text(
        """
tickers:
  - SPY
start_date: 2024-01-05
end_date: 2024-01-06
extra_field: nope
""".strip()
    )

    with pytest.raises(ValueError) as exc:
        batch_runner.load_batch_config(config_path)

    assert "Unexpected configuration keys" in str(exc.value)


def test_main_dispatches_to_run_batch(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
tickers:
  - SPY
start_date: 2024-01-05
end_date: 2024-01-06
""".strip()
    )

    captured = {}

    def fake_run_batch(**options):
        captured.update(options)

    monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)

    batch_runner.main([str(config_path)])

    assert captured["tickers"] == ["SPY"]
    assert captured["start_date"] == dt.date(2024, 1, 5)
    assert captured["end_date"] == dt.date(2024, 1, 6)
    assert captured["config_path"] == config_path


def test_run_batch_scopes_results_dir_by_project(monkeypatch, tmp_path: Path):
    base_results_dir = tmp_path / "outputs"
    base_config = {
        "backend_url": "base-url",
        "quick_think_llm": "quick",
        "deep_think_llm": "deep",
        "llm_provider": "openai",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "results_dir": str(base_results_dir),
    }
    monkeypatch.setattr(batch_runner.cli_main, "DEFAULT_CONFIG", base_config.copy())

    captured_dirs = []

    def fake_run_analysis():
        captured_dirs.append(batch_runner.cli_main.DEFAULT_CONFIG["results_dir"])

    monkeypatch.setattr(batch_runner.cli_main, "run_analysis", fake_run_analysis)

    batch_runner.run_batch(
        tickers=["SPY"],
        start_date=dt.date(2024, 1, 5),
        end_date=dt.date(2024, 1, 5),
        backend_url="override-url",
        shallow_thinker="fast",
        deep_thinker="deep",
        project="demo",
    )

    expected_project_dir = base_results_dir / "demo"

    assert captured_dirs == [str(expected_project_dir)]
    assert expected_project_dir.is_dir()
    assert batch_runner.cli_main.DEFAULT_CONFIG == base_config.copy()


def test_run_batch_copies_config_into_project_dir(monkeypatch, tmp_path: Path):
    config_source = tmp_path / "batch.yaml"
    config_source.write_text("sample: value\n")

    base_results_dir = tmp_path / "outputs"
    base_config = {
        "backend_url": "base-url",
        "quick_think_llm": "quick",
        "deep_think_llm": "deep",
        "llm_provider": "openai",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "results_dir": str(base_results_dir),
    }
    monkeypatch.setattr(batch_runner.cli_main, "DEFAULT_CONFIG", base_config.copy())

    monkeypatch.setattr(batch_runner.cli_main, "run_analysis", lambda: None)

    batch_runner.run_batch(
        tickers=["SPY"],
        start_date=dt.date(2024, 1, 5),
        end_date=dt.date(2024, 1, 5),
        backend_url="override-url",
        shallow_thinker="fast",
        deep_thinker="deep",
        project="demo",
        config_path=config_source,
    )

    copied = (base_results_dir / "demo" / config_source.name)

    assert copied.is_file()
    assert copied.read_text() == config_source.read_text()
    assert batch_runner.cli_main.DEFAULT_CONFIG == base_config.copy()
