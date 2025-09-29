import datetime as dt

import pytest

import cli.batch_runner as batch_runner
from cli.models import AnalystType


def test_normalize_date_accepts_date_instance():
    sample_date = dt.date(2024, 5, 17)

    normalized = batch_runner._normalize_date(sample_date)

    assert normalized == "2024-05-17"


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
        batch_runner.run_batch([{"ticker": "SPY"}])

    assert "analysis_date" in str(exc.value)


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

    jobs = [
        {"ticker": "spy", "analysis_date": "2024-01-01", "research_depth": 5},
        {
            "ticker": "QQQ",
            "analysis_date": dt.date(2024, 2, 1),
            "llm_provider": "Anthropic",
            "analysts": ["market", AnalystType.NEWS],
            "config_overrides": {
                "backend_url": "override-url",
                "quick_think_llm": "fast",
            },
        },
    ]

    batch_runner.run_batch(jobs, pause_seconds=0.25)

    assert len(calls) == 2
    assert sleep_calls == [0.25]

    first_defaults = calls[0]["defaults"]
    second_defaults = calls[1]["defaults"]

    assert first_defaults["backend_url"] == "base-url"
    assert first_defaults["quick_think_llm"] == "quick"
    assert second_defaults["backend_url"] == "override-url"
    assert second_defaults["quick_think_llm"] == "fast"

    assert calls[0]["selections"]["ticker"] == "SPY"
    assert calls[0]["selections"]["research_depth"] == 5
    assert calls[1]["selections"]["analysis_date"] == "2024-02-01"
    assert calls[1]["selections"]["llm_provider"] == "anthropic"
    assert calls[1]["selections"]["analysts"] == [
        AnalystType.MARKET,
        AnalystType.NEWS,
    ]

    assert calls[0]["buffer_type"] is cli_main.MessageBuffer
    assert calls[1]["buffer_type"] is cli_main.MessageBuffer
    assert calls[0]["buffer_id"] != id(sentinel_buffer)
    assert calls[0]["buffer_id"] != calls[1]["buffer_id"]

    # DEFAULT_CONFIG should be restored after the run
    assert cli_main.DEFAULT_CONFIG == base_config.copy()

