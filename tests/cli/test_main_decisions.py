import csv
from pathlib import Path
from types import SimpleNamespace

import cli.main as cli_main
from cli.models import AnalystType


class DummyLive:
    def __init__(self, layout, refresh_per_second=4):
        self.layout = layout
        self.refresh_per_second = refresh_per_second

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DummyGraph:
    chunks = []

    def __init__(self, analysts, config, debug):
        self.analysts = analysts
        self.config = config
        self.debug = debug
        self.propagator = SimpleNamespace(
            create_initial_state=lambda ticker, date: {"ticker": ticker, "date": date},
            get_graph_args=lambda: {},
        )
        self.graph = SimpleNamespace(stream=self._stream)

    def _stream(self, init_state, **kwargs):
        for chunk in self.__class__.chunks:
            yield chunk

    def process_signal(self, text):
        return text or ""


def _setup_cli_harness(monkeypatch, tmp_path: Path) -> Path:
    results_dir = tmp_path / "outputs"
    monkeypatch.setitem(cli_main.DEFAULT_CONFIG, "results_dir", str(results_dir))
    monkeypatch.setattr(cli_main, "TradingAgentsGraph", DummyGraph)
    monkeypatch.setattr(cli_main, "create_layout", lambda: object())
    monkeypatch.setattr(cli_main, "update_display", lambda layout, spinner_text=None: None)
    monkeypatch.setattr(cli_main, "display_complete_report", lambda final_state: None)
    monkeypatch.setattr(cli_main, "Live", DummyLive)
    monkeypatch.setattr(cli_main.console, "print", lambda *args, **kwargs: None)
    return results_dir


def _run_analysis_once(monkeypatch, selection, chunk):
    DummyGraph.chunks = [chunk]
    cli_main.message_buffer = cli_main.MessageBuffer()
    monkeypatch.setattr(cli_main, "get_user_selections", lambda: selection)
    cli_main.run_analysis()


def _make_selection(**overrides):
    selection = {
        "ticker": "SPY",
        "analysis_date": "2024-01-01",
        "analysts": [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ],
        "research_depth": 1,
        "llm_provider": "openai",
        "backend_url": "http://backend",
        "shallow_thinker": "quick",
        "deep_thinker": "deep",
    }
    selection.update(overrides)
    return selection


def test_run_analysis_writes_decisions_csv_with_normalized_values(monkeypatch, tmp_path):
    results_dir = _setup_cli_harness(monkeypatch, tmp_path)

    selection = _make_selection(
        ticker="SPY",
        analysis_date="2024-08-19",
        llm_provider="OpenAI",
    )

    final_chunk = {
        "messages": [],
        "market_report": "Momentum suggests we should buy the breakout.",
        "sentiment_report": "Community sentiment leans hold at the moment.",
        "news_report": "Breaking news headline screams SELL right now!",
        "fundamentals_report": "Latest filings make this a buy candidate.",
        "investment_plan": "Research summary keeps everyone aligned.",
        "trader_investment_plan": "Trader verdict: buy with a tight stop.",
        "final_trade_decision": "Portfolio manager final decision: Buy asap.",
        "investment_debate_state": {
            "bull_history": "\n".join(
                [
                    "Bull Analyst: Initial pass",
                    "Bull Analyst: BUY we must",
                ]
            ),
            "bear_history": "Bear Analyst: SELL before it drops",
            "judge_decision": "Research Manager decides HOLD for now",
        },
        "risk_debate_state": {
            "current_risky_response": "Risky Analyst: SELL aggressively",
            "current_safe_response": "Safe Analyst: HOLD until clarity",
            "current_neutral_response": "Neutral Analyst: maybe buy soon",
            "judge_decision": "Portfolio Manager chooses BUY long term",
        },
    }

    _run_analysis_once(monkeypatch, selection, final_chunk)

    decisions_file = results_dir / "decisions.csv"
    assert decisions_file.exists()

    with decisions_file.open() as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert len(rows) == 1
    row = rows[0]

    assert row["ticker"] == "SPY"
    assert row["analysis_date"] == "2024-08-19"
    assert row["final_decision"] == "BUY"
    assert row["market_analyst"] == "BUY"
    assert row["social_analyst"] == "HOLD"
    assert row["news_analyst"] == "SELL"
    assert row["fundamentals_analyst"] == "BUY"
    assert row["bull_researcher"] == "BUY"
    assert row["bear_researcher"] == "SELL"
    assert row["research_manager"] == "HOLD"
    assert row["trader"] == "BUY"
    assert row["risky_analyst"] == "SELL"
    assert row["safe_analyst"] == "HOLD"
    assert row["neutral_analyst"] == "BUY"
    assert row["portfolio_manager"] == "BUY"


def test_decisions_csv_appends_and_fills_missing_agent_outputs(monkeypatch, tmp_path):
    results_dir = _setup_cli_harness(monkeypatch, tmp_path)

    first_selection = _make_selection(ticker="AAA", analysis_date="2024-01-01")
    first_chunk = {
        "messages": [],
        "market_report": "Market outlook: buy the strength.",
        "sentiment_report": "Sentiment says hold your positions.",
        "news_report": "Latest coverage implies SELL pressure.",
        "fundamentals_report": "Fundamentals tell us to buy.",
        "investment_plan": "Research team aligned on direction.",
        "trader_investment_plan": "Trader wants to buy dip.",
        "final_trade_decision": "Final verdict: Buy it.",
        "investment_debate_state": {
            "bull_history": "Bull Analyst: BUY ready",
            "bear_history": "Bear Analyst: SELL soon",
            "judge_decision": "Research Manager chooses BUY",
        },
        "risk_debate_state": {
            "current_risky_response": "Risky Analyst: SELL now",
            "current_safe_response": "Safe Analyst: HOLD tight",
            "current_neutral_response": "Neutral Analyst: consider buy",
            "judge_decision": "Portfolio Manager: BUY decision",
        },
    }

    _run_analysis_once(monkeypatch, first_selection, first_chunk)

    second_selection = _make_selection(
        ticker="BBB",
        analysis_date="2024-01-02",
        analysts=[AnalystType.MARKET],
    )
    second_chunk = {
        "messages": [],
        "market_report": None,
        "sentiment_report": "",
        "news_report": None,
        "fundamentals_report": None,
        "investment_plan": None,
        "trader_investment_plan": None,
        "final_trade_decision": "Outcome: hold steady.",
        "investment_debate_state": {},
        "risk_debate_state": {"judge_decision": ""},
    }

    _run_analysis_once(monkeypatch, second_selection, second_chunk)

    decisions_file = results_dir / "decisions.csv"
    with decisions_file.open() as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert len(rows) == 2

    first_row, second_row = rows

    assert first_row["ticker"] == "AAA"
    assert first_row["final_decision"] == "BUY"

    assert second_row["ticker"] == "BBB"
    assert second_row["analysis_date"] == "2024-01-02"
    assert second_row["final_decision"] == "HOLD"
    assert second_row["portfolio_manager"] == "HOLD"
    assert second_row["market_analyst"] == ""
    assert second_row["social_analyst"] == ""
    assert second_row["news_analyst"] == ""
    assert second_row["fundamentals_analyst"] == ""
    assert second_row["bull_researcher"] == ""
    assert second_row["bear_researcher"] == ""
    assert second_row["research_manager"] == ""
    assert second_row["trader"] == ""
    assert second_row["risky_analyst"] == ""
    assert second_row["safe_analyst"] == ""
    assert second_row["neutral_analyst"] == ""
