import datetime as dt
import statistics

import pandas as pd
import pytest

from cli import backtester


def test_run_backtests_fetches_prices_and_returns_results(monkeypatch, tmp_path):
    decisions_path = tmp_path / "decisions.csv"
    decisions_frame = pd.DataFrame(
        [
            {"ticker": "AAPL", "analysis_date": "2024-01-02", "final_decision": "BUY"},
            {"ticker": "AAPL", "analysis_date": "2024-01-03", "final_decision": "SELL"},
        ]
    )
    decisions_frame.to_csv(decisions_path, index=False)

    price_calls = []

    def fake_get_stock_data(symbol, start_date, end_date, save_path=None):
        price_calls.append((symbol, start_date, end_date))
        index = pd.date_range("2023-12-30", periods=8, freq="D")
        return pd.DataFrame(
            {
                "Open": range(10, 18),
                "High": range(11, 19),
                "Low": range(9, 17),
                "Close": range(10, 18),
                "Volume": [1000] * 8,
                "OpenInterest": [0] * 8,
            },
            index=index,
        )

    monkeypatch.setattr(backtester.YFinanceUtils, "get_stock_data", fake_get_stock_data)

    created_cerebros = []

    class DummyBroker:
        def __init__(self):
            self.cash = None
            self.commission = None
            self.value = None

        def setcash(self, cash):
            self.cash = cash
            self.value = cash

        def setcommission(self, commission):
            self.commission = commission

        def getvalue(self):
            return self.value

        def set_coc(self, value):
            self.coc = value

    class DummyAnalyzer:
        def __init__(self, data):
            self._data = data

        def get_analysis(self):
            return self._data

    class DummyCerebro:
        def __init__(self):
            self.broker = DummyBroker()
            self.sizers = []
            self.data = []
            self.strategy = None
            self.analyzer_specs = []
            created_cerebros.append(self)

        def addsizer(self, sizer_cls, **kwargs):
            self.sizers.append((sizer_cls, kwargs))

        def adddata(self, feed, name=None):
            self.data.append((feed, name))

        def addstrategy(self, strategy_cls, **kwargs):
            self.strategy = (strategy_cls, kwargs)

        def addanalyzer(self, analyzer_cls, *args, **kwargs):
            self.analyzer_specs.append(kwargs.get("_name"))

        def run(self):
            self.broker.value = (self.broker.cash or 0) + 123.45
            analyzers_container = type("Analyzers", (), {})()

            returns_data = {
                dt.datetime(2024, 1, 2): 0.01,
                dt.datetime(2024, 1, 3): -0.005,
            }
            drawdown_data = {"maxdrawdown": 12.0}

            for name in self.analyzer_specs:
                if name == "returns":
                    setattr(analyzers_container, name, DummyAnalyzer(returns_data))
                elif name == "drawdown":
                    setattr(analyzers_container, name, DummyAnalyzer(drawdown_data))

            strategy = type("DummyStrategy", (), {"analyzers": analyzers_container})()
            return [strategy]

    class DummyPandasData:
        def __init__(self, dataname):
            self.dataname = dataname

    class DummySizer:
        pass

    monkeypatch.setattr(backtester.bt, "Cerebro", DummyCerebro)
    monkeypatch.setattr(backtester.bt.feeds, "PandasData", DummyPandasData)
    monkeypatch.setattr(backtester.bt.sizers, "SizerFix", DummySizer)
    monkeypatch.setattr(backtester.bt, "TimeFrame", type("TimeFrame", (), {"Days": object()}))
    monkeypatch.setattr(
        backtester.bt,
        "analyzers",
        type("Analyzers", (), {"TimeReturn": object(), "DrawDown": object()}),
    )

    config = {
        "decisions_csv": "decisions.csv",
        "cash": 5000,
        "commission": 0.002,
        "stake": 4,
        "price_padding_days": 3,
        "price_data_columns": {"openinterest_col": "OpenInterest"},
        "risk_free_rate": 0.02,
        "trading_days_per_year": 252,
    }

    baseline_tracker = {}

    def fake_evaluate_baselines(**kwargs):
        baseline_tracker.update(kwargs)
        return [
            {
                "model": "Buy and Hold",
                "starting_cash": kwargs["initial_cash"],
                "ending_value": kwargs["initial_cash"] * 1.1,
                "return_pct": 10.0,
                "trades_executed": 2,
                "cumulative_return": 10.0,
                "annualized_return": 12.0,
                "sharpe_ratio": 1.5,
                "max_drawdown": -5.0,
            }
        ]

    monkeypatch.setattr(backtester, "_evaluate_baselines", fake_evaluate_baselines)

    results = backtester.run_backtests(config, base_path=tmp_path)

    assert price_calls == [("AAPL", "2023-12-30", "2024-01-06")]
    assert len(results) == 2
    backtester.write_results(results, {"output_csv": "results.csv"}, tmp_path)
    written = pd.read_csv(tmp_path / "results.csv")
    assert list(written.columns[:2]) == ["ticker", "model"]

    assert baseline_tracker["initial_cash"] == 5000
    assert baseline_tracker["commission"] == 0.002

    agent_result = next(item for item in results if item["model"] == "TradingAgents")
    baseline_result = next(item for item in results if item["model"] == "Buy and Hold")

    result = agent_result
    assert result["ticker"] == "AAPL"
    assert result["starting_cash"] == 5000
    assert result["ending_value"] == pytest.approx(5123.45)
    assert result["trades_executed"] == 2
    assert result["cumulative_return"] == pytest.approx(((5123.45 / 5000) - 1.0) * 100.0)

    duration_years = ((dt.date(2024, 1, 3) - dt.date(2024, 1, 2)).days + 1) / 365.25
    expected_annualized = ((5123.45 / 5000) ** (1.0 / duration_years) - 1.0) * 100.0
    assert result["annualized_return"] == pytest.approx(expected_annualized)

    returns_series = [0.01, -0.005]
    expected_sharpe = (
        (sum(returns_series) / len(returns_series))
        - (0.02 / 252)
    ) / statistics.pstdev(returns_series)
    assert result["sharpe_ratio"] == pytest.approx(expected_sharpe)
    assert result["max_drawdown"] == pytest.approx(12.0)

    assert len(created_cerebros) == 1
    cerebro = created_cerebros[0]

    assert cerebro.sizers == [(DummySizer, {"stake": 4})]
    assert len(cerebro.data) == 1
    feed, name = cerebro.data[0]
    assert isinstance(feed, DummyPandasData)
    assert name == "AAPL"

    prepared = feed.dataname
    assert list(prepared.columns) == ["open", "high", "low", "close", "volume", "openinterest"]
    assert prepared.index.name == "datetime"

    strategy_cls, strategy_kwargs = cerebro.strategy
    assert strategy_cls is backtester.DecisionStrategy
    assert set(strategy_kwargs["decisions"].keys()) == {
        dt.date(2024, 1, 2),
        dt.date(2024, 1, 3),
    }
    assert strategy_kwargs["hold_value"] == "HOLD"
    assert strategy_kwargs["buy_value"] == "BUY"
    assert strategy_kwargs["sell_value"] == "SELL"

    assert baseline_result["ticker"] == "AAPL"
    assert baseline_result["return_pct"] == 10.0
    assert baseline_result["trades_executed"] == 2


def test_run_backtests_errors_when_price_data_missing(monkeypatch, tmp_path):
    decisions_path = tmp_path / "decisions.csv"
    pd.DataFrame(
        [{"ticker": "MSFT", "analysis_date": "2024-02-01", "final_decision": "BUY"}]
    ).to_csv(decisions_path, index=False)

    def fake_get_stock_data(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(backtester.YFinanceUtils, "get_stock_data", fake_get_stock_data)

    config = {"decisions_csv": "decisions.csv"}

    with pytest.raises(ValueError) as exc:
        backtester.run_backtests(config, base_path=tmp_path)

    assert "No price data returned" in str(exc.value)


def test_evaluate_baselines_produces_entries():
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    base_prices = pd.Series(range(120), index=index, dtype=float) + 100
    price_frame = pd.DataFrame(
        {
            "Open": base_prices + 0.1,
            "High": base_prices + 1.0,
            "Low": base_prices - 1.0,
            "Close": base_prices,
            "Volume": 1_000,
        },
        index=index,
    )

    baselines = backtester._evaluate_baselines(
        price_frame=price_frame,
        initial_cash=10_000,
        commission=0.001,
        start_date=index[0].date(),
        end_date=index[-1].date(),
        risk_free_rate=0.02,
        periods_per_year=252,
    )

    model_names = {row["model"] for row in baselines}
    expected_models = {"Buy and Hold", "MACD", "KDJ+RSI", "ZMR", "SMA"}
    assert model_names == expected_models
    for row in baselines:
        assert "ending_value" in row
        assert "cumulative_return" in row
