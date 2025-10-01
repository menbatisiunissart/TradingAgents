import datetime as dt
from pathlib import Path

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

    class DummyCerebro:
        def __init__(self):
            self.broker = DummyBroker()
            self.sizers = []
            self.data = []
            self.strategy = None
            created_cerebros.append(self)

        def addsizer(self, sizer_cls, **kwargs):
            self.sizers.append((sizer_cls, kwargs))

        def adddata(self, feed, name=None):
            self.data.append((feed, name))

        def addstrategy(self, strategy_cls, **kwargs):
            self.strategy = (strategy_cls, kwargs)

        def run(self):
            self.broker.value = (self.broker.cash or 0) + 123.45
            return []

    class DummyPandasData:
        def __init__(self, dataname):
            self.dataname = dataname

    class DummySizer:
        pass

    monkeypatch.setattr(backtester.bt, "Cerebro", DummyCerebro)
    monkeypatch.setattr(backtester.bt.feeds, "PandasData", DummyPandasData)
    monkeypatch.setattr(backtester.bt.sizers, "SizerFix", DummySizer)

    config = {
        "decisions_csv": "decisions.csv",
        "cash": 5000,
        "commission": 0.002,
        "stake": 4,
        "price_padding_days": 3,
        "price_data_columns": {"openinterest_col": "OpenInterest"},
    }

    results = backtester.run_backtests(config, base_path=tmp_path)

    assert price_calls == [("AAPL", "2023-12-30", "2024-01-06")]
    assert len(results) == 1

    result = results[0]
    assert result["ticker"] == "AAPL"
    assert result["starting_cash"] == 5000
    assert result["ending_value"] == pytest.approx(5123.45)
    assert result["trades_executed"] == 2

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
