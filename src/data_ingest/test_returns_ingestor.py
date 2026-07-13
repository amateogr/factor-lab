import numpy as np
import pandas as pd
import pytest

import returns_ingestor as ri


def _fake_multiindex_raw(tickers, dates, seed=0, dead_ticker=None):
    """Replica la forma real de yf.download con group_by='column': columnas
    MultiIndex (field, ticker), fields = Open/High/Low/Close/Adj Close/Volume."""
    rng = np.random.default_rng(seed)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = {}
    for field in fields:
        for tk in tickers:
            if tk == dead_ticker:
                data[(field, tk)] = np.full(len(dates), np.nan)
            else:
                base = 100 + rng.normal(0, 1, size=len(dates)).cumsum()
                data[(field, tk)] = base if field != "Volume" else np.full(len(dates), 1_000_000)
    cols = pd.MultiIndex.from_tuples(list(data.keys()))
    df = pd.DataFrame(data, index=dates)
    df.columns = cols
    return df


@pytest.fixture
def dates():
    return pd.date_range("2023-01-02", periods=30, freq="B")


# ---------------------------------------------------------------------------
# download_market_data
# ---------------------------------------------------------------------------

def test_download_market_data_extracts_adj_close_and_includes_spy(monkeypatch, dates):
    tickers = ["AAPL", "MSFT"]
    raw = _fake_multiindex_raw(tickers + ["SPY"], dates)
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: raw)

    prices = ri.download_market_data(tickers, "2023-01-01", "2023-02-15")
    assert set(prices.columns) == {"AAPL", "MSFT", "SPY"}
    assert len(prices) == len(dates)


def test_download_market_data_drops_delisted_ticker(monkeypatch, dates):
    tickers = ["AAPL", "GHOST"]
    raw = _fake_multiindex_raw(tickers + ["SPY"], dates, dead_ticker="GHOST")
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: raw)

    prices = ri.download_market_data(tickers, "2023-01-01", "2023-02-15")
    assert "GHOST" not in prices.columns
    assert "AAPL" in prices.columns and "SPY" in prices.columns


def test_download_market_data_raises_if_spy_fails(monkeypatch, dates):
    tickers = ["AAPL"]
    raw = _fake_multiindex_raw(tickers + ["SPY"], dates, dead_ticker="SPY")
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: raw)

    with pytest.raises(RuntimeError, match="SPY"):
        ri.download_market_data(tickers, "2023-01-01", "2023-02-15")


def test_download_market_data_raises_on_empty_response(monkeypatch):
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(RuntimeError):
        ri.download_market_data(["AAPL"], "2023-01-01", "2023-02-15")


# ---------------------------------------------------------------------------
# compute_daily_returns
# ---------------------------------------------------------------------------

def test_compute_daily_returns_pct_change_and_first_row_nan():
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 99.0]})
    rets = ri.compute_daily_returns(prices)
    assert pd.isna(rets["AAPL"].iloc[0])
    assert rets["AAPL"].iloc[1] == pytest.approx(0.10)
    assert rets["AAPL"].iloc[2] == pytest.approx(99 / 110 - 1)


# ---------------------------------------------------------------------------
# build_returns_panel
# ---------------------------------------------------------------------------

def test_build_returns_panel_shape_and_mkt_ret_broadcast(monkeypatch, dates):
    tickers = ["AAPL", "MSFT"]
    raw = _fake_multiindex_raw(tickers + ["SPY"], dates)
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: raw)

    panel = ri.build_returns_panel(tickers, "2023-01-01", "2023-02-15")
    assert set(panel.columns) == {"date", "ticker", "ret", "mkt_ret"}
    assert set(panel["ticker"].unique()) == {"AAPL", "MSFT"}

    # mkt_ret debe ser igual para AAPL y MSFT en la misma fecha (broadcast de SPY)
    pivot = panel.pivot(index="date", columns="ticker", values="mkt_ret")
    assert (pivot["AAPL"].dropna() == pivot["MSFT"].dropna()).all()


# ---------------------------------------------------------------------------
# attach_cik
# ---------------------------------------------------------------------------

def test_attach_cik_merges_and_flags_unmatched():
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-03", "2023-01-03", "2023-01-03"]),
        "ticker": ["AAPL", "MSFT", "NOTINUNIVERSE"],
        "ret": [0.01, 0.02, 0.03],
        "mkt_ret": [0.005, 0.005, 0.005],
    })
    universe = pd.DataFrame({"ticker": ["AAPL", "MSFT"], "cik": [320193, 789019]})

    out = ri.attach_cik(panel, universe)
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    assert out.loc[out["ticker"] == "AAPL", "cik"].iloc[0] == 320193
    assert pd.api.types.is_integer_dtype(out["cik"])
