import pandas as pd
import pytest

import universe_selector as us


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Aísla cada test en su propio directorio de caché — evita falsos positivos/negativos
    por estado compartido entre tests."""
    monkeypatch.setattr(us, "CACHE_DIR", tmp_path / "cache_universe")
    yield


def _fake_sec_json():
    return {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "Berkshire Hathaway"},
        "2": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }


# --------------------------------------------------------------------------
# fetch_sec_ticker_to_cik_map — mock de requests.get + verificación de caché
# --------------------------------------------------------------------------

def test_fetch_sec_ticker_map_parses_and_caches(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(_fake_sec_json())

    monkeypatch.setattr(us.requests, "get", fake_get)

    mapping = us.fetch_sec_ticker_to_cik_map()
    assert mapping["AAPL"] == 320193
    assert mapping["BRK-B"] == 1067983
    assert calls["n"] == 1

    # segunda llamada debe leer la caché en disco, no volver a pegarle a la red
    mapping2 = us.fetch_sec_ticker_to_cik_map()
    assert mapping2 == mapping
    assert calls["n"] == 1


def test_fetch_sec_ticker_map_bypass_cache_forces_refetch(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(_fake_sec_json())

    monkeypatch.setattr(us.requests, "get", fake_get)
    us.fetch_sec_ticker_to_cik_map()
    us.fetch_sec_ticker_to_cik_map(bypass_cache=True)
    assert calls["n"] == 2


# --------------------------------------------------------------------------
# _map_ticker_to_cik — variantes de separador y caso sin CIK
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ticker,expected_cik", [
    ("AAPL", 320193),
    ("BRK-B", 1067983),   # match directo
    ("BRK.B", 1067983),   # variante con punto -> debe resolver al mismo CIK
    ("BRKB", 1067983),    # variante sin separador
    ("brk-b", 1067983),   # insensible a mayúsculas
    ("MSFT", 789019),
])
def test_map_ticker_to_cik_handles_separator_variants(ticker, expected_cik):
    sec_map = {"AAPL": 320193, "BRK-B": 1067983, "MSFT": 789019}
    assert us._map_ticker_to_cik(ticker, sec_map) == expected_cik


def test_map_ticker_to_cik_returns_none_when_unmapped():
    sec_map = {"AAPL": 320193}
    assert us._map_ticker_to_cik("NOSUCHTICKER", sec_map) is None


# --------------------------------------------------------------------------
# get_sp500_constituents — mock de pd.read_html + requests.get, caché parquet
# --------------------------------------------------------------------------

def _fake_wiki_table():
    return pd.DataFrame({
        "Symbol": ["AAPL", "BRK.B", "ZZZZ"],          # ZZZZ no tiene CIK -> debe quedar fuera
        "Security": ["Apple Inc.", "Berkshire Hathaway", "Ghost Corp"],
        "GICS Sector": ["Information Technology", "Financials", "Industrials"],
        "GICS Sub-Industry": ["Tech Hardware", "Multi-Sector Holdings", "N/A"],
    })


def test_get_sp500_constituents_maps_and_flags_unmapped(monkeypatch):
    monkeypatch.setattr(us.pd, "read_html", lambda url: [_fake_wiki_table()])
    monkeypatch.setattr(us.requests, "get",
                         lambda url, headers=None, timeout=None: _FakeResponse(_fake_sec_json()))

    universe = us.get_sp500_constituents(bypass_cache=True)

    assert set(universe["ticker"]) == {"AAPL", "BRK.B"}
    assert universe.loc[universe["ticker"] == "AAPL", "cik"].iloc[0] == 320193
    assert universe.loc[universe["ticker"] == "BRK.B", "cik"].iloc[0] == 1067983
    assert pd.api.types.is_integer_dtype(universe["cik"])


def test_get_sp500_constituents_respects_parquet_cache(monkeypatch):
    call_count = {"n": 0}

    def fake_read_html(url):
        call_count["n"] += 1
        return [_fake_wiki_table()]

    monkeypatch.setattr(us.pd, "read_html", fake_read_html)
    monkeypatch.setattr(us.requests, "get",
                         lambda url, headers=None, timeout=None: _FakeResponse(_fake_sec_json()))

    us.get_sp500_constituents(bypass_cache=True)
    assert call_count["n"] == 1

    us.get_sp500_constituents(bypass_cache=False)   # debe leer del parquet cacheado, sin red
    assert call_count["n"] == 1


def test_get_sp500_constituents_bypass_cache_forces_rescrape(monkeypatch):
    call_count = {"n": 0}

    def fake_read_html(url):
        call_count["n"] += 1
        return [_fake_wiki_table()]

    monkeypatch.setattr(us.pd, "read_html", fake_read_html)
    monkeypatch.setattr(us.requests, "get",
                         lambda url, headers=None, timeout=None: _FakeResponse(_fake_sec_json()))

    us.get_sp500_constituents(bypass_cache=True)
    us.get_sp500_constituents(bypass_cache=True)
    assert call_count["n"] == 2
