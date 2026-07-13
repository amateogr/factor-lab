"""test_xbrl_tag_mapper.py — pytest, sin red: valida fallback/merge/as_of contra fixtures sintéticos."""
import xbrl_tag_mapper as m


def _facts(**tags):
    return {"facts": {"us-gaap": tags}}


def test_fallback_prefers_primary_tag_when_present(monkeypatch):
    facts = _facts(
        NetIncomeLoss={"units": {"USD": [
            {"end": "2023-12-31", "start": "2023-01-01", "val": 100, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "A1"},
        ]}},
        ProfitLoss={"units": {"USD": [
            {"end": "2023-12-31", "start": "2023-01-01", "val": 999, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "A2"},
        ]}},
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_concept_timeseries(1, "net_income")
    assert len(df) == 1
    assert df.iloc[0]["val"] == 100
    assert df.iloc[0]["source_tag"] == "NetIncomeLoss"


def test_fallback_uses_secondary_tag_when_primary_missing_for_period(monkeypatch):
    facts = _facts(
        NetIncomeLoss={"units": {"USD": [
            {"end": "2022-12-31", "start": "2022-01-01", "val": 50, "fp": "FY",
             "form": "10-K", "filed": "2023-02-10", "accn": "B1"},
        ]}},
        ProfitLoss={"units": {"USD": [
            {"end": "2023-12-31", "start": "2023-01-01", "val": 999, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "B2"},
        ]}},
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_concept_timeseries(1, "net_income")
    assert len(df) == 2
    row_2023 = df[df["end"] == "2023-12-31"].iloc[0]
    assert row_2023["val"] == 999
    assert row_2023["source_tag"] == "ProfitLoss"


def test_instant_excludes_entries_with_start(monkeypatch):
    facts = _facts(
        StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest={"units": {"USD": [
            {"end": "2023-12-31", "start": None, "val": 500, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "C1"},
            {"end": "2023-06-30", "start": "2023-01-01", "val": 9999, "fp": "Q2",  # duración espuria
             "form": "10-Q", "filed": "2023-08-01", "accn": "C2"},
        ]}},
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_concept_timeseries(1, "stockholders_equity")
    assert len(df) == 1
    assert df.iloc[0]["val"] == 500


def test_as_of_respects_filed_not_end(monkeypatch):
    facts = _facts(
        NetIncomeLoss={"units": {"USD": [
            {"end": "2022-12-31", "start": "2022-01-01", "val": 50, "fp": "FY",
             "form": "10-K", "filed": "2023-02-10", "accn": "D1"},
            {"end": "2023-12-31", "start": "2023-01-01", "val": 80, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "D2"},
        ]}},
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_concept_timeseries(1, "net_income")

    row = m.as_of(df, "2023-06-01")   # antes de que se publique el 10-K de FY2023
    assert row["val"] == 50
    row = m.as_of(df, "2024-03-01")   # tras el filing, FY2023 pasa a ser lo "conocido"
    assert row["val"] == 80


def test_as_of_picks_latest_restated_vintage_for_same_period(monkeypatch):
    facts = _facts(
        NetIncomeLoss={"units": {"USD": [
            {"end": "2022-12-31", "start": "2022-01-01", "val": 50, "fp": "FY",
             "form": "10-K", "filed": "2023-02-10", "accn": "E1"},
            {"end": "2022-12-31", "start": "2022-01-01", "val": 45, "fp": "FY",   # restatement
             "form": "10-K/A", "filed": "2023-09-01", "accn": "E2"},
        ]}},
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_concept_timeseries(1, "net_income")
    assert len(df) == 2  # ambas versiones se conservan

    assert m.as_of(df, "2023-05-01")["val"] == 50   # antes del restatement
    assert m.as_of(df, "2023-10-01")["val"] == 45   # después del restatement


def test_liabilities_identity_fallback(monkeypatch):
    facts = _facts(
        LiabilitiesAndStockholdersEquity={"units": {"USD": [
            {"end": "2023-12-31", "start": None, "val": 1000, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "F1"},
        ]}},
        StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest={"units": {"USD": [
            {"end": "2023-12-31", "start": None, "val": 300, "fp": "FY",
             "form": "10-K", "filed": "2024-02-15", "accn": "F2"},
        ]}},
        # sin tag `Liabilities` directo -> fuerza el fallback derivado
    )
    monkeypatch.setattr(m, "fetch_companyfacts", lambda cik: facts)
    df = m.resolve_liabilities_with_identity_fallback(1)
    assert len(df) == 1
    assert df.iloc[0]["val"] == 700
    assert "derived" in df.iloc[0]["source_tag"]


def test_frame_panel_fallback_merges_across_tags(monkeypatch):
    frame_primary = {"data": [
        {"cik": 1, "entityName": "AAA", "val": 10, "end": "2023-12-31", "accn": "G1"},
    ]}
    frame_secondary = {"data": [
        {"cik": 1, "entityName": "AAA", "val": 999, "end": "2023-12-31", "accn": "G2"},  # cik ya cubierto, ignorar
        {"cik": 2, "entityName": "BBB", "val": 20, "end": "2023-12-31", "accn": "G3"},
    ]}

    def fake_fetch_frame(tag, period):
        return frame_primary if tag == "NetIncomeLoss" else frame_secondary

    monkeypatch.setattr(m, "fetch_frame", fake_fetch_frame)
    df = m.resolve_concept_panel("net_income", 2023)
    assert set(df["cik"]) == {1, 2}
    assert df[df["cik"] == 1].iloc[0]["val"] == 10
    assert df[df["cik"] == 2].iloc[0]["val"] == 20
