import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

import run_pipeline as rp
import data_ingest.returns_ingestor as ri


def _synthetic_universe(n=12):
    sectors = ["Technology", "Financials", "Healthcare", "Industrials"]
    return pd.DataFrame({
        "ticker": [f"T{i}" for i in range(n)],
        "entity_name": [f"Company {i}" for i in range(n)],
        "gics_sector": [sectors[i % len(sectors)] for i in range(n)],
        "gics_sub_industry": ["Sub"] * n,
        "cik": list(range(1000, 1000 + n)),
    })


def _fake_yf_multiindex(tickers, dates, seed=1):
    rng = np.random.default_rng(seed)
    fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    data = {}
    for field in fields:
        for tk in tickers:
            base = 100 + rng.normal(0, 1, size=len(dates)).cumsum()
            data[(field, tk)] = base if field != "Volume" else np.full(len(dates), 1_000_000)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(list(data.keys()))
    return df


@pytest.fixture
def isolated_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def test_run_pipeline_end_to_end(monkeypatch, isolated_workdir):
    universe = _synthetic_universe(n=12)
    monkeypatch.setattr(rp, "get_sp500_constituents", lambda: universe)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=2)
    dates = pd.bdate_range(start_date, end_date)
    fake_raw = _fake_yf_multiindex(universe["ticker"].tolist() + ["SPY"], dates)
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: fake_raw)

    weights_df = rp.main(
        years_history=2,
        use_real_sec=False,
        max_weight=0.3,
        max_leverage=1.0,
        risk_aversion=1.0,
        beta_window=30,
        min_group_size=2,
        output_path="reports/optimal_weights.csv",
    )

    assert set(weights_df.columns) == {"ticker", "cik", "weight", "gics_sector"}
    assert not weights_df.empty
    assert weights_df["weight"].sum() == pytest.approx(0, abs=1e-4)
    assert weights_df["weight"].abs().sum() <= 1.0 + 1e-4
    assert (weights_df["weight"].abs() <= 0.3 + 1e-4).all()
    assert (isolated_workdir / "reports" / "optimal_weights.csv").exists()


def test_run_pipeline_raises_when_universe_limit_leaves_too_few_assets(monkeypatch, isolated_workdir):
    universe = _synthetic_universe(n=12)
    monkeypatch.setattr(rp, "get_sp500_constituents", lambda: universe)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=2)
    dates = pd.bdate_range(start_date, end_date)
    fake_raw = _fake_yf_multiindex(universe["ticker"].tolist() + ["SPY"], dates)
    monkeypatch.setattr(ri.yf, "download", lambda *a, **k: fake_raw)

    # con solo 1 activo, sector_neutralize (min_group_size>=2) descarta todo -> debe fallar explícito
    with pytest.raises(RuntimeError, match="neutralización"):
        rp.main(universe_limit=1, years_history=2, min_group_size=2, output_path="reports/optimal_weights.csv")
