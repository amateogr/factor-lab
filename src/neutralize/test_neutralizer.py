import numpy as np
import pandas as pd
import pytest

import neutralizer as nz


# ---------------------------------------------------------------------------
# sector_neutralize
# ---------------------------------------------------------------------------

def test_sector_neutralize_sum_zero_and_unit_std_per_group():
    df = pd.DataFrame({
        "period": ["P1"] * 6,
        "gics_sector": ["Tech", "Tech", "Tech", "Fin", "Fin", "Fin"],
        "score": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    out = nz.sector_neutralize(df, "score", "gics_sector", min_group_size=3)
    tech = out[df["gics_sector"] == "Tech"]
    fin = out[df["gics_sector"] == "Fin"]
    assert tech.sum() == pytest.approx(0, abs=1e-9)
    assert fin.sum() == pytest.approx(0, abs=1e-9)
    assert tech.std(ddof=0) == pytest.approx(1, abs=1e-9)


def test_sector_neutralize_demean_only_when_standardize_false():
    df = pd.DataFrame({
        "period": ["P1"] * 3,
        "gics_sector": ["Tech"] * 3,
        "score": [1.0, 2.0, 3.0],
    })
    out = nz.sector_neutralize(df, "score", "gics_sector", standardize=False, min_group_size=3)
    assert out.sum() == pytest.approx(0, abs=1e-9)
    assert out.std(ddof=0) == pytest.approx(np.std([1, 2, 3]), abs=1e-9)  # varianza original preservada


def test_sector_neutralize_nan_below_min_group_size():
    df = pd.DataFrame({
        "period": ["P1"] * 2,
        "gics_sector": ["Tech"] * 2,
        "score": [1.0, 2.0],
    })
    out = nz.sector_neutralize(df, "score", "gics_sector", min_group_size=3)
    assert out.isna().all()


def test_sector_neutralize_respects_date_grouping():
    # mismo sector, distinta fecha -> grupos separados, no deben mezclarse
    df = pd.DataFrame({
        "period": ["P1", "P1", "P2", "P2"],
        "gics_sector": ["Tech"] * 4,
        "score": [1.0, 3.0, 100.0, 300.0],
    })
    out = nz.sector_neutralize(df, "score", "gics_sector", min_group_size=2)
    assert out.iloc[0] == pytest.approx(-out.iloc[1])
    assert out.iloc[2] == pytest.approx(-out.iloc[3])
    assert abs(out.iloc[0]) < 10  # no contaminado por la escala de P2


# ---------------------------------------------------------------------------
# compute_rolling_beta
# ---------------------------------------------------------------------------

def _synthetic_returns(beta_true=1.2, n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    market = rng.normal(0, 0.01, size=n)
    noise = rng.normal(0, 0.001, size=n)
    asset_ret = beta_true * market + noise
    return pd.DataFrame({"cik": [1] * n, "period": dates, "ret": asset_ret, "mkt_ret": market})


def test_compute_rolling_beta_recovers_true_beta():
    df = _synthetic_returns(beta_true=1.2, n=60)
    betas = nz.compute_rolling_beta(df, "ret", "mkt_ret", asset_col="cik", date_col="period", window=30)
    last_beta = betas.dropna().iloc[-1]
    assert last_beta == pytest.approx(1.2, abs=0.15)


def test_compute_rolling_beta_nan_before_min_nobs():
    df = _synthetic_returns(beta_true=1.0, n=60)
    betas = nz.compute_rolling_beta(df, "ret", "mkt_ret", asset_col="cik", date_col="period", window=30)
    assert betas.iloc[:29].isna().all()
    assert betas.iloc[29:].notna().all()


def test_compute_rolling_beta_insufficient_history_returns_all_nan():
    df = _synthetic_returns(beta_true=1.0, n=10)  # menos que window
    betas = nz.compute_rolling_beta(df, "ret", "mkt_ret", asset_col="cik", date_col="period", window=30)
    assert betas.isna().all()


# ---------------------------------------------------------------------------
# beta_neutralize
# ---------------------------------------------------------------------------

def test_beta_neutralize_residuals_orthogonal_to_beta():
    rng = np.random.default_rng(1)
    n_dates, n_assets = 40, 8
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D")
    market = rng.normal(0, 0.01, size=n_dates)

    rows = []
    for asset in range(n_assets):
        beta_true = 0.5 + 0.2 * asset
        noise = rng.normal(0, 0.001, size=n_dates)
        asset_ret = beta_true * market + noise
        score = rng.normal(0, 1, size=n_dates) + 0.8 * beta_true  # score correlacionado con beta a propósito
        for d, r, m, s in zip(dates, asset_ret, market, score):
            rows.append({"cik": asset, "period": d, "ret": r, "mkt_ret": m, "score": s})
    df = pd.DataFrame(rows)

    out = nz.beta_neutralize(df, "score", "ret", "mkt_ret", asset_col="cik", date_col="period", window=20)
    valid = out.dropna(subset=["beta", "score_beta_neutral"])

    # propiedad matemática de OLS: el residuo es ortogonal al regresor dentro de cada fecha
    for date, g in valid.groupby("period"):
        if len(g) < 3:
            continue
        cov = np.cov(g["score_beta_neutral"], g["beta"])[0, 1]
        assert abs(cov) < 1e-6


def test_beta_neutralize_nan_when_group_too_small():
    df = _synthetic_returns(beta_true=1.0, n=60)
    df["score"] = 1.0
    out = nz.beta_neutralize(df, "score", "ret", "mkt_ret", asset_col="cik", date_col="period",
                              window=30, min_group_size=3)
    # un solo activo por fecha -> nunca alcanza min_group_size=3
    assert out["score_beta_neutral"].isna().all()
