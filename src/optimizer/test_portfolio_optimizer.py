import numpy as np
import pandas as pd
import pytest

import portfolio_optimizer as po


# ---------------------------------------------------------------------------
# estimate_covariance
# ---------------------------------------------------------------------------

def _synthetic_returns(n_assets=5, n_days=100, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"A{i}" for i in range(n_assets)]
    factor = rng.normal(0, 0.01, size=n_days)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    return pd.DataFrame(
        {tk: betas[i] * factor + rng.normal(0, 0.005, size=n_days) for i, tk in enumerate(tickers)}
    )


def test_estimate_covariance_shape_symmetric_psd():
    returns = _synthetic_returns()
    cov = po.estimate_covariance(returns)
    assert cov.shape == (5, 5)
    assert np.allclose(cov.values, cov.values.T)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert eigvals.min() > -1e-10  # PSD (tolerancia numérica)


def test_estimate_covariance_drops_sparse_assets():
    returns = _synthetic_returns(n_assets=4, n_days=100)
    returns["A3"] = np.nan  # activo sin historial -> debe excluirse
    cov = po.estimate_covariance(returns, min_periods=10)
    assert "A3" not in cov.columns
    assert set(cov.columns) == {"A0", "A1", "A2"}


def test_estimate_covariance_raises_when_too_few_complete_rows():
    returns = pd.DataFrame({"A0": [np.nan] * 10, "A1": [np.nan] * 10})
    with pytest.raises(ValueError):
        po.estimate_covariance(returns, min_periods=1)


# ---------------------------------------------------------------------------
# optimize_portfolio
# ---------------------------------------------------------------------------

def test_optimize_portfolio_respects_all_constraints():
    tickers = ["A0", "A1", "A2", "A3"]
    scores = pd.Series([1.0, -1.0, 0.5, -0.5], index=tickers)
    cov = pd.DataFrame(np.eye(4) * 0.01, index=tickers, columns=tickers)

    result = po.optimize_portfolio(scores, cov, max_weight=0.3, max_leverage=1.0, risk_aversion=1.0)

    assert result.status == "optimal"
    assert result.weights.sum() == pytest.approx(0, abs=1e-6)
    assert result.weights.abs().sum() <= 1.0 + 1e-6
    assert (result.weights.abs() <= 0.3 + 1e-6).all()


def test_optimize_portfolio_higher_score_gets_higher_weight():
    tickers = ["A0", "A1"]
    scores = pd.Series([2.0, -2.0], index=tickers)
    cov = pd.DataFrame(np.eye(2) * 0.01, index=tickers, columns=tickers)

    result = po.optimize_portfolio(scores, cov, max_weight=0.5, max_leverage=1.0, risk_aversion=0.1)
    assert result.weights["A0"] > result.weights["A1"]
    assert result.weights["A0"] == pytest.approx(0.5, abs=1e-4)   # tope de concentración activo


def test_optimize_portfolio_aligns_indices_drops_unmatched():
    scores = pd.Series([1.0, -1.0, 0.5], index=["A0", "A1", "EXTRA_NO_COV"])
    cov = pd.DataFrame(np.eye(2) * 0.01, index=["A0", "A1"], columns=["A0", "A1"])

    result = po.optimize_portfolio(scores, cov, max_weight=0.5, max_leverage=1.0)
    assert set(result.weights.index) == {"A0", "A1"}


def test_optimize_portfolio_raises_when_no_common_assets():
    scores = pd.Series([1.0], index=["ONLY_HERE"])
    cov = pd.DataFrame(np.eye(1) * 0.01, index=["OTHER"], columns=["OTHER"])
    with pytest.raises(ValueError):
        po.optimize_portfolio(scores, cov)


def test_optimize_portfolio_max_leverage_binding():
    tickers = ["A0", "A1", "A2", "A3"]
    scores = pd.Series([3.0, -3.0, 2.0, -2.0], index=tickers)
    cov = pd.DataFrame(np.eye(4) * 0.001, index=tickers, columns=tickers)

    result = po.optimize_portfolio(scores, cov, max_weight=1.0, max_leverage=0.4, risk_aversion=0.01)
    assert result.weights.abs().sum() == pytest.approx(0.4, abs=1e-4)
