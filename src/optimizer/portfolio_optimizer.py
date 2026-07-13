"""
portfolio_optimizer.py — Optimizador mean-variance dollar-neutral: covarianza shrinkage
(Ledoit-Wolf) + programa cuadrático convexo (cvxpy) sobre scores QMJ neutralizados.

Target: factor-lab/src/optimizer/portfolio_optimizer.py

NOTA DE DISEÑO: la firma pedida para `optimize_portfolio` no incluía el coeficiente de
aversión al riesgo (lambda) del objetivo "scores^T w - lambda * w^T Cov w" — se añadió
`risk_aversion: float = 1.0` como parámetro explícito, ya que sin él el objetivo no está
definido. También se expone `solver` para poder fijar el solver de cvxpy si el default
automático no converge bien en tu universo real.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("portfolio_optimizer")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 1. Covarianza shrinkage (Ledoit-Wolf)
# ---------------------------------------------------------------------------

def estimate_covariance(returns_matrix: pd.DataFrame, min_periods: Optional[int] = None) -> pd.DataFrame:
    """
    Covarianza shrinkage (Ledoit-Wolf) sobre `returns_matrix` (índice=fecha,
    columnas=activo). Excluye activos con menos de `min_periods` observaciones no
    nulas (default: la mitad de las fechas disponibles), luego usa solo filas
    completas (sin NaN) entre los activos restantes — LedoitWolf no tolera NaNs.

    n_obs < n_activos no es un error: es justo el caso para el que existe el
    shrinkage (la covarianza muestral simple sería singular ahí), solo se loguea.

    Retorna DataFrame cuadrado indexado/columnado por activo.
    """
    min_periods = min_periods if min_periods is not None else len(returns_matrix) // 2
    coverage = returns_matrix.notna().sum()
    sparse_assets = coverage[coverage < min_periods].index.tolist()
    if sparse_assets:
        logger.warning("Excluidos por historial insuficiente (<%d obs): %s", min_periods, sparse_assets)
    dense = returns_matrix.drop(columns=sparse_assets)

    complete = dense.dropna(how="any")
    if len(complete) < 2:
        raise ValueError(
            f"Solo {len(complete)} filas completas tras alinear activos — insuficiente para estimar covarianza"
        )
    if complete.shape[0] < complete.shape[1]:
        logger.warning(
            "n_obs (%d) < n_activos (%d) — LedoitWolf sigue siendo válido, es justo el caso para el que existe shrinkage",
            complete.shape[0], complete.shape[1],
        )

    lw = LedoitWolf().fit(complete.values)
    return pd.DataFrame(lw.covariance_, index=complete.columns, columns=complete.columns)


# ---------------------------------------------------------------------------
# 2. Optimizador QP dollar-neutral
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    status: str
    objective_value: Optional[float]


def optimize_portfolio(
    expected_scores: pd.Series,
    cov_matrix: pd.DataFrame,
    max_weight: float = 0.05,
    max_leverage: float = 1.0,
    risk_aversion: float = 1.0,
    solver: Optional[str] = None,
) -> OptimizationResult:
    """
    Maximiza `expected_scores^T w - risk_aversion * w^T cov_matrix w` sujeto a:
      1. ||w||_1 <= max_leverage           (control de apalancamiento)
      2. -max_weight <= w_i <= max_weight  (control de concentración)
      3. sum(w) == 0                       (dollar-neutral)

    Alinea `expected_scores` y `cov_matrix` por la intersección de índices/columnas
    antes de optimizar — activos presentes en uno pero no en el otro se excluyen
    (logueado), nunca se rellenan con score o covarianza ficticios.
    """
    common = expected_scores.index.intersection(cov_matrix.index).intersection(cov_matrix.columns)
    dropped_scores = expected_scores.index.difference(common)
    dropped_cov = cov_matrix.index.difference(common)
    if len(dropped_scores) or len(dropped_cov):
        logger.warning(
            "Excluidos por falta de score o covarianza — sin_cov=%s sin_score=%s",
            dropped_scores.tolist(), dropped_cov.tolist(),
        )
    if len(common) == 0:
        raise ValueError("Sin activos en común entre expected_scores y cov_matrix")

    common = list(common)
    mu = expected_scores.loc[common].values
    sigma = cov_matrix.loc[common, common].values
    n = len(common)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - risk_aversion * cp.quad_form(w, sigma, assume_PSD=True))
    constraints = [
        cp.norm(w, 1) <= max_leverage,
        w >= -max_weight,
        w <= max_weight,
        cp.sum(w) == 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        logger.error("Optimización no convergió: status=%s", problem.status)
        return OptimizationResult(
            weights=pd.Series(np.nan, index=common), status=problem.status, objective_value=None
        )

    weights = pd.Series(np.asarray(w.value).ravel(), index=common)
    return OptimizationResult(weights=weights, status=problem.status, objective_value=problem.value)


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n_assets, n_days = 6, 120
    tickers = [f"A{i}" for i in range(n_assets)]
    factor = rng.normal(0, 0.01, size=n_days)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    returns = pd.DataFrame(
        {tk: betas[i] * factor + rng.normal(0, 0.005, size=n_days) for i, tk in enumerate(tickers)}
    )

    cov = estimate_covariance(returns)
    scores = pd.Series(rng.normal(0, 1, size=n_assets), index=tickers)

    result = optimize_portfolio(scores, cov, max_weight=0.3, max_leverage=1.0, risk_aversion=5.0)
    print("status:", result.status)
    print(result.weights)
    print("suma pesos (dollar-neutral, ~0):", result.weights.sum())
    print("apalancamiento (suma |w|):", result.weights.abs().sum())
