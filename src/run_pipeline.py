"""
run_pipeline.py — Orquestador end-to-end de factor-lab: universo -> mercado ->
fundamentals -> factor QMJ -> neutralización -> optimización -> pesos objetivo.

Target: factor-lab/src/run_pipeline.py
Ejecución: desde la raíz del proyecto, `python src/run_pipeline.py`.

NOTA DE ARQUITECTURA: los fundamentals son una foto de un solo periodo (el último
10-K disponible), pero el beta de mercado se estima con Rolling OLS sobre 2 años de
retornos diarios. Para combinar ambos en un único rebalanceo "de hoy": se calcula
beta sobre TODO el historial de retornos y se toma el valor más reciente por activo
(no se re-corre `neutralizer.beta_neutralize` completo, que asume un score por fecha
en el mismo panel — aquí el score es constante por activo); la neutralización de
beta del snapshot usa `residualize_cross_sectional` directamente (misma función que
usa `beta_neutralize` internamente, sin duplicar la lógica de OLS).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_ingest.universe_selector import get_sp500_constituents
from data_ingest.xbrl_tag_mapper import resolve_concept_panel
from data_ingest.returns_ingestor import build_returns_panel, attach_cik
from factor_construct.factor_builder import assemble_accounting_panel, build_qmj_panel, QMJConfig
from neutralize.neutralizer import sector_neutralize, compute_rolling_beta, residualize_cross_sectional
from optimizer.portfolio_optimizer import estimate_covariance, optimize_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Fundamentals: carga cacheada, simulada por defecto (SEC real es lento a escala)
# ---------------------------------------------------------------------------

def _parse_period(period: str) -> tuple[int, int]:
    """"CY2024" -> (2024, 4). Asume panel anual (Q4 = último 10-K del año)."""
    return int(period.replace("CY", "")), 4


def _simulate_fundamentals_panel(universe: pd.DataFrame, period: str, seed: int = 42) -> pd.DataFrame:
    """Genera un panel contable SINTÉTICO con valores plausibles por CIK. No es
    dato real de SEC — solo para poder correr el pipeline end-to-end sin esperar
    ~500 llamadas rate-limited a EDGAR. Reproducible vía `seed`."""
    rng = np.random.default_rng(seed)
    n = len(universe)
    total_assets = rng.lognormal(mean=15, sigma=1.5, size=n)
    leverage_ratio = rng.uniform(0.2, 0.85, size=n)
    liabilities = total_assets * leverage_ratio
    stockholders_equity = total_assets - liabilities
    net_income = stockholders_equity * rng.normal(0.10, 0.08, size=n)

    tag = "SIMULATED"
    return pd.DataFrame({
        "period": period, "cik": universe["cik"].values,
        "net_income": net_income, "net_income_source_tag": tag,
        "stockholders_equity": stockholders_equity, "stockholders_equity_source_tag": tag,
        "total_assets": total_assets, "total_assets_source_tag": tag,
        "liabilities": liabilities, "liabilities_source_tag": tag,
    })


def load_or_fetch_fundamentals(
    universe: pd.DataFrame,
    period: str,
    cache_path: Path,
    use_real_sec: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Panel contable ancho para `period` (columnas: period, cik, net_income,
    stockholders_equity, total_assets, liabilities, + *_source_tag).

    use_real_sec=False (default): si `cache_path` no existe, genera un panel
    SIMULADO (source_tag="SIMULATED") en vez de pegarle a la SEC — pensado para
    correr el pipeline completo sin esperar ~500 llamadas rate-limited a EDGAR
    (~10 req/s -> minutos por corrida en frío). Cachea para reruns reproducibles.

    use_real_sec=True: si el caché no existe, pega a EDGAR vía
    `xbrl_tag_mapper.resolve_concept_panel` para los 4 conceptos y ensambla con
    `factor_builder.assemble_accounting_panel`. Úsalo solo cuando ya validaste el
    pipeline con datos simulados — una corrida en frío contra 500 CIKs reales tarda.
    """
    if cache_path.exists():
        logger.info("Fundamentals: cargando panel cacheado desde %s", cache_path)
        return pd.read_parquet(cache_path)

    if use_real_sec:
        logger.info("Fundamentals: caché ausente, descargando de SEC EDGAR (puede tardar varios minutos)...")
        year, quarter = _parse_period(period)
        panel = assemble_accounting_panel(
            resolve_concept_panel("net_income", year, quarter),
            resolve_concept_panel("stockholders_equity", year, quarter),
            resolve_concept_panel("total_assets", year, quarter),
            resolve_concept_panel("liabilities", year, quarter),
            period=period,
        )
    else:
        logger.warning(
            "Fundamentals: caché ausente -> GENERANDO PANEL SIMULADO (no es dato real de SEC). "
            "Usa use_real_sec=True para producción."
        )
        panel = _simulate_fundamentals_panel(universe, period, seed=seed)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path, index=False)
    return panel


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------

def main(
    universe_limit: Optional[int] = None,
    years_history: int = 2,
    period_label: Optional[str] = None,
    use_real_sec: bool = False,
    max_weight: float = 0.05,
    max_leverage: float = 1.0,
    risk_aversion: float = 5.0,
    beta_window: int = 60,
    min_group_size: int = 3,
    output_path: str = "reports/optimal_weights.csv",
) -> pd.DataFrame:
    """Ejecuta el pipeline de rebalanceo completo y devuelve/guarda los pesos objetivo."""
    logger.info("=== factor-lab | pipeline de rebalanceo — inicio ===")

    # 1. Universo
    logger.info("[1/6] Universo: descargando constituyentes S&P 500...")
    universe = get_sp500_constituents()
    if universe_limit:
        universe = universe.head(universe_limit).copy()
        logger.info("Universo recortado a %d activos (universe_limit)", universe_limit)
    logger.info("Universo: %d activos con CIK mapeado", len(universe))

    # 2. Mercado
    logger.info("[2/6] Mercado: descargando retornos (%d años de historia)...", years_history)
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(years=years_history)
    returns_panel = build_returns_panel(
        universe["ticker"].tolist(), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
    )
    returns_panel = attach_cik(returns_panel, universe)
    logger.info("Retornos: %d observaciones para %d activos", len(returns_panel), returns_panel["cik"].nunique())

    # 3. Fundamentals + Factor
    logger.info("[3/6] Fundamentals: cargando panel contable...")
    period_label = period_label or f"CY{end_date.year - 1}"
    cache_path = Path("cache/fundamentals") / f"{period_label}.parquet"
    accounting_panel = load_or_fetch_fundamentals(universe, period_label, cache_path, use_real_sec=use_real_sec)

    logger.info("Factor: construyendo panel QMJ...")
    qmj_panel = build_qmj_panel(accounting_panel, universe=universe, config=QMJConfig(sector_relative=False))
    qmj_snapshot = qmj_panel.reset_index()
    logger.info(
        "QMJ: %d/%d activos con score calculado",
        qmj_snapshot["qmj_score"].notna().sum(), len(qmj_snapshot),
    )

    # 4. Neutralización
    logger.info("[4/6] Neutralización: beta histórico + sector + residualización...")
    returns_panel["beta"] = compute_rolling_beta(
        returns_panel, "ret", "mkt_ret", asset_col="cik", date_col="date", window=beta_window
    )
    latest_beta = (
        returns_panel.dropna(subset=["beta"]).sort_values("date").groupby("cik")["beta"].last()
    )
    qmj_snapshot = qmj_snapshot.merge(latest_beta.rename("beta"), on="cik", how="left")

    qmj_snapshot["qmj_sector_neutral"] = sector_neutralize(
        qmj_snapshot, "qmj_score", "gics_sector", date_col="period", min_group_size=min_group_size
    )
    qmj_snapshot["qmj_final_score"] = residualize_cross_sectional(
        qmj_snapshot, "qmj_sector_neutral", "beta", date_col="period", min_group_size=min_group_size
    )
    n_final = qmj_snapshot["qmj_final_score"].notna().sum()
    logger.info("Neutralización: %d activos con score final utilizable", n_final)
    if n_final == 0:
        raise RuntimeError("Ningún activo sobrevivió la neutralización — revisa cobertura de sector/beta")

    # 5. Optimización
    logger.info("[5/6] Optimización: covarianza Ledoit-Wolf + QP dollar-neutral...")
    returns_matrix = returns_panel.pivot(index="date", columns="cik", values="ret")
    cov = estimate_covariance(returns_matrix)

    scores = qmj_snapshot.dropna(subset=["qmj_final_score"]).set_index("cik")["qmj_final_score"]
    result = optimize_portfolio(
        scores, cov, max_weight=max_weight, max_leverage=max_leverage, risk_aversion=risk_aversion
    )
    if result.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimización no convergió: status={result.status}")
    n_active = (result.weights.abs() > 1e-6).sum()
    logger.info("Optimización: status=%s, %d activos con peso != 0, apalancamiento=%.3f",
                result.status, n_active, result.weights.abs().sum())

    # 6. Serialización
    logger.info("[6/6] Serialización: guardando %s...", output_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    weights_df = result.weights.rename("weight").rename_axis("cik").reset_index()
    lookup = qmj_snapshot[["cik", "ticker", "gics_sector"]].drop_duplicates("cik")
    weights_df = weights_df.merge(lookup, on="cik", how="left")
    weights_df = weights_df[["ticker", "cik", "weight", "gics_sector"]].sort_values("weight", ascending=False)
    weights_df.to_csv(out_path, index=False)
    logger.info("Guardado: %s (%d filas)", out_path, len(weights_df))

    logger.info("=== pipeline de rebalanceo — fin ===")
    return weights_df


if __name__ == "__main__":
    main()
