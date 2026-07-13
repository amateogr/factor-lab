"""
neutralizer.py — Neutralización de riesgo de scores cross-sectionales (QMJ u otros):
sectorial (de-mean por sector) y de beta de mercado (residualización vs exposición
histórica, estimada con Rolling OLS).

Target: factor-lab/src/neutralize/neutralizer.py

NOTA DE DISEÑO: las firmas pedidas (`sector_neutralize(df, score_col, sector_col)` y
`beta_neutralize(df, returns_col, market_returns_col, window=60)`) no incluían columnas
de fecha/activo/score explícitas, necesarias para agrupar cross-sectionalmente y para
identificar qué serie regredir. Se añadieron `date_col`, `asset_col` y `score_col` (en
beta_neutralize) con defaults ("period", "cik") consistentes con el índice de salida de
`factor_builder.build_qmj_panel`. Ambas funciones esperan columnas planas — si el input
viene de `build_qmj_panel` (indexado por (period, cik)), hacer `.reset_index()` antes.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

logger = logging.getLogger("neutralizer")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 1. Neutralización sectorial: de-mean (+ opcionalmente estandarización) por
#    (date_col, sector_col)
# ---------------------------------------------------------------------------

def sector_neutralize(
    df: pd.DataFrame,
    score_col: str,
    sector_col: str,
    date_col: str = "period",
    standardize: bool = True,
    min_group_size: int = 3,
) -> pd.Series:
    """
    De-mean transversal de `score_col` dentro de cada grupo (date_col, sector_col).

    standardize=True (default): también divide por el std del grupo -> z-score
    sectorial real (suma cero y varianza unitaria). standardize=False: solo de-mean
    (suma cero, varianza original) — usar si `score_col` ya viene en una escala
    (ej. un z-score global previo) que quieres preservar en magnitud.

    Grupos con n < min_group_size, o std=0 cuando standardize=True, devuelven NaN
    en vez de un neutralizado espurio o una división por cero.

    Retorna una Series alineada al índice original de `df`.
    """
    def _neutralize_group(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < min_group_size:
            return pd.Series(np.nan, index=s.index)
        mean = valid.mean()
        if not standardize:
            return s - mean
        std = valid.std(ddof=0)
        if std == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - mean) / std

    return df.groupby([date_col, sector_col], group_keys=False)[score_col].apply(_neutralize_group)


# ---------------------------------------------------------------------------
# 2. Beta histórico por activo vía Rolling OLS (asset_returns ~ market_returns)
# ---------------------------------------------------------------------------

def compute_rolling_beta(
    df: pd.DataFrame,
    returns_col: str,
    market_returns_col: str,
    asset_col: str = "cik",
    date_col: str = "period",
    window: int = 60,
    min_nobs: Optional[int] = None,
) -> pd.Series:
    """
    Beta rolling por activo: para cada `asset_col`, ordena por `date_col` y corre
    RollingOLS(returns_col ~ const + market_returns_col, window=window). Devuelve
    la pendiente (beta) alineada al índice original de `df`.

    min_nobs (default = window): observaciones mínimas para producir una estimación;
    ventanas con menos historia devuelven NaN — no rellena con beta=1 ni con el
    primer valor disponible, para no fabricar exposición donde no hay evidencia.
    """
    min_nobs = min_nobs or window
    betas = pd.Series(np.nan, index=df.index, dtype=float)

    for _, g in df.groupby(asset_col):
        g_sorted = g.sort_values(date_col)
        if len(g_sorted) < min_nobs:
            continue
        y = g_sorted[returns_col]
        X = sm.add_constant(g_sorted[[market_returns_col]])
        try:
            res = RollingOLS(y, X, window=window, min_nobs=min_nobs).fit()
        except ValueError as e:
            logger.warning("RollingOLS falló para un activo (n=%d, window=%d): %s",
                            len(g_sorted), window, e)
            continue
        betas.loc[g_sorted.index] = res.params[market_returns_col].values

    return betas


# ---------------------------------------------------------------------------
# 3. Regresión cross-sectional genérica: score ~ const + exposure, por fecha.
#    Building block reutilizable — usado por beta_neutralize, y directamente
#    utilizable para neutralizar contra cualquier otra exposición continua
#    (tamaño, momentum, iliquidez) sin duplicar la lógica de OLS.
# ---------------------------------------------------------------------------

def residualize_cross_sectional(
    df: pd.DataFrame,
    score_col: str,
    exposure_col: str,
    date_col: str = "period",
    min_group_size: int = 3,
) -> pd.Series:
    """
    Regresión cross-sectional `score_col ~ const + exposure_col` dentro de cada
    grupo de `date_col`. Devuelve el residuo — la parte de `score_col` no
    explicada linealmente por `exposure_col` — alineado al índice original de `df`.

    Fechas con < min_group_size observaciones válidas, o con `exposure_col` sin
    varianza (todos el mismo valor), devuelven NaN en vez de un residuo mal definido.
    """
    def _residualize_group(g: pd.DataFrame) -> pd.Series:
        result = pd.Series(np.nan, index=g.index)
        valid = g[[score_col, exposure_col]].dropna()
        if len(valid) < min_group_size or valid[exposure_col].std(ddof=0) == 0:
            return result
        X = sm.add_constant(valid[exposure_col])
        resid = sm.OLS(valid[score_col], X).fit().resid
        result.loc[resid.index] = resid.values
        return result

    # NOTA: DataFrameGroupBy.apply() con un único grupo total tiene una ambigüedad
    # conocida de pandas -- puede interpretar el Series devuelto como una fila ancha
    # (columnas = posiciones) en vez de apilarlo por fila. Se evita con loop+concat
    # explícito en vez de .apply(), robusto sin importar el número de grupos.
    parts = [_residualize_group(g) for _, g in df.groupby(date_col, sort=False)]
    return pd.concat(parts).loc[df.index]


# ---------------------------------------------------------------------------
# 4. Neutralización de beta: beta histórico por activo (Rolling OLS) + residualizar
#    el score contra esa exposición, cross-sectionalmente por fecha.
# ---------------------------------------------------------------------------

def beta_neutralize(
    df: pd.DataFrame,
    score_col: str,
    returns_col: str,
    market_returns_col: str,
    asset_col: str = "cik",
    date_col: str = "period",
    window: int = 60,
    min_nobs: Optional[int] = None,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """
    Pipeline completo: (1) beta histórico por activo vía `compute_rolling_beta`,
    (2) `residualize_cross_sectional(score_col, "beta")` en cada fecha — el residuo
    reemplaza al score original, restando la componente explicada linealmente por
    la exposición a mercado y dejando la parte "idiosincrática" de la señal.

    Retorna una copia de `df` con columnas nuevas: `beta` y `{score_col}_beta_neutral`.
    """
    out = df.copy()
    out["beta"] = compute_rolling_beta(
        out, returns_col, market_returns_col, asset_col, date_col, window, min_nobs
    )
    out[f"{score_col}_beta_neutral"] = residualize_cross_sectional(
        out, score_col, "beta", date_col, min_group_size
    )
    return out


if __name__ == "__main__":
    # Demo con datos sintéticos — no depende de red.
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=80, freq="D")
    market = rng.normal(0, 0.01, size=len(dates))

    rows = []
    true_betas = {"A": 1.5, "B": 0.5, "C": 1.0}
    for asset, beta_true in true_betas.items():
        noise = rng.normal(0, 0.005, size=len(dates))
        asset_ret = beta_true * market + noise
        for d, r, m in zip(dates, asset_ret, market):
            rows.append({"cik": asset, "period": d, "ret": r, "mkt_ret": m,
                          "gics_sector": "Tech" if asset != "C" else "Financials",
                          "qmj_score": rng.normal(0, 1)})
    demo = pd.DataFrame(rows)

    sector_result = sector_neutralize(demo, "qmj_score", "gics_sector", min_group_size=2)
    print("sector_neutralize (head):", sector_result.head().values)

    beta_result = beta_neutralize(demo, "qmj_score", "ret", "mkt_ret", window=20)
    recovered = beta_result.dropna(subset=["beta"]).groupby("cik")["beta"].last()
    print("beta recuperado (último) vs verdadero:", dict(recovered), true_betas)
