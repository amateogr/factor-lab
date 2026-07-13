"""
factor_builder.py — Construcción cross-sectional del factor Quality-Minus-Junk (QMJ):
ratios contables -> limpieza -> winsorización -> z-score -> score compuesto.

Target: factor-lab/src/factor_construct/factor_builder.py

Input esperado: paneles crudos por concepto (salida de xbrl_tag_mapper.resolve_concept_panel),
cada uno con columnas [cik, val, end, source_tag], para un periodo dado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("factor_builder")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# 1. Ensamblado del panel contable: concept panels crudos -> panel ancho por CIK
# ---------------------------------------------------------------------------

def assemble_accounting_panel(
    net_income: pd.DataFrame,
    stockholders_equity: pd.DataFrame,
    total_assets: pd.DataFrame,
    liabilities: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """
    Combina 4 paneles crudos (columnas [cik, val, source_tag]) en un panel ancho:
    una fila por CIK con las 4 métricas + provenance por columna. `period` es la
    etiqueta del periodo (ej. "CY2023") que se propaga como columna/índice.
    """
    def _prep(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["cik", name, f"{name}_source_tag"])
        return df[["cik", "val", "source_tag"]].rename(
            columns={"val": name, "source_tag": f"{name}_source_tag"}
        )

    panel = _prep(net_income, "net_income")
    for df, name in [
        (stockholders_equity, "stockholders_equity"),
        (total_assets, "total_assets"),
        (liabilities, "liabilities"),
    ]:
        panel = panel.merge(_prep(df, name), on="cik", how="outer")

    panel.insert(0, "period", period)
    return panel


def stack_periods(panels: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatena paneles de `assemble_accounting_panel` de distintos periodos en
    un único DataFrame multi-periodo, input esperado de `build_qmj_panel`."""
    return pd.concat(panels, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Componentes QMJ: ROE (profitability) y Leverage (safety) — con máscaras
#    para evitar ratios económicamente sin sentido (equity negativo, división por ~0)
# ---------------------------------------------------------------------------

def compute_roe(panel: pd.DataFrame, min_abs_equity: float = 1e-6) -> pd.DataFrame:
    """
    ROE = NetIncomeLoss / StockholdersEquity.
    Enmascara (NaN) cuando equity <= 0: con equity negativo, el signo del ROE se invierte
    de forma económicamente engañosa (empresa en distress con pérdidas puede mostrar ROE
    positivo). AQR excluye estos casos de la métrica de calidad en vez de usarlos tal cual.
    También enmascara equity cercano a cero (ambos signos) para evitar ratios explosivos
    por división casi por cero.
    """
    out = panel.copy()
    equity = out["stockholders_equity"]

    out["equity_negative_flag"] = equity <= 0
    safe_equity = equity.where((equity > min_abs_equity) | (equity < -min_abs_equity))

    with np.errstate(divide="ignore", invalid="ignore"):
        roe = out["net_income"] / safe_equity
    roe = roe.replace([np.inf, -np.inf], np.nan)
    out["roe"] = roe.where(~out["equity_negative_flag"])
    return out


def compute_leverage(panel: pd.DataFrame, min_assets: float = 1e-6) -> pd.DataFrame:
    """
    Leverage = Liabilities / Assets. Enmascara cuando assets <= 0 (guardia defensiva,
    económicamente casi imposible en un filer real, pero previene división por cero silenciosa).
    """
    out = panel.copy()
    assets = out["total_assets"]
    safe_assets = assets.where(assets > min_assets)

    with np.errstate(divide="ignore", invalid="ignore"):
        lev = out["liabilities"] / safe_assets
    out["leverage"] = lev.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------------
# 3. Winsorización cross-sectional
# ---------------------------------------------------------------------------

def winsorize_cross_sectional(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str] = ("period",),
    method: str = "percentile",
    lower: float = 0.01,
    upper: float = 0.99,
    n_mad: float = 5.0,
) -> pd.Series:
    """
    Trunca outliers dentro de cada grupo cross-seccional (por defecto, por periodo).
    method="percentile": clip a [percentil `lower`, percentil `upper`].
    method="mad": clip a mediana ± n_mad * MAD escalado (1.4826x, consistente con std
    bajo normalidad) — más robusto que percentiles en universos pequeños o muy sesgados.
    Grupos sin observaciones válidas devuelven la serie sin cambios (no error).
    """
    def _winsorize_group(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if valid.empty:
            return s
        if method == "percentile":
            lo, hi = valid.quantile([lower, upper])
        elif method == "mad":
            med = valid.median()
            mad = (valid - med).abs().median() * 1.4826
            if mad == 0:
                return s
            lo, hi = med - n_mad * mad, med + n_mad * mad
        else:
            raise ValueError(f"method desconocido: {method!r}")
        return s.clip(lower=lo, upper=hi)

    if not group_cols:
        return _winsorize_group(df[value_col])
    return df.groupby(list(group_cols), group_keys=False)[value_col].apply(_winsorize_group)


# ---------------------------------------------------------------------------
# 4. Z-score cross-sectional (global o sector-relativo)
# ---------------------------------------------------------------------------

def zscore_cross_sectional(
    df: pd.DataFrame,
    value_col: str,
    group_cols: Sequence[str] = ("period",),
    min_group_size: int = 5,
) -> pd.Series:
    """
    Z-score dentro de cada grupo (por defecto, por periodo -> estandarización global;
    añade "gics_sector" a group_cols para estandarización sector-relativa). Grupos con
    std=0 o n < min_group_size devuelven NaN en vez de +/-inf o un z-score sin
    significancia estadística real.
    """
    def _zscore_group(s: pd.Series) -> pd.Series:
        valid = s.dropna()
        if len(valid) < min_group_size or valid.std(ddof=0) == 0:
            return pd.Series(np.nan, index=s.index)
        return (s - valid.mean()) / valid.std(ddof=0)

    return df.groupby(list(group_cols), group_keys=False)[value_col].apply(_zscore_group)


# ---------------------------------------------------------------------------
# 5. Orquestador: panel crudo -> señales QMJ listas para neutralización/optimizador
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QMJConfig:
    winsor_method: str = "percentile"
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    winsor_n_mad: float = 5.0
    sector_relative: bool = False
    min_group_size: int = 5


def build_qmj_panel(
    accounting_panel: pd.DataFrame,
    universe: Optional[pd.DataFrame] = None,
    config: QMJConfig = QMJConfig(),
) -> pd.DataFrame:
    """
    Pipeline completo: accounting_panel (salida de assemble_accounting_panel/stack_periods,
    puede cubrir múltiples periodos apilados) -> ROE + Leverage -> winsorización -> z-score
    (global o sector-relativo si `universe` trae `gics_sector`) -> score compuesto QMJ.

    `universe`: DataFrame de universe_selector.get_sp500_constituents(), columnas
    [ticker, cik, entity_name, gics_sector, ...] — se cruza por `cik` para (a) traer ticker
    legible y (b) habilitar estandarización sectorial si config.sector_relative=True.

    Nota de signo: Leverage alto = menor calidad, así que su z-score se invierte
    (`safety_zscore = -leverage_zscore`) antes de combinar — sin esto, el score compuesto
    premiaría el apalancamiento en vez de castigarlo.

    Retorna DataFrame indexado por (period, cik), listo para neutralize/ y el optimizador.
    """
    panel = compute_roe(accounting_panel)
    panel = compute_leverage(panel)

    if universe is not None:
        panel = panel.merge(
            universe[["cik", "ticker", "gics_sector"]], on="cik", how="left"
        )
        n_unmatched = panel["ticker"].isna().sum()
        if n_unmatched:
            logger.warning("%d filas del panel contable sin match en el universo (sin ticker/sector)",
                            n_unmatched)

    if config.sector_relative and "gics_sector" not in panel.columns:
        logger.warning("sector_relative=True pero no se pasó `universe` con gics_sector — cae a z-score global")
    group_cols = ["period", "gics_sector"] if (config.sector_relative and "gics_sector" in panel.columns) \
        else ["period"]

    panel["roe_winsorized"] = winsorize_cross_sectional(
        panel, "roe", group_cols=["period"], method=config.winsor_method,
        lower=config.winsor_lower, upper=config.winsor_upper, n_mad=config.winsor_n_mad,
    )
    panel["leverage_winsorized"] = winsorize_cross_sectional(
        panel, "leverage", group_cols=["period"], method=config.winsor_method,
        lower=config.winsor_lower, upper=config.winsor_upper, n_mad=config.winsor_n_mad,
    )

    panel["roe_zscore"] = zscore_cross_sectional(
        panel, "roe_winsorized", group_cols=group_cols, min_group_size=config.min_group_size,
    )
    panel["leverage_zscore"] = zscore_cross_sectional(
        panel, "leverage_winsorized", group_cols=group_cols, min_group_size=config.min_group_size,
    )
    panel["safety_zscore"] = -panel["leverage_zscore"]

    component_cols = ["roe_zscore", "safety_zscore"]
    panel["qmj_score"] = panel[component_cols].mean(axis=1, skipna=True)
    panel["qmj_n_components"] = panel[component_cols].notna().sum(axis=1)

    return panel.set_index(["period", "cik"]).sort_index()


if __name__ == "__main__":
    # Demo con datos sintéticos — no depende de red.
    net_income = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [100, -50, 20, 30, 5],
                                "source_tag": ["NetIncomeLoss"] * 5})
    equity = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [500, -10, 200, 150, 400],
                            "source_tag": ["StockholdersEquity"] * 5})
    assets = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [1000, 800, 600, 900, 700],
                            "source_tag": ["Assets"] * 5})
    liabs = pd.DataFrame({"cik": [1, 2, 3, 4, 5], "val": [400, 700, 100, 300, 600],
                           "source_tag": ["Liabilities"] * 5})

    raw = assemble_accounting_panel(net_income, equity, assets, liabs, period="CY2023")
    result = build_qmj_panel(raw, config=QMJConfig(min_group_size=3))
    print(result[["roe", "leverage", "roe_zscore", "safety_zscore", "qmj_score"]])
