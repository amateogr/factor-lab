"""
returns_ingestor.py — Descarga de precios (yfinance) y cálculo de retornos diarios
para el universo y el proxy de mercado (SPY), en formato panel largo.

Target: factor-lab/src/data_ingest/returns_ingestor.py

NOTA: la superficie de la API de yfinance ha cambiado entre versiones (ej. el default
de `auto_adjust` cambió de False a True en yfinance>=0.2.31, lo que afecta si 'Adj Close'
existe como columna separada de 'Close'). Este módulo fuerza `auto_adjust=False`
explícitamente para no depender del default de la versión instalada. finance.yahoo.com
no está en el whitelist de red de este entorno -> no se pudo probar contra la API real;
los tests usan `monkeypatch` sobre `yf.download` con la forma de respuesta documentada.
Valida manualmente contra una descarga real antes de producción.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger("returns_ingestor")
logging.basicConfig(level=logging.INFO)

MARKET_TICKER = "SPY"


def download_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    include_market: bool = True,
) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustado (Adj Close) para `tickers` (+ SPY si
    include_market=True) entre start_date y end_date. Tickers delisted, sin
    historial en el rango, o que fallan la descarga quedan como columnas
    completamente NaN en yfinance -> se detectan y se excluyen explícitamente,
    con warning, en vez de propagarse silenciosamente al resto del pipeline.

    Retorna DataFrame ancho: índice=fecha, columnas=ticker, valores=Adj Close.
    """
    request_tickers = list(dict.fromkeys(tickers + ([MARKET_TICKER] if include_market else [])))

    raw = yf.download(
        request_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError(f"yfinance no devolvió datos para {request_tickers} en [{start_date}, {end_date}]")

    prices = _extract_adj_close(raw, request_tickers)

    failed = prices.columns[prices.isna().all()].tolist()
    if failed:
        logger.warning("Tickers sin datos (delisted/inválidos), excluidos: %s", failed)
        prices = prices.drop(columns=failed)

    if include_market and MARKET_TICKER not in prices.columns:
        raise RuntimeError("Descarga de SPY (proxy de mercado) falló — no se puede construir mkt_ret")

    return prices


def _extract_adj_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normaliza la forma de salida de yfinance (varía según 1 vs N tickers) a un
    DataFrame ancho consistente: índice=fecha, columnas=ticker."""
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" not in raw.columns.get_level_values(0):
            raise KeyError("'Adj Close' no está en la respuesta de yfinance — revisa auto_adjust/versión instalada")
        return raw["Adj Close"].copy()

    if "Adj Close" not in raw.columns:
        raise KeyError("'Adj Close' no está en la respuesta de yfinance para ticker único")
    single_ticker = [t for t in tickers if t != MARKET_TICKER] or tickers
    return raw[["Adj Close"]].rename(columns={"Adj Close": single_ticker[0]})


def compute_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Retornos porcentuales diarios simples: (P_t / P_{t-1}) - 1. La primera fila
    de cada serie queda NaN (sin precio previo) — se conserva así, no se rellena."""
    return prices_df.pct_change(fill_method=None)


def build_returns_panel(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Orquestador: descarga precios (universo + SPY), calcula retornos, y devuelve
    panel largo con columnas [date, ticker, ret, mkt_ret] — mkt_ret es el retorno
    de SPY en esa fecha, repetido para cada ticker (broadcast), listo para
    `neutralizer.beta_neutralize`.
    """
    prices = download_market_data(tickers, start_date, end_date, include_market=True)
    returns = compute_daily_returns(prices)

    if MARKET_TICKER not in returns.columns:
        raise RuntimeError("mkt_ret no disponible — SPY no se descargó correctamente")
    mkt_ret = returns[MARKET_TICKER]

    asset_returns = returns.drop(columns=[MARKET_TICKER])
    long_panel = asset_returns.stack().reset_index()
    long_panel.columns = ["date", "ticker", "ret"]
    long_panel["mkt_ret"] = long_panel["date"].map(mkt_ret)

    return long_panel.dropna(subset=["ret"]).reset_index(drop=True)


def attach_cik(returns_panel: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza el panel de retornos (columna `ticker`) con el universo de
    `universe_selector.get_sp500_constituents()` (columnas [ticker, cik, ...])
    para añadir `cik`. Tickers del panel sin match en el universo se reportan
    y se excluyen — no se puede correr `beta_neutralize` sin un `cik` para agrupar.
    """
    merged = returns_panel.merge(universe[["ticker", "cik"]], on="ticker", how="left")
    unmatched = merged.loc[merged["cik"].isna(), "ticker"].unique().tolist()
    if unmatched:
        logger.warning("Tickers en el panel de retornos sin CIK en el universo: %s", unmatched)
    merged = merged.dropna(subset=["cik"]).copy()
    merged["cik"] = merged["cik"].astype(int)
    return merged


if __name__ == "__main__":
    import sys
    demo_tickers = sys.argv[1:] or ["AAPL", "MSFT"]
    panel = build_returns_panel(demo_tickers, "2023-01-01", "2023-06-30")
    print(panel.head())
