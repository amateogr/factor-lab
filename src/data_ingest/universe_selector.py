"""
universe_selector.py — Descarga y mapeo del universo de activos (S&P 500)
cruzando tickers de mercado con identificadores CIK de la SEC de forma local.

Target: factor-lab/src/data_ingest/universe_selector.py

NOTA DE AUDITORÍA (vs. versión original de Gemini): se corrigieron 2 bugs bloqueantes
encontrados antes de escribir tests:
1. Faltaba `from typing import Optional` — rompía `_map_ticker_to_cik` en tiempo de
   ejecución del módulo (NameError al definir la función con esa anotación).
2. Normalización de separador de ticker inconsistente entre el lado SEC ("-"->".")
   y el lado Wikipedia ("."->"-") — direcciones opuestas garantizaban un miss en
   cualquier ticker de clase de acción (ej. BRK.B / BRK-B). Reemplazado por lookup
   multi-candidato que no asume una dirección fija.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("universe_selector")
logging.basicConfig(level=logging.INFO)

CACHE_DIR = Path("cache/universe")
SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "SWD Research contact@example.com"}  # TODO: reemplazar con contacto real


def fetch_sec_ticker_to_cik_map(bypass_cache: bool = False) -> dict[str, int]:
    """Descarga el mapeo oficial ticker -> CIK de la SEC. Cachea en disco tras la
    primera descarga. Claves normalizadas a mayúsculas, sin forzar separador —
    el lookup multi-candidato de `_map_ticker_to_cik` absorbe la ambigüedad."""
    cache_file = CACHE_DIR / "sec_ticker_map.json"
    if cache_file.exists() and not bypass_cache:
        return json.loads(cache_file.read_text())

    logger.info("Descargando mapa de tickers oficial de la SEC...")
    resp = requests.get(SEC_TICKER_MAP_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    sec_data = resp.json()

    mapping: dict[str, int] = {
        item["ticker"].upper(): item["cik_str"] for item in sec_data.values()
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(mapping))
    return mapping


def _strip_separators(ticker: str) -> str:
    return ticker.replace(".", "").replace("-", "")


def _map_ticker_to_cik(
    ticker: str,
    sec_map: dict[str, int],
    stripped_map: Optional[dict[str, int]] = None,
) -> Optional[int]:
    """Prueba variantes de separador contra el mapa SEC sin asumir una dirección fija de
    normalización — cubre BRK.B / BRK-B / BRKB indistintamente en ambos sentidos.
    `stripped_map` (ticker sin separador -> CIK) se recibe precomputado para evitar
    reconstruirlo en cada llamada dentro de un `.apply()` sobre cientos de tickers;
    si no se pasa, se reconstruye aquí (más lento, cómodo para tests aislados)."""
    tk = ticker.upper()
    for candidate in (tk, tk.replace(".", "-"), tk.replace("-", ".")):
        if candidate in sec_map:
            return sec_map[candidate]

    stripped_map = stripped_map or {_strip_separators(k): v for k, v in sec_map.items()}
    return stripped_map.get(_strip_separators(tk))


def get_sp500_constituents(bypass_cache: bool = False) -> pd.DataFrame:
    """Raspa los componentes actuales del S&P 500 desde Wikipedia y los mapea con su CIK.
    Cachea el resultado final en Parquet."""
    cache_file = CACHE_DIR / "sp500_universe.parquet"
    if cache_file.exists() and not bypass_cache:
        logger.info("Cargando universo S&P 500 desde la caché local.")
        return pd.read_parquet(cache_file)

    logger.info("Raspando componentes del S&P 500 desde Wikipedia...")
    try:
        tables = pd.read_html(WIKI_SP500_URL)
        df_wiki = tables[0]
    except Exception as e:
        raise RuntimeError(f"Error al raspar Wikipedia: {e}") from e

    df_wiki = df_wiki.rename(columns={
        "Symbol": "ticker",
        "Security": "entity_name",
        "GICS Sector": "gics_sector",
        "GICS Sub-Industry": "gics_sub_industry",
    })
    df_wiki["ticker"] = df_wiki["ticker"].str.upper().str.strip()

    sec_map = fetch_sec_ticker_to_cik_map(bypass_cache=bypass_cache)
    stripped_map = {_strip_separators(k): v for k, v in sec_map.items()}
    df_wiki["cik"] = df_wiki["ticker"].apply(lambda tk: _map_ticker_to_cik(tk, sec_map, stripped_map))

    unmapped = df_wiki[df_wiki["cik"].isna()]
    if not unmapped.empty:
        logger.warning("No se pudo mapear CIK para: %s", unmapped["ticker"].tolist())

    df_universe = df_wiki.dropna(subset=["cik"]).copy()
    df_universe["cik"] = df_universe["cik"].astype(int)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df_universe.to_parquet(cache_file, index=False)
    logger.info("Universo S&P 500 inicializado con %d activos válidos mapeados.", len(df_universe))
    return df_universe


if __name__ == "__main__":
    universe = get_sp500_constituents(bypass_cache=True)
    print(universe[["ticker", "cik", "gics_sector"]].head())
