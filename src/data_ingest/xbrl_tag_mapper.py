"""
xbrl_tag_mapper.py — Fallback resolution para conceptos US-GAAP XBRL con tagging
inconsistente entre filers (SEC EDGAR companyfacts + frames).

Target: factor-lab/src/data_ingest/xbrl_tag_mapper.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("xbrl_tag_mapper")
logging.basicConfig(level=logging.INFO)

HEADERS = {"User-Agent": "SWD Research contact@example.com"}  # TODO: EDGAR banea IP sin User-Agent identificable real
CACHE_DIR = Path("cache/xbrl")
RATE_LIMIT_S = 0.15          # ~6-7 req/s, conservador bajo el límite de 10 req/s de EDGAR
MAX_RETRIES = 4

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"


# ---------------------------------------------------------------------------
# 1. Tabla de fallback: concepto económico -> tags US-GAAP candidatos (orden = prioridad)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptSpec:
    tags: tuple[str, ...]   # primer tag con datos para un `end` dado gana esa fecha
    instant: bool           # True = balance sheet (point-in-time) | False = flow (duración)


CONCEPT_MAP: dict[str, ConceptSpec] = {
    "net_income": ConceptSpec(
        tags=(
            "NetIncomeLoss",
            "ProfitLoss",
            "NetIncomeLossAvailableToCommonStockholdersBasic",
            "IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest",
        ),
        instant=False,
    ),
    "stockholders_equity": ConceptSpec(
        tags=(
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "StockholdersEquity",
        ),
        instant=True,
    ),
    "total_assets": ConceptSpec(tags=("Assets",), instant=True),
    "liabilities": ConceptSpec(tags=("Liabilities",), instant=True),
    "liabilities_and_equity": ConceptSpec(tags=("LiabilitiesAndStockholdersEquity",), instant=True),
}


# ---------------------------------------------------------------------------
# 2. Capa HTTP cacheada en disco, retry/backoff, rate limit
# ---------------------------------------------------------------------------

def _cached_get(url: str, cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(data))
            time.sleep(RATE_LIMIT_S)
            return data
        if resp.status_code in (429, 503):
            logger.warning("rate limited (%s), backoff %.1fs, intento %d/%d",
                            resp.status_code, backoff, attempt, MAX_RETRIES)
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"fallo tras {MAX_RETRIES} intentos: {url}")


def fetch_companyfacts(cik: int) -> dict:
    url = FACTS_URL.format(cik=cik)
    return _cached_get(url, CACHE_DIR / "companyfacts" / f"CIK{cik:010d}.json")


def fetch_frame(tag: str, period: str) -> dict:
    url = FRAME_URL.format(tag=tag, period=period)
    return _cached_get(url, CACHE_DIR / "frames" / f"{tag}_{period}.json")


# ---------------------------------------------------------------------------
# 3. Serie temporal por compañía (companyfacts) con fallback multi-tag.
#    Conserva TODAS las versiones (accn) por `end` del tag ganador, para que
#    as_of() pueda distinguir entre filing original y restatements.
# ---------------------------------------------------------------------------

def resolve_concept_timeseries(cik: int, concept: str, fiscal_period: str = "FY") -> pd.DataFrame:
    spec = CONCEPT_MAP[concept]
    facts = fetch_companyfacts(cik).get("facts", {}).get("us-gaap", {})

    def _valid(e: dict) -> bool:
        if e.get("fp") != fiscal_period:
            return False
        if spec.instant and e.get("start") is not None:
            return False
        if not spec.instant and e.get("start") is None:
            return False
        return True

    tag_entries = {
        tag: [e for e in facts.get(tag, {}).get("units", {}).get("USD", []) if _valid(e)]
        for tag in spec.tags
    }

    covered_ends: set[str] = set()
    winning_tag_for_end: dict[str, str] = {}
    for tag in spec.tags:                                   # prioridad descendente
        ends_here = {e["end"] for e in tag_entries[tag]}
        for end in ends_here - covered_ends:
            winning_tag_for_end[end] = tag
        covered_ends |= ends_here

    rows = []
    for tag, entries in tag_entries.items():
        for e in entries:
            if winning_tag_for_end.get(e["end"]) != tag:
                continue
            rows.append({
                "end": e["end"], "start": e.get("start"), "val": e["val"],
                "form": e.get("form"), "fy": e.get("fy"), "filed": e.get("filed"),
                "accn": e.get("accn"), "source_tag": tag,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("cik=%s concept=%s: sin datos tras fallback %s", cik, concept, spec.tags)
        return df
    return df.sort_values(["end", "filed"]).reset_index(drop=True)


def _latest_vintage(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa a la versión con `filed` más reciente por `end`. Solo para inputs
    secundarios de fallbacks derivados — pierde histórico de restatements a
    propósito, para evitar cartesian blow-up en el merge de identidad contable."""
    if df.empty or "filed" not in df.columns:
        return df
    return df.sort_values("filed").drop_duplicates("end", keep="last")


def resolve_liabilities_with_identity_fallback(cik: int, fiscal_period: str = "FY") -> pd.DataFrame:
    """Liabilities = LiabilitiesAndStockholdersEquity - StockholdersEquity(NCI-inclusive).
    Fallback de última instancia, solo para periodos sin tag directo `Liabilities`."""
    direct = resolve_concept_timeseries(cik, "liabilities", fiscal_period)
    l_and_e = _latest_vintage(resolve_concept_timeseries(cik, "liabilities_and_equity", fiscal_period))
    equity = _latest_vintage(resolve_concept_timeseries(cik, "stockholders_equity", fiscal_period))

    if l_and_e.empty or equity.empty:
        return direct

    merged = l_and_e.merge(equity, on="end", suffixes=("_le", "_eq"))
    merged["val"] = merged["val_le"] - merged["val_eq"]
    merged["source_tag"] = "derived:LiabilitiesAndStockholdersEquity-StockholdersEquity"
    derived = merged[["end", "val", "source_tag"]]

    if direct.empty:
        return derived
    missing_ends = set(derived["end"]) - set(direct["end"])
    fill = derived[derived["end"].isin(missing_ends)]
    out = pd.concat([direct[["end", "val", "source_tag"]], fill], ignore_index=True)
    return out.sort_values("end").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Lookup point-in-time — usa `filed`, no `end`, para evitar look-ahead bias.
#    Entre versiones (original vs restatement) del mismo `end`, toma la de
#    `filed` más reciente que aún sea <= asof_date.
# ---------------------------------------------------------------------------

def as_of(df: pd.DataFrame, asof_date: str) -> Optional[dict]:
    if df.empty:
        return None
    if "filed" in df.columns and df["filed"].notna().any():
        known = df[df["filed"] <= asof_date]
        sort_cols = ["end", "filed"]
    else:
        logger.warning("sin columna 'filed' — usando 'end' (riesgo de look-ahead bias)")
        known = df[df["end"] <= asof_date]
        sort_cols = ["end"]
    if known.empty:
        return None
    return known.sort_values(sort_cols).iloc[-1].to_dict()


# ---------------------------------------------------------------------------
# 5. Panel cross-sectional (frames) con fallback multi-tag y provenance.
#    frames NO trae `filed` -> usar para universo/breadth, luego point-in-time-ficar
#    con filed_date_for_accn() antes de usar en backtest.
# ---------------------------------------------------------------------------

def resolve_concept_panel(concept: str, year: int, quarter: int = 4) -> pd.DataFrame:
    spec = CONCEPT_MAP[concept]
    period = f"CY{year}Q{quarter}I" if spec.instant else f"CY{year}"

    frames: dict[str, dict] = {}
    for tag in spec.tags:
        try:
            frames[tag] = fetch_frame(tag, period)
        except requests.HTTPError:
            logger.info("tag %s sin frame para %s (404 esperado si no aplica)", tag, period)

    rows: dict[int, dict] = {}
    for tag in spec.tags:                       # prioridad descendente, no sobreescribe cik ya cubierto
        for e in frames.get(tag, {}).get("data", []):
            cik = e["cik"]
            if cik in rows:
                continue
            rows[cik] = {
                "cik": cik, "entityName": e["entityName"], "val": e["val"],
                "end": e["end"], "accn": e["accn"], "source_tag": tag,
            }
    return pd.DataFrame(rows.values())


def filed_date_for_accn(cik: int, accn: str) -> Optional[str]:
    """Cruza un accession number de un frame con companyfacts para recuperar `filed`
    y convertir un panel de frames en look-ahead-safe."""
    facts = fetch_companyfacts(cik).get("facts", {}).get("us-gaap", {})
    for tag_data in facts.values():
        for e in tag_data.get("units", {}).get("USD", []):
            if e.get("accn") == accn:
                return e.get("filed")
    return None


if __name__ == "__main__":
    import sys
    cik = int(sys.argv[1]) if len(sys.argv) > 1 else 320193  # Apple, demo
    ni = resolve_concept_timeseries(cik, "net_income")
    print(ni.tail())
    print("valor conocido al 2024-01-01:", as_of(ni, "2024-01-01"))
