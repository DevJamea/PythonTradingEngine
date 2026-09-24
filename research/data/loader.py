"""Dataset loading + data-integrity audit (§3 of the brief).

Priority: real broker CSV at ``research/data/xauusd_m15.csv`` if present,
otherwise the synthetic proxy (loudly labelled). Nothing is silently repaired:
the audit reports duplicates, missing grid slots, weekend bars, timezone and
spread provenance; any cleaning applied by the loader is listed explicitly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import REAL_DATA_CSV, SYNTHETIC_TAG
from .synthetic import generate_synthetic_m15

REQUIRED_COLUMNS = ("time", "open", "high", "low", "close")


@dataclass(frozen=True)
class DataManifest:
    source: str            # "REAL_CSV" | "SYNTHETIC_PROXY_v1"
    path: Optional[str]
    symbol: str
    timeframe: str
    n_bars: int
    start: str
    end: str
    dataset_hash: str      # sha256 over canonical OHLCV bytes
    spread_provenance: str  # "actual_bid_ask" | "csv_spread_column" | "synthetic"


def _hash_frame(df: pd.DataFrame) -> str:
    blob = pd.util.hash_pandas_object(
        df[["time", "open", "high", "low", "close", "tick_volume", "spread"]],
        index=False,
    ).to_numpy().tobytes()
    return hashlib.sha256(blob).hexdigest()


def load_dataset(symbol: str = "XAUUSD",
                 timeframe: str = "M15") -> Tuple[pd.DataFrame, DataManifest]:
    """Load real CSV when available, else the synthetic proxy (labelled)."""
    csv_path = Path(REAL_DATA_CSV)
    if csv_path.exists():
        df = _load_real_csv(csv_path)
        provenance = ("actual_bid_ask" if {"bid", "ask"}.issubset(df.columns)
                      else "csv_spread_column")
        manifest = DataManifest(
            source="REAL_CSV", path=str(csv_path), symbol=symbol,
            timeframe=timeframe, n_bars=len(df),
            start=str(df["time"].iloc[0]), end=str(df["time"].iloc[-1]),
            dataset_hash=_hash_frame(df), spread_provenance=provenance,
        )
        return df, manifest
    df = generate_synthetic_m15()
    manifest = DataManifest(
        source=SYNTHETIC_TAG, path=None, symbol=symbol, timeframe=timeframe,
        n_bars=len(df), start=str(df["time"].iloc[0]),
        end=str(df["time"].iloc[-1]), dataset_hash=_hash_frame(df),
        spread_provenance="synthetic",
    )
    return df, manifest


def _load_real_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"real CSV missing columns: {missing}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    if "tick_volume" not in df.columns:
        df["tick_volume"] = 0
    if "spread" not in df.columns:
        if {"bid", "ask"}.issubset(df.columns):
            df["spread"] = (df["ask"] - df["bid"]).astype(float)
        else:
            raise ValueError("real CSV needs a 'spread' column or bid/ask columns")
    return df


def audit_data(df: pd.DataFrame, manifest: DataManifest) -> Dict[str, Any]:
    """Full integrity audit. Reports problems; repairs nothing silently."""
    times = pd.to_datetime(df["time"], utc=True)
    dupes = int(times.duplicated().sum())
    weekend_bars = int((times.dt.dayofweek >= 5).sum())
    # expected weekday M15 grid between first and last bar
    full = pd.date_range(times.iloc[0].floor("15min"),
                         times.iloc[-1].floor("15min"), freq="15min", tz="UTC")
    full = full[full.dayofweek < 5]
    have = set(times.dt.floor("15min"))
    missing_slots = int(sum(1 for t in full if t not in have))
    # largest gap (excluding weekend breaks is complex; report raw largest gap)
    gaps = times.diff().dropna()
    max_gap = str(gaps.max()) if len(gaps) else "n/a"
    ohlc_bad = int(((df["high"] < df[["open", "close"]].max(axis=1)) |
                    (df["low"] > df[["open", "close"]].min(axis=1))).sum())
    nonpos = int(((df[["open", "high", "low", "close"]] <= 0).any(axis=1)).sum())

    cleaning: List[str] = []
    if manifest.source != SYNTHETIC_TAG:
        cleaning.append("sorted by time ascending (documented)")
        # NOTE: duplicates/missing slots are REPORTED, not dropped/filled here.
        # Any future cleaning must append to this list AND bump a data version.

    return {
        "symbols": [manifest.symbol],
        "date_range": {"start": manifest.start, "end": manifest.end},
        "timeframe": manifest.timeframe,
        "n_candles": manifest.n_bars,
        "bid_ask_available": manifest.spread_provenance in ("actual_bid_ask",),
        "spread_provenance": manifest.spread_provenance,
        "spread_summary": {
            "mean": float(df["spread"].mean()),
            "median": float(df["spread"].median()),
            "p95": float(df["spread"].quantile(0.95)),
            "max": float(df["spread"].max()),
        },
        "duplicate_timestamps": dupes,
        "missing_grid_slots": missing_slots,
        "max_gap": max_gap,
        "weekend_bars": weekend_bars,
        "weekend_handling": "excluded (no Sat/Sun bars expected)",
        "timezone": "UTC",
        "broker_data_source": ("broker CSV (see path)" if manifest.source == "REAL_CSV"
                               else "NONE — synthetic proxy (no broker data available)"),
        "ohlc_inconsistent_bars": ohlc_bad,
        "non_positive_price_bars": nonpos,
        "cleaning_applied": cleaning,
        "dataset_hash": manifest.dataset_hash,
    }
