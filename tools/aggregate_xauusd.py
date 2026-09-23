"""Build XAUUSD M5/M15 backtest frames with a REAL per-bar spread column.

Input: the monthly M1 bid/ask CSVs cached by ``tools/fetch_xauusd_dukascopy.py``.
Output: ``xauusd_m5_<span>.csv.gz`` and ``xauusd_m15_<span>.csv.gz`` with
``time, open, high, low, close, ask_open, tick_volume, spread, spread_p90,
spread_max`` -- the exact schema ``gold_trader.backtest.data.load_candles_csv``
and ``CostModel.from_frame`` expect.

Methodology notes baked into the code (they matter for the study's honesty):

* bars are built from the **bid** side (what a retail terminal prints), and
  ``spread`` is ``ask_close - bid_close` averaged over the minutes of each bar;
* M5/M15 are always **aggregated from real M1**, never fetched pre-aggregated;
* ``ask_open`` is carried so the backtest can price a long entry at the actual
  ask instead of a guessed constant;
* minutes missing on one side are dropped by the inner merge, and bars with no
  ticks are removed.

    XAU_RAW_DIR   cached M1 directory   (default: ~/.cache/xauusd_raw)
    XAU_OUT_DIR   where to write the gz (default: current directory)
    XAU_SPAN      "<start> <end>" bar filter (default: 2021-01-01 2026-01-01)
    XAU_LABEL     file-name suffix        (default: 2021_2025)
"""
from __future__ import annotations

import glob
import os
import sys

import pandas as pd

RAW = os.environ.get("XAU_RAW_DIR", os.path.expanduser("~/.cache/xauusd_raw"))
OUT = os.environ.get("XAU_OUT_DIR", ".")
_span = os.environ.get("XAU_SPAN", "2021-01-01 2026-01-01").split()
START, END = _span[0], _span[1]
LABEL = os.environ.get("XAU_LABEL", "2021_2025")

RULES = {"5min": "m5", "15min": "m15", "60min": "h1"}


def months_on_disk() -> list[str]:
    return sorted({os.path.basename(p).split("_m1_")[1][:7]
                   for p in glob.glob(f"{RAW}/*.csv")})


def load_month(side: str, ym: str) -> pd.DataFrame:
    path = f"{RAW}/xauusd_{side}_m1_{ym}.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["time", "open", "high", "low", "close"]].sort_values("time")


def build() -> int:
    frames = []
    for ym in months_on_disk():
        bid, ask = load_month("bid", ym), load_month("ask", ym)
        if bid.empty or ask.empty:
            print(f"[warn] missing month {ym}: bid={len(bid)} ask={len(ask)}", flush=True)
            continue
        merged = bid.merge(ask, on="time", suffixes=("_bid", "_ask"))
        merged["spread"] = merged["close_ask"] - merged["close_bid"]
        frames.append(merged[["time", "open_bid", "high_bid", "low_bid", "close_bid",
                              "open_ask", "high_ask", "low_ask", "spread"]])
        print(f"[ok] {ym}: {len(merged)} m1 rows", flush=True)
    if not frames:
        print("no cached months found -- run tools/fetch_xauusd_dukascopy.py first",
              file=sys.stderr)
        return 1

    m1 = pd.concat(frames, ignore_index=True).sort_values("time")
    m1 = m1.drop_duplicates(subset="time", keep="first").reset_index(drop=True)
    spread = m1["spread"].dropna()
    print(f"[m1] rows={len(m1)} range={m1.time.min()}..{m1.time.max()}", flush=True)
    print("[m1 spread] mean=%.4f median=%.4f p90=%.4f p99=%.4f max=%.4f" % (
        spread.mean(), spread.median(), spread.quantile(0.9),
        spread.quantile(0.99), spread.max()), flush=True)

    for rule, name in RULES.items():
        grouped = m1.set_index("time").resample(rule)
        bars = pd.DataFrame({
            "open": grouped["open_bid"].first(),
            "high": grouped["high_bid"].max(),
            "low": grouped["low_bid"].min(),
            "close": grouped["close_bid"].last(),
            "ask_open": grouped["open_ask"].first(),
            "tick_volume": grouped["close_bid"].count(),
            "spread": grouped["spread"].mean(),
            "spread_p90": grouped["spread"].quantile(0.9),
            "spread_max": grouped["spread"].max(),
        }).dropna(subset=["open", "high", "low", "close"])
        bars = bars.reset_index().rename(columns={"index": "time"})
        keep = bars[(bars["time"] >= START) & (bars["time"] < END)].copy()
        keep = keep[keep["tick_volume"] > 0]
        if keep.empty:
            print(f"[{name}] no rows in {START}..{END} -- skipped", flush=True)
            continue
        os.makedirs(OUT, exist_ok=True)
        path = f"{OUT}/xauusd_{name}_{LABEL}.csv.gz"
        keep.to_csv(path, index=False, compression="gzip")
        print(f"[{name}] rows={len(keep)} -> {path} "
              f"({os.path.getsize(path) / 1e6:.2f} MB)", flush=True)
        print(f"[{name}] spread mean={keep['spread'].mean():.4f} "
              f"median={keep['spread'].median():.4f} "
              f"p90={keep['spread'].quantile(0.9):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
