"""Download real XAUUSD M1 bid+ask history and cache it month by month.

Why this script exists: the spread assumption is the whole argument behind the
scalping study, so the dataset must be a real two-sided feed, not a constant.
This pulls Dukascopy-sourced XAUUSD M1 BID and ASK bars (UTC, epoch-millisecond
timestamps) from the public ``kevingtlin/Market-Data-Lab`` repository through
the ``gh api`` contents endpoint (raw.githubusercontent.com is often
unreachable from sandboxes; the API is not).

    XAU_RAW_DIR   where to cache the monthly CSVs
                  (default: ~/.cache/xauusd_raw)
    XAU_MONTHS    "<first> <last>" inclusive, e.g. "2020_10 2025_12"
                  (default: 2020_10 2025_12 -- three warm-up months before 2021)

Already-downloaded months are skipped, so the run is resumable. Pair this with
``tools/aggregate_xauusd.py``, which builds the M5/M15 frames with a real
spread column that the backtest reads.
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import os
import subprocess
import sys

REPO = os.environ.get("XAU_REPO", "kevingtlin/Market-Data-Lab")
RAW = os.environ.get("XAU_RAW_DIR", os.path.expanduser("~/.cache/xauusd_raw"))

_DEFAULT_FIRST, _DEFAULT_LAST = "2020_10", "2025_12"


def month_range(first: str, last: str) -> list[str]:
    """Every ``YYYY_MM`` between ``first`` and ``last``, inclusive."""
    out: list[str] = []
    fy, ly = int(first[:4]), int(last[:4])
    fm, lm = int(first[5:7]), int(last[5:7])
    for year in range(fy, ly + 1):
        lo = fm if year == fy else 1
        hi = lm if year == ly else 12
        out += [f"{year}_{m:02d}" for m in range(lo, hi + 1)]
    return out


def fetch_job(job: tuple[str, str]) -> tuple[str, object]:
    side, ym = job
    out = f"{RAW}/xauusd_{side}_m1_{ym}.csv"
    if os.path.exists(out) and os.path.getsize(out) > 100_000:
        return out, "cached"
    path = f"repos/{REPO}/contents/xauusd/{side}/m1/xauusd_{side}_m1_{ym}.csv"
    with open(out, "wb") as fh:
        proc = subprocess.run(
            ["gh", "api", path, "-H", "Accept: application/vnd.github.raw"],
            stdout=fh, stderr=subprocess.PIPE, timeout=600,
        )
    if proc.returncode != 0:
        try:
            os.remove(out)
        except OSError:
            pass
        return out, proc.stderr.decode(errors="replace")[:200]
    return out, os.path.getsize(out)


def main() -> int:
    if shutil_which_gh() is None:
        print("gh (GitHub CLI) is required and was not found on PATH", file=sys.stderr)
        return 2
    months = os.environ.get("XAU_MONTHS", "").split() or [_DEFAULT_FIRST, _DEFAULT_LAST]
    first, last = months[0], months[1]
    os.makedirs(RAW, exist_ok=True)
    tasks = [(side, ym) for ym in month_range(first, last) for side in ("bid", "ask")]
    print(f"[fetch] {len(tasks)} monthly files from {REPO} -> {RAW}", flush=True)

    done = failures = 0
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        for out, res in pool.map(fetch_job, tasks):
            done += 1
            if done % 20 == 0:
                print(f"[fetch] {done}/{len(tasks)}", flush=True)
            if isinstance(res, str) and res != "cached":
                failures += 1
                print(f"[fetch] FAIL {out}: {res}", flush=True)
    found = len(glob.glob(f"{RAW}/*.csv"))
    print(f"[fetch] files on disk: {found} (failures: {failures})", flush=True)
    return 1 if failures else 0


def shutil_which_gh():
    import shutil

    return shutil.which("gh")


if __name__ == "__main__":
    raise SystemExit(main())
