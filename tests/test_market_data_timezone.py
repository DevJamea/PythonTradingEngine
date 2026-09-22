"""Regression tests for ``drop_unclosed_candle`` timezone handling.

Bug reproduced on Windows + real MT5 (``python -m gold_trader.main --once``):

    ValueError: Cannot pass a datetime or Timestamp with tzinfo with the tz
    parameter. Use tz_convert instead.

caused by ``pd.Timestamp(now or datetime.now(timezone.utc), tz="UTC")``
(and by ``pd.Timestamp(df["time"].iloc[-1], tz="UTC")`` on the tz-aware
``time`` column produced by :func:`get_candles`).

These tests pin the fixed contract:
  * ``now=None`` (fallback = aware UTC now)
  * naive ``now``  -> interpreted as UTC
  * aware UTC ``now`` (what ``utcnow()`` returns in main.py)
  * aware ``now`` in a non-UTC zone -> same instant, same decision
  * tz-aware and naive ``time`` columns
  * the closed/unclosed decision itself is unchanged
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from gold_trader.mt5.market_data import drop_unclosed_candle

# Last candle of the fixtures below opens at 2026-01-05 00:30 UTC and is
# therefore closed from 2026-01-05 00:45 UTC onwards.
LAST_CANDLE_OPEN = datetime(2026, 1, 5, 0, 30, tzinfo=timezone.utc)
CLOSE_AT = datetime(2026, 1, 5, 0, 45, tzinfo=timezone.utc)


def make_df(*, tz: str | None = "UTC") -> pd.DataFrame:
    """Three M15 candles: 00:00, 00:15 and the still-forming 00:30 one."""
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-05 00:00:00", periods=3, freq="15min", tz=tz),
            "open": [2000.0, 2001.0, 2002.0],
            "high": [2002.0, 2003.0, 2004.0],
            "low": [1999.0, 2000.0, 2001.0],
            "close": [2001.0, 2002.0, 2003.0],
            "tick_volume": [100, 110, 120],
            "spread": [10, 10, 10],
            "real_volume": [0, 0, 0],
        }
    )


def make_empty_df() -> pd.DataFrame:
    return make_df().iloc[0:0]


# ---------------------------------------------------------------------------
# now = None
# ---------------------------------------------------------------------------


def test_now_none_far_past_candles_are_all_closed(monkeypatch):
    """now=None must not raise and must treat long-past candles as closed."""
    df = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-05 00:00:00", periods=3, freq="15min", tz="UTC"),
            "open": [2000.0] * 3,
            "high": [2001.0] * 3,
            "low": [1999.0] * 3,
            "close": [2000.5] * 3,
            "tick_volume": [1] * 3,
            "spread": [10] * 3,
            "real_volume": [0] * 3,
        }
    )
    out = drop_unclosed_candle(df, "M15", None)
    assert len(out) == 3
    assert out is df  # nothing dropped -> the frame is returned untouched


def test_now_none_future_last_candle_is_dropped():
    """now=None vs. a candle dated far in the future -> it is still forming."""
    df = make_df().copy()
    df.loc[df.index[-1], "time"] = pd.Timestamp("2999-01-05 00:30:00", tz="UTC")
    out = drop_unclosed_candle(df, "M15", None)
    assert len(out) == 2
    assert out["time"].iloc[-1] == pd.Timestamp("2026-01-05 00:15:00", tz="UTC")


# ---------------------------------------------------------------------------
# naive now -> interpreted as UTC
# ---------------------------------------------------------------------------


def test_naive_now_before_close_drops_last_candle():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", datetime(2026, 1, 5, 0, 44, 59))
    assert len(out) == 2
    assert out["time"].iloc[-1] == pd.Timestamp(LAST_CANDLE_OPEN - timedelta(minutes=15))


def test_naive_now_exactly_at_close_keeps_last_candle():
    """Candle closes at open + timeframe; the boundary is inclusive."""
    df = make_df()
    naive_close = CLOSE_AT.replace(tzinfo=None)
    out = drop_unclosed_candle(df, "M15", naive_close)
    assert len(out) == 3
    assert out is df


def test_naive_now_is_interpreted_as_utc_not_local_time():
    """A naive value means UTC, so +3h of wall clock must not be assumed."""
    df = make_df()
    # 03:40 local would be 00:40 UTC -> still forming; as UTC it is closed.
    out = drop_unclosed_candle(df, "M15", datetime(2026, 1, 5, 3, 40))
    assert len(out) == 3


# ---------------------------------------------------------------------------
# aware UTC now (the exact Windows/main.py scenario)
# ---------------------------------------------------------------------------


def test_aware_utc_now_before_close_drops_last_candle():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))
    assert len(out) == 2


def test_aware_utc_now_exactly_at_close_keeps_last_candle():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", CLOSE_AT)
    assert len(out) == 3


def test_regression_aware_now_does_not_raise_value_error():
    """The original crash: aware datetime passed together with ``tz="UTC"``.

    Reproduces the Windows traceback by calling the function exactly the way
    ``main.py`` does (``utcnow()``) against a ``get_candles``-style tz-aware
    frame. This test fails with
    ``ValueError: Cannot pass a datetime or Timestamp with tzinfo with the
    tz parameter`` on the pre-fix code.
    """
    from gold_trader.utils.time_utils import utcnow

    df = make_df()
    now = utcnow()  # tz-aware UTC, as produced on Windows during --once
    assert now.tzinfo is not None

    out = drop_unclosed_candle(df, "M15", now)  # must not raise

    assert len(out) == 3  # candles from 2026 are closed "now"
    assert isinstance(out["time"].dtype, pd.DatetimeTZDtype)


def test_regression_old_pandas_pattern_would_still_raise():
    """Guard: pandas still rejects tz-aware input + ``tz=`` (root cause)."""
    with pytest.raises(ValueError, match="tzinfo"):
        pd.Timestamp(datetime.now(timezone.utc), tz="UTC")


# ---------------------------------------------------------------------------
# aware now in a non-UTC zone -> converted with tz_convert
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tz",
    [
        timezone(timedelta(hours=3)),
        timezone(timedelta(hours=-5)),
        timezone(timedelta(hours=5, minutes=30)),
        ZoneInfo("Asia/Riyadh"),
        ZoneInfo("America/New_York"),
    ],
)
def test_aware_other_timezone_now_same_decision_as_utc(tz):
    df = make_df()
    before_close = (CLOSE_AT - timedelta(seconds=1)).astimezone(tz)
    at_close = CLOSE_AT.astimezone(tz)

    assert len(drop_unclosed_candle(df, "M15", before_close)) == 2
    assert len(drop_unclosed_candle(df, "M15", at_close)) == 3


def test_other_timezone_now_converted_not_relabelled():
    """The instant is preserved: +03 means 00:45 UTC, i.e. the candle closed."""
    df = make_df()
    riyadh_now = CLOSE_AT.astimezone(ZoneInfo("Asia/Riyadh"))
    assert riyadh_now.hour == 3  # sanity: same instant, different wall clock
    assert len(drop_unclosed_candle(df, "M15", riyadh_now)) == 3


# ---------------------------------------------------------------------------
# `time` column variants (tz-aware / naive / pandas Timestamp `now`)
# ---------------------------------------------------------------------------


def test_tz_aware_time_column_is_supported():
    """``get_candles`` returns tz-aware UTC timestamps (second crash site)."""
    df = make_df(tz="UTC")
    assert isinstance(df["time"].dtype, pd.DatetimeTZDtype)
    assert len(drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))) == 2


def test_non_utc_aware_time_column_is_supported():
    """A non-UTC aware column is compared by instant, not by wall clock."""
    df = make_df(tz="Asia/Riyadh")  # last candle opens 00:30+03:00
    riyadh_close = pd.Timestamp("2026-01-05 00:45:00", tz="Asia/Riyadh")
    assert len(drop_unclosed_candle(df, "M15", riyadh_close - timedelta(seconds=1))) == 2
    assert drop_unclosed_candle(df, "M15", riyadh_close) is df


def test_naive_time_column_is_treated_as_utc():
    df = make_df(tz=None)
    assert drop_unclosed_candle(df, "M15", CLOSE_AT) is df
    assert len(drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))) == 2


def test_pandas_timestamp_now_aware_and_naive():
    df = make_df()
    assert len(drop_unclosed_candle(df, "M15", pd.Timestamp(CLOSE_AT))) == 3
    assert len(
        drop_unclosed_candle(
            df, "M15", pd.Timestamp(CLOSE_AT.astimezone(ZoneInfo("Asia/Riyadh")))
        )
    ) == 3
    assert len(drop_unclosed_candle(df, "M15", pd.Timestamp(CLOSE_AT.replace(tzinfo=None)))) == 3


# ---------------------------------------------------------------------------
# end-to-end: the exact ``python -m gold_trader.main --once`` data path
# ---------------------------------------------------------------------------


def _fake_mt5_rates():
    """A ``copy_rates_from_pos`` payload shaped like the real MT5 return."""
    import numpy as np

    dtype = [
        ("time", "<i8"),
        ("open", "<f8"),
        ("high", "<f8"),
        ("low", "<f8"),
        ("close", "<f8"),
        ("tick_volume", "<u8"),
        ("spread", "<i4"),
        ("real_volume", "<u8"),
    ]
    base = int(pd.Timestamp("2026-01-05 00:00:00", tz="UTC").timestamp())
    rows = []
    for i in range(3):
        price = 2000.0 + i
        rows.append((base + i * 900, price, price + 2, price - 1, price + 1, 100, 10, 0))
    return np.array(rows, dtype=dtype)


def test_regression_windows_main_once_data_path(monkeypatch):
    """``get_candles`` -> ``drop_unclosed_candle(df, tf, utcnow())`` as in main.py.

    This mirrors ``TradingBot._load_data`` (main.py lines ~194/640) with a
    mocked terminal: the MT5 API returns tz-naive epoch seconds, ``get_candles``
    makes the ``time`` column tz-aware UTC, and ``utcnow()`` is passed as
    ``now``. Pre-fix this raised the Windows ValueError both on ``now_ts``
    (aware ``now``) and on ``last_time`` (aware column).
    """
    from unittest.mock import MagicMock

    import gold_trader.mt5.connection as mt5_conn
    from gold_trader.mt5.market_data import get_candles
    from gold_trader.utils.time_utils import utcnow

    fake = MagicMock()
    fake.copy_rates_from_pos.return_value = _fake_mt5_rates()
    monkeypatch.setattr(mt5_conn, "MT5_AVAILABLE", True)
    monkeypatch.setattr(mt5_conn, "_mt5", fake)

    df = get_candles("XAUUSD", "M15", 3)
    assert df["time"].dt.tz is not None  # tz-aware UTC, as on Windows

    now = utcnow()
    out = drop_unclosed_candle(df, "M15", now)  # must not raise

    assert len(out) == 3  # 2026 candles are closed relative to "now"
    assert isinstance(out["time"].dtype, pd.DatetimeTZDtype)


def test_regression_aware_utc_now_drops_only_the_open_candle():
    """A still-forming last candle is removed with an aware UTC ``now``."""
    from datetime import timezone as _tz

    df = make_df()
    now = CLOSE_AT.astimezone(_tz.utc) - timedelta(seconds=1)
    out = drop_unclosed_candle(df, "M15", now)
    assert len(out) == 2
    assert out["time"].iloc[-1] == pd.Timestamp("2026-01-05 00:15:00", tz="UTC")


# ---------------------------------------------------------------------------
# behaviour that must stay unchanged
# ---------------------------------------------------------------------------


def test_empty_dataframe_returned_as_is():
    df = make_empty_df()
    assert drop_unclosed_candle(df, "M15", None) is df
    assert drop_unclosed_candle(df, "M15", CLOSE_AT) is df


def test_only_the_last_row_is_ever_removed():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))
    assert list(out.index) == [0, 1]
    assert out["time"].tolist() == df["time"].tolist()[:2]
    assert out["close"].tolist() == [2001.0, 2002.0]


def test_returned_frame_is_a_copy_when_a_row_is_dropped():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))
    assert out is not df
    out.iloc[0, out.columns.get_loc("close")] = -1.0
    assert df["close"].iloc[0] == 2001.0


def test_columns_and_dtypes_are_preserved():
    df = make_df()
    out = drop_unclosed_candle(df, "M15", CLOSE_AT - timedelta(seconds=1))
    assert list(out.columns) == list(df.columns)
    assert out["time"].dtype == df["time"].dtype
    assert out["tick_volume"].dtype == df["tick_volume"].dtype


@pytest.mark.parametrize("timeframe,seconds", [("M1", 60), ("M5", 300), ("M15", 900), ("H1", 3600)])
def test_timeframe_length_drives_the_close_boundary(timeframe, seconds):
    df = make_df()
    assert len(drop_unclosed_candle(df, timeframe, LAST_CANDLE_OPEN)) == 2
    assert len(
        drop_unclosed_candle(df, timeframe, LAST_CANDLE_OPEN + timedelta(seconds=seconds))
    ) == 3


def test_unsupported_timeframe_still_raises_value_error():
    with pytest.raises(ValueError):
        drop_unclosed_candle(make_df(), "M7", CLOSE_AT)
