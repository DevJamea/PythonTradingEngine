# Gold Trading Engine (MVP)

An **experimental** automatic trading system for gold (XAUUSD-style symbols)
built with **Python + MetaTrader 5**. It discovers the broker's gold symbol,
analyzes closed candles, generates BUY / SELL / NO-TRADE signals, sizes the
position from a risk percentage, manages stops (break-even, partial close,
optional trailing) and runs a basic backtest — all behind a hard
safety gate that keeps you in **Dry Run / Demo** until you explicitly change
the configuration.

> ## IMPORTANT
> **This software is experimental and does not guarantee profits.**
> The bundled strategy is a simple, understandable baseline — it is **not**
> claimed to be profitable. Trading foreign exchange and precious metals
> carries a high level of risk and may result in the loss of your capital.
> Only run it on a **Demo** account until you have fully understood and
> tested every component. Nothing in this repository is financial advice.

---

## 1. Requirements

- A **Windows** machine (the `MetaTrader5` Python package is Windows-only
  and automates a local MT5 terminal).
- The **MetaTrader 5** terminal installed and logged into a **Demo**
  account (any broker).
- **Python 3.10 – 3.12** (3.11 recommended).
- No paid services, no database, no LLM/AI — by design (MVP scope).

## 2. Python version

The project targets **Python 3.10+**. It was developed and tested on
Python 3.11. (Strategy, risk and backtest modules are platform-neutral
and run on any OS — only the live MT5 layer requires Windows.)

## 3. MT5 installation

1. Download MetaTrader 5 from your broker (or metaquotes.net).
2. Install the terminal on Windows.
3. Open the terminal and log into your **Demo** account.
4. Keep the terminal running — the bot connects to it (optionally via
   `MT5_TERMINAL_PATH` if it is not on the standard path).

## 4. Demo account

Create (or select) a **Demo** account in the terminal. The bot is designed
to be used on Demo accounts. Even if you flip the safety switches later,
you should keep the account on Demo until you have strong reasons — and
full understanding — to consider anything else.

## 5. Installation (project)

```bash
git clone <repo-url>
cd PythonTradingEngine
```

Project layout:

```
gold_trader/
├── main.py                  # entry point & main loop
├── config.py                # central configuration (env-overridable)
├── models.py                # shared domain types (no MT5 dependency)
├── mt5/                     # connection, data, orders, positions, symbols
├── strategy/                # candles, indicators, trend, signals, levels
├── risk/                    # position sizing + independent risk gate
├── trade_management/        # break-even, partial close, trailing, pending
├── backtest/                # engine + metrics
└── utils/                   # logging, validators, time helpers
tests/                       # deterministic unit tests (no MT5 needed)
```

## 6. Virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS (for testing only)
```

## 7. pip install

```bash
pip install -r requirements.txt
```

> On non-Windows machines `MetaTrader5` cannot be installed — install only
> `pandas`, `numpy`, `python-dotenv`, `pytest` and run the unit tests and
> (with your own OHLCV data) the backtest engine.

## 8. Configuration

Copy `.env.example` to `.env` and adjust:

```bash
copy .env.example .env          # Windows
```

Key settings (defaults in `gold_trader/config.py`, overridable via env):

| Variable | Default | Meaning |
|---|---|---|
| `TRADING_ENABLED` | `false` | Master kill-switch. No real order is ever sent while `false`. |
| `DRY_RUN` | `true` | Simulated execution: prints the order it *would* send, sends nothing. |
| `SYMBOL` | *(empty)* | Leave empty for **automatic gold symbol discovery**. |
| `GOLD_SYMBOL_CANDIDATES` | `XAUUSD,GOLD,XAUUSDm,...` | Priority list for discovery. |
| `TIMEFRAME` | `M15` | M1, M5, M15, M30, H1, H4. |
| `MAGIC_NUMBER` | `123456789` | Identifies the bot's orders/positions. **Change it per broker account.** |
| `RISK_PER_TRADE` | `0.005` | Risk 0.5% of equity per trade. |
| `MAX_DAILY_LOSS` | `500` | Stop opening trades when daily loss (realized + floating) exceeds this. |
| `MAX_OPEN_POSITIONS` | `1` | Bot positions limit. |
| `MAX_PENDING_ORDERS` | `2` | Bot pending orders limit. |
| `MAX_SPREAD` | `0.50` | Skip entries when the spread (price units) is wider. |
| `TRADING_START_TIME` / `TRADING_END_TIME` | `00:00` / `23:59` | UTC session window (supports overnight windows). |
| `ATR_SL_MULTIPLIER` / `ATR_TP_MULTIPLIER` | `1.5` / `2.5` | SL/TP distance in ATR units. |
| `BREAK_EVEN_R` / `BREAK_EVEN_BUFFER` | `1.0` / `0.10` | Move SL to entry ± buffer after +1R. |
| `PARTIAL_CLOSE_LEVELS` | `1.0:0.5,2.0:0.3,3.0:1.0` | Close 50% at +1R, 30% more at +2R, remainder at +3R. |
| `TRAILING_STOP_ENABLED` | `false` | ATR trailing stop, **off by default** (untested feature stays off). |
| `EXPECTED_ACCOUNT_MODE` | *(empty)* | Optional: abort if the account is not `HEDGING`/`NETTING` as expected. |
| `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` | *(empty)* | Optional — otherwise the already-logged-in terminal is used. **Never commit `.env`.** |

There are deliberately **no passwords or account numbers in the code**;
credentials (if used at all) come from the environment and are never logged.

## 9. Symbol discovery

The bot does **not** assume the symbol is `XAUUSD`. `find_gold_symbol()`:

1. tries your configured `SYMBOL` (if any),
2. then the `GOLD_SYMBOL_CANDIDATES` list in order,
3. then a deterministic scan of all broker symbols containing `XAU`/`GOLD`.

A candidate is only accepted when it is **visible** and in **FULL trade
mode** with sane volume constraints. The selected symbol and its properties
(point, digits, volume min/max/step, stops level, freeze level, contract
size, tick value/size) are printed and logged:

```
Selected gold symbol: XAUUSD | point=0.01 digits=2 | volume=0.01-100.0 step=0.01 ...
```

If nothing suitable is found the bot stops with `GoldSymbolNotFoundError`
listing what it tried — it never guesses.

## 10. Dry Run

With `DRY_RUN=true` (the default) the full pipeline runs — data, signals,
SL/TP, sizing, every risk gate — but **no `order_send` is performed**.
Instead you get:

```
SIGNAL: BUY
SYMBOL: XAUUSD
ENTRY: 2385.40000
SL: 2380.10000
TP: 2395.90000
VOLUME: 0.25
RISK: 50.00
WOULD EXECUTE: BUY
```

Run one cycle on your Demo terminal:

```bash
python -m gold_trader.main --once
```

## 11. Backtesting

Backtest the strategy on real broker history from the MT5 terminal:

```bash
python -m gold_trader.main --backtest
```

The engine (1) uses closed candles only, (2) evaluates the signal at closed
candle *i* using data `0..i` only, (3) enters at the **open of candle i+1**,
(4) simulates SL/TP from the entry candle onward (if one candle touches both,
SL wins — conservative), (5) applies a configurable spread cost and broker
position sizing. It reports total trades, wins/losses, win rate, gross
profit/loss, net profit, max drawdown, profit factor and average win/loss.
No future data can leak into the strategy (a unit test asserts the
precomputation is identical to window-limited computation).

## 12. Demo trading (graduated switch-over)

The safe path, in order:

1. `python -m gold_trader.main --check` — verify terminal, account mode, symbol.
2. `python -m gold_trader.main --backtest` — look at historical behaviour.
3. `DRY_RUN=true` (default) — watch what the bot *would* do for real.
4. `TRADING_ENABLED=true`, `DRY_RUN=false` — real orders, **on the Demo
   account**.
5. Keep Demo until the system behaves exactly as you expect over time.

The main loop sleeps between cycles (default 45 s) and evaluates a new entry
signal **only when a new closed candle appears** — no busy loop, no
re-triggering on the same candle.

## 13. Risk management

- **Sizing**: volume = (equity × risk%) / (stop distance × loss per lot),
  using the broker's exact tick value/size (contract size fallback),
  rounded **down** to the volume step, clamped to min/max. If no safe size
  exists → **NO TRADE** (never a magic fixed lot).
- **Risk gate** (`risk/risk_manager.py`, independent of the strategy) blocks
  real orders when: trading disabled, dry run, no connection, invalid symbol,
  market closed, server trading off, spread too wide, position/pending
  limits reached, daily loss limit hit, outside trading hours, sizing
  failed, SL/TP invalid or closer than the broker `stops_level`, or a
  duplicate entry (same-direction position / same-side order at the same
  price).
- **Break-even**: at +1R the SL moves to entry ± buffer — idempotent, and
  state survives restarts (encoded in the position comment).
- **Partial close**: R-multiple levels from config (50% / 30% / remainder);
  broker `volume_min`/`volume_step` are enforced — invalid orders are logged
  and skipped, never sent.
- **Trailing stop**: ATR-based, disabled by default.
- **Duplicate protection**: new entries only on a new closed candle, plus
  position/pending checks, all filtered by the magic number.
- **Account mode**: NETTING vs HEDGING is detected and logged; set
  `EXPECTED_ACCOUNT_MODE` to abort on a mismatch. Manual (non-bot) trades
  are never read for decisions or modified.

## 14. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `MetaTrader5 package is not available` | You are not on Windows, or the package is missing. Live trading requires Windows + `pip install MetaTrader5`. Tests still work anywhere. |
| `mt5.initialize() failed` | Start the MT5 terminal and log into the Demo account; or set `MT5_TERMINAL_PATH`. |
| `GoldSymbolNotFoundError` | Your broker's gold symbol is unusual. Add its exact name to `GOLD_SYMBOL_CANDIDATES` in `.env`. |
| `no valid tick` | Symbol not selected / market closed. The bot selects the discovered symbol; on weekends ticks are unavailable. |
| `TRADE BLOCKED (spread ...)` | Spread above `MAX_SPREAD` — normal protection; raise it only if your broker quotes gold differently. |
| `required volume ... below broker minimum` | Stop distance too small for the chosen risk — the bot correctly refuses to trade. Widen the ATR SL multiplier or lower risk. |
| Order rejected by the server | The error log contains `retcode`, `comment` and the full request — common causes: wrong filling mode (auto-selected from symbol properties), stop distance below `stops_level`, insufficient margin, volume step mismatch. |
| Nothing logged | Check `logs/app.log`, `logs/trades.log`, `logs/errors.log` (created under `gold_trader/logs/`). |

---

## Development

```bash
# run the full deterministic test suite (no MT5/Windows needed)
python -m pytest tests/ -q
```

### Design notes

- `mt5/` is the only layer that imports `MetaTrader5`; the import is
  guarded so everything else (strategy, risk, backtest, management) is pure
  pandas/numpy and unit-testable on any platform.
- All decisions are pure functions over dataclasses — no hidden state.
  This keeps the architecture open for future modules (e.g. AI agents)
  to be added later **without rewriting the system**: a new signal
  source or decision layer only needs to produce the same
  `TradeSignal` / `TradePlan` contracts.
- No LLM/AI, no martingale, no grid, no fixed lots, no large risk —
  these are out of scope for this MVP by design.
