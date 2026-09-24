"""Phase-2 XAUUSD A/B/C research experiment (RESEARCH ONLY).

This package is fully isolated from the production engine in ``gold_trader/``.
It MAY import production modules read-only (Config, models, signals, levels,
position sizing) so the Baseline stays faithful, but it MUST NEVER modify
production behaviour, enable live trading, or be imported by production code.

Nothing in here runs on Demo/REAL. Backtests only.
"""

__version__ = "2.0.0-phase2"
