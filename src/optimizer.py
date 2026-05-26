from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .backtester import BacktestConfig, run_backtest
from .metrics import summarize_performance
from .strategy import MovingAverageCrossover


@dataclass(frozen=True)
class OptimizationResult:
    short_window: int
    long_window: int
    sharpe: float
    total_return: float
    max_drawdown: float


def optimize_moving_average(
    data: pd.DataFrame,
    short_windows: tuple[int, ...] = (10, 20, 30, 40),
    long_windows: tuple[int, ...] = (60, 80, 120, 180),
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    rows: list[OptimizationResult] = []
    for short_window, long_window in product(short_windows, long_windows):
        if short_window >= long_window:
            continue
        strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
        result = run_backtest(data, strategy, config=config)
        summary = summarize_performance(result)
        rows.append(
            OptimizationResult(
                short_window=short_window,
                long_window=long_window,
                sharpe=summary["sharpe"],
                total_return=summary["total_return"],
                max_drawdown=summary["max_drawdown"],
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows]).sort_values(
        ["sharpe", "total_return"], ascending=False
    )
