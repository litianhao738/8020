from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class Strategy(Protocol):
    name: str

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Return target position for each bar: -1 short, 0 flat, 1 long."""


@dataclass(frozen=True)
class BuyAndHold:
    name: str = "buy_and_hold"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=data.index)


@dataclass(frozen=True)
class MovingAverageCrossover:
    short_window: int = 20
    long_window: int = 80
    name: str = "moving_average_crossover"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")

        short_ma = data["close"].rolling(self.short_window).mean()
        long_ma = data["close"].rolling(self.long_window).mean()

        signal = pd.Series(0, index=data.index, dtype=float)
        signal[short_ma > long_ma] = 1.0
        signal[short_ma < long_ma] = -1.0
        return signal.fillna(0.0)


@dataclass(frozen=True)
class LiquidityRegimeAwareSMA:
    short_window: int
    long_window: int
    vol_window: int
    trend_threshold: float
    vol_low: float
    vol_high: float
    tradable_sessions: tuple[str, ...]
    name: str = "liquidity_regime_aware_sma"

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window.")
        if "log_return" not in data.columns:
            raise ValueError("LiquidityRegimeAwareSMA requires a log_return column.")

        short_ma = data["close"].rolling(self.short_window).mean()
        long_ma = data["close"].rolling(self.long_window).mean()
        trend_strength = (short_ma - long_ma).abs() / data["close"]
        realized_vol = data["log_return"].rolling(self.vol_window).std()

        raw_signal = pd.Series(0, index=data.index, dtype=float)
        raw_signal[short_ma > long_ma] = 1.0
        raw_signal[short_ma < long_ma] = -1.0

        regime_ok = (
            (trend_strength > self.trend_threshold)
            & (realized_vol > self.vol_low)
            & (realized_vol < self.vol_high)
        )
        if "session" in data.columns:
            liquidity_ok = data["session"].isin(self.tradable_sessions)
        else:
            liquidity_ok = pd.Series(True, index=data.index)

        final_signal = raw_signal.where(regime_ok & liquidity_ok, 0.0)
        return final_signal.fillna(0.0)
