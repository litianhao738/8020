from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .strategy import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 1_000_000.0
    contract_multiplier: float = 50.0
    margin_rate: float = 0.1
    commission_per_contract: float = 10.0
    slippage_points: float = 1.0
    max_position: int = 1
    flatten_at_day_end: bool = True


def _flatten_at_session_end(position: pd.Series, data: pd.DataFrame) -> pd.Series:
    if "session_id" in data.columns:
        session_id = data["session_id"].astype(str)
        day_end = session_id != session_id.shift(-1)
    else:
        dates = position.index.date
        next_dates = pd.Series(dates, index=position.index).shift(-1)
        day_end = pd.Series(dates, index=position.index) != next_dates

    adjusted = position.copy()
    adjusted[day_end] = 0.0
    return adjusted


def run_backtest(
    data: pd.DataFrame,
    strategy: Strategy,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Run a single-strategy vectorized backtest."""
    config = config or BacktestConfig()

    signals = strategy.generate_signals(data).clip(-config.max_position, config.max_position)
    target_position = signals.round()
    if config.flatten_at_day_end:
        target_position = _flatten_at_session_end(target_position, data)

    executed_position = target_position.shift(1).fillna(0.0)
    price_change = data["close"].diff().fillna(0.0)
    turnover = target_position.diff().abs().fillna(target_position.abs())

    gross_pnl = executed_position * price_change * config.contract_multiplier
    trading_cost = turnover * (
        config.commission_per_contract
        + config.slippage_points * config.contract_multiplier
    )
    net_pnl = gross_pnl - trading_cost
    equity = config.initial_capital + net_pnl.cumsum()

    result = data.copy()
    result["signal"] = signals
    result["target_position"] = target_position
    result["position"] = executed_position
    result["turnover"] = turnover
    result["gross_pnl"] = gross_pnl
    result["trading_cost"] = trading_cost
    result["net_pnl"] = net_pnl
    result["equity"] = equity
    result["returns"] = result["equity"].pct_change().fillna(0.0)
    result["contract_multiplier"] = config.contract_multiplier
    result["notional_exposure"] = (
        result["position"].abs() * result["close"] * config.contract_multiplier
    )
    result["margin_required"] = result["notional_exposure"] * config.margin_rate
    result["exposure_to_capital"] = result["notional_exposure"] / config.initial_capital
    return result
