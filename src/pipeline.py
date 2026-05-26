from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .backtester import BacktestConfig, run_backtest
from .data_loader import (
    add_returns,
    clean_minute_data,
    load_intraday_data,
    resample_ohlcv,
    split_sample,
)
from .metrics import summarize_performance
from .reporting import save_report_plots
from .strategy import BuyAndHold, LiquidityRegimeAwareSMA, MovingAverageCrossover, Strategy


@dataclass(frozen=True)
class PipelineConfig:
    raw_path: Path = Path("dataset/raw/hi1_20170701_20200609.csv")
    output_dir: Path = Path("dataset")
    in_sample_end: str = "2019-06-30"
    out_sample_start: str = "2019-07-01"
    bar_frequency: str = "5min"
    sma_short_window: int = 20
    sma_long_window: int = 60
    regime_vol_window: int = 48
    trend_quantile: float = 0.60
    vol_low_quantile: float = 0.30
    vol_high_quantile: float = 0.90
    initial_capital: float = 1_000_000.0
    contract_multiplier: float = 50.0
    margin_rate: float = 0.1
    commission_per_contract: float = 0.0
    slippage_points: float = 0.0


def build_clean_datasets(config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = load_intraday_data(config.raw_path)
    minute = clean_minute_data(raw)
    bars = add_returns(resample_ohlcv(minute, config.bar_frequency))
    in_sample, out_sample = split_sample(
        bars,
        in_sample_end=config.in_sample_end,
        out_sample_start=config.out_sample_start,
    )
    return bars, in_sample, out_sample


DISPLAY_NAMES = {
    "total_return": "Total Return",
    "annualized_return": "Annualized Return",
    "annualized_volatility": "Annualized Volatility",
    "sharpe": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown",
    "calmar": "Calmar Ratio",
    "trades": "Trades",
    "win_rate": "Win Rate",
    "total_cost": "Total Cost",
    "final_equity": "Final Equity",
    "max_notional_exposure": "Max Notional Exposure",
    "max_margin_required": "Max Margin Required",
    "max_exposure_to_capital": "Max Exposure / Capital",
}


def _backtest_config(config: PipelineConfig, flatten_at_day_end: bool) -> BacktestConfig:
    return BacktestConfig(
        initial_capital=config.initial_capital,
        contract_multiplier=config.contract_multiplier,
        margin_rate=config.margin_rate,
        commission_per_contract=config.commission_per_contract,
        slippage_points=config.slippage_points,
        flatten_at_day_end=flatten_at_day_end,
    )


def _evaluate_strategy_table(
    samples: dict[str, pd.DataFrame],
    strategies: dict[str, tuple[Strategy, bool]],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    columns: dict[str, dict[str, float]] = {}
    oos_results: dict[str, pd.DataFrame] = {}
    for sample_name, sample in samples.items():
        for strategy_name, (strategy, flatten) in strategies.items():
            result = run_backtest(sample, strategy, config=_backtest_config(config, flatten))
            columns[f"{strategy_name} ({sample_name})"] = summarize_performance(result)
            if sample_name == "OOS":
                oos_results[strategy_name] = result

    table = pd.DataFrame(columns).rename(index=DISPLAY_NAMES)
    return table, oos_results


def evaluate_strategies(
    in_sample: pd.DataFrame,
    out_sample: pd.DataFrame,
    config: PipelineConfig,
    adaptive_strategy: LiquidityRegimeAwareSMA,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    strategies = {
        "Buy-and-Hold": (BuyAndHold(), False),
        "SMA Crossover": (
            MovingAverageCrossover(
                short_window=config.sma_short_window,
                long_window=config.sma_long_window,
            ),
            True,
        ),
        "Liquidity-Regime SMA": (adaptive_strategy, True),
    }
    return _evaluate_strategy_table({"IS": in_sample, "OOS": out_sample}, strategies, config)


def optimize_sma_parameters(in_sample: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rows = []
    for short_window in (8, 12, 20, 30, 48):
        for long_window in (40, 60, 96, 144, 240):
            if short_window >= long_window:
                continue
            strategy = MovingAverageCrossover(short_window=short_window, long_window=long_window)
            result = run_backtest(in_sample, strategy, config=_backtest_config(config, True))
            summary = summarize_performance(result)
            rows.append(
                {
                    "short_window": short_window,
                    "long_window": long_window,
                    "total_return": summary["total_return"],
                    "sharpe": summary["sharpe"],
                    "max_drawdown": summary["max_drawdown"],
                    "calmar": summary["calmar"],
                    "trades": summary["trades"],
                }
            )
    return pd.DataFrame(rows).sort_values(["sharpe", "calmar"], ascending=False)


def calibrate_liquidity_regime_sma(
    in_sample: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[LiquidityRegimeAwareSMA, pd.DataFrame]:
    """Calibrate all adaptive-strategy thresholds from in-sample data only."""
    short_ma = in_sample["close"].rolling(config.sma_short_window).mean()
    long_ma = in_sample["close"].rolling(config.sma_long_window).mean()
    trend_strength = ((short_ma - long_ma).abs() / in_sample["close"]).dropna()
    realized_vol = in_sample["log_return"].rolling(config.regime_vol_window).std().dropna()

    trend_threshold = float(trend_strength.quantile(config.trend_quantile))
    vol_low = float(realized_vol.quantile(config.vol_low_quantile))
    vol_high = float(realized_vol.quantile(config.vol_high_quantile))

    session_avg_volume = in_sample.groupby("session")["volume"].mean()
    liquidity_threshold = float(session_avg_volume.median())
    tradable_sessions = tuple(
        session_avg_volume[session_avg_volume >= liquidity_threshold].index.astype(str)
    )

    strategy = LiquidityRegimeAwareSMA(
        short_window=config.sma_short_window,
        long_window=config.sma_long_window,
        vol_window=config.regime_vol_window,
        trend_threshold=trend_threshold,
        vol_low=vol_low,
        vol_high=vol_high,
        tradable_sessions=tradable_sessions,
    )
    calibration = pd.DataFrame(
        [
            {"parameter": "short_window", "value": config.sma_short_window},
            {"parameter": "long_window", "value": config.sma_long_window},
            {"parameter": "vol_window", "value": config.regime_vol_window},
            {"parameter": "trend_quantile", "value": config.trend_quantile},
            {"parameter": "trend_threshold", "value": trend_threshold},
            {"parameter": "vol_low_quantile", "value": config.vol_low_quantile},
            {"parameter": "vol_low", "value": vol_low},
            {"parameter": "vol_high_quantile", "value": config.vol_high_quantile},
            {"parameter": "vol_high", "value": vol_high},
            {"parameter": "liquidity_threshold", "value": liquidity_threshold},
            {"parameter": "tradable_sessions", "value": ",".join(tradable_sessions)},
            {"parameter": "morning_avg_volume", "value": session_avg_volume.get("morning", pd.NA)},
            {"parameter": "afternoon_avg_volume", "value": session_avg_volume.get("afternoon", pd.NA)},
            {"parameter": "night_avg_volume", "value": session_avg_volume.get("night", pd.NA)},
        ]
    )
    return strategy, calibration


def compute_intraday_stats(bars: pd.DataFrame) -> pd.DataFrame:
    """Summarize session-level volume and volatility from the current cleaned bars."""
    if "session" not in bars.columns or "session_id" not in bars.columns:
        raise ValueError("Intraday statistics require session and session_id columns.")

    stats: dict[str, float] = {}
    session_order = ("morning", "afternoon", "night")
    opening_bars = bars.groupby("session_id", sort=True).head(1)

    for session in session_order:
        session_bars = bars[bars["session"] == session]
        session_opening_bars = opening_bars[opening_bars["session"] == session]
        stats[f"{session}_open_vol"] = float(session_opening_bars["log_return"].abs().mean())
        stats[f"{session}_avg_volume"] = float(session_bars["volume"].mean())

    absolute_returns = bars["log_return"].abs()
    for percentile in (25, 50, 75, 90, 95):
        stats[f"vol_percentile_{percentile}"] = float(absolute_returns.quantile(percentile / 100))

    stats["total_5min_bars"] = float(len(bars))
    stats["unique_trading_days"] = float(pd.Series(bars["trading_date"]).nunique())
    return pd.DataFrame.from_dict(stats, orient="index", columns=["value"])


def slippage_sensitivity(
    out_sample: pd.DataFrame,
    config: PipelineConfig,
    strategy: Strategy,
    slippage_points: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 3.0),
) -> pd.DataFrame:
    rows = []
    for slippage in slippage_points:
        adjusted = PipelineConfig(
            raw_path=config.raw_path,
            output_dir=config.output_dir,
            in_sample_end=config.in_sample_end,
            out_sample_start=config.out_sample_start,
            bar_frequency=config.bar_frequency,
            sma_short_window=config.sma_short_window,
            sma_long_window=config.sma_long_window,
            regime_vol_window=config.regime_vol_window,
            trend_quantile=config.trend_quantile,
            vol_low_quantile=config.vol_low_quantile,
            vol_high_quantile=config.vol_high_quantile,
            initial_capital=config.initial_capital,
            contract_multiplier=config.contract_multiplier,
            margin_rate=config.margin_rate,
            commission_per_contract=config.commission_per_contract,
            slippage_points=slippage,
        )
        result = run_backtest(out_sample, strategy, config=_backtest_config(adjusted, True))
        summary = summarize_performance(result)
        rows.append(
            {
                "slippage_points": slippage,
                "total_return": summary["total_return"],
                "sharpe": summary["sharpe"],
                "max_drawdown": summary["max_drawdown"],
                "calmar": summary["calmar"],
                "total_cost": summary["total_cost"],
                "final_equity": summary["final_equity"],
                "max_notional_exposure": summary["max_notional_exposure"],
                "max_margin_required": summary["max_margin_required"],
            }
        )
    return pd.DataFrame(rows)


def save_outputs(
    bars: pd.DataFrame,
    in_sample: pd.DataFrame,
    out_sample: pd.DataFrame,
    performance: pd.DataFrame,
    sma_optimization: pd.DataFrame,
    adaptive_calibration: pd.DataFrame,
    intraday_stats: pd.DataFrame,
    slippage_table: pd.DataFrame,
    oos_results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bars.to_csv(output_dir / "hi1_5min_full.csv")
    in_sample.to_csv(output_dir / "hi1_5min_insample.csv")
    out_sample.to_csv(output_dir / "hi1_5min_outofsample.csv")
    performance.to_csv(output_dir / "table1_baseline_performance.csv")
    sma_optimization.to_csv(output_dir / "table2_sma_parameter_search.csv", index=False)
    adaptive_calibration.to_csv(output_dir / "table3_adaptive_sma_calibration.csv", index=False)
    intraday_stats.to_csv(output_dir / "intraday_stats_for_B.csv")
    slippage_table.to_csv(output_dir / "table4_slippage_sensitivity.csv", index=False)
    save_report_plots(bars, oos_results, sma_optimization, output_dir)


def run_pipeline(config: PipelineConfig) -> dict[str, pd.DataFrame]:
    bars, in_sample, out_sample = build_clean_datasets(config)
    intraday_stats = compute_intraday_stats(bars)
    sma_optimization = optimize_sma_parameters(in_sample, config)
    adaptive_strategy, adaptive_calibration = calibrate_liquidity_regime_sma(in_sample, config)
    performance, oos_results = evaluate_strategies(
        in_sample,
        out_sample,
        config,
        adaptive_strategy,
    )
    recommended_strategy: Strategy = adaptive_strategy
    slippage_table = slippage_sensitivity(out_sample, config, recommended_strategy)
    save_outputs(
        bars,
        in_sample,
        out_sample,
        performance,
        sma_optimization,
        adaptive_calibration,
        intraday_stats,
        slippage_table,
        oos_results,
        config.output_dir,
    )
    return {
        "performance": performance,
        "sma_optimization": sma_optimization,
        "adaptive_calibration": adaptive_calibration,
        "intraday_stats": intraday_stats,
        "slippage": slippage_table,
    }
