from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STAT8020 intraday futures backtest")
    parser.add_argument("--data", default="dataset/raw/hi1_20170701_20200609.csv")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--in-sample-end", default="2019-06-30")
    parser.add_argument("--out-sample-start", default="2019-07-01")
    parser.add_argument("--frequency", default="5min")
    parser.add_argument("--short-window", type=int, default=20)
    parser.add_argument("--long-window", type=int, default=60)
    parser.add_argument("--regime-vol-window", type=int, default=48)
    parser.add_argument("--trend-quantile", type=float, default=0.60)
    parser.add_argument("--vol-low-quantile", type=float, default=0.30)
    parser.add_argument("--vol-high-quantile", type=float, default=0.90)
    parser.add_argument("--contract-multiplier", type=float, default=50.0)
    parser.add_argument("--margin-rate", type=float, default=0.1)
    parser.add_argument("--commission", type=float, default=0.0)
    parser.add_argument("--slippage", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        raw_path=Path(args.data),
        output_dir=Path(args.output_dir),
        in_sample_end=args.in_sample_end,
        out_sample_start=args.out_sample_start,
        bar_frequency=args.frequency,
        sma_short_window=args.short_window,
        sma_long_window=args.long_window,
        regime_vol_window=args.regime_vol_window,
        trend_quantile=args.trend_quantile,
        vol_low_quantile=args.vol_low_quantile,
        vol_high_quantile=args.vol_high_quantile,
        contract_multiplier=args.contract_multiplier,
        margin_rate=args.margin_rate,
        commission_per_contract=args.commission,
        slippage_points=args.slippage,
    )
    outputs = run_pipeline(config)
    performance = outputs["performance"]
    print("Baseline performance:")
    print(performance.round(4).to_string())
    print("\nTop SMA parameter sets:")
    print(outputs["sma_optimization"].head(5).round(4).to_string(index=False))
    print("\nLiquidity-Regime SMA calibration:")
    print(outputs["adaptive_calibration"].to_string(index=False))
    print("\nIntraday session statistics:")
    print(outputs["intraday_stats"].round(6).to_string())
    print("\nLiquidity-Regime SMA slippage sensitivity:")
    print(outputs["slippage"].round(4).to_string(index=False))
    print(f"\nSaved cleaned datasets and baseline table to: {config.output_dir}")


if __name__ == "__main__":
    main()
