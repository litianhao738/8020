# STAT8020 Intraday Trading Backtest Framework

This is a lightweight framework for the STAT8020 project requirement:
back-test intraday trading algorithms on Hang Seng Index futures minute data.

## Expected Data

Place the provided raw CSV file in `dataset/raw/`, for example:

```text
dataset/raw/hi1_20170701_20200609.csv
```

The loader accepts common column names for date, time, OHLC, volume, and price.
If your cleaned file uses different names, edit `src/data_loader.py`.

## Quick Start

```powershell
pip install -r requirements.txt
python -m src.main --data dataset/raw/hi1_20170701_20200609.csv
```

The script cleans raw 1-minute data, removes duplicated minutes, removes lunch
break and weekend rows, resamples to 5-minute OHLCV bars, calculates returns,
splits in-sample and out-of-sample data, evaluates Buy-and-Hold, SMA Crossover,
and Liquidity-Regime SMA performance, runs SMA parameter search, calibrates
adaptive thresholds from in-sample data only, tests slippage sensitivity, saves
plots, and saves the cleaned datasets.

The backtest uses one HI1 contract by default. PnL is calculated as
`position * price_change * contract_multiplier`, with `contract_multiplier=50`.
Initial capital is used to normalize the equity curve. The outputs also report
notional exposure and margin requirement using `margin_rate=0.1`.

## Project Layout

```text
src/
  data_loader.py   # Load and normalize minute futures data
  strategy.py      # Strategy interface and sample moving-average strategy
  backtester.py    # Position, PnL, transaction cost, and risk controls
  metrics.py       # Return, drawdown, Sharpe ratio, trade statistics
  optimizer.py     # Simple grid search for strategy parameters
  pipeline.py      # End-to-end raw-data cleaning and baseline evaluation
  reporting.py     # Report-ready charts
  main.py          # Command-line entry point
```

## Generated Outputs

```text
dataset/hi1_5min_full.csv
dataset/hi1_5min_insample.csv
dataset/hi1_5min_outofsample.csv
dataset/table1_baseline_performance.csv
dataset/table2_sma_parameter_search.csv
dataset/table3_adaptive_sma_calibration.csv
dataset/intraday_stats_for_B.csv
dataset/table4_slippage_sensitivity.csv     # Liquidity-Regime SMA under different slippage assumptions
dataset/figure1_price_volume.png
dataset/figure2_return_distribution.png
dataset/figure3_oos_equity_curves.png
dataset/figure4_oos_drawdown.png
dataset/figure5_sma_parameter_heatmap.png
```

## Next Steps

1. Confirm the cleaned outputs in `dataset/`.
2. Use `table2_sma_parameter_search.csv` and `figure5_sma_parameter_heatmap.png` for parameter discussion.
3. Report Liquidity-Regime SMA performance with and without slippage assumptions.
