# STAT8020 HI1 Futures Intraday Backtesting Project

This repository contains the code and data outputs for a STAT8020 intraday
trading project on Hang Seng Index futures (`HI1`). The project builds a
reproducible backtesting pipeline from raw 1-minute futures data to cleaned
5-minute bars, strategy evaluation, parameter analysis, slippage testing, and
report-ready figures.

The raw dataset is already included in this repository:

```text
dataset/raw/hi1_20170701_20200609.csv
```

## Project Objective

The project evaluates whether transparent intraday trading rules can outperform
a simple Buy-and-Hold benchmark on HI1 futures. The main strategy is a
5-minute SMA Crossover, and the innovation is a risk-managed
Liquidity-Regime SMA variant that only trades when both market-regime and
liquidity conditions are suitable.

The code is designed to support the report requirements:

- clean and transform minute-level HI1 futures data
- construct 5-minute OHLCV bars
- split the data into in-sample and out-of-sample periods
- backtest Buy-and-Hold, SMA Crossover, and Liquidity-Regime SMA
- evaluate return, Sharpe ratio, drawdown, Calmar ratio, trading cost, and exposure
- conduct SMA parameter search
- test slippage sensitivity
- generate tables and figures for the report

## Quick Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python -m src.main --data dataset/raw/hi1_20170701_20200609.csv --output-dir dataset
```

The command regenerates the cleaned datasets, performance tables, slippage
analysis, intraday statistics, and figures under `dataset/`.

## Data Processing Flow

The pipeline follows this process:

```text
Raw 1-minute HI1 futures data
-> datetime conversion
-> duplicate-minute aggregation
-> invalid price filtering
-> trading session identification
-> valid night-session preservation
-> 5-minute OHLCV resampling
-> simple return and log return calculation
-> in-sample / out-of-sample split
-> strategy backtesting
-> performance, slippage, and exposure analysis
-> CSV and figure output
```

The session-aware cleaning step preserves valid Friday night sessions that
cross into Saturday early morning, assigning them to the previous trading date.

## Sample Split

The default split is:

```text
In-sample:      2017-07-03 to 2019-06-30
Out-of-sample:  2019-07-01 to 2020-06-09
```

In-sample data is used for parameter search and adaptive-threshold calibration.
Out-of-sample data is used for final validation and slippage sensitivity
testing.

## Strategies

### Buy-and-Hold

The benchmark strategy holds one long HI1 futures contract throughout the
sample period. It is used to evaluate whether active trading improves over a
passive futures exposure.

### SMA Crossover 20/60

The main return-seeking strategy uses 5-minute close prices:

```text
SMA(20) > SMA(60) -> long one contract
SMA(20) < SMA(60) -> short one contract
Session end       -> flatten position
```

This is a transparent intraday trend-following rule.

### Liquidity-Regime SMA

The innovation extends the SMA Crossover strategy with two filters:

```text
1. Regime filter:
   trade only when trend strength is high enough and realized volatility is
   neither too low nor extreme.

2. Liquidity filter:
   open new positions only in sessions that are liquid enough according to
   in-sample average volume.
```

All thresholds are calibrated from in-sample data only and saved in:

```text
dataset/table3_adaptive_sma_calibration.csv
```

This avoids using out-of-sample information when designing the trading rule.

## Backtesting Assumptions

The backtest uses one HI1 contract by default.

```text
PnL = position * price_change * contract_multiplier
contract_multiplier = 50
initial_capital = 1,000,000
margin_rate = 0.1
```

The equity curve is normalized by initial capital. The output also reports:

```text
Max Notional Exposure
Max Margin Required
Max Exposure / Capital
```

Slippage sensitivity is tested on the out-of-sample period.

## Key Results

Current results from `dataset/table1_baseline_performance.csv`:

| Strategy | Sample | Total Return | Sharpe | Max Drawdown | Calmar |
| --- | --- | ---: | ---: | ---: | ---: |
| Buy-and-Hold | IS | 14.54% | 0.31 | -32.12% | 0.21 |
| SMA Crossover 20/60 | IS | 55.74% | 1.48 | -10.08% | 2.31 |
| Liquidity-Regime SMA | IS | 42.00% | 2.10 | -4.49% | 4.01 |
| Buy-and-Hold | OOS | -20.27% | -0.48 | -40.31% | -0.47 |
| SMA Crossover 20/60 | OOS | 56.67% | 2.82 | -9.29% | 5.60 |
| Liquidity-Regime SMA | OOS | 8.54% | 1.12 | -4.93% | 1.61 |

Interpretation:

- SMA Crossover 20/60 has the strongest out-of-sample return.
- Liquidity-Regime SMA sacrifices return but reduces drawdown and trading
  frequency, so it can be discussed as a risk-managed version.
- Buy-and-Hold performs poorly in the out-of-sample period.

## Slippage Sensitivity

`dataset/table4_slippage_sensitivity.csv` evaluates the Liquidity-Regime SMA
under different slippage assumptions on the out-of-sample period:

```text
0.0, 0.5, 1.0, 2.0, 3.0 points
```

This table supports the report discussion on real trading costs and execution
risk.

## Project Layout

```text
src/
  data_loader.py   # Raw data loading, datetime parsing, session-aware cleaning
  strategy.py      # Buy-and-Hold, SMA Crossover, Liquidity-Regime SMA
  backtester.py    # Vectorized futures backtest, PnL, slippage, exposure
  metrics.py       # Return, Sharpe, drawdown, Calmar, cost, exposure metrics
  optimizer.py     # Legacy SMA parameter search helper
  pipeline.py      # End-to-end project pipeline and output generation
  reporting.py     # Report-ready chart generation
  main.py          # Command-line entry point
```

## Generated Outputs

Cleaned datasets:

```text
dataset/hi1_5min_full.csv
dataset/hi1_5min_insample.csv
dataset/hi1_5min_outofsample.csv
```

Report tables:

```text
dataset/table1_baseline_performance.csv
dataset/table2_sma_parameter_search.csv
dataset/table3_adaptive_sma_calibration.csv
dataset/table4_slippage_sensitivity.csv
dataset/intraday_stats_for_B.csv
```

Report figures:

```text
dataset/figure1_price_volume.png
dataset/figure2_return_distribution.png
dataset/figure3_oos_equity_curves.png
dataset/figure4_oos_drawdown.png
dataset/figure5_sma_parameter_heatmap.png
```

## tips

The recommended framing is:

```text
SMA Crossover 20/60 is the main return-seeking strategy.
Liquidity-Regime SMA is a risk-managed extension that uses only in-sample
statistics to identify suitable trading regimes and liquid sessions.
```

