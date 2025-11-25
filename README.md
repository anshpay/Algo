# Portfolio Optimization

Walk-forward portfolio optimization with transaction cost modeling.

## What it does

- Optimizes weights across NIFTY, S&P 500, and Gold
- Uses 36-month rolling lookback windows
- Rebalances annually with 1-period execution delay
- Estimates transaction costs using Almgren-Chriss market impact model
- Applies Ledoit-Wolf shrinkage to covariance estimation

## Usage

```bash
python portfolio.py
```

## Dependencies

- numpy
- pandas
- scipy
- yfinance
- matplotlib

Optional: `numba` (for faster impact calculations), `pyarrow` (for parquet export)

## Data

Requires two NIFTY 50 CSV files in the same directory:
- `Nifty 50 Historical Data (1).csv`
- `Nifty 50 Historical Data (2).csv`

S&P 500 and Gold data are fetched from Yahoo Finance.

## Output

- Performance metrics printed to console
- PNG charts saved to working directory
- Parquet files with backtest results (if pyarrow installed)
