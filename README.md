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

All data is fetched from Yahoo Finance:
- S&P 500 (^GSPC)
- Gold futures (GC=F)
- USD/INR exchange rate (INR=X)
- 13-week T-bill rate (^IRX)

NIFTY 50 data is loaded from local CSV files:
- `Nifty 50 Historical Data (1).csv`
- `Nifty 50 Historical Data (2).csv`

## Output

- Performance metrics printed to console
- PNG charts saved to working directory
- Parquet files with backtest results (if pyarrow installed)
