# Portfolio Optimization

Walk-forward portfolio optimization with transaction cost modeling.

## What it does

Optimizes a 3-asset portfolio:
- **NIFTY 50** - Indian equity index (converted to USD)
- **S&P 500** - US equity index
- **Gold** - Commodity / safe haven asset

The optimizer maximizes Sharpe ratio using:
- 36-month rolling lookback windows
- Annual rebalancing with 1-period execution delay
- Almgren-Chriss market impact cost model
- Ledoit-Wolf covariance shrinkage

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
