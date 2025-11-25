# Portfolio Optimization

Walk-forward portfolio optimization with transaction cost modeling.

## What it does

Optimizes a 3-asset portfolio:
- **NIFTY 50** - Indian equity index (converted to USD)
- **S&P 500** - US equity index
- **Gold** - Commodity / safe haven asset

### Optimization Process

1. **Data Preparation**: Fetches historical prices, converts NIFTY to USD using exchange rates, resamples to monthly frequency
2. **Return Calculation**: Computes monthly returns for each asset
3. **Covariance Estimation**: Uses Ledoit-Wolf shrinkage to get a more stable covariance matrix
4. **Mean-Variance Optimization**: Finds portfolio weights that maximize the Sharpe ratio (return per unit risk)
5. **Transaction Costs**: Estimates trading costs using the Almgren-Chriss square-root market impact model plus bid-ask spreads
6. **Walk-Forward Backtest**: Rolls the optimization forward in time using 36-month lookback windows, rebalancing annually

### Output Metrics

- Sharpe, Sortino, and Calmar ratios
- Max drawdown
- VaR and CVaR (95%)
- Cumulative performance vs benchmarks
- Rolling Sharpe ratio chart
- Portfolio weight evolution over time

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
