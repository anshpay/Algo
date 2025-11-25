# Portfolio Optimization Engine

A portfolio backtesting tool with walk-forward optimization, transaction cost modeling, and basic risk metrics.

## Features

- Walk-forward backtesting with rolling lookback windows
- Execution delay to avoid look-ahead bias
- Transaction cost model (spread + square-root impact)
- Ledoit-Wolf covariance shrinkage
- Supports S&P 500, NIFTY 50 (USD-adjusted), Gold
- Sharpe, Sortino, Max Drawdown, VaR/CVaR
- Optional Numba acceleration

## Installation

```bash
git clone https://github.com/anshpay/Algo.git
cd Algo
pip install numpy pandas scipy yfinance matplotlib
```

Optional:
```bash
pip install numba      # faster transaction cost calc
pip install pyarrow    # parquet export
```

## Usage

```python
python portfolio.py
```

Or import as a module:

```python
from portfolio import QuantConfig, DataEngine, BacktestEngine

config = QuantConfig(
    start_date="2004-01-01",
    end_date="2025-11-01",
    lookback_window=36,
    rebalance_freq=12,
)

data_engine = DataEngine(config)
prices, rf_series = data_engine.load_all()

engine = BacktestEngine(prices, config, rf_series=rf_series)
results = engine.run()
```

## Files

- `portfolio.py` — Full version with comments
- `portfolio1.py` — Compact version

## Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookback_window` | 36 | Months of history for optimization |
| `rebalance_freq` | 12 | Months between rebalances |
| `execution_delay` | 1 | Periods between signal and execution |
| `ledoit_wolf_shrinkage` | True | Shrinkage covariance estimator |

## Transaction Costs

Uses spread + square-root market impact:

```
cost = spread + η × daily_vol × √(participation_rate)
```

Default spreads: SPX 2bps, NIFTY 12bps, GOLD 4bps

## Output

- Console: performance metrics (Sharpe, drawdown, etc.)
- `performance.png` — cumulative returns chart
- `weights_evolution.png` — allocation over time
- `rolling_sharpe.png` — rolling Sharpe ratio

## Limitations

- Uses price indices (no dividends)
- Survivorship bias in indices
- Simplified ADV assumptions
- No intraday modeling

## Requirements

numpy, pandas, scipy, yfinance, matplotlib

## License

MIT
