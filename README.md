# Portfolio Optimization Engine

A production-grade quantitative portfolio optimization and backtesting framework with walk-forward analysis, realistic transaction costs, and institutional-quality risk metrics.

## Features

### Core Capabilities
- **Walk-Forward Backtesting** — Out-of-sample testing with rolling lookback windows
- **Execution Delay Modeling** — Proper signal-to-execution lag to prevent look-ahead bias
- **Almgren-Chriss Market Impact** — Square-root impact model with asset-specific spreads and ADV constraints
- **Ledoit-Wolf Shrinkage** — Robust covariance estimation for small sample sizes
- **Multi-Asset Support** — S&P 500, NIFTY 50 (USD-adjusted), Gold futures

### Risk Metrics
- Sharpe Ratio, Sortino Ratio, Calmar Ratio
- Maximum Drawdown
- Value-at-Risk (VaR) and Conditional VaR (CVaR) at 95%
- Rolling performance analytics

### Performance Optimizations
- **Numba JIT** — Optional acceleration for transaction cost calculations
- **Vectorized Operations** — NumPy-based computations throughout
- **Batched Data Downloads** — Efficient yfinance API usage
- **Pre-allocated Arrays** — Memory-efficient weight tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/anshpay/Algo.git
cd Algo

# Install dependencies
pip install numpy pandas scipy yfinance matplotlib

# Optional: Install Numba for performance boost
pip install numba

# Optional: Install pyarrow for parquet export
pip install pyarrow
```

## Quick Start

```python
from portfolio import QuantConfig, DataEngine, BacktestEngine, Reporter

# Configure backtest
config = QuantConfig(
    start_date="2004-01-01",
    end_date="2025-11-01",
    lookback_window=36,        # 3-year rolling window
    rebalance_freq=12,         # Annual rebalancing
    execution_delay=1,         # 1-period execution lag
    portfolio_value=1e8,       # $100M portfolio
)

# Load data
data_engine = DataEngine(config)
prices, rf_series = data_engine.load_all()

# Run backtest
engine = BacktestEngine(prices, config, rf_series=rf_series)
results = engine.run()

# Print performance summary
Reporter.print_summary(
    results["return"],
    engine.get_benchmark_returns(),
    engine.rebalance_events,
    engine.asset_names,
    config,
)
```

## File Structure

```
├── portfolio.py      # Full version with detailed docstrings and comments
├── portfolio1.py     # Streamlined production version
├── README.md
└── data/
    ├── Nifty 50 Historical Data (1).csv
    └── Nifty 50 Historical Data (2).csv
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `start_date` | `"2004-01-01"` | Backtest start date |
| `end_date` | `"2025-11-01"` | Backtest end date |
| `lookback_window` | `36` | Months of history for optimization |
| `rebalance_freq` | `12` | Months between rebalances |
| `execution_delay` | `1` | Periods between signal and execution |
| `risk_free_rate` | `None` | Static rate or `None` for time-varying |
| `portfolio_value` | `1e8` | Portfolio size for ADV calculations |
| `ledoit_wolf_shrinkage` | `True` | Use shrinkage covariance estimator |

## Transaction Cost Model

The engine implements an **Almgren-Chriss** style market impact model:

```
Total Cost = Spread Cost + Market Impact

Spread Cost = half_spread × trade_size
Market Impact = η × σ_daily × (participation_rate)^0.5 × trade_size
```

Where:
- `η` = temporary impact coefficient (default: 0.1)
- `σ_daily` = daily volatility
- `participation_rate` = trade_value / ADV (capped at 10%)

### Default Market Parameters

| Asset | Spread (bps) | ADV ($) |
|-------|-------------|---------|
| SPX | 2.0 | $500B |
| NIFTY | 12.0 | $100M |
| GOLD | 4.0 | $200B |

## Output

### Console Output
```
============================================================
PERFORMANCE SUMMARY
============================================================

Walk-Forward Portfolio Metrics:
----------------------------------------
  Annual Return:          8.45%
  Annual Volatility:     12.32%
  Sharpe Ratio:           0.68
  Sortino Ratio:          0.92
  Max Drawdown:         -28.45%
  Calmar Ratio:           0.30
  VaR (95%):             -4.21%
  CVaR (95%):            -6.15%
  Total Return:         412.56%
  Win Rate:              58.33%
```

### Generated Files
- `performance.png` — Cumulative returns and drawdown chart
- `weights_evolution.png` — Portfolio allocation over time
- `rolling_sharpe.png` — 36-month rolling Sharpe ratio
- `backtest_results.parquet` — Full results for further analysis
- `actual_weights.parquet` — Weight history

## Methodology

### Walk-Forward Process
1. **Signal Generation (T-1)**: Optimize portfolio using data `[T-1-lookback : T-1)`
2. **Execution (T)**: Execute trades at period T prices with transaction costs
3. **Return Realization (T)**: Compute portfolio return net of costs
4. **Weight Drift**: Update weights based on differential asset returns

### Optimizer
- **Objective**: Maximize Sharpe Ratio with soft L1 penalty for dust positions
- **Method**: SLSQP (Sequential Least Squares Programming)
- **Constraints**: 
  - Long-only (`w ≥ 0`)
  - Fully invested (`Σw = 1`)
  - Liquidity limits (`w ≤ 5% of ADV`)

## Known Limitations

- Uses price indices (dividends excluded) — for production, use total return indices
- Index-level survivorship bias present
- No intraday effects modeled
- Constant ADV assumptions

## Dependencies

| Package | Version | Required |
|---------|---------|----------|
| numpy | ≥1.20 | ✓ |
| pandas | ≥1.3 | ✓ |
| scipy | ≥1.7 | ✓ |
| yfinance | ≥0.2 | ✓ |
| matplotlib | ≥3.4 | ✓ |
| numba | ≥0.54 | Optional |
| pyarrow | ≥8.0 | Optional |

## License

MIT License

## Author

Ansh

---

*Built for quantitative research and portfolio management applications.*
