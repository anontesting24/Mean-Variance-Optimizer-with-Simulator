# Mean-Variance Portfolio Optimizer with Monte Carlo Simulation

## Overview
A Python implementation of Modern Portfolio Theory (MPT) that optimizes asset allocation using mean-variance analysis and includes Monte Carlo simulation for risk assessment. The tool optimizes portfolio weights to maximize the Sharpe ratio while allowing for flexible weight constraints, and provides Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations through simulation.

## Features
- Mean-Variance portfolio optimization using Sharpe ratio maximization
- Monte Carlo simulation for portfolio performance analysis
- Flexible portfolio weight constraints
- Risk metrics calculation (VaR and CVaR)
- Real-time data fetching using yfinance
- Visualization of Monte Carlo simulation results
- Support for custom risk-free rates

## Prerequisites
```
numpy
pandas
matplotlib
scipy
yfinance
```

## Usage
### Basic Example
```python
from MVO import mvo

# Initialize portfolio optimizer
portfolio = mvo()

# Add tickers to portfolio
portfolio.add_tickers(['GLD', 'QQQ', 'SPY', 'VTI', '^NSEI'])

# Optimize portfolio
# Parameters: years of historical data, lower bound, upper bound
portfolio.optimize(years=5, lower_bound=0, upper_bound=0.5)

# Run Monte Carlo simulation
# Parameters: number of simulations, number of days, initial investment
portfolio.MonteCarloSim(n_sims=100, n_days=100, initialval=1000)
```

### Key Functions

#### Portfolio Optimization (`MVO.py`)
- `add_tickers(tickers)`: Add assets to the portfolio
- `remove_tickers(tickers)`: Remove assets from the portfolio
- `empty_tickers()`: Clear all assets
- `optimize(years, lower_bound, upper_bound, rf_rate=None)`: Optimize portfolio weights
- `results()`: Display optimization results
- `MonteCarloSim(n_sims, n_days, initialval)`: Run Monte Carlo simulation

#### Monte Carlo Simulation (`MonteCarlo.py`)
- `VaR(returns, cl=5)`: Calculate Value at Risk
- `CVaR(returns, cl=5)`: Calculate Conditional Value at Risk
- `Simulate(tickers, weights, mean_returns, covmatrix, n_sims, n_days, initialval)`: Run simulation

## Implementation Details

### Optimization Method
- Uses Sharpe ratio maximization via SciPy's SLSQP optimizer
- Supports custom weight constraints for each asset
- Automatically fetches risk-free rate from ^IRX (13-week Treasury Bill rate)
- Calculates returns using simple returns (can be modified for log returns)

### Risk Analysis
- Monte Carlo simulation using Cholesky decomposition for correlated returns
- Visual representation of simulated paths
- VaR and CVaR calculations at 5% confidence level
- Interactive plots showing risk metrics

## Example Output
The optimizer provides:
- Optimal weights for each asset
- Expected annual return
- Expected volatility
- Sharpe ratio

The Monte Carlo simulation generates:
- Visual plot of simulated portfolio paths
- VaR and CVaR lines on the plot
- Portfolio value projections

## Acknowledgments
- yfinance for free market data access
