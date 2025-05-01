# Mean-Reversion & Trend-Following Trading System

## Overview
This project is an advanced algorithmic trading system for US equities, featuring both mean-reversion and trend-following strategies. It includes robust backtesting, live trading integration with Interactive Brokers (IBKR), and machine learning-based signal filtering. The codebase is designed for research, simulation, and live deployment.

## Features
- **Live Trading with IBKR**: Automated execution using the Interactive Brokers API (via `ibapi`).
- **Mean-Reversion Strategy**: Classic mean-reversion logic with RSI(2), ATR-based volatility filters, and machine learning signal filtering.
- **Trend-Following Strategy**: Moving average crossover system with ATR-based stop-loss/take-profit and volatility filtering.
- **Backtesting Engine**: Fast, vectorized backtesters for both strategies with performance analytics (Sharpe, drawdown, win rate).
- **Parameter Grid Search**: Optimize strategy parameters for best performance.
- **Data Acquisition**: Automated download and CSV export of historical 5-minute bar data from IBKR.
- **Trade Logging & Analytics**: Detailed trade logs, equity curve plots, and CSV export for further analysis.

## Directory Structure
```
mean-rev_trader/
├── bb_ml_stable3.py                # Main live trading engine (IBKR + ML filter)
├── mean-rev_backtester.py          # Mean-reversion backtester
├── trend_follow_backtester.py      # Trend-following backtester
├── get_data.py                     # Download historical data from IBKR
├── requirements.txt                # Python dependencies
├── *_5min-bar_3mo_historical_data.csv # Historical data (per-symbol)
├── trade_log_export.csv            # Mean-reversion trade log
├── trend_trade_log_export.csv      # Trend-following trade log
└── README.md                       # Project documentation
```

## Quick Start
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Historical Data
Edit the symbol list in `get_data.py` and run:
```bash
python get_data.py
```
This will create CSVs like `SPY_5min-bar_3mo_historical_data.csv`.

### 3. Run Backtests
**Mean-Reversion:**
```bash
python mean-rev_backtester.py
```
**Trend-Following:**
```bash
python trend_follow_backtester.py
```

### 4. Live Trading (IBKR TWS/Gateway Required)
- Ensure IBKR TWS or Gateway is running and API is enabled.
- Place your trained ML model as `trading_signal_model.joblib` in the project folder.
- Edit symbols and parameters in `bb_ml_stable3.py` as needed.
- Run:
```bash
python bb_ml_stable3.py
```

## File Descriptions
- **bb_ml_stable3.py**: Main live trading engine. Connects to IBKR, streams data, generates signals (mean-reversion), filters with ML, and places orders.
- **mean-rev_backtester.py**: Backtesting engine for the mean-reversion strategy. Includes parameter grid search and analytics.
- **trend_follow_backtester.py**: Backtesting engine for the trend-following strategy. Includes grid search and analytics.
- **get_data.py**: Downloads historical 5-min bar data from IBKR and saves as CSV.
- **requirements.txt**: Python dependencies (pandas, numpy, matplotlib, talib, joblib).
- ***_5min-bar_3mo_historical_data.csv**: Historical data files for each symbol.
- **trade_log_export.csv**: Trade log from mean-reversion backtests.
- **trend_trade_log_export.csv**: Trade log from trend-following backtests.

## Machine Learning Model
- Place your trained model as `trading_signal_model.joblib` in the project root.
- The ML model is used to filter trade signals in live trading (see `SignalFilter` class in `bb_ml_stable3.py`).

## Requirements
- Python 3.8+
- Interactive Brokers TWS or Gateway (for live trading)
- IB API (`ibapi`), pandas, numpy, matplotlib, talib, joblib

## Notes
- **Risk Warning:** This code is for research and educational purposes. Trading involves risk. Use at your own discretion.
- **Data:** Only 5-minute bar data is supported. Ensure your IBKR account has proper permissions for data access.
- **Customization:** Strategies and parameters can be modified in the respective scripts.

## License
MIT License. See `LICENSE` file if present.

---
_Last updated: 2025-05-01_
