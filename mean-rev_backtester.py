import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

from bb_ml_stable3 import SignalFilter

# --- PARAMETERS FOR TUNING ---
RSI_ENTRY = 10   # Entry threshold for RSI(2)
RSI_EXIT = 50   # Exit threshold for RSI(2)
SL_MULT = 0.5    # Stop-loss multiple of ATR
TP_MULT = 1.5    # Take-profit multiple of ATR

# --- Lightweight Bar object to mimic IBKR's bar ---
class Bar:
    def __init__(self, date, open_, high, low, close, volume):
        self.date = date
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

# --- Simple Performance Analyzer (reuse from bb_ml_stable3 if possible) ---
class PerformanceAnalyzer:
    def __init__(self):
        self.trade_log = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'symbol', 'direction',
            'quantity', 'entry_price', 'exit_price', 'pnl', 'exit_reason'])

    def log_trade(self, entry_time, exit_time, symbol, direction, quantity, entry_price, exit_price, exit_reason=None):
        pnl = (exit_price - entry_price) * quantity if direction == 'BUY' else (entry_price - exit_price) * quantity
        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason
        }])
        self.trade_log = pd.concat([self.trade_log, new_trade], ignore_index=True)

    def generate_report(self):
        if self.trade_log.empty:
            print("No trades to report.")
            return

        # Plot: Per-symbol equity curve (SPY only)
        plt.figure(figsize=(12, 6))
        for symbol in self.trade_log['symbol'].unique():
            symbol_trades = self.trade_log[self.trade_log['symbol'] == symbol]
            symbol_trades = symbol_trades.sort_values('exit_time')
            plt.plot(pd.to_datetime(symbol_trades['exit_time']),
                    symbol_trades['pnl'].cumsum(),
                    label=symbol)
        plt.title('Strategy Equity Curve (SPY)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Analytics ---
        pnl = self.trade_log['pnl']
        returns = pnl / self.trade_log['entry_price']
        # Sharpe ratio (assume 252*78 trading periods/year for 5min bars)
        sharpe = np.nan
        if len(returns.dropna()) > 1:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78)
        # Max drawdown
        cum_pnl = pnl.cumsum()
        roll_max = cum_pnl.cummax()
        drawdown = cum_pnl - roll_max
        max_dd = drawdown.min()
        # Win rate
        wins = (pnl > 0).sum()
        total = len(pnl)
        win_rate = wins / total if total > 0 else np.nan
        print("\n--- Performance Analytics ---")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f}")
        print(f"Win Rate: {win_rate:.2%}")

        # Export trade log to CSV
        csv_path = os.path.join(os.getcwd(), 'trade_log_export.csv')
        self.trade_log.to_csv(csv_path, index=False)
        print(f"\nTrade log exported to: {csv_path}")

        print(self.trade_log)

# --- Mock Order Manager ---
class MockOrderManager:
    def __init__(self, app):
        self.app = app
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.direction = None
        self.quantity = 0
        self.stop_loss = None
        self.take_profit = None
        self.bars_held = 0

    def evaluate_long_entry(self, symbol, bar, atr):
        if self.position == 0:
            self.position = 1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'BUY'
            self.quantity = 100
            self.stop_loss = self.entry_price - SL_MULT * atr
            self.take_profit = self.entry_price + TP_MULT * atr
            self.bars_held = 0

    def evaluate_short_entry(self, symbol, bar, atr):
        if self.position == 0:
            self.position = -1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'SELL'
            self.quantity = 100
            self.stop_loss = self.entry_price + SL_MULT * atr
            self.take_profit = self.entry_price - TP_MULT * atr
            self.bars_held = 0

    def check_exit(self, symbol, bar):
        if self.position != 0:
            self.bars_held += 1
            exit_reason = None
            # Check stop-loss/take-profit
            if self.position == 1:
                if bar.low <= self.stop_loss:
                    exit_reason = 'SL'
                elif bar.high >= self.take_profit:
                    exit_reason = 'TP'
            elif self.position == -1:
                if bar.high >= self.stop_loss:
                    exit_reason = 'SL'
                elif bar.low <= self.take_profit:
                    exit_reason = 'TP'
            # Time-based exit (after 10 bars for demo)
            if self.bars_held >= 10:
                exit_reason = 'TIME'
            if exit_reason is not None:
                self.close_position(symbol, bar, exit_reason)

    def close_position(self, symbol, bar, exit_reason=None):
        if self.position == 0:
            return
        exit_price = bar.close
        exit_time = bar.date
        self.app.performance_analyzer.log_trade(
            self.entry_time, exit_time, symbol, self.direction, self.quantity, self.entry_price, exit_price, exit_reason)
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.direction = None
        self.quantity = 0
        self.stop_loss = None
        self.take_profit = None
        self.bars_held = 0

# --- Grid Search for Parameter Optimization ---
from itertools import product

def run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, verbose=False):
    # Override global params for this run
    global RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT
    RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT = rsi_entry, rsi_exit, sl_mult, tp_mult
    perf = PerformanceAnalyzer()
    for symbol in symbols:
        if symbol != 'SPY':
            continue
        csv_path = os.path.join(data_dir, f"{symbol}_5min-bar_3mo_historical_data.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            continue
        bars = [Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']) for idx, row in df.iterrows()]
        order_manager = MockOrderManager(app=type('App', (), {'performance_analyzer': perf})())
        closes, highs, lows = [], [], []
        for bar in bars:
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < 15:
                continue
            import talib
            close_arr = np.array(closes[-15:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            rsi2 = talib.RSI(close_arr, timeperiod=2)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)[-1]
            if order_manager.position == 0 and rsi2 < RSI_ENTRY:
                order_manager.evaluate_long_entry(symbol, bar, atr)
            elif order_manager.position == 1 and rsi2 > RSI_EXIT:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            order_manager.check_exit(symbol, bar)
    # Return total PnL
    if perf.trade_log.empty:
        return -np.inf if not verbose else (None, -np.inf)
    total_pnl = perf.trade_log['pnl'].sum()
    if verbose:
        perf.generate_report()
    return total_pnl

def grid_search(symbols, data_dir):
    rsi_entry_range = [2, 5, 10]
    rsi_exit_range = [50, 60, 70]
    sl_mult_range = [0.5, 1.0, 1.5]
    tp_mult_range = [1.0, 1.5, 2.0]
    best_pnl = -np.inf
    best_params = None
    results = []
    for rsi_entry, rsi_exit, sl_mult, tp_mult in product(rsi_entry_range, rsi_exit_range, sl_mult_range, tp_mult_range):
        pnl = run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult)
        results.append((rsi_entry, rsi_exit, sl_mult, tp_mult, pnl))
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = (rsi_entry, rsi_exit, sl_mult, tp_mult)
    print("Grid Search Results (top 10):")
    results.sort(key=lambda x: x[-1], reverse=True)
    for row in results[:10]:
        print(f"RSI_ENTRY={row[0]}, RSI_EXIT={row[1]}, SL_MULT={row[2]}, TP_MULT={row[3]} => Total PnL: {row[4]:.2f}")
    print(f"\nBest Params: RSI_ENTRY={best_params[0]}, RSI_EXIT={best_params[1]}, SL_MULT={best_params[2]}, TP_MULT={best_params[3]} => Total PnL: {best_pnl:.2f}")
    return best_params

# --- Main Backtest Logic ---
def run_backtest(symbols, data_dir, model_path=None):
    perf = PerformanceAnalyzer()
    for symbol in symbols:
        if symbol != 'SPY':
            continue  # Only trade SPY for this strategy
        print(f"Backtesting {symbol}...")
        csv_path = os.path.join(data_dir, f"{symbol}_5min-bar_3mo_historical_data.csv")
        if not os.path.exists(csv_path):
            print(f"Data file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            print(f"Missing columns in {csv_path}")
            continue
        bars = [Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']) for idx, row in df.iterrows()]
        order_manager = MockOrderManager(app=type('App', (), {'performance_analyzer': perf})())
        closes = []
        highs = []
        lows = []
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < 15:
                continue
            import talib
            close_arr = np.array(closes[-15:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            rsi2 = talib.RSI(close_arr, timeperiod=2)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)[-1]
            # Entry: RSI(2) < RSI_ENTRY and not in position
            if order_manager.position == 0 and rsi2 < RSI_ENTRY:
                order_manager.evaluate_long_entry(symbol, bar, atr)
            # Exit: RSI(2) > RSI_EXIT or SL/TP
            elif order_manager.position == 1 and rsi2 > RSI_EXIT:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            # Always check for SL/TP
            order_manager.check_exit(symbol, bar)
    perf.generate_report()

if __name__ == "__main__":
    symbols = ['SPY', 'NVDA', 'RGTI', 'GOLD', 'QBTS', 'QUBT', 'TSLA', 'IONQ']
    data_dir = r'c:/Users/chris/Desktop/workspace/trading/IBKR/BB_trader/train_model/tech_liquid_set'
    model_path = 'trading_signal_model.joblib'  # Adjust if needed
    # Uncomment below to run grid search
    #grid_search(symbols, data_dir)
    # Or run with chosen params:
    # run_backtest_param(symbols, data_dir, 5, 60, 0.5, 1.0, verbose=True)
    run_backtest(symbols, data_dir, None)
