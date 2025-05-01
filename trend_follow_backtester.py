import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# --- PARAMETERS FOR TUNING ---
FAST_MA = 10   # Fast moving average window
SLOW_MA = 50   # Slow moving average window
SL_MULT = 1.0  # Stop-loss multiple of ATR
TP_MULT = 1.0  # Take-profit multiple of ATR
ATR_THRESHOLD = 3.0  # Volatility filter threshold

# --- Lightweight Bar object ---
class Bar:
    def __init__(self, date, open_, high, low, close, volume):
        self.date = date
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

# --- Simple Performance Analyzer ---
class PerformanceAnalyzer:
    def __init__(self, starting_cash=25000):
        self.trade_log = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'symbol', 'direction',
            'quantity', 'entry_price', 'exit_price', 'pnl', 'exit_reason'])
        self.starting_cash = starting_cash

    def log_trade(self, entry_time, exit_time, symbol, direction, quantity, entry_price, exit_price, pnl, exit_reason):
        self.trade_log = pd.concat([
            self.trade_log,
            pd.DataFrame([{
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
        ], ignore_index=True)

    def generate_report(self):
        if self.trade_log.empty:
            print('No trades to analyze.')
            return
        pnl = self.trade_log['pnl']
        returns = pnl / self.trade_log['entry_price']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78) if returns.std() != 0 else 0
        equity = pnl.cumsum() + self.starting_cash
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / self.starting_cash) * 100
        win_rate = (self.trade_log['pnl'] > 0).mean() * 100
        print("--- Performance Analytics ---")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f} ({max_dd_pct:.2f}%)")
        print(f"Win Rate: {win_rate:.2f}%")
        plt.figure(figsize=(10,5))
        plt.plot(pd.to_datetime(self.trade_log['exit_time']), equity, label='SPY')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.title('Strategy Equity Curve (SPY)')
        plt.legend()
        plt.show()
        self.trade_log.to_csv('trend_trade_log_export.csv', index=False)

# --- Order Manager ---
class OrderManager:
    def __init__(self, perf_analyzer):
        self.position = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.perf_analyzer = perf_analyzer

    def evaluate_entry(self, symbol, bar, atr, direction):
        if self.position == 0:
            self.position = 1 if direction == 'long' else -1
            self.entry_bar = bar
            self.entry_price = bar.close
            self.stop_loss = bar.close - SL_MULT*atr if direction == 'long' else bar.close + SL_MULT*atr
            self.take_profit = bar.close + TP_MULT*atr if direction == 'long' else bar.close - TP_MULT*atr

    def check_exit(self, symbol, bar):
        if self.position == 0:
            return
        direction = 'long' if self.position == 1 else 'short'
        hit_sl = (bar.low <= self.stop_loss) if direction == 'long' else (bar.high >= self.stop_loss)
        hit_tp = (bar.high >= self.take_profit) if direction == 'long' else (bar.low <= self.take_profit)
        exit_reason = None
        if hit_sl:
            exit_reason = 'StopLoss'
        elif hit_tp:
            exit_reason = 'TakeProfit'
        if exit_reason:
            pnl = (bar.close - self.entry_price)*self.position
            self.perf_analyzer.log_trade(
                self.entry_bar.date, bar.date, symbol, direction, 1, self.entry_price, bar.close, pnl, exit_reason)
            self.position = 0
            self.entry_bar = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None

    def close_position(self, symbol, bar, exit_reason):
        if self.position == 0:
            return
        direction = 'long' if self.position == 1 else 'short'
        pnl = (bar.close - self.entry_price)*self.position
        self.perf_analyzer.log_trade(
            self.entry_bar.date, bar.date, symbol, direction, 1, self.entry_price, bar.close, pnl, exit_reason)
        self.position = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

# --- Main Backtest Logic ---
def run_backtest(symbols, data_dir, fast_ma=FAST_MA, slow_ma=SLOW_MA, show_report=True, starting_cash=25000):
    perf = PerformanceAnalyzer(starting_cash=starting_cash)
    for symbol in symbols:
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
        closes, highs, lows = [], [], []
        order_manager = OrderManager(perf)
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < slow_ma or len(closes) < 15:
                continue
            import talib
            close_arr = np.array(closes[-slow_ma:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            fast = talib.SMA(close_arr[-fast_ma:], timeperiod=fast_ma)[-1]
            slow = talib.SMA(close_arr, timeperiod=slow_ma)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            # Volatility filter
            if atr >= ATR_THRESHOLD:
                continue
            # Entry logic: crossover
            if order_manager.position == 0:
                if fast > slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='long')
                elif fast < slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='short')
            # Exit logic: cross in opposite direction
            elif order_manager.position == 1 and fast < slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossDown')
            elif order_manager.position == -1 and fast > slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossUp')
            order_manager.check_exit(symbol, bar)
    if show_report:
        perf.generate_report()
    if perf.trade_log.empty:
        return -np.inf
    pnl = perf.trade_log['pnl'].sum()
    return pnl

# --- Grid Search for Parameter Optimization ---
from itertools import product

def grid_search(symbols, data_dir):
    fast_ma_range = [5, 10, 15]
    slow_ma_range = [20, 30, 50]
    sl_mult_range = [0.5, 1.0, 1.5]
    tp_mult_range = [0.5, 1.0, 1.5]
    atr_thresh_range = [1.0, 2.0, 3.0]
    best_pnl = -np.inf
    best_params = None
    results = []
    for fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh in product(fast_ma_range, slow_ma_range, sl_mult_range, tp_mult_range, atr_thresh_range):
        if fast_ma >= slow_ma:
            continue
        global FAST_MA, SLOW_MA, SL_MULT, TP_MULT, ATR_THRESHOLD
        FAST_MA, SLOW_MA, SL_MULT, TP_MULT, ATR_THRESHOLD = fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh
        pnl = run_backtest(symbols, data_dir, fast_ma, slow_ma, show_report=False)
        results.append((fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, pnl))
        if pnl > best_pnl:
            best_pnl = pnl
            best_params = (fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh)
    print("Grid Search Results (top 10):")
    results.sort(key=lambda x: x[-1], reverse=True)
    for row in results[:10]:
        print(f"FAST_MA={row[0]}, SLOW_MA={row[1]}, SL_MULT={row[2]}, TP_MULT={row[3]}, ATR_THRESH={row[4]} => Total PnL: {row[5]:.2f}")
    if best_params is not None and best_pnl != -np.inf:
        print(f"\nBest Params: FAST_MA={best_params[0]}, SLOW_MA={best_params[1]}, SL_MULT={best_params[2]}, TP_MULT={best_params[3]}, ATR_THRESH={best_params[4]} => Total PnL: {best_pnl:.2f}")
    else:
        print("\nNo valid trades for any parameter combination.")
    return best_params

if __name__ == "__main__":
    symbols = ['SPY']
    data_dir = os.getcwd()
    # Uncomment to run grid search
    #grid_search(symbols, data_dir)
    # Or run with chosen params:
    run_backtest(symbols, data_dir, fast_ma=15, slow_ma=50)
