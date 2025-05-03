import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import csv
import talib


# --- USER CONFIGURABLE DATA FILE ---
# Set this variable to the filename of your historical data CSV (must be in data_dir)
CSV_FILENAME = "../../data/SPY_1hour_bar_20Y_historical_data.csv"

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
        # Only add non-empty, non-all-NA trades to the log to avoid FutureWarning
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
        if not new_trade.isna().all(axis=None):
            self.trade_log = pd.concat([
                self.trade_log,
                new_trade
            ], ignore_index=True)

    def generate_report(self, overlay_buy_and_hold=False, csv_filename=CSV_FILENAME):
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
        plt.plot(pd.to_datetime(self.trade_log['exit_time']), equity, label='Strategy')
        if overlay_buy_and_hold:
            if os.path.exists(csv_filename):
                df = pd.read_csv(csv_filename)
                df = df.sort_values('date').reset_index(drop=True)
                entry_price = df['close'].iloc[0]
                shares = self.starting_cash / entry_price
                equity_bh = shares * df['close']
                plt.plot(pd.to_datetime(df['date']), equity_bh, label='Buy & Hold (SPY)', color='gray', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.title('Equity Curve Comparison (SPY)')
        plt.legend()
        plt.show()
        self.trade_log.to_csv('trend_trade_log_export.csv', index=False)

# --- Order Manager ---
class OrderManager:
    def __init__(self, perf_analyzer, sl_mult=1.0, tp_mult=1.0):
        self.perf_analyzer = perf_analyzer
        self.position = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.quantity = 0

    def evaluate_entry(self, symbol, bar, atr, direction):
        if self.position == 0:
            # Calculate position size to risk 1% of equity per trade
            equity = self.perf_analyzer.trade_log['pnl'].cumsum().iloc[-1] + self.perf_analyzer.starting_cash if not self.perf_analyzer.trade_log.empty else self.perf_analyzer.starting_cash
            risk_per_trade = equity * 0.01
            stop_dist = abs((bar.close - (bar.close - self.sl_mult*atr)) if direction == 'long' else (bar.close - (bar.close + self.sl_mult*atr)))
            quantity = max(int(risk_per_trade / stop_dist), 1) if stop_dist > 0 else 1
            self.position = 1 if direction == 'long' else -1
            self.entry_bar = bar
            self.entry_price = bar.close
            self.stop_loss = bar.close - self.sl_mult*atr if direction == 'long' else bar.close + self.sl_mult*atr
            self.take_profit = bar.close + self.tp_mult*atr if direction == 'long' else bar.close - self.tp_mult*atr
            self.quantity = quantity
            # Suppressed verbose entry print

    def close_position(self, symbol, bar, exit_reason):
        if self.position == 0:
            return
        direction = 'long' if self.position == 1 else 'short'
        pnl = (bar.close - self.entry_price)*self.position * self.quantity
        self.perf_analyzer.log_trade(
            self.entry_bar.date, bar.date, symbol, direction, self.quantity, self.entry_price, bar.close, pnl, exit_reason)
        self.position = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.quantity = 0
        # Suppressed verbose close print

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
            pnl = (bar.close - self.entry_price)*self.position * self.quantity
            self.perf_analyzer.log_trade(
                self.entry_bar.date, bar.date, symbol, direction, self.quantity, self.entry_price, bar.close, pnl, exit_reason)
            self.position = 0
            self.entry_bar = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.quantity = 0
            # Suppressed verbose exit print

# --- Main Backtest Logic ---
def run_backtest(symbols, data_dir, fast_ma=FAST_MA, slow_ma=SLOW_MA, sl_mult=SL_MULT, tp_mult=TP_MULT, atr_thresh=ATR_THRESHOLD, show_report=True, starting_cash=25000, csv_filename=CSV_FILENAME, overlay_buy_and_hold=False):
    perf = PerformanceAnalyzer(starting_cash=starting_cash)
    for symbol in symbols:
        print(f"Backtesting {symbol}...")
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Data file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            continue
        bars = [Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']) for idx, row in df.iterrows()]
        order_manager = OrderManager(perf, sl_mult=sl_mult, tp_mult=tp_mult)
        fast_ma_series = pd.Series([bar.close for bar in bars]).rolling(window=fast_ma).mean()
        slow_ma_series = pd.Series([bar.close for bar in bars]).rolling(window=slow_ma).mean()
        atr_series = pd.Series([np.nan] * len(bars))
        for i in range(len(bars)):
            if i >= 14:
                highs = [bar.high for bar in bars[i-14:i+1]]
                lows = [bar.low for bar in bars[i-14:i+1]]
                closes = [bar.close for bar in bars[i-14:i+1]]
                atr_series.iloc[i] = talib.ATR(np.array(highs), np.array(lows), np.array(closes), timeperiod=14)[-1]
        for i, bar in enumerate(bars):
            fast = fast_ma_series.iloc[i]
            slow = slow_ma_series.iloc[i]
            atr = atr_series.iloc[i]
            if np.isnan(fast) or np.isnan(slow) or np.isnan(atr):
                continue
            # Volatility filter
            if atr >= atr_thresh:
                continue
            if order_manager.position == 0:
                if fast > slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='long')
                elif fast < slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='short')
            elif order_manager.position == 1 and fast < slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossDown')
            elif order_manager.position == -1 and fast > slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossUp')
            order_manager.check_exit(symbol, bar)
    if show_report:
        perf.generate_report(overlay_buy_and_hold=overlay_buy_and_hold, csv_filename=csv_filename)
    if perf.trade_log.empty:
        return -np.inf
    pnl = perf.trade_log['pnl'].sum()
    return pnl

# --- Split Test Logic ---
def run_backtest_param_split(symbols, data_dir, fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, n_splits=3, min_train_size=1000, test_size=500, verbose=True, starting_cash=25000, csv_filename=CSV_FILENAME):
    perf = PerformanceAnalyzer(starting_cash=starting_cash)
    for symbol in symbols:
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Data file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            continue
        bars = [Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']) for idx, row in df.iterrows()]
        total_bars = len(bars)
        split_points = []
        start = 0
        for split in range(n_splits):
            train_end = start + min_train_size
            test_end = train_end + test_size
            if test_end > total_bars:
                break
            split_points.append((start, train_end, test_end))
            start = train_end
        for i, (train_start, train_end, test_end) in enumerate(split_points):
            # Train split
            train_perf = PerformanceAnalyzer(starting_cash=starting_cash)
            train_order_manager = OrderManager(train_perf, sl_mult=sl_mult, tp_mult=tp_mult)
            closes, highs, lows = [], [], []
            for bar in bars[train_start:train_end]:
                closes.append(bar.close)
                highs.append(bar.high)
                lows.append(bar.low)
                if len(closes) < slow_ma or len(closes) < 15:
                    continue
                close_arr = np.array(closes[-slow_ma:])
                high_arr = np.array(highs[-15:])
                low_arr = np.array(lows[-15:])
                # Defensive: all ATR arrays must be length 15
                if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                    continue
                import talib
                fast = talib.SMA(close_arr[-fast_ma:], timeperiod=fast_ma)[-1]
                slow = talib.SMA(close_arr, timeperiod=slow_ma)[-1]
                atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
                # Volatility filter
                if atr >= atr_thresh:
                    continue
                if train_order_manager.position == 0:
                    if fast > slow:
                        train_order_manager.evaluate_entry(symbol, bar, atr, direction='long')
                    elif fast < slow:
                        train_order_manager.evaluate_entry(symbol, bar, atr, direction='short')
                elif train_order_manager.position == 1 and fast < slow:
                    train_order_manager.close_position(symbol, bar, exit_reason='CrossDown')
                elif train_order_manager.position == -1 and fast > slow:
                    train_order_manager.close_position(symbol, bar, exit_reason='CrossUp')
                train_order_manager.check_exit(symbol, bar)
            train_log = train_perf.trade_log.copy()
            train_log.to_csv(f"{symbol}_walk_train_trades_{i}.csv", index=False)
            # Test split
            test_perf = PerformanceAnalyzer(starting_cash=starting_cash)
            test_order_manager = OrderManager(test_perf, sl_mult=sl_mult, tp_mult=tp_mult)
            closes, highs, lows = [], [], []
            for bar in bars[train_end:test_end]:
                closes.append(bar.close)
                highs.append(bar.high)
                lows.append(bar.low)
                if len(closes) < slow_ma or len(closes) < 15:
                    continue
                close_arr = np.array(closes[-slow_ma:])
                high_arr = np.array(highs[-15:])
                low_arr = np.array(lows[-15:])
                # Defensive: all ATR arrays must be length 15
                if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                    continue
                import talib
                fast = talib.SMA(close_arr[-fast_ma:], timeperiod=fast_ma)[-1]
                slow = talib.SMA(close_arr, timeperiod=slow_ma)[-1]
                atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
                # Volatility filter
                if atr >= atr_thresh:
                    continue
                if test_order_manager.position == 0:
                    if fast > slow:
                        test_order_manager.evaluate_entry(symbol, bar, atr, direction='long')
                    elif fast < slow:
                        test_order_manager.evaluate_entry(symbol, bar, atr, direction='short')
                elif test_order_manager.position == 1 and fast < slow:
                    test_order_manager.close_position(symbol, bar, exit_reason='CrossDown')
                elif test_order_manager.position == -1 and fast > slow:
                    test_order_manager.close_position(symbol, bar, exit_reason='CrossUp')
                test_order_manager.check_exit(symbol, bar)
            test_log = test_perf.trade_log.copy()
            test_log.to_csv(f"{symbol}_walk_test_trades_{i}.csv", index=False)
            if verbose:
                print(f"Split {i}: Train trades: {len(train_log)}, Test trades: {len(test_log)}")

# --- Walk-Forward Validation Logic ---
def walk_forward_validation(symbols, data_dir, best_params, n_splits=3, min_train_size=1000, test_size=500, verbose=True, starting_cash=25000, csv_filename=CSV_FILENAME):
    fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh = best_params
    run_backtest_param_split(symbols, data_dir, fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, n_splits=n_splits, min_train_size=min_train_size, test_size=test_size, verbose=verbose, starting_cash=starting_cash, csv_filename=csv_filename)

# --- Grid Search for Parameter Optimization ---
from itertools import product

def grid_search(symbols, data_dir, summary_csv='trend_grid_search_summary.csv', top_n_walk=5, run_walk_forward=True, csv_filename=CSV_FILENAME):
    from itertools import product
    import time
    fast_ma_range = [10, 20, 30]
    slow_ma_range = [50, 100, 200]
    sl_mult_range = [1.0, 1.5]
    tp_mult_range = [1.0, 1.5, 2.0]
    atr_thresh_range = [1.0, 2.0, 3.0, 5.0, 7.0]
    best_pnl = -np.inf
    best_params = None
    results = []
    summary_fields = ['fast_ma','slow_ma','sl_mult','tp_mult','atr_thresh','pnl','sharpe','max_drawdown','win_rate']
    param_combos = list(product(fast_ma_range, slow_ma_range, sl_mult_range, tp_mult_range, atr_thresh_range))
    total = len(param_combos)
    start_time = time.time()
    with open(summary_csv, 'w', newline='') as f:
        for i, (fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh) in enumerate(param_combos):
            # Ensure there are enough bars for the moving average calculation
            min_bars = max(fast_ma, slow_ma)
            df_path = os.path.join(data_dir, csv_filename)
            if not os.path.exists(df_path):
                continue
            df = pd.read_csv(df_path)
            if len(df) < min_bars:
                continue  # skip this combo if not enough data
            try:
                pnl, sharpe, max_dd, win_rate = run_backtest_param_metrics(symbols, data_dir, fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, csv_filename=csv_filename)
            except Exception as e:
                print(f"[SKIP] fast_ma={fast_ma}, slow_ma={slow_ma}, sl_mult={sl_mult}, tp_mult={tp_mult}, atr_thresh={atr_thresh} | Error: {e}")
                continue
            results.append([fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, pnl, sharpe, max_dd, win_rate])
            elapsed = time.time() - start_time
            pct = (i+1)/total*100
            print(f"[{i+1}/{total} | {pct:.1f}% | {elapsed:.1f}s] Tested: fast_ma={fast_ma}, slow_ma={slow_ma}, sl_mult={sl_mult}, tp_mult={tp_mult}, atr_thresh={atr_thresh} | PnL={pnl:.2f} Sharpe={sharpe:.2f} DD={max_dd:.2f} Win%={win_rate:.2f}")
        else:
            print("\nNo valid trades for any parameter combination.")
        # Optionally run walk-forward validation on top N sets
        if run_walk_forward:
            print(f"\nRunning walk-forward validation on top {top_n_walk} parameter sets...")
            for row in results[:top_n_walk]:
                params = row[:5]
                print(f"Walk-forward for params: {params}")
                walk_forward_validation(symbols, data_dir, params, n_splits=3, min_train_size=1000, test_size=500, verbose=True, csv_filename=csv_filename)
    df = pd.DataFrame(results, columns=summary_fields)
    df.to_csv(summary_csv, index=False)
    return best_params

# Helper to run backtest and return key metrics

def run_backtest_param_metrics(symbols, data_dir, fast_ma, slow_ma, sl_mult, tp_mult, atr_thresh, csv_filename=CSV_FILENAME):
    perf = PerformanceAnalyzer()
    for symbol in symbols:
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        required_cols = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            continue
        bars = [Bar(row['date'], row['open'], row['high'], row['low'], row['close'], row['volume']) for idx, row in df.iterrows()]
        closes, highs, lows = [], [], []
        order_manager = OrderManager(perf, sl_mult=sl_mult, tp_mult=tp_mult)
        for bar in bars:
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < slow_ma or len(closes) < 15:
                continue
            close_arr = np.array(closes[-slow_ma:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            # Defensive: all ATR arrays must be length 15
            if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                continue
            import talib
            fast = talib.SMA(close_arr[-fast_ma:], timeperiod=fast_ma)[-1]
            slow = talib.SMA(close_arr, timeperiod=slow_ma)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            if atr >= atr_thresh:
                continue
            if order_manager.position == 0:
                if fast > slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='long')
                elif fast < slow:
                    order_manager.evaluate_entry(symbol, bar, atr, direction='short')
            elif order_manager.position == 1 and fast < slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossDown')
            elif order_manager.position == -1 and fast > slow:
                order_manager.close_position(symbol, bar, exit_reason='CrossUp')
            order_manager.check_exit(symbol, bar)
    pnl = perf.trade_log['pnl'].sum() if not perf.trade_log.empty else -np.inf
    returns = perf.trade_log['pnl'] / perf.trade_log['entry_price'] if not perf.trade_log.empty else pd.Series([0])
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78) if returns.std() != 0 else 0
    equity = perf.trade_log['pnl'].cumsum() + 25000
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd = drawdown.min() if not perf.trade_log.empty else 0
    win_rate = (perf.trade_log['pnl'] > 0).mean() * 100 if not perf.trade_log.empty else 0
    return pnl, sharpe, max_dd, win_rate

SHOW_BUY_AND_HOLD_OVERLAY = False

def show_buy_and_hold():
    global SHOW_BUY_AND_HOLD_OVERLAY
    SHOW_BUY_AND_HOLD_OVERLAY = True

if __name__ == "__main__":
    symbols = ['SPY']
    data_dir = os.getcwd()
    # Set your data file here:
    csv_filename = CSV_FILENAME
    # Example: enable buy and hold overlay
    #show_buy_and_hold()
    run_backtest(symbols, data_dir, fast_ma=30, slow_ma=200, sl_mult=1.5, tp_mult=2.0, atr_thresh= 1.0, csv_filename=csv_filename, overlay_buy_and_hold=SHOW_BUY_AND_HOLD_OVERLAY)
    # Example: run param split
    #run_backtest_param_split(symbols, data_dir, 15, 50, 1.0, 1.0, 3.0, csv_filename=csv_filename)
    # Example: run param metrics
    #run_backtest_param_metrics(symbols, data_dir, 15, 50, 1.0, 1.0, 3.0, csv_filename=csv_filename)
    # Example: run grid search (uncomment to run)
    #grid_search(symbols, data_dir, csv_filename=csv_filename)
