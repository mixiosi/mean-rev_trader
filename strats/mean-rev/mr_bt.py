import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

from bb_ml_stable3 import SignalFilter

# --- USER CONFIGURABLE DATA FILE ---
# Set this variable to the filename of your historical data CSV (must be in data_dir)
CSV_FILENAME = "../../data/SPY_1hour_bar_20Y_historical_data.csv"

# --- PARAMETERS FOR TUNING ---
RSI_ENTRY = 15   # Entry threshold for RSI(2)
RSI_EXIT = 60   # Exit threshold for RSI(2)
SL_MULT = 1.0    # Stop-loss multiple of ATR
TP_MULT = 1.0    # Take-profit multiple of ATR
ATR_THRESHOLD = 5.0 # Volatility filter threshold

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
    def __init__(self, starting_cash=25000, commission=1.0, slippage_bps=1):
        self.trade_log = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'symbol', 'direction',
            'quantity', 'entry_price', 'exit_price', 'pnl', 'exit_reason', 'commission', 'slippage'])
        self.starting_cash = starting_cash
        self.commission = commission
        self.slippage_bps = slippage_bps  # basis points (1 bps = 0.01%)

    def log_trade(self, entry_time, exit_time, symbol, direction, quantity, entry_price, exit_price, exit_reason=None):
        # Only add non-empty, non-all-NA trades to the log to avoid FutureWarning
        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': (exit_price - entry_price) * quantity - self.commission * 2 - (entry_price + exit_price) * 0.5 * self.slippage_bps / 10000 * quantity,
            'exit_reason': exit_reason,
            'commission': self.commission * 2,
            'slippage': (entry_price + exit_price) * 0.5 * self.slippage_bps / 10000 * quantity
        }])
        if not new_trade.isna().all(axis=None):
            self.trade_log = pd.concat([
                self.trade_log,
                new_trade
            ], ignore_index=True)

    def generate_report(self):
        if self.trade_log.empty:
            print("No trades to report.")
            return

        # Plot: Per-symbol equity curve (SPY only)
        plt.figure(figsize=(12, 6))
        for symbol in self.trade_log['symbol'].unique():
            symbol_trades = self.trade_log[self.trade_log['symbol'] == symbol]
            symbol_trades = symbol_trades.sort_values('exit_time')
            equity = symbol_trades['pnl'].cumsum() + self.starting_cash
            plt.plot(pd.to_datetime(symbol_trades['exit_time']),
                    equity,
                    label=symbol)
        plt.title('Strategy Equity Curve (SPY)')
        plt.xlabel('Date')
        plt.ylabel('Equity ($)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # --- Analytics ---
        pnl = self.trade_log['pnl']
        returns = pnl / self.trade_log['entry_price']
        sharpe = np.nan
        if len(returns.dropna()) > 1:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78)
        cum_pnl = pnl.cumsum()
        equity = cum_pnl + self.starting_cash
        roll_max = equity.cummax()
        drawdown = equity - roll_max
        max_dd = drawdown.min()
        max_dd_pct = (max_dd / self.starting_cash) * 100
        wins = (pnl > 0).sum()
        total = len(pnl)
        win_rate = wins / total if total > 0 else np.nan
        print("\n--- Performance Analytics ---")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd:.2f} ({max_dd_pct:.2f}%)")
        print(f"Win Rate: {win_rate:.2%}")

        csv_path = os.path.join(os.getcwd(), 'mr_trade_log.csv')
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
            # Calculate position size to risk 1% of equity per trade
            equity = self.app.trade_log['pnl'].cumsum().iloc[-1] + self.app.starting_cash if not self.app.trade_log.empty else self.app.starting_cash
            risk_per_trade = equity * 0.01
            stop_dist = abs(bar.close - (bar.close - SL_MULT * atr))
            quantity = max(int(risk_per_trade / stop_dist), 1) if stop_dist > 0 else 1
            self.position = 1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'BUY'
            self.quantity = quantity
            self.stop_loss = bar.close - SL_MULT * atr
            self.take_profit = bar.close + TP_MULT * atr
            self.bars_held = 0
            # print(f"[ORDERMANAGER] ENTER LONG: {bar.date} | Price={bar.close} | SL={self.stop_loss} | TP={self.take_profit}")
        else:
            pass  # print(f"[ORDERMANAGER] SKIP LONG ENTRY: Already in position")

    def evaluate_short_entry(self, symbol, bar, atr):
        if self.position == 0:
            # Calculate position size to risk 1% of equity per trade
            equity = self.app.trade_log['pnl'].cumsum().iloc[-1] + self.app.starting_cash if not self.app.trade_log.empty else self.app.starting_cash
            risk_per_trade = equity * 0.01
            stop_dist = abs(bar.close - (bar.close + SL_MULT * atr))
            quantity = max(int(risk_per_trade / stop_dist), 1) if stop_dist > 0 else 1
            self.position = -1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'SELL'
            self.quantity = quantity
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
                    exit_reason = 'StopLoss'
                elif bar.high >= self.take_profit:
                    exit_reason = 'TakeProfit'
            elif self.position == -1:
                if bar.high >= self.stop_loss:
                    exit_reason = 'StopLoss'
                elif bar.low <= self.take_profit:
                    exit_reason = 'TakeProfit'
            # Time-based exit (after 10 bars for demo)
            if self.bars_held >= 10:
                exit_reason = 'TIME'
            if exit_reason is not None:
                # print(f"[ORDERMANAGER] EXIT LONG: {bar.date} | Price={bar.close} | Reason={exit_reason}")
                self.close_position(symbol, bar, exit_reason=exit_reason)

    def close_position(self, symbol, bar, exit_reason=None):
        if self.position == 0:
            # print(f"[ORDERMANAGER] CLOSE: No open position to close at {bar.date}")
            return
        # print(f"[ORDERMANAGER] CLOSE POSITION: {bar.date} | Price={bar.close} | Reason={exit_reason}")
        pa = getattr(self.app, 'performance_analyzer', None)
        if pa is not None:
            pa.log_trade(
                self.entry_time, bar.date, symbol, self.direction, self.quantity, self.entry_price, bar.close, exit_reason=exit_reason)
        elif hasattr(self.app, 'log_trade'):
            self.app.log_trade(
                self.entry_time, bar.date, symbol, self.direction, self.quantity, self.entry_price, bar.close, exit_reason=exit_reason)
        else:
            raise AttributeError('No suitable log_trade method found on app or app.performance_analyzer')
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
import csv
import time

def grid_search(symbols, data_dir, summary_csv='gs_summary.csv', top_n_walk=5, run_walk_forward=True, csv_filename=CSV_FILENAME):
    from itertools import product
    rsi_entry_range = list(range(5, 31, 5))  # 5, 10, 15, 20, 25, 30
    rsi_exit_range = list(range(40, 81, 10)) # 40, 50, 60, 70, 80, 90
    sl_mult_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    tp_mult_range = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    sma_len_range = [20, 30, 50, 75, 100]
    atr_thresh_range = [1.0, 2.0, 3.0, 5.0]
    best_pnl = -np.inf
    best_params = None
    results = []
    summary_fields = ['rsi_entry','rsi_exit','sl_mult','tp_mult','sma_len','atr_thresh','pnl','sharpe','max_drawdown','win_rate']
    param_combos = list(product(rsi_entry_range, rsi_exit_range, sl_mult_range, tp_mult_range, sma_len_range, atr_thresh_range))
    total = len(param_combos)
    start_time = time.time()
    with open(summary_csv, 'w', newline='') as f:
        for i, (rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh) in enumerate(param_combos):
            pnl, sharpe, max_dd, win_rate = run_backtest_param_metrics(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, csv_filename=csv_filename)
            results.append([rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, pnl, sharpe, max_dd, win_rate])
            if pnl > best_pnl:
                best_pnl = pnl
                best_params = (rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh)
            elapsed = time.time() - start_time
            pct = (i+1)/total*100
            print(f"[{i+1}/{total} | {pct:.1f}% | {elapsed:.1f}s] Tested: rsi_entry={rsi_entry}, rsi_exit={rsi_exit}, sl_mult={sl_mult}, tp_mult={tp_mult}, sma_len={sma_len}, atr_thresh={atr_thresh} | PnL={pnl:.2f} Sharpe={sharpe:.2f} DD={max_dd:.2f} Win%={win_rate:.2f}")
        else:
            print("\nNo valid trades for any parameter combination.")
        # Optionally run walk-forward validation on top N sets
        if run_walk_forward:
            print(f"\nRunning walk-forward validation on top {top_n_walk} parameter sets...")
            for row in results[:top_n_walk]:
                params = row[:6]
                print(f"Walk-forward for params: {params}")
                walk_forward_validation(symbols, data_dir, params, n_splits=3, min_train_size=1000, test_size=500, verbose=True, csv_filename=csv_filename)
    df = pd.DataFrame(results, columns=summary_fields)
    df.to_csv(summary_csv, index=False)
    return best_params

# Helper to run backtest and return key metrics

def run_backtest_param_metrics(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, csv_filename=CSV_FILENAME):
    # This function should run the backtest and return (pnl, sharpe, max_drawdown, win_rate)
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
        order_manager = MockOrderManager(perf)
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < sma_len or len(closes) < 15:
                continue
            import talib
            close_arr = np.array(closes[-sma_len:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            # Defensive: all ATR arrays must be length 15
            if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                continue
            rsi2 = talib.RSI(close_arr, timeperiod=2)[-1]
            sma = pd.Series(close_arr).rolling(window=sma_len).mean().iloc[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            if bar.close > sma and atr < atr_thresh:
                if order_manager.position == 0 and rsi2 < rsi_entry:
                    order_manager.evaluate_long_entry(symbol, bar, atr)
            elif order_manager.position == 1 and rsi2 > rsi_exit:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
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

# --- Walk-Forward Validation ---
import warnings
def walk_forward_validation(symbols, data_dir, param_set, n_splits=5, min_train_size=1000, test_size=500, verbose=True, csv_filename=CSV_FILENAME):
    import copy
    rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh = param_set
    all_logs = []
    for symbol in symbols:
        if symbol != 'SPY':
            continue
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Data file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        n_bars = len(df)
        print(f"Symbol: {symbol}, Total Bars: {n_bars}")
        start = 0
        split = 0
        while start + min_train_size + test_size <= n_bars and split < n_splits:
            train_end = start + min_train_size
            test_end = train_end + test_size
            df_train = df.iloc[start:train_end].copy()
            df_test = df.iloc[train_end:test_end].copy()
            print(f"\nSplit {split}: Train rows {start}-{train_end-1} ({len(df_train)} bars), Test rows {train_end}-{test_end-1} ({len(df_test)} bars)")
            print(f"  Train Start: {df_train.iloc[0]['date'] if not df_train.empty else 'N/A'} | End: {df_train.iloc[-1]['date'] if not df_train.empty else 'N/A'}")
            print(f"  Test Start: {df_test.iloc[0]['date'] if not df_test.empty else 'N/A'} | End: {df_test.iloc[-1]['date'] if not df_test.empty else 'N/A'}")
            temp_train_path = os.path.join(data_dir, f"{symbol}_walk_train_{split}.csv")
            temp_test_path = os.path.join(data_dir, f"{symbol}_walk_test_{split}.csv")
            df_train.to_csv(temp_train_path, index=False)
            df_test.to_csv(temp_test_path, index=False)
            # Run in-sample (train)
            train_pnl = run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, verbose=False, commission=1.0, slippage_bps=1, csv_override=None, debug_signals=(split==0), debug_split_id=f"train{split}", csv_filename=csv_filename)
            train_log_path = os.path.join(data_dir, f"{symbol}_walk_train_trades_{split}.csv")
            if os.path.exists('mr_trade_log.csv'):
                os.replace('mr_trade_log.csv', train_log_path)
            # Run out-of-sample (test)
            test_pnl = run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, verbose=False, commission=1.0, slippage_bps=1, csv_override=None, debug_signals=(split==0), debug_split_id=f"test{split}", csv_filename=csv_filename)
            test_log_path = os.path.join(data_dir, f"{symbol}_walk_test_trades_{split}.csv")
            if os.path.exists('mr_trade_log.csv'):
                os.replace('mr_trade_log.csv', test_log_path)
            train_log = pd.read_csv(train_log_path) if os.path.exists(train_log_path) else pd.DataFrame()
            test_log = pd.read_csv(test_log_path) if os.path.exists(test_log_path) else pd.DataFrame()
            print(f"  Train Trades: {len(train_log)} | Test Trades: {len(test_log)}")
            if len(train_log) == 0:
                print("    [DEBUG] No trades in train split. Check if signals triggered.")
            if len(test_log) == 0:
                print("    [DEBUG] No trades in test split. Check if signals triggered.")
            all_logs.append({'split': split, 'train_log': train_log, 'test_log': test_log})
            if verbose:
                print(f"Split {split}: Train PnL={train_pnl:.2f}, Test PnL={test_pnl:.2f}, Train Trades={len(train_log)}, Test Trades={len(test_log)}")
            os.remove(temp_train_path)
            os.remove(temp_test_path)
            start = train_end
            split += 1
    if verbose and all_logs:
        test_pnls = [log['test_log']['pnl'].sum() for log in all_logs if not log['test_log'].empty]
        test_trade_counts = [len(log['test_log']) for log in all_logs]
        train_trade_counts = [len(log['train_log']) for log in all_logs]
        print(f"\nWalk-Forward Validation Summary (Test Sets):")
        print(f"Splits: {len(test_pnls)}")
        print(f"Mean Test PnL: {np.mean(test_pnls):.2f}")
        print(f"Median Test PnL: {np.median(test_pnls):.2f}")
        print(f"Std Test PnL: {np.std(test_pnls):.2f}")
        print(f"All Test PnLs: {test_pnls}")
        print(f"Train Trades per Split: {train_trade_counts}")
        print(f"Test Trades per Split: {test_trade_counts}")
        print(f"Total Train Trades: {sum(train_trade_counts)} | Total Test Trades: {sum(test_trade_counts)}")
    return all_logs

# --- Main Backtest Logic ---
def run_backtest(symbols, data_dir, model_path=None, starting_cash=25000, csv_filename=CSV_FILENAME):
    perf = PerformanceAnalyzer(starting_cash=starting_cash)
    for symbol in symbols:
        if symbol != 'SPY':
            continue  # Only trade SPY for this strategy
        print(f"Backtesting {symbol}...")
        csv_path = os.path.join(data_dir, csv_filename)
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
        # We'll store signals for each bar, but only act on them at the NEXT bar
        signals = []  # Each element: dict with keys 'long_entry', 'long_exit', 'atr', 'sma50', 'rsi2', 'bar'
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < 50:
                signals.append(None)
                continue
            import talib
            close_arr = np.array(closes[-50:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            # Defensive: all ATR arrays must be length 15
            if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                continue
            rsi2 = talib.RSI(close_arr[-15:], timeperiod=2)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            sma = pd.Series(close_arr).rolling(window=50).mean().iloc[-1]
            if bar.close > sma and atr < ATR_THRESHOLD:
                if order_manager.position == 0 and rsi2 < RSI_ENTRY:
                    order_manager.evaluate_long_entry(symbol, bar, atr)
            elif order_manager.position == 1 and rsi2 > RSI_EXIT:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            order_manager.check_exit(symbol, bar)
    if perf.trade_log.empty:
        return
    perf.generate_report()

def run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, verbose=False, commission=1.0, slippage_bps=1, csv_override=None, debug_signals=False, debug_split_id=None, csv_filename=CSV_FILENAME):
    perf = PerformanceAnalyzer(commission=commission, slippage_bps=slippage_bps)
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
        order_manager = MockOrderManager(perf)
        for bar in bars:
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < sma_len or len(closes) < 15:
                continue
            import talib
            close_arr = np.array(closes[-sma_len:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            # Defensive: all ATR arrays must be length 15
            if len(high_arr) < 15 or len(low_arr) < 15 or len(close_arr[-15:]) < 15:
                continue
            rsi2 = talib.RSI(close_arr, timeperiod=2)[-1]
            sma = pd.Series(close_arr).rolling(window=sma_len).mean().iloc[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            if bar.close > sma and atr < atr_thresh:
                if order_manager.position == 0 and rsi2 < rsi_entry:
                    order_manager.evaluate_long_entry(symbol, bar, atr)
            elif order_manager.position == 1 and rsi2 > rsi_exit:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            order_manager.check_exit(symbol, bar)
    if perf.trade_log.empty:
        return -np.inf
    if csv_override:
        perf.trade_log.to_csv(csv_override, index=False)
    if verbose:
        perf.generate_report()
    pnl = perf.trade_log['pnl'].sum()
    return pnl

def grid_search(symbols, data_dir, summary_csv='gs_summary.csv', csv_filename=CSV_FILENAME):
    from itertools import product
    import time
    lookback_range = [10, 20, 30]
    sl_mult_range = [1.0, 1.5]
    tp_mult_range = [1.0, 1.5, 2.0]
    atr_thresh_range = [1.0, 2.0]
    summary_fields = ['lookback','sl_mult','tp_mult','atr_thresh','pnl','sharpe','max_drawdown','win_rate']
    param_combos = list(product(lookback_range, sl_mult_range, tp_mult_range, atr_thresh_range))
    total = len(param_combos)
    start_time = time.time()
    with open(summary_csv, 'w', newline='') as f:
        results = []
        for i, (lookback, sl_mult, tp_mult, atr_thresh) in enumerate(param_combos):
            pnl, sharpe, max_dd, win_rate = run_meanrev_backtest(symbols, data_dir, lookback, sl_mult, tp_mult, atr_thresh, csv_filename=csv_filename)
            results.append([lookback, sl_mult, tp_mult, atr_thresh, pnl, sharpe, max_dd, win_rate])
            elapsed = time.time() - start_time
            pct = (i+1)/total*100
            print(f"[{i+1}/{total} | {pct:.1f}% | {elapsed:.1f}s] Tested: lookback={lookback}, sl_mult={sl_mult}, tp_mult={tp_mult}, atr_thresh={atr_thresh} | PnL={pnl:.2f} Sharpe={sharpe:.2f} DD={max_dd:.2f} Win%={win_rate:.2f}")
        df = pd.DataFrame(results, columns=summary_fields)
        df.to_csv(summary_csv, index=False)
        print(f"Grid search complete. Results saved to {summary_csv}.")

if __name__ == "__main__":
    symbols = ['SPY']
    data_dir = os.getcwd()
    csv_filename = CSV_FILENAME
    # Example: run backtest
    run_backtest(symbols, data_dir, csv_filename=csv_filename)
    # Example: run param
    run_backtest_param(symbols, data_dir, 15, 60, 1.0, 1.0, 50, 5.0, csv_filename=csv_filename)
    # Example: run param metrics
    run_backtest_param_metrics(symbols, data_dir, 15, 60, 1.0, 1.0, 50, 5.0, csv_filename=csv_filename)
    # Example walk-forward validation run (uncomment to use):
    #best_params = (10, 50, 1.5, 1.0, 30, 5.0)
    #walk_forward_validation(symbols, data_dir, best_params, n_splits=3, min_train_size=1000, test_size=500, verbose=True, csv_filename=csv_filename)

    grid_search(symbols, data_dir, csv_filename=csv_filename)
