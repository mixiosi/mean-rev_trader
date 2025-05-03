import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# --- USER CONFIGURABLE DATA FILE ---
# Set this variable to the filename of your historical data CSV (must be in data_dir)
CSV_FILENAME = "../../data/SPY_1hour_bar_20Y_historical_data.csv"

# --- PARAMETERS FOR TUNING ---
LOOKBACK_RANGE = [5, 10, 15, 20, 30, 50, 100, 150]
SL_MULT_RANGE = [0.5, 1.0, 1.5, 2.0]
TP_MULT_RANGE = [0.5, 1.0, 1.5, 2.0]
ATR_THRESH_RANGE = [1.0, 2.0, 3.0, 5.0]

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
    def __init__(self, starting_cash=25000, commission=1.0, slippage_bps=1):
        self.trade_log = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'symbol', 'direction',
            'quantity', 'entry_price', 'exit_price', 'pnl', 'exit_reason', 'commission', 'slippage'])
        self.starting_cash = starting_cash
        self.commission = commission
        self.slippage_bps = slippage_bps
    def log_trade(self, entry_time, exit_time, symbol, direction, quantity, entry_price, exit_price, exit_reason=None):
        slippage = (entry_price + exit_price) * 0.5 * self.slippage_bps / 10000 * quantity
        commission = self.commission * 2
        if direction == 'BUY':
            pnl = (exit_price - entry_price) * quantity - commission - slippage
        else:
            pnl = (entry_price - exit_price) * quantity - commission - slippage
        new_trade = pd.DataFrame([{
            'entry_time': entry_time, 'exit_time': exit_time, 'symbol': symbol, 'direction': direction,
            'quantity': quantity, 'entry_price': entry_price, 'exit_price': exit_price, 'pnl': pnl,
            'exit_reason': exit_reason, 'commission': commission, 'slippage': slippage
        }])
        # Only add non-empty, non-all-NA trades to the log to avoid FutureWarning
        if not new_trade.empty and new_trade.notna().any().any():
            self.trade_log = pd.concat([self.trade_log, new_trade], ignore_index=True)
    def generate_report(self):
        if self.trade_log.empty:
            print("No trades to report.")
            return
        plt.figure(figsize=(12, 6))
        for symbol in self.trade_log['symbol'].unique():
            symbol_trades = self.trade_log[self.trade_log['symbol'] == symbol]
            symbol_trades = symbol_trades.sort_values('exit_time')
            equity = symbol_trades['pnl'].cumsum() + self.starting_cash
            plt.plot(pd.to_datetime(symbol_trades['exit_time']), equity, label=symbol)
        plt.title('Breakout Strategy Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.legend()
        plt.show()
        pnl = self.trade_log['pnl']
        returns = pnl / self.trade_log['entry_price']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78) if returns.std() != 0 else 0
        equity = pnl.cumsum() + 25000
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
        csv_path = os.path.join(os.getcwd(), 'bo_trade_log.csv')
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
            stop_dist = abs(bar.close - (bar.close - self.app.sl_mult * atr))
            quantity = max(int(risk_per_trade / stop_dist), 1) if stop_dist > 0 else 1
            self.position = 1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'BUY'
            self.quantity = quantity
            self.stop_loss = bar.close - self.app.sl_mult * atr
            self.take_profit = bar.close + self.app.tp_mult * atr
            self.bars_held = 0
    def evaluate_short_entry(self, symbol, bar, atr):
        if self.position == 0:
            # Calculate position size to risk 1% of equity per trade
            equity = self.app.trade_log['pnl'].cumsum().iloc[-1] + self.app.starting_cash if not self.app.trade_log.empty else self.app.starting_cash
            risk_per_trade = equity * 0.01
            stop_dist = abs(bar.close - (bar.close + self.app.sl_mult * atr))
            quantity = max(int(risk_per_trade / stop_dist), 1) if stop_dist > 0 else 1
            self.position = -1
            self.entry_price = bar.close
            self.entry_time = bar.date
            self.direction = 'SELL'
            self.quantity = quantity
            self.stop_loss = bar.close + self.app.sl_mult * atr
            self.take_profit = bar.close - self.app.tp_mult * atr
            self.bars_held = 0
    def close_position(self, symbol, bar, exit_reason=None):
        if self.position == 0:
            return
        self.app.log_trade(
            self.entry_time, bar.date, symbol, self.direction, self.quantity, self.entry_price, bar.close, exit_reason=exit_reason)
        self.position = 0
        self.entry_price = None
        self.entry_time = None
        self.direction = None
        self.quantity = 0
        self.stop_loss = None
        self.take_profit = None
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
                self.close_position(symbol, bar, exit_reason=exit_reason)

# --- Backtest Logic ---
def run_breakout_backtest(symbols, data_dir, lookback, sl_mult, tp_mult, atr_thresh, verbose=False, csv_filename=CSV_FILENAME):
    perf = PerformanceAnalyzer()
    perf.sl_mult = sl_mult
    perf.tp_mult = tp_mult
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
        closes, highs, lows = [], [], []
        order_manager = MockOrderManager(perf)
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < lookback or len(closes) < 15:
                continue
            close_arr = np.array(closes[-lookback:])
            high_arr = np.array(highs[-lookback:])
            low_arr = np.array(lows[-lookback:])
            atr_arr_high = np.array(highs[-15:])
            atr_arr_low = np.array(lows[-15:])
            atr_arr_close = np.array(closes[-15:])
            if len(atr_arr_high) < 15 or len(atr_arr_low) < 15 or len(atr_arr_close) < 15:
                continue
            import talib
            atr = talib.ATR(atr_arr_high, atr_arr_low, atr_arr_close, timeperiod=14)[-1]
            # Breakout logic: Enter long if price breaks above highest high of lookback
            if len(high_arr[:-1]) == 0 or len(low_arr[:-1]) == 0:
                continue  # skip if not enough bars for breakout calculation
            breakout_high = np.max(high_arr[:-1])
            breakout_low = np.min(low_arr[:-1])
            if bar.close > breakout_high and atr > atr_thresh:
                order_manager.evaluate_long_entry(symbol, bar, atr)
            elif bar.close < breakout_low and atr > atr_thresh:
                order_manager.evaluate_short_entry(symbol, bar, atr)
            order_manager.check_exit(symbol, bar)
    if perf.trade_log.empty:
        return -np.inf, 0, 0, 0
    pnl = perf.trade_log['pnl'].sum()
    returns = perf.trade_log['pnl'] / perf.trade_log['entry_price']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252*78) if returns.std() != 0 else 0
    equity = perf.trade_log['pnl'].cumsum() + 25000
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    max_dd = drawdown.min()
    win_rate = (perf.trade_log['pnl'] > 0).mean() * 100
    if verbose:
        perf.generate_report()
    return pnl, sharpe, max_dd, win_rate

# --- Split/Walk-Forward Backtest Logic ---
def run_backtest_split(symbols, data_dir, lookback, sl_mult, tp_mult, atr_thresh, n_splits=3, min_train_size=1000, test_size=500, verbose=True, starting_cash=25000, csv_filename=CSV_FILENAME):
    import math
    for symbol in symbols:
        csv_path = os.path.join(data_dir, csv_filename)
        if not os.path.exists(csv_path):
            print(f"Data file missing: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        # Ensure date column is datetime and sort by date
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        n_bars = len(df)
        print(f"Symbol: {symbol}, Total Bars: {n_bars}")
        split_starts = []
        train_logs, test_logs = [], []
        split_idx = 0
        start = 0
        while True:
            train_start = split_idx * test_size
            train_end = train_start + min_train_size
            test_start = train_end
            test_end = test_start + test_size
            if test_end > n_bars:
                break
            # Print date ranges for each split
            train_dates = (df.iloc[train_start]['date'], df.iloc[train_end-1]['date'])
            test_dates = (df.iloc[test_start]['date'], df.iloc[test_end-1]['date'])
            print(f"Split {split_idx}: Train {train_start}-{train_end-1} ({train_dates[0].date()} to {train_dates[1].date()}), "
                  f"Test {test_start}-{test_end-1} ({test_dates[0].date()} to {test_dates[1].date()})")
            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]
            train_perf = PerformanceAnalyzer(starting_cash=starting_cash)
            train_perf.sl_mult = sl_mult
            train_perf.tp_mult = tp_mult
            order_manager = MockOrderManager(train_perf)
            high_arr = train_df['high'].values
            low_arr = train_df['low'].values
            close_arr = train_df['close'].values
            for i in range(lookback, len(train_df)):
                bar = Bar(train_df.iloc[i]['date'], train_df.iloc[i]['open'], train_df.iloc[i]['high'], train_df.iloc[i]['low'], train_df.iloc[i]['close'], train_df.iloc[i]['volume'])
                atr_arr_high = high_arr[i-lookback:i+1]
                atr_arr_low = low_arr[i-lookback:i+1]
                atr_arr_close = close_arr[i-lookback:i+1]
                import talib
                atr = talib.ATR(atr_arr_high, atr_arr_low, atr_arr_close, timeperiod=14)[-1]
                if len(high_arr[i-lookback:i]) == 0 or len(low_arr[i-lookback:i]) == 0:
                    continue
                breakout_high = np.max(high_arr[i-lookback:i])
                breakout_low = np.min(low_arr[i-lookback:i])
                if bar.close > breakout_high and atr > atr_thresh:
                    order_manager.evaluate_long_entry(symbol, bar, atr)
                elif bar.close < breakout_low and atr > atr_thresh:
                    order_manager.evaluate_short_entry(symbol, bar, atr)
                order_manager.check_exit(symbol, bar)
            train_log = train_perf.trade_log.copy()
            train_log['split'] = split_idx
            train_log['set'] = 'train'
            train_logs.append(train_log)
            test_perf = PerformanceAnalyzer(starting_cash=starting_cash)
            test_perf.sl_mult = sl_mult
            test_perf.tp_mult = tp_mult
            test_order_manager = MockOrderManager(test_perf)
            high_arr = test_df['high'].values
            low_arr = test_df['low'].values
            close_arr = test_df['close'].values
            for i in range(lookback, len(test_df)):
                bar = Bar(test_df.iloc[i]['date'], test_df.iloc[i]['open'], test_df.iloc[i]['high'], test_df.iloc[i]['low'], test_df.iloc[i]['close'], test_df.iloc[i]['volume'])
                atr_arr_high = high_arr[i-lookback:i+1]
                atr_arr_low = low_arr[i-lookback:i+1]
                atr_arr_close = close_arr[i-lookback:i+1]
                import talib
                atr = talib.ATR(atr_arr_high, atr_arr_low, atr_arr_close, timeperiod=14)[-1]
                if len(high_arr[i-lookback:i]) == 0 or len(low_arr[i-lookback:i]) == 0:
                    continue
                breakout_high = np.max(high_arr[i-lookback:i])
                breakout_low = np.min(low_arr[i-lookback:i])
                if bar.close > breakout_high and atr > atr_thresh:
                    test_order_manager.evaluate_long_entry(symbol, bar, atr)
                elif bar.close < breakout_low and atr > atr_thresh:
                    test_order_manager.evaluate_short_entry(symbol, bar, atr)
                test_order_manager.check_exit(symbol, bar)
            test_log = test_perf.trade_log.copy()
            test_log['split'] = split_idx
            test_log['set'] = 'test'
            test_logs.append(test_log)
            # Save logs
            train_log.to_csv(os.path.join(data_dir, f"bo_train_trades_split{split_idx}.csv"), index=False)
            test_log.to_csv(os.path.join(data_dir, f"bo_test_trades_split{split_idx}.csv"), index=False)
            split_idx += 1
        if verbose:
            print(f"\nWalk-Forward Validation Summary (Test Sets):")
            test_pnls = [log['pnl'].sum() for log in test_logs if not log.empty]
            print(f"Splits: {len(test_pnls)}")
            print(f"Mean Test PnL: {np.mean(test_pnls):.2f}")
            print(f"Median Test PnL: {np.median(test_pnls):.2f}")
            print(f"Std Test PnL: {np.std(test_pnls):.2f}")
            print(f"All Test PnLs: {test_pnls}")
            print(f"Train Trades per Split: {[len(log) for log in train_logs]}")
            print(f"Test Trades per Split: {[len(log) for log in test_logs]}")
            print(f"Total Train Trades: {sum(len(log) for log in train_logs)} | Total Test Trades: {sum(len(log) for log in test_logs)}")
    return train_logs, test_logs

def walk_forward_validation(symbols, data_dir, param_set, n_splits=3, min_train_size=1000, test_size=500, verbose=True, starting_cash=25000, csv_filename=CSV_FILENAME):
    lookback, sl_mult, tp_mult, atr_thresh = param_set
    train_logs, test_logs = run_backtest_split(
        symbols, data_dir, lookback, sl_mult, tp_mult, atr_thresh,
        n_splits=n_splits, min_train_size=min_train_size, test_size=test_size, verbose=verbose, starting_cash=starting_cash, csv_filename=csv_filename)
    # Optionally analyze logs here
    if verbose:
        print("\nWalk-Forward Validation Summary (Test Sets):")
        test_pnls = [log['pnl'].sum() for log in test_logs if not log.empty]
        print(f"Splits: {len(test_pnls)}")
        print(f"Mean Test PnL: {np.mean(test_pnls):.2f}")
        print(f"Median Test PnL: {np.median(test_pnls):.2f}")
        print(f"Std Test PnL: {np.std(test_pnls):.2f}")
        print(f"All Test PnLs: {test_pnls}")
        print(f"Train Trades per Split: {[len(log) for log in train_logs]}")
        print(f"Test Trades per Split: {[len(log) for log in test_logs]}")
        print(f"Total Train Trades: {sum(len(log) for log in train_logs)} | Total Test Trades: {sum(len(log) for log in test_logs)}")
    return train_logs, test_logs

# --- Grid Search for Parameter Optimization ---
def grid_search(symbols, data_dir, summary_csv='bo_gs_summary.csv', csv_filename=CSV_FILENAME):
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
            pnl, sharpe, max_dd, win_rate = run_breakout_backtest(symbols, data_dir, lookback, sl_mult, tp_mult, atr_thresh, csv_filename=csv_filename)
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

    # run grid search (uncomment to run)
    grid_search(symbols, data_dir, csv_filename=csv_filename)

    # walk-forward validation
    #param_set = (20, 1.0, 1.0, 2.0)  # (lookback, sl_mult, tp_mult, atr_thresh)
    #walk_forward_validation(symbols, data_dir, param_set, n_splits=3, min_train_size=1000, test_size=500, verbose=True, starting_cash=25000, csv_filename=csv_filename)

    # backtest
    #run_breakout_backtest(symbols, data_dir, lookback=3, sl_mult=2.0, tp_mult=1.5, atr_thresh=1.0, verbose=True, csv_filename=csv_filename)

    # backtest split
    #run_backtest_split(symbols, data_dir, lookback=70, sl_mult=1.5, tp_mult=1.5, atr_thresh=5.0, n_splits=1, min_train_size=35500, test_size=0, verbose=True, csv_filename=csv_filename)
