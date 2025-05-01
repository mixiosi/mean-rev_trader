import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

from bb_ml_stable3 import SignalFilter

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
        # Calculate slippage: applied to both entry and exit
        slippage = (entry_price + exit_price) * 0.5 * self.slippage_bps / 10000 * quantity
        commission = self.commission * 2  # round trip
        if direction == 'BUY':
            pnl = (exit_price - entry_price) * quantity - commission - slippage
        else:
            pnl = (entry_price - exit_price) * quantity - commission - slippage
        new_trade = pd.DataFrame([{
            'entry_time': entry_time,
            'exit_time': exit_time,
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'commission': commission,
            'slippage': slippage
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

def run_backtest_param(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len=50, atr_thresh=None, verbose=False, commission=1.0, slippage_bps=1):
    global RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT, ATR_THRESHOLD
    RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT = rsi_entry, rsi_exit, sl_mult, tp_mult
    if atr_thresh is not None:
        ATR_THRESHOLD = atr_thresh
    perf = PerformanceAnalyzer(commission=commission, slippage_bps=slippage_bps)
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
        closes = []
        highs = []
        lows = []
        for i, bar in enumerate(bars):
            closes.append(bar.close)
            highs.append(bar.high)
            lows.append(bar.low)
            if len(closes) < sma_len:
                continue
            import talib
            close_arr = np.array(closes[-sma_len:])
            high_arr = np.array(highs[-15:])
            low_arr = np.array(lows[-15:])
            rsi2 = talib.RSI(close_arr[-15:], timeperiod=2)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            sma = pd.Series(close_arr).rolling(window=sma_len).mean().iloc[-1]
            if bar.close > sma and atr < ATR_THRESHOLD:
                if order_manager.position == 0 and rsi2 < RSI_ENTRY:
                    order_manager.evaluate_long_entry(symbol, bar, atr)
            elif order_manager.position == 1 and rsi2 > RSI_EXIT:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            order_manager.check_exit(symbol, bar)
    if perf.trade_log.empty:
        return -np.inf if not verbose else (None, -np.inf)
    total_pnl = perf.trade_log['pnl'].sum()
    if verbose:
        perf.generate_report()
    return total_pnl

def run_backtest_param_split(symbols, data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len=50, atr_thresh=None, verbose=False, commission=1.0, slippage_bps=1):
    global RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT, ATR_THRESHOLD
    RSI_ENTRY, RSI_EXIT, SL_MULT, TP_MULT = rsi_entry, rsi_exit, sl_mult, tp_mult
    if atr_thresh is not None:
        ATR_THRESHOLD = atr_thresh
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
        n = len(df)
        split_idx = int(n * 0.7)
        df_in = df.iloc[:split_idx]
        df_out = df.iloc[split_idx:]
        print(f"\nIn-sample: {df_in['date'].iloc[0]} to {df_in['date'].iloc[-1]}")
        print(f"Out-of-sample: {df_out['date'].iloc[0]} to {df_out['date'].iloc[-1]}")
        print("\n--- IN-SAMPLE ---")
        run_backtest_param([symbol], data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, verbose=True, commission=commission, slippage_bps=slippage_bps)
        print("\n--- OUT-OF-SAMPLE ---")
        # Save a temp CSV for out-of-sample and point to it
        temp_path = os.path.join(data_dir, f"{symbol}_temp_out_of_sample.csv")
        df_out.to_csv(temp_path, index=False)
        run_backtest_param([symbol], data_dir, rsi_entry, rsi_exit, sl_mult, tp_mult, sma_len, atr_thresh, verbose=True, commission=commission, slippage_bps=slippage_bps)
        os.remove(temp_path)

# --- Main Backtest Logic ---
def run_backtest(symbols, data_dir, model_path=None, starting_cash=25000):
    perf = PerformanceAnalyzer(starting_cash=starting_cash)
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
            rsi2 = talib.RSI(close_arr[-15:], timeperiod=2)[-1]
            atr = talib.ATR(high_arr, low_arr, close_arr[-15:], timeperiod=14)[-1]
            sma50 = pd.Series(close_arr).rolling(window=50).mean().iloc[-1]
            # Compute signals using only data up to the PREVIOUS bar
            if i == 0:
                signals.append(None)
                continue
            prev_bar = bars[i-1]
            prev_close_arr = np.array(closes[-51:-1])
            prev_high_arr = np.array(highs[-16:-1])
            prev_low_arr = np.array(lows[-16:-1])
            prev_rsi2 = talib.RSI(prev_close_arr[-15:], timeperiod=2)[-1]
            prev_atr = talib.ATR(prev_high_arr, prev_low_arr, prev_close_arr[-15:], timeperiod=14)[-1]
            prev_sma50 = pd.Series(prev_close_arr).rolling(window=50).mean().iloc[-1]
            long_entry = (prev_bar.close > prev_sma50 and prev_atr < ATR_THRESHOLD and order_manager.position == 0 and prev_rsi2 < RSI_ENTRY)
            long_exit = (order_manager.position == 1 and prev_rsi2 > RSI_EXIT)
            signals.append({'long_entry': long_entry, 'long_exit': long_exit, 'atr': prev_atr, 'sma50': prev_sma50, 'rsi2': prev_rsi2, 'bar': bar})
        # Now, execute trades based on the *previous* bar's signal, at the current bar
        for i, signal in enumerate(signals):
            if signal is None:
                continue
            bar = signal['bar']
            if signal['long_entry']:
                order_manager.evaluate_long_entry(symbol, bar, signal['atr'])
            elif signal['long_exit']:
                order_manager.close_position(symbol, bar, exit_reason='RSI')
            order_manager.check_exit(symbol, bar)
    perf.generate_report()

if __name__ == "__main__":
    symbols = ['SPY']
    data_dir = os.getcwd()
    # Example: run with transaction costs, slippage, and out-of-sample split
    run_backtest_param_split(symbols, data_dir, 15, 60, 1.0, 1.0, 50, 5.0, verbose=True, commission=1.0, slippage_bps=1)
    #run_backtest_param(symbols, data_dir, 15, 60, 1.0, 1.0, 50, 5.0, verbose=True)
    #run_backtest(symbols, data_dir, None)
