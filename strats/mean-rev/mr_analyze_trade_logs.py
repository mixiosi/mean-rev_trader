import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

# Automatically find all trade log CSVs in the current directory
trade_log_files = glob.glob("*trade*.csv")

for file in trade_log_files:
    print(f"\n--- Analyzing {file} ---")
    df = pd.read_csv(file)
    if df.empty:
        print("No trades in this file.")
        continue
    # Basic stats
    print(f"Total Trades: {len(df)}")
    print(f"Win Rate: {(df['pnl'] > 0).mean() * 100:.2f}%")
    print(f"Average Trade: {df['pnl'].mean():.2f}")
    print(f"Median Trade: {df['pnl'].median():.2f}")
    print(f"Max Win: {df['pnl'].max():.2f}")
    print(f"Max Loss: {df['pnl'].min():.2f}")
    print(f"Total PnL: {df['pnl'].sum():.2f}")
    print(f"Sharpe (simple): {df['pnl'].mean() / df['pnl'].std() * (len(df) ** 0.5):.2f}" if df['pnl'].std() > 0 else "Sharpe: N/A")
    print("Exit Reason Breakdown:")
    print(df['exit_reason'].value_counts())

    # Trade duration (in minutes)
    try:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df['trade_duration_min'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 60
        print(f"Average Trade Duration (min): {df['trade_duration_min'].mean():.2f}")
    except Exception as e:
        print("Could not parse trade times for duration.")

    # Last 5 trades
    print("Last 5 trades:")
    print(df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'exit_reason']].tail(5))

    # Plot equity curve
    plt.figure(figsize=(10,5))
    equity = df['pnl'].cumsum() + 25000  # Start equity at $25,000
    plt.plot(df['exit_time'], equity, label='Equity Curve')
    plt.title(f"Equity Curve: {os.path.basename(file)}")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot histogram of trade PnL
    plt.figure(figsize=(8,4))
    plt.hist(df['pnl'], bins=20, edgecolor='black')
    plt.title(f"Trade PnL Distribution: {os.path.basename(file)}")
    plt.xlabel("PnL per Trade")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Plot rolling drawdown
    equity = df['pnl'].cumsum() + 25000  # Start equity at $25,000
    roll_max = equity.cummax()
    drawdown = equity - roll_max
    plt.figure(figsize=(10,4))
    plt.plot(df['exit_time'], drawdown, color='red')
    plt.title(f"Rolling Drawdown: {os.path.basename(file)}")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.show()

    # Print drawdown stats
    print(f"Max Drawdown: {drawdown.min():.2f}")
    print(f"Max Drawdown (percent): {100 * drawdown.min() / 25000:.2f}%")

    # Optional: print trade PnL quantiles
    print("Trade PnL Quantiles:")
    print(df['pnl'].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]))

print("\nAnalysis complete.")