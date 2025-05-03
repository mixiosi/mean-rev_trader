import pandas as pd
import matplotlib.pyplot as plt
import os

# Configuration
SUMMARY_CSV = 'tf_gs_summary.csv'  # Adjust path as needed
TOP_N = 10

# Load grid search summary
summary_path = os.path.join(os.path.dirname(__file__), SUMMARY_CSV)
df = pd.read_csv(summary_path)

# Rank by PnL, Sharpe, Win Rate, etc.
df_sorted = df.sort_values(by=['pnl', 'sharpe', 'win_rate'], ascending=[False, False, False])
print('Top parameter sets (Trend-Following):')
print(df_sorted.head(TOP_N))

# Save top-N to CSV
out_csv = os.path.join(os.path.dirname(__file__), 'tf_top_gs_results.csv')
df_sorted.head(TOP_N).to_csv(out_csv, index=False)
print(f"Saved top {TOP_N} parameter sets to {out_csv}")

# Plot PnL vs. Sharpe
plt.figure(figsize=(8,6))
plt.scatter(df['sharpe'], df['pnl'], c=df['win_rate'], cmap='viridis', alpha=0.7)
plt.xlabel('Sharpe Ratio')
plt.ylabel('Total PnL')
plt.title('Trend Grid Search: PnL vs. Sharpe (color=Win Rate)')
plt.colorbar(label='Win Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'tf_gs_scatter.png'))
plt.show()

# (Optional) Parameter heatmap for further analysis
# Example: FAST_MA vs. SLOW_MA, colored by PnL
pivot = df.pivot_table(index='fast_ma', columns='slow_ma', values='pnl', aggfunc='mean')
plt.figure(figsize=(8,6))
plt.title('PnL Heatmap: FAST_MA vs. SLOW_MA')
plt.xlabel('SLOW_MA')
plt.ylabel('FAST_MA')
plt.imshow(pivot, cmap='RdYlGn', aspect='auto', origin='lower')
plt.colorbar(label='Mean PnL')
plt.xticks(ticks=range(len(pivot.columns)), labels=pivot.columns)
plt.yticks(ticks=range(len(pivot.index)), labels=pivot.index)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'tf_pnl_heatmap.png'))
plt.show()
