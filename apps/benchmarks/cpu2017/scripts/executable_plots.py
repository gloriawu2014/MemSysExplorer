import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_size(size_str):
    if pd.isna(size_str) or size_str == '':
        return np.nan
    size_str = str(size_str).strip().upper()
    if size_str.endswith('M'):
        return float(size_str[:-1]) * 1e6
    elif size_str.endswith('K'):
        return float(size_str[:-1]) * 1e3
    else:
        try:
            return float(size_str)
        except ValueError:
            return np.nan
        
def plot_data(df):
    opt_levels = ['O1', 'O2', 'O3', 'Ofast']
    for col in opt_levels:
        df[col] = df[col].apply(parse_size)

    suites = df['Suite'].unique()

    for suite in suites:
        suite_df = df[df['Suite'] == suite].dropna(subset=opt_levels)
        benchmarks = suite_df['Benchmarks']

        x = np.arange(len(benchmarks))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 9))

        steps = [('O1', None), ('O2', 'O1'), ('O3', 'O2'), ('Ofast', 'O3')]

        for i, (current, prev) in enumerate(steps):
            bars = ax.bar(x + (i - 1.5) * width, suite_df[current], width, label=current)
        
            if prev:
                for j, bar in enumerate(bars):
                    val_prev = suite_df.iloc[j][prev]
                    val_curr = suite_df.iloc[j][current]
                    
                    if val_prev > 0:
                        pct_inc = ((val_curr / val_prev) - 1) * 100
                        
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                                f'{pct_inc:+.1f}%', ha='center', va='bottom', 
                                rotation=30, fontsize=6, fontweight='light', color='navy')

        ax.set_title(f' Benchmark Sizes: {suite} (HPC)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Size in Bytes (Log Scale)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=40, ha='center')
        ax.legend(title="Optimization Level")
        ax.set_yscale('log')
        ax.grid(True, which="major", linestyle='--', alpha=0.3)
        plt.tight_layout()
        #plt.show()
        plt.savefig(f'{suite}_executable_sizes_HPC.png')

if __name__ == "__main__":
    data = pd.read_csv("benchmarks_HPC.csv")
    plot_data(data)