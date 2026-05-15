import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def cache_hits_by_suite(data, suite, level):
    df = pd.DataFrame(data)
    if df.empty:
        print("No data extracted. Verify JSON paths.")
        return
    
    df = df.groupby(["Benchmark", "Arch"])["Total Hits"].mean().reset_index()
    pivot_df = df.pivot(index="Benchmark", columns="Arch", values="Total Hits").fillna(0)
    
    #if "HPC" in pivot_df.columns:
        #pivot_df = pivot_df.sort_values(by="HPC", ascending=False)

    pivot_df = pivot_df.sort_index()

    x = np.arange(len(pivot_df.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 13))
    rects1 = ax.bar(x - width/2, pivot_df.get('HPC', 0), width, label='HPC', color='navy', edgecolor='black')
    rects2 = ax.bar(x + width/2, pivot_df.get('Desktop', 0), width, label='Desktop', color='purple', edgecolor='black')
    ax.set_xlabel('Benchmark Name', fontweight='bold')
    ax.set_ylabel('Total Cache Hits', fontweight='bold')
    ax.set_title(f'L{level} Cache Hits Comparison: HPC vs Desktop ({suite})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1e}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=45)
        
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout
    #plt.show()
    plt.savefig(f'{suite}_l{level}_cache_hits.jpg')
    plt.close()
                
def read_freq_by_suite(data, suite, level):
    df = pd.DataFrame(data)
    if df.empty:
        print("No data extracted. Verify JSON paths.")
        return
    
    df = df.groupby(["Benchmark", "Arch"])["Read Frequency"].mean().reset_index()
    pivot_df = df.pivot(index="Benchmark", columns="Arch", values="Read Frequency").fillna(0)
    
    #if "HPC" in pivot_df.columns:
        #pivot_df = pivot_df.sort_values(by="HPC", ascending=False)

    pivot_df = pivot_df.sort_index()

    x = np.arange(len(pivot_df.index))
    width = 0.35

    fig, ax = plt.subplots(figsize=(18, 13))
    rects1 = ax.bar(x - width/2, pivot_df.get('HPC', 0), width, label='HPC', color='navy', edgecolor='black')
    rects2 = ax.bar(x + width/2, pivot_df.get('Desktop', 0), width, label='Desktop', color='purple', edgecolor='black')
    ax.set_xlabel('Benchmark Name', fontweight='bold')
    ax.set_ylabel('Read Frequency', fontweight='bold')
    ax.set_title(f'L{level} Read Frequency Comparison: HPC vs Desktop ({suite})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1e}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=45)
        
    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout
    #plt.show()
    plt.savefig(f'{suite}_l{level}_read_freq.jpg')
    plt.close()

def collect_json(suite, level):
    all_data = []

    archs = ["HPC", "Desktop"]
    for arch in archs:
        base_path = Path(f"/Users/gloriawu/Documents/BEAM/Devs Dataset/Perf_AMD_{arch}/l{level}")
        json_files = list(base_path.rglob(f"{suite}/baseline/**/memsyspatternconfig_*.json"))
       
        for file in json_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    load_hits = data.get("load_hits", 0)
                    store_hits = data.get("store_hits", 0)
                    total_hits = load_hits + store_hits
                    read_freq = data.get("read_freq", 0)

                    benchmark = file.parts[-2]

                    all_data.append({
                        "Arch": arch,
                        "Benchmark": benchmark,
                        "Total Hits": total_hits,
                        "Read Frequency": read_freq
                    })
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
    return all_data

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--level', type=int, default=1, help="Cache level")
    #parser.add_argument('--filter_dir', type=str, default="intrate", help="intrate, intspeed, fprate, fpspeed")
    #args = parser.parse_args()
    
    #data = collect_json(args.level, args.filter_dir)
    #cache_hits(data, args.filter_dir, args.level)
    #read_freq(data, args.filter_dir, args.level)

    
    levels = ["1", "3"]
    suites = ["intrate", "intspeed", "fprate", "fpspeed"]

    for level in levels:
        for suite in suites:
            data = collect_json(suite, level)
            cache_hits_by_suite(data, suite, level)
            read_freq_by_suite(data, suite, level)