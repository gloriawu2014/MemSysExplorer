import json
from pathlib import Path
from scipy.stats import ttest_rel
from statistics import mean
import pandas as pd

def dataset():
    all_data = []

    for arch in ["HPC", "Desktop"]:
        
        for level in ["1", "3"]:
            base_path = Path(f"/Users/gloriawu/Documents/BEAM/Devs Dataset/Perf_AMD_{arch}/l{level}")
            
            for suite in ["intrate", "intspeed", "fprate", "fpspeed"]:
                json_files = list(base_path.rglob(f"{suite}/baseline/**/memsyspatternconfig_*.json"))
                
                for file in json_files:
                    try:
                        with open(file, "r") as f:
                            data = json.load(f)

                        load_hits = data.get("load_hits", 0)
                        store_hits = data.get("store_hits", 0)
                        total_hits = load_hits + store_hits
                        read_freq = data.get("read_freq", 0)
                        benchmark = file.parts[-2]

                        all_data.append({
                            "arch": arch,
                            "suite": suite,
                            "name": benchmark,
                            "level": level,
                            "load_hits": load_hits,
                            "store_hits": store_hits,
                            "total_hits": total_hits,
                            "read_freq": read_freq,
                        })
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

    return all_data

def paired_t_test(dataset, level, stat, suite=None):
    filtered = [row for row in dataset 
                if row["level"] == level 
                and (suite is None or row["suite"] == suite)
                and row.get(stat, 0) > 0]

    pairs = {}
    for row in filtered:
        key = (row["suite"], row["name"]) if suite is None else row["name"]
        value = row[stat]
        arch = row["arch"]

        if key not in pairs:
            pairs[key] = {"Desktop": [], "HPC": []}
        pairs[key][arch].append(value)

    grouped = {}
    for key, values in pairs.items():
        if values.get("HPC") and values.get("Desktop"):
            suite_key = key[0] if suite is None else suite
            grouped.setdefault(suite_key, {"Desktop": [], "HPC": []})
            grouped[suite_key]["Desktop"].append(mean(values["Desktop"]))
            grouped[suite_key]["HPC"].append(mean(values["HPC"]))

    results = {}
    for suite_key, values in grouped.items():
        n = len(values["Desktop"])
        if n < 2:
            continue

        test = ttest_rel(values["Desktop"], values["HPC"])
        results[suite_key] = {
            "n": n,
            "statistic": float(test.statistic),
            "pvalue": float(test.pvalue),
        }
        
        print(f"\nSuite: {suite_key} | Level: {level} | Comparing: {stat}")
        print(f"{'Benchmark':<25} | {'Desktop (avg)':<15} | {'HPC (avg)':<15}")
        print("-" * 65)

        for key, b_values in pairs.items():
            b_suite, b_name = key if isinstance(key, tuple) else (suite_key, key)

            if b_suite == suite_key and b_values.get("Desktop") and b_values.get("HPC"):
                d_avg = mean(b_values["Desktop"])
                h_avg = mean(b_values["HPC"])
                print(f"{b_name:<25} | {d_avg:<15.2f} | {h_avg:<15.2f}")
        
        print(f"Result: n={n}, t={results[suite_key]['statistic']:.4f}, p={results[suite_key]['pvalue']:.4f}")

    return results

if __name__ == "__main__":
    dataset = dataset()
    for level in ["1", "3"]:
        for stat in ["total_hits", "read_freq"]:
            results = paired_t_test(dataset, level, stat)