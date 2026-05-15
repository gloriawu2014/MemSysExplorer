import json
import argparse
from pathlib import Path

def fix_frequency_metrics(json_files):
    changed_files = []

    for file in json_files:
        try:
            with open(file, 'r') as f:
                data = json.load(f)

            total_reads = data.get("total_reads", 0)
            current_freq = data.get("read_freq", 0)
            exec_time = data.get("execution_time", 0)

            if exec_time > 0:
                expected_freq = total_reads / exec_time

                if abs(current_freq - expected_freq) / (expected_freq + 1e-9) > 0.01:
                    data["read_freq"] = expected_freq
                    #print(f"{current_freq}, {expected_freq}")

                    with open(file, 'w') as f:
                        json.dump(data, f, indent=2)

                    changed_files.append(file)

        except Exception as e:
            print(f"{file.nameL<30} | ERROR: {e}")

    return changed_files

if __name__ == "__main__":
    archs = ["HPC", "Desktop"]
    suites = ["intrate", "intspeed", "fprate", "fpspeed"]
    levels = ["l1", "l3"]

    for arch in archs:
        base_path = f"/Users/gloriawu/Documents/BEAM/Devs Dataset/Perf_AMD_{arch}"
        
        for level in levels:
            base = Path(f"{base_path}/{level}")

            for suite in suites:
                json_files = list(base.rglob(f"{suite}/baseline/**/memsyspatternconfig_*.json"))

                changed_files = fix_frequency_metrics(json_files)

                if changed_files:
                    print(f"TOTAL FILES CHANGED: {len(changed_files)}")
                    for file in changed_files:
                        print(f" - {file}")