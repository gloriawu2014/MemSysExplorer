#!/bin/bash

# -------- Configurable Paths --------
CMD_DIR="/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/commands"
RUN_TYPE="refrate"
SPEC_ROOT="/home/gwu28/spec2017"

# -------- Ranges to Loop Through --------
FILTER_DIRS=("intrate" "intspeed" "fprate" "fpspeed")
LEVELS=("l1" "l3")
OPTIMIZATIONS=("O1" "O3" "Ofast")

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')

echo "BenchID, Filter, Level, Opt, ConfigFile, ExecTime"

for FILTER_DIR in "${FILTER_DIRS[@]}"; do
    for LEVEL in "${LEVELS[@]}"; do
        for OPTIMIZATION in "${OPTIMIZATIONS[@]}"; do
            
            # Find the command files for the current filter category
            find "$CMD_DIR/$FILTER_DIR" -name "*.${CMD_TYPE}.cmd" 2>/dev/null | while read -r CMD_FILE; do
                BENCH_ID=$(basename "$CMD_FILE" .${CMD_TYPE}.cmd)
                RUN_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/${LEVEL}_${OPTIMIZATION}"

                if [ -d "$RUN_DIR" ]; then
                    (
                        cd "$RUN_DIR" || exit
                        
                        # Find all matching JSON config files
                        find . -maxdepth 1 -name "memsyspatternconfig*.json" | while read -r CONFIG_FILE; do
                            
                            # Extract execution_time; defaults to 0 if missing
                            EXEC_TIME=$(jq -r '.execution_time // 0' "$CONFIG_FILE" | tr -d '\r')

                            # Print the results for all files found
                            # Note: I added FILTER_DIR to the output so you can distinguish the results
                            echo "${BENCH_ID}, ${FILTER_DIR}, ${LEVEL}, ${OPTIMIZATION}, $(basename "$CONFIG_FILE"), ${EXEC_TIME}s"
                        done
                    )
                fi
            done
        done
    done
done