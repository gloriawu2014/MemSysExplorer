#!/bin/bash

# -------- Configurable Paths --------
SPEC_ROOT="/home/gloriawu/cpu2017"
OPTIMIZATION="O2"
RUN_TYPE="refrate"
CMD_DIR="/home/gloriawu/MemSysExplorer/apps/benchmarks/cpu2017/commands"

# Lists to iterate through
SUITES=("intrate" "intspeed" "fprate" "fpspeed")
LEVELS=("l1")

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')  # refrate -> ref, etc.

for FILTER_DIR in "${SUITES[@]}"; do
    for LEVEL in "${LEVELS[@]}"; do

        # -------- Iterate through benchmarks --------
        find "$CMD_DIR/$FILTER_DIR" -name "*.${CMD_TYPE}.cmd" | while read -r CMD_FILE; do

            CMD_NAME=$(basename "$CMD_FILE" .${CMD_TYPE}.cmd)
            echo -e "\n==> Processing $CMD_NAME"

            BENCH_ID="${CMD_NAME}"
            BENCH_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID"
            
            [ ! -d "$BENCH_DIR" ] && continue
            
            RUN_DIR="$BENCH_DIR/run/${LEVEL}_${OPTIMIZATION}"
            [ ! -d "$RUN_DIR" ] && continue
            
            SH_FILE="$RUN_DIR/${BENCH_ID}.${RUN_TYPE}.sh"
            
            if [ -f "$SH_FILE" ]; then
                echo -e "\n--- Scanning $BENCH_ID ($FILTER_DIR) ---"
                
                PY_COUNT=$(grep -c "python3" "$SH_FILE")
                JSON_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name "memsys*.json" | wc -l)
                TARGET_JSON=$((PY_COUNT * 2))

                if [ "$PY_COUNT" -gt 0 ] && [ "$JSON_COUNT" -ge "$TARGET_JSON" ]; then
                    echo "Result check passed: $JSON_COUNT/$TARGET_JSON JSONs found. Skipping."
                    continue
                fi

                echo "Results missing: Found $JSON_COUNT/$TARGET_JSON. Executing..."

                while IFS= read -u 3 -r line || [[ -n "$line" ]]; do
                    [[ "$line" != *"python3"* ]] && continue
                    
                    while pgrep -f "_base.none" > /dev/null || pgrep -f "python3.*memsys" > /dev/null; do
                        echo "    [WAIT] Another benchmark is currently running... checking again in 60s."
                        sleep 60
                    done

                    echo "  [LAUNCH] Creating tmux session for $BENCH_ID..."
                    SESSION_NAME="${BENCH_ID}_$(date +%s)"
                    
                    # Execute in subshell to keep the main script's path stable
                    (
                        cd "$RUN_DIR" || exit
                        tmux new-session -d -s "$SESSION_NAME" "$line"
                    )
                    
                    echo "           Launched: $SESSION_NAME"
                    sleep 30
                    
                done 3< "$SH_FILE"
            fi
        done
    done
done
