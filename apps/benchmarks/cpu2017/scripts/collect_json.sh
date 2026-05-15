#!/bin/bash

# -------- Configurable Paths --------
DEST_ROOT="./collected_results"
SPEC_ROOT="/home/gwu28/spec2017"
LEVELS=("l1" "l3")
OPTIMIZATION="O2"
TRIAL="5"

mkdir -p "$DEST_ROOT"

for LEVEL in "${LEVELS[@]}"; do
    echo "=== Processing Level: $LEVEL ==="

    if [ ! -d "$SPEC_ROOT/benchspec/CPU" ]; then
        echo "Error: $SPEC_ROOT/benchspec/CPU not found. Check your SPEC_ROOT path."
        exit 1
    fi

    for BENCH_DIR in "$SPEC_ROOT/benchspec/CPU"/*; do
        [ ! -d "$BENCH_DIR" ] && continue
        
        FULL_ID=$(basename "$BENCH_DIR")
        SHORT_NAME=$(echo "$FULL_ID" | sed 's/^[0-9]*\.//')
        
        # 1. Determine Category (int or fp)
        if [[ "$FULL_ID" =~ (gcc|perl|xz|mcf|omnetpp|deepsjeng|leela|x264|exchange2) ]]; then
            CAT_TYPE="int"
        else
            CAT_TYPE="fp"
        fi

        # 2. Determine Rate vs Speed
        RUN_BASE_DIR="$BENCH_DIR/run"
        [ ! -d "$RUN_BASE_DIR" ] && continue
        
        BENCH_NUM=$(echo "$FULL_ID" | grep -oP '^\d{3}')

        if [[ "$BENCH_NUM" -ge 600 ]]; then
            SUFFIX="speed"
        elif [[ "$BENCH_NUM" -ge 500 ]]; then
            SUFFIX="rate"
        else
            # Fallback if the number doesn't fit the expected SPEC pattern
            SUFFIX="unknown"
        fi

        CATEGORY="${CAT_TYPE}${SUFFIX}"
        
        LEVEL_DIR="$RUN_BASE_DIR/${LEVEL}_${OPTIMIZATION}/trial_${TRIAL}"

        if [ -d "$LEVEL_DIR" ]; then
            TARGET_DIR="$DEST_ROOT/${OPTIMIZATION}/$LEVEL/$CATEGORY/baseline/$SHORT_NAME/trial_${TRIAL}"
            mkdir -p "$TARGET_DIR"

            # --- STEP 1: Copy JSON files safely ---
            if ls "$LEVEL_DIR"/*.json >/dev/null 2>&1; then
                cp "$LEVEL_DIR"/*.json "$TARGET_DIR/"
                
                # Loop through copied files to check for empty ones without crashing
                for json_file in "$TARGET_DIR"/*.json; do
                    if [ ! -s "$json_file" ]; then
                            echo "  [!] WARNING: $(basename "$json_file") is EMPTY"
                    fi
                done
                echo "  [✓] $SHORT_NAME: JSON(s) copied to $TARGET_DIR"
            fi
        fi
    done
done


