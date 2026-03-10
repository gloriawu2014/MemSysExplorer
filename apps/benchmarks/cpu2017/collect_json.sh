#!/bin/bash

# -------- Configurable Paths --------
DEST_ROOT="./collected_results"
SPEC_ROOT_1="/home/gwu28/spec2017"
SPEC_ROOT_2="/home/gwu28/cpu2017"
LEVELS=("l1" "l3")

# Keywords to flag as errors in logs
ERROR_KEYWORDS="Error|error|Aborted|Segmentation fault|FAILED|panic"

mkdir -p "$DEST_ROOT"

for LEVEL in "${LEVELS[@]}"; do
    echo "=== Processing Level: $LEVEL ==="

    for SPEC_ROOT in "$SPEC_ROOT_1" "$SPEC_ROOT_2"; do
        [ ! -d "$SPEC_ROOT/benchspec/CPU" ] && continue

        for BENCH_DIR in "$SPEC_ROOT/benchspec/CPU"/*; do
            [ ! -d "$BENCH_DIR" ] && continue
            
            FULL_ID=$(basename "$BENCH_DIR")
            SHORT_NAME=$(echo "$FULL_ID" | sed 's/^[0-9]*\.//')
            
            # 1. Determine Category (int or fp)
            if [[ "$FULL_ID" =~ (gcc|perl|xz|mcf|omnetpp|deepsjeng|leela|x264|renesas|astrate|astar|label) ]]; then
                CAT_TYPE="int"
            else
                CAT_TYPE="fp"
            fi

            # 2. Determine Rate vs Speed
            RUN_BASE_DIR="$BENCH_DIR/run"
            [ ! -d "$RUN_BASE_DIR" ] && continue
            
            ACTUAL_RUN_FOLDER=$(ls -d "$RUN_BASE_DIR"/run_base_* 2>/dev/null | head -n 1)
            [ -z "$ACTUAL_RUN_FOLDER" ] && continue
            
            [[ "$ACTUAL_RUN_FOLDER" == *"rate"* ]] && SUFFIX="rate" || SUFFIX="speed"
            CATEGORY="${CAT_TYPE}${SUFFIX}"
            
            LEVEL_DIR="$RUN_BASE_DIR/$LEVEL"

            if [ -d "$LEVEL_DIR" ]; then
                TARGET_DIR="$DEST_ROOT/$LEVEL/$CATEGORY/$SHORT_NAME"
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

                # --- STEP 2: Log Audit with FULL PATHS ---
                # grep -l returns the full path provided in the argument
                ERRORS=$(grep -lEi "$ERROR_KEYWORDS" "$LEVEL_DIR"/*.{out,err} 2>/dev/null)
                
                if [ ! -z "$ERRORS" ]; then
                    echo "      -> ALERT: Errors found in following logs:"
                    echo "$ERRORS" | sed 's/^/         - /'
                fi
            fi
        done
    done
done

echo -e "\nSummary: Results organized in $DEST_ROOT"