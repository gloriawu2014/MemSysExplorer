#!/bin/bash

# NOTE: FILTER_DIR doesn't do anything here, will just do all benchmarks

# -------- Configurable Paths --------
SPEC_ROOT="/home/gwu28/spec2017"
FILTER_DIR="intrate"   # Change this to match your target suite
RUN_TYPE="refrate"     # Change as needed (refrate, testrate, etc.)
OLD_LEVEL="l1"         # The source level you are copying FROM
NEW_LEVEL="l2"         # The target level you are copying TO
OPTIMIZATION="O3"

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')
echo "==> Starting migration from ${OLD_LEVEL} to ${NEW_LEVEL}..."

find "$SPEC_ROOT/benchspec/CPU" -maxdepth 1 -type d -name "[56]*" | while read -r BENCH_DIR; do
    BENCH_ID=$(basename "$BENCH_DIR")
    
    # Define source and destination directories
    # Logic follows your previous script's naming: ${LEVEL}_O1
    SRC_DIR="$BENCH_DIR/run/${OLD_LEVEL}_${OPTIMIZATION}"
    DEST_DIR="$BENCH_DIR/run/${NEW_LEVEL}_${OPTIMIZATION}"

    if [[ -d "$SRC_DIR" ]]; then
        echo "--> Processing $BENCH_ID"

        # 1. Copy the directory (using -p to preserve permissions)
        cp -rp "$SRC_DIR" "$DEST_DIR"
        
        # 2. Delete any existing memsys*.json files in the new folder
        rm -f "$DEST_DIR"/memsys*.json
        
        # 3. Locate the .sh script in the new folder
        # Naming convention from your script: ${BENCH_ID}.${RUN_TYPE}.sh
        RUN_SH="$DEST_DIR/${BENCH_ID}.${RUN_TYPE}.sh"

        if [[ -f "$RUN_SH" ]]; then
            # Use sed to replace the level flag: --level l1 -> --level l2
            # Also update any internal path references if they were hardcoded
            sed -i "s/--level ${OLD_LEVEL}/--level ${NEW_LEVEL}/g" "$RUN_SH"
                        
            echo "    ✓ Copied to ${NEW_LEVEL}_${OPTIMIZATION}, cleaned JSONs, and updated $(basename "$RUN_SH")"
        else
            echo "    ! Warning: Script $RUN_SH not found in $DEST_DIR"
        fi
    fi
done

echo -e "\n==> Migration Complete."