#!/bin/bash

# -------- Configurable Paths --------
SPEC_ROOT="/home/gwu28/spec2017"
LEVELS=("l1" "l3")
OPTIMIZATIONS=("O1" "O2" "O3")

echo "==> Starting Fixer Script..."

for LVL in "${LEVELS[@]}"; do
    for OPT in "${OPTIMIZATIONS[@]}"; do

    TARGET_SUFFIX="${LVL}_${OPT}"
    echo "--> Processing level/opt: $TARGET_SUFFIX"

    find "$SPEC_ROOT/benchspec/CPU" -path "*/run/${TARGET_SUFFIX}/*.sh" | while read -r RUN_SH; do
        
        SCRIPT_DIR=$(dirname "$RUN_SH")
        OLD_EXE_PATH=$(grep -oP '(?<=--executable )[^ ]+' "$RUN_SH" | head -n 1)

        if [[ -z "$OLD_EXE_PATH" ]]; then
                echo "    [!] Could not find --executable path in $RUN_SH. Skipping."
                continue
        fi

        EXE_FILENAME=$(basename "$OLD_EXE_PATH")
        NEW_EXE_PATH="$SCRIPT_DIR/$EXE_FILENAME"

        sed -i "s|^# Executable:.*|# Executable: $NEW_EXE_PATH|" "$RUN_SH"
        sed -i "s|--executable $OLD_EXE_PATH|--executable $NEW_EXE_PATH|" "$RUN_SH"

        echo "    [✓] Updated $(basename "$RUN_SH") -> $EXE_FILENAME"
        done
    done
done