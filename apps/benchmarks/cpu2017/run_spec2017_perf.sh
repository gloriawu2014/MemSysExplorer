#!/bin/bash

### Modified version of run_spec2017.sh to run SPEC 2017 benchmarks using Perf

# -------- Configurable Paths --------
SPEC_ROOT="/home/gwu28/cpu2017"
MAIN_SCRIPT="/home/gwu28/MemSysExplorer/apps/main.py"
CMD_DIR="./commands"
WORKDIR="/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/spec_runs"
FILTER_DIR="intrate"   # One of: intrate, intspeed, fprate, fpspeed
RUN_TYPE="refrate"     # One of: refrate, testrate, trainrate
mkdir -p "$WORKDIR"

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')  # refrate -> ref, etc.

export PERF_PATH=/usr/bin/perf

# -------- Loop through .<CMD_TYPE>.cmd files --------
find "$CMD_DIR/$FILTER_DIR" -name "*.${CMD_TYPE}.cmd" | while read -r CMD_FILE; do
    CMD_NAME=$(basename "$CMD_FILE" .${CMD_TYPE}.cmd)
    echo -e "\n==> Processing $CMD_NAME (${RUN_TYPE})"

    BENCH_ID="${CMD_NAME}"
    BENCH_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID"
    EXE_DIR="$BENCH_DIR/exe"
    INPUT_DIR="$BENCH_DIR/data/${RUN_TYPE}/input"
    OUTPUT_DIR="$BENCH_DIR/data/${RUN_TYPE}/output"

    # Locate executable (assume only one)
    EXE_PATH=$(find "$EXE_DIR" -maxdepth 1 -type f -executable | head -n 1)
    [[ ! -x "$EXE_PATH" ]] && echo " Skipping $CMD_NAME: Executable not found" && continue
    EXE_BASENAME=$(basename "$EXE_PATH")

    # Setup run directory
    RUN_DIR="$WORKDIR/${BENCH_ID}.${RUN_TYPE}"
    mkdir -p "$RUN_DIR"
    cp "$EXE_PATH" "$RUN_DIR/"

    # Copy input files
    if [[ -d "$INPUT_DIR" ]]; then
        echo "  Copying inputs from: $INPUT_DIR"
        cp -r "$INPUT_DIR/"* "$RUN_DIR/" 2>/dev/null
    fi

    # Copy expected output files
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "  Copying golden outputs from: $OUTPUT_DIR"
        cp -r "$OUTPUT_DIR/"* "$RUN_DIR/" 2>/dev/null
    fi

    # Copy .out/.err references if present
    CMD_DIRNAME=$(dirname "$CMD_FILE")
    for ext in out err; do
        AUX_FILE="${CMD_DIRNAME}/${CMD_NAME}.${CMD_TYPE}.${ext}"
        [[ -f "$AUX_FILE" ]] && cp "$AUX_FILE" "$RUN_DIR/"
    done

    # ---------- Generate run script ----------
    RUN_SH="${RUN_DIR}/${BENCH_ID}.${RUN_TYPE}.sh"
    echo "#!/bin/bash" > "$RUN_SH"
    echo "#SBATCH --job-name=${BENCH_ID}" >> "$RUN_SH"
    echo "#SBATCH --partition=cpu-q" >> "$RUN_SH"
    echo "#SBATCH --cpus-per-task=1" >> "$RUN_SH"
    echo "#SBATCH --time=01:00:00" >> "$RUN_SH"
    echo "#SBATCH --output=${BENCH_ID}.out" >> "$RUN_SH"
    echo "#SBATCH --error=${BENCH_ID}.err" >> "$RUN_SH"
    echo "# Generated from $CMD_FILE" >> "$RUN_SH"
    echo "# Executable: $EXE_BASENAME" >> "$RUN_SH"
    echo "" >> "$RUN_SH"

    PREFIX="python3 ${MAIN_SCRIPT} -p perf -a both --level l1 --arch amd"

    # Wrap each line with executable
    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')
        [[ -z "$trimmed" ]] && continue

        # Copy missing input files if needed
        for f in $trimmed; do
            if [[ "$f" == *.* ]]; then
                if [[ ! -f "./$f" ]]; then
                    echo "Copying $f..."
                    cp "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000/$f" "${RUN_DIR}/"
                fi
                if [[ ! -f "./$f" ]]; then
                    echo "Warning: $f not found in source path"
                fi
            fi
        done >> "${RUN_DIR}/output.log" 2>&1

        echo "${PREFIX} --executable ./$EXE_BASENAME --executable_args $trimmed" >> "$RUN_SH"
    done < "$CMD_FILE"

    # Hard code copy files for 520 & 620, 500, 548
    if [[ "$BENCH_ID" == "520.omnetpp_r" || "$BENCH_ID" == "620.omnetpp_s" ]]; then
        echo "Copying ned/"
        cp -r "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000/ned/" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "500.perlbench_r" ]]; then
        echo "Copying lib/"
        echo "Copying cpu2017_mhonarc.rc"
        cp -r "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000/lib/" "${RUN_DIR}/"
        cp "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000/cpu2017_mhonarc.rc" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "548.exchange2_r" ]]; then
        echo "Copying puzzles.txt"
        rm -f "puzzles.txt"
        cp "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000/puzzles.txt" "${RUN_DIR}/"
    fi

    chmod +x "$RUN_SH"

    # Execute using the renamed script
    cd "$RUN_DIR"
    echo "   → Entering dir: $(pwd)"
    echo "   → Executing ${BENCH_ID} (${RUN_TYPE}) with Perf..."
    sbatch "$RUN_SH"
    echo "   → Finished ${BENCH_ID} (${RUN_TYPE}) with Perf..."

    cd - > /dev/null
done
