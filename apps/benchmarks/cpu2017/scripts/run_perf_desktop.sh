#!/bin/bash

### Run SPEC 2017 benchmarks with Perf directly through spec2017/

# -------- Configurable Paths --------
MAIN_SCRIPT="/home/gloriawu/MemSysExplorer/apps/main.py"
CMD_DIR="/home/gloriawu/MemSysExplorer/apps/benchmarks/cpu2017/commands"
SPEC_ROOT="/home/gloriawu/cpu2017"
FILTER_DIR="fprate"   # One of: intrate, intspeed, fprate, fpspeed
RUN_TYPE="refrate"     # One of: refrate, testrate, trainrate
LEVEL="l3"             # One of: l1, l2, l3
OPTIMIZATION="Ofast"      # One of: O1, O2, O3, Ofast

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')  # refrate -> ref, etc.

export PERF_PATH=/usr/bin/perf

# -------- Loop through .<CMD_TYPE>.cmd files --------
find "$CMD_DIR/$FILTER_DIR" -name "*.${CMD_TYPE}.cmd" | while read -r CMD_FILE; do
    CMD_NAME=$(basename "$CMD_FILE" .${CMD_TYPE}.cmd)
    echo -e "\n==> Processing $CMD_NAME (${RUN_TYPE})"

    BENCH_ID="${CMD_NAME}"
    BENCH_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID"
    #EXE_DIR="$BENCH_DIR/exe"

    if [[ "$FILTER_DIR" == "intrate" || "$FILTER_DIR" == "fprate" ]]; then
        COPY="$BENCH_DIR/run/run_base_refrate_none.0000"
    else
        COPY="$BENCH_DIR/run/run_base_refspeed_none.0000"
    fi

    cd "$BENCH_DIR/run"
    COPY_DIR="$BENCH_DIR/run/${LEVEL}_${OPTIMIZATION}"
    cp -r "$COPY" "$COPY_DIR"
    EXE_DIR="$COPY_DIR"

    # Locate executable
    if [[ "$BENCH_ID" == "625.x264_s" ]]; then 
        EXE_PATH="$EXE_DIR/x264_s_base.none"
    elif [[ "$BENCH_ID" == "525.x264_r" ]]; then
        EXE_PATH="$EXE_DIR/x264_r_base.none"
    elif [[ "$BENCH_ID" == "503.bwaves_r" ]]; then
        EXE_PATH="$EXE_DIR/bwaves_r_base.none"
    elif [[ "$BENCH_ID" == "638.imagick_s" ]]; then
        EXE_PATH="$EXE_DIR/imagick_s_base.none"
    elif [[ "$BENCH_ID" == "621.wrf_s" ]]; then
        EXE_PATH="$EXE_DIR/wrf_s_base.none"
    elif [[ "$BENCH_ID" == "521.wrf_r" ]]; then
        EXE_PATH="$EXE_DIR/wrf_r_base.none"
    elif [[ "$BENCH_ID" == "527.cam4_r" ]]; then
        EXE_PATH="$EXE_DIR/cam4_r_base.none"
    elif [[ "$BENCH_ID" == "511.povray_r" ]]; then
        EXE_PATH="$EXE_DIR/povray_r_base.none"
    elif [[ "$BENCH_ID" == "526.blender_r" ]]; then
        EXE_PATH="$EXE_DIR/blender_r_base.none"
    elif [[ "$BENCH_ID" == "538.imagick_r" ]]; then
        EXE_PATH="$EXE_DIR/imagick_r_base.none"
    else
        EXE_PATH=$(find "$EXE_DIR" -maxdepth 1 -type f -executable | head -n 1)
    fi

    [[ ! -x "$EXE_PATH" ]] && echo " Skipping $CMD_NAME: Executable not found" && continue

    # ---------- Generate run script ----------
    RUN_SH="${COPY_DIR}/${BENCH_ID}.${RUN_TYPE}.sh"
    echo "#!/bin/bash" > "$RUN_SH"
    echo "# Generated from $CMD_FILE" >> "$RUN_SH"
    echo "# Executable: $EXE_PATH" >> "$RUN_SH"
    echo "" >> "$RUN_SH"

    # Nohup prefix
    #PREFIX="nohup python3 ${MAIN_SCRIPT} -p perf -a both --level ${LEVEL} --arch amd"
    PREFIX="python3 ${MAIN_SCRIPT} --profiler perf --action both --level ${LEVEL} --arch amd"


    # Wrap each line with executable
    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')
        [[ -z "$trimmed" ]] && continue

        # Nohup suffix
        #echo "${PREFIX} --executable $EXE_PATH --executable_args $trimmed &" >> "$RUN_SH"
        echo "${PREFIX} --executable $EXE_PATH --executable_args $trimmed" >> "$RUN_SH"
    done < "$CMD_FILE"

    chmod +x "$RUN_SH"

    # Execute using the renamed script
    cd "$COPY_DIR"
    echo "   → Entering dir: $(pwd)"
    echo "   → Executing ${BENCH_ID} (${RUN_TYPE}) with Perf..."
    #tmux split-window -h "bash $RUN_SH > ${BENCH_ID}_${LEVEL}.out 2>&1"
    echo "   → Finished ${BENCH_ID} (${RUN_TYPE}) with Perf..."

    cd - > /dev/null
done
