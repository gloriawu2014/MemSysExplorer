#!/bin/bash

### Modified version of run_spec2017.sh to run SPEC 2017 benchmarks using Perf

# -------- Configurable Paths --------
SPEC_ROOT="/home/gwu28/spec2017"
MAIN_SCRIPT="/home/gwu28/MemSysExplorer/apps/main.py"
CMD_DIR="./commands"
WORKDIR="/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/spec_runs_2"
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

    # Locate executable
    if [[ "$BENCH_ID" == "625.x264_s" ]]; then 
        EXE_PATH="$EXE_DIR/x264_s_base.none"
    elif [[ "$BENCH_ID" == "503.bwaves_r" ]]; then
        EXE_PATH="$EXE_DIR/bwaves_r_base.none"
    elif [[ "$BENCH_ID" == "638.imagick_s" ]]; then
        EXE_PATH="$EXE_DIR/imagick_s_base.none"
    else
        EXE_PATH=$(find "$EXE_DIR" -maxdepth 1 -type f -executable | head -n 1)
    fi

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
    echo "#SBATCH --time=04:00:00" >> "$RUN_SH"
    echo "#SBATCH --output=${BENCH_ID}.out" >> "$RUN_SH"
    echo "#SBATCH --error=${BENCH_ID}.err" >> "$RUN_SH"
    echo "# Generated from $CMD_FILE" >> "$RUN_SH"
    echo "# Executable: $EXE_BASENAME" >> "$RUN_SH"
    echo "" >> "$RUN_SH"

    PREFIX="python3 ${MAIN_SCRIPT} -p perf -a both --level l1 --arch amd"

    if [[ "$FILTER_DIR" == "intrate" || "$FILTER_DIR" == "fprate" || "$BENCH_ID" == "554.roms_r" ]]; then
        COPY_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refrate_none.0000"
    else
        COPY_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID/run/run_base_refspeed_none.0000"
    fi

    # Wrap each line with executable
    while IFS= read -r line || [[ -n "$line" ]]; do
        trimmed=$(echo "$line" | sed 's/^[ \t]*//;s/[ \t]*$//')
        [[ -z "$trimmed" ]] && continue

        # Copy missing input files if needed
        for f in $trimmed; do
            if [[ "$f" == *.* ]]; then
                if [[ ! -f "./$f" ]]; then
                    echo "Copying $f..."
                    cp "$COPY_DIR/$f" "${RUN_DIR}/"
                fi
                if [[ ! -f "./$f" ]]; then
                    echo "Warning: $f not found in source path"
                fi
            fi
        done >> "${RUN_DIR}/output.log" 2>&1

        echo "${PREFIX} --executable ./$EXE_BASENAME --executable_args $trimmed" >> "$RUN_SH"
    done < "$CMD_FILE"

    # Hard code copy files
    if [[ "$BENCH_ID" == "520.omnetpp_r" ]]; then
        echo "Copying ned/"
        cp -r "${$COPY_DIR}/ned/" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "500.perlbench_r" || "600.perlbench_s" ]]; then
        echo "Copying lib/"
        echo "Copying cpu2017_mhonarc.rc"
        cp -r "{$COPY_DIR}/lib/" "${RUN_DIR}/"
        cp "{$COPY_DIR}/cpu2017_mhonarc.rc" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "548.exchange2_r" || "$BENCH_ID" == "648.exchange2_s" ]]; then
        echo "Copying puzzles.txt"
        rm -f "puzzles.txt"
        cp "{$COPY_DIR}/puzzles.txt" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "605.mcf_s" ]]; then
        echo "Copying inp.in"
        cp "{$COPY_DIR}/inp.in" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "620.omnetpp_s" ]]; then
        echo "Copying ned/"
        echo "Copying omnetpp.ini"
        cp -r "{$COPY_DIR}/ned/" "${RUN_DIR}/"
        cp "{$COPY_DIR}/omnetpp.ini" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "511.povray_r" ]]; then
        echo "Copying shapes.inc"
        echo "Copying shapes_old.inc"
        echo "Copying consts.inc"
        echo "Copying transforms.inc"
        echo "Copying math.inc"
        echo "Copying functions.inc"
        echo "Copying strings.inc"
        echo "Copying colors.inc"
        echo "Copying textures.inc"
        echo "Copying finish.inc"
        echo "Copying skies.inc"
        echo "Copying metals.inc"
        echo "Copying golds.inc"
        echo "Copying woods.inc"
        echo "Copying woodmaps.inc"
        cp "{$COPY_DIR}/shapes.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/shapes_old.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/consts.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/transforms.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/math.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/functions.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/strings.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/colors.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/textures.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/finish.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/skies.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/metals.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/golds.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/woods.inc" "${RUN_DIR}/"
        cp "{$COPY_DIR}/woodmaps.inc" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "554.roms_r" || "$BENCH_ID" == "654.roms_s" ]]; then
        echo "Copying varinfo.dat"
        cp "{$COPY_DIR}/varinfo.dat" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "549.fotonik3d_r" || "$BENCH_ID" == "649.fotonik3d_s" ]]; then
        echo "Copying material.fppized.f90"
        echo "Copying OBJ.dat"
        cp "$SPEC_ROOT/benchspec/CPU/$BENCH_ID/build/build_base_none.0000/material.fppized.f90" "${RUN_DIR}/"
        cp "{$COPY_DIR}/OBJ.dat" "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "649.fotonik3d_s" ]]; then
        echo "Copying yee.dat"
        echo "Copying PSI.dat"
        echo "Copying TEwaveguide.m"
        echo "Copying power1.dat"
        echo "Copying incident_W3PC_25nm.def"
        echo "Copying power2.dat"
        echo "Copying trans_W3PC_25nm.def"
        cp "{$COPY_DIR}/yee.dat" "${RUN_DIR}/"
        cp "{$COPY_DIR}/PSI.dat" "${RUN_DIR}/"
        cp "{$COPY_DIR}/TEwaveguide.m" "${RUN_DIR}/"
        cp "{$COPY_DIR}/power1.dat" "${RUN_DIR}/"
        cp "{$COPY_DIR}/incident_W3PC_25nm.def" "${RUN_DIR}/"
        cp "{$COPY_DIR}/power2.dat" "${RUN_DIR}/"
        cp "{$COPY_DIR}/trans_W3PC_25nm.def " "${RUN_DIR}/"
    fi
    if [[ "$BENCH_ID" == "644.nab_s" ]]; then
        echo "Copying 3j1n"
        cp -r "{$COPY_DIR}/3j1n" "${RUN_DIR}"
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