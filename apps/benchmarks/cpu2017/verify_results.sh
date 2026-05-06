#!/bin/bash

# -------- Configurable Paths --------
MAIN_SCRIPT="/home/gwu28/MemSysExplorer/apps/main.py"
CMD_DIR="/home/gwu28/MemSysExplorer/apps/benchmarks/cpu2017/commands"
FILTER_DIR="intspeed"
RUN_TYPE="refrate"
LEVEL="l1"
SPEC_ROOT="/home/gwu28/spec2017"
OPTIMIZATION="O1"

CMD_TYPE=$(echo "$RUN_TYPE" | sed 's/rate//')

declare -A RULES
# Keep your existing colon format
RULES["500.perlbench_r:splitmail"]="mbox.201011: MWCd4AIhgN+tfu+VrKG2CTrWXcPurWNNqUMaFgQd1LmURiTyoJxXLkUOEYDGcQsTnltA8CtkSvZEK1WPZoJc3g"
RULES["500.perlbench_r:diffmail"]="'two017' long"
RULES["500.perlbench_r:checkspam"]="...replaced by ee8c056e01de367b1dc4bd77a4fc5c1b"
RULES["502.gcc_r:opts-O3_-finline"]="gcc-pp.c:463588:22: warning: 'is_too_expensive' used but never defined"
RULES["502.gcc_r:opts-O2"]="gcc-pp.c:463588:22: warning: 'is_too_expensive' used but never defined"
RULES["502.gcc_r:smaller.opts-O3"]="gcc-smaller.c:25469:7: warning: right shift count >= width of type"
RULES["502.gcc_r:ref32.opts-O5"]="ref32.c:6213:17: warning: conflicting types for built-in function 'imaxabs'"
RULES["502.gcc_r:ref32.opts-O3"]="ref32.c:6213:17: warning: conflicting types for built-in function 'imaxabs'"
RULES["505.mcf_r:inp.out"]="done"
RULES["520.omnetpp_r:omnetpp"]="End."
RULES["525.x264_r:pass1"]="x264 [info]: kb/s:1091.73"
RULES["525.x264_r:pass2"]="x264 [info]: kb/s:1006.68"
RULES["525.x264_r:0500"]="x264 [info]: kb/s:2089.52"
RULES["531.deepsjeng_r:ref.out"]="Nodes: 70448753 (63.75% qnodes)"
RULES["541.leela_r:ref.out"]="Hash: F1FB28A473FB5734 Ko-Hash: 3BE0C910509C113F"
RULES["548.exchange2_r:exchange2"]="Puzzle, count, changes:   6 19 12"
RULES["557.xz_r:cld"]="Compressed data 60400412 bytes in length"
RULES["557.xz_r:cpu"]="Compressed data 23280580 bytes in length"
RULES["557.xz_r:input"]="Compressed data 40809580 bytes in length"
RULES["600.perlbench_s:checkspam"]="...replaced by ee8c056e01de367b1dc4bd77a4fc5c1b"
RULES["600.perlbench_s:diffmail"]="'two017' long"
RULES["600.perlbench_s:splitmail"]="<!-- MHonArc v2.6.19&#45;CPU2017 -->"
RULES["602.gcc_s:fipa"]="gcc-pp.c:25469:7: warning: right shift count >= width of type"
RULES["602.gcc_s:1000"]="gcc-pp.c:25469:7: warning: right shift count >= width of type"
RULES["602.gcc_s:24000"]="gcc-pp.c:25469:7: warning: right shift count >= width of type"
RULES["605.mcf_s:inp.out"]="done"
RULES["620.omnetpp_s:omnetpp"]="End."
RULES["625.x264_s:pass1"]="x264 [info]: kb/s:1091.73"
RULES["625.x264_s:pass2"]="x264 [info]: kb/s:1006.68"
RULES["625.x264_s:0500"]="x264 [info]: kb/s:2089.52"
RULES["631.deepsjeng_s:ref.out"]="Nodes: 60404098 (63.35% qnodes)"
RULES["641.leela_s:ref.out"]="Hash: F1FB28A473FB5734 Ko-Hash: 3BE0C910509C113F"
RULES["648.exchange2_s:exchange2.txt"]="Puzzle, count, changes:   6 19 12"
RULES["657.xz_s:cpu"]="Compressed data 1036078272 bytes in length"
RULES["657.xz_s:cld"]="Compressed data 539938872 bytes in length"
RULES["507.cactuBSSN_r:spec"]="Done."
RULES["508.namd_r:namd"]="SUCCESSFUL COMPLETION"
RULES["510.parest_r:ref"]="main 0 [0]: Done! Closing down master"
RULES["511.povray_r:stdout"]=""
RULES["519.lbm_r:lbm"]="minU  : 0.000001 maxU  : 0.005068"
RULES["521.wrf_r:rsl"]="d01 2000-01-25_12:00:00 wrf: SUCCESS COMPLETE WRF"
RULES["526.blender_r:spec"]="Blender quit"
RULES["527.cam4_r:cam4"]="******* END OF MODEL RUN *******"
RULES["538.imagick_r:refrate"]=""
RULES["544.nab_r:1am0"]="...Done, md returns 0"
RULES["549.fotonik3d_r:fotonik3d"]="Ending main loop"
RULES["607.cactuBSSN_s:spec"]="Done."
RULES["619.lbm_s:lbm"]="minU  : 0.000000 maxU  : 0.004780"
RULES["621.wrf_s:rsl"]="d01 2000-01-25_12:00:00 wrf: SUCCESS COMPLETE WRF"
RULES["627.cam4_s:cam4"]="******* END OF MODEL RUN *******"
RULES["628.pop2_s:pop"]="MCT::m_Router::initp_: GSMap indices not increasing...Will correct"
RULES["638.imagick_s:refspeed"]=""
RULES["644.nab_s:3j1n"]="...Done, md returns 0"
RULES["649.fotonik3d_s:fotonik3d"]="Ending main loop"

# -------- Loop through .cmd files --------
find "$CMD_DIR/$FILTER_DIR" -name "*.${CMD_TYPE}.cmd" | while read -r CMD_FILE; do
    CMD_NAME=$(basename "$CMD_FILE" .${CMD_TYPE}.cmd)
    echo -e "\n==> Processing $CMD_NAME (${RUN_TYPE})"

    BENCH_ID="${CMD_NAME}"
    BENCH_DIR="$SPEC_ROOT/benchspec/CPU/$BENCH_ID"
    
    # Define the directory where files actually live
    RUN_DIR="$BENCH_DIR/run/${LEVEL}_${OPTIMIZATION}"

    # Check if directory exists before trying to enter
    if [ ! -d "$RUN_DIR" ]; then
        echo "Skip: Directory not found"
        continue
    fi

    cd "$RUN_DIR"
    RUN_SH="${RUN_DIR}/${BENCH_ID}.${RUN_TYPE}.sh"

    if [[ -f "$RUN_SH" ]]; then
        # Use one loop to find all output filenames from the script
        OUT_FILES=$(grep "^python3" "$RUN_SH" | grep -oP '(?<!2)>+ \s*\K\S+' | grep -v "\.err$")
        
        for OUT_FILE in $OUT_FILES; do
            FULL_PATH="${RUN_DIR}/$OUT_FILE"
            
            if [[ ! -f "$FULL_PATH" ]]; then continue; fi

            RULE_FOUND=false
            TARGET_STR=""
            
            for PATTERN_KEY in "${!RULES[@]}"; do
                # Simple check: does the key contain a colon?
                if [[ "$PATTERN_KEY" == *":"* ]]; then
                    # Split key by colon to match your RULES definition
                    REQUIRED_ID="${PATTERN_KEY%:*}"
                    REQUIRED_WORKLOAD="${PATTERN_KEY#*:}"

                    if [[ "$BENCH_ID" == "$REQUIRED_ID" && "$OUT_FILE" == *"$REQUIRED_WORKLOAD"* ]]; then
                        TARGET_STR="${RULES[$PATTERN_KEY]}"
                        RULE_FOUND=true
                    fi
                else
                    # General match for simple keys
                    if [[ "$OUT_FILE" == *"$PATTERN_KEY"* ]]; then
                        TARGET_STR="${RULES[$PATTERN_KEY]}"
                        RULE_FOUND=true
                    fi
                fi

                if [ "$RULE_FOUND" = true ]; then
                    COUNT=$(grep -F -c "$TARGET_STR" "$FULL_PATH" 2>/dev/null || echo 0)
                    echo "${BENCH_ID}, ${LEVEL}, ${OPTIMIZATION}, ${OUT_FILE}, ${COUNT}"
                    break
                fi
            done
        done
    fi
done
