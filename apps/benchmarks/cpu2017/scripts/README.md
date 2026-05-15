*Note: I moved these files to a separate scripts/ folder so some of the file paths may be broken, just move back to ../cpu2017/ and it should be fine*

### HPC

1. Compile SPEC 2017 benchmarks
- The config files I used are included, note that some of the flags for certain benchmarks have been changed for Ofast

2. Run and collect data through MemSysExplorer
- `./run_perf.sh` (change FILTER_DIR, LEVEL, and OPTIMIZATION)
- `./verify_results.sh` (change FILTER_DIR, LEVEL, and OPTIMIZATION)
- `./collect_json.sh` (change OPTIMIZATION and TRIAL)

3. Run multiple trials
- `./perf_trials.sh` (change FILTER_DIR, LEVEL, and OPTIMIZATION)

### Desktop

The same files have been slightly modified for a desktop machine without a Slurm job scheduler
- `./run_perf_desktop.sh` (change FILTER_DIR, LEVEL, and OPTIMIZATION)
- `./submit_perf_desktop.sh <FILTER_DIR> <LEVEL>` - I like to submit this script with `sleep 45m;`
- `./perf_trials_desktop.sh` (change FILTER_DIR, LEVEL, and OPTIMIZATION, and then run `./submit_perf_desktop.sh` again)
- `./run_single_benchmark.sh` - I had plans to collect another run with only running one benchmark at a time but wasn't able to, submitting this script will do so automatically

### Data Analysis

- `./fix_metrics.py` - some `memsyspatternconfig*.json` files have wrong calculations due to error in parsing `execution_time` earlier, this should fix it
- `./executable_plots.py` - plots executable size by optimization
- `./plots_py` - plots read_freq and cache_hits
- `./stats_py` - runs paired t-tests for read_freq and cache_hits