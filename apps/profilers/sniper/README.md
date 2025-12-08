# Profiling with Sniper Architectural Simulator

The **SniperProfiler** integrates the Sniper architectural simulator into MemSysExplorer to evaluate memory system behavior at cycle-level precision. It enables users to run real workloads in a simulated environment and analyze cache/memory interactions, latency, and access statistics.

## Features

- **Cycle-accurate simulation** of multi-core systems
- **Memory hierarchy modeling** (L1, L2, L3, DRAM) with configurable latencies
- **ROI-based simulation control** via instruction count (roi-icount)
- **Size histogram tracking** for memory access patterns
- **Per-core statistics** extraction and aggregation

---

## Workflow Overview

The Sniper integration supports the following actions:

1. **Profiling (`profiling`)** – Runs Sniper with the specified application and configuration.
2. **Metric Extraction (`extract_metrics`)** – Parses Sniper outputs and summarizes memory access statistics.
3. **Both (`both`)** – Executes profiling and extraction sequentially.

---

## Required Arguments

### CLI Flags

| Flag | Required For | Description |
|------|--------------|-------------|
| `-p sniper` | All | Select Sniper profiler |
| `-a <action>` | All | Action: `profiling`, `extract_metrics`, or `both` |
| `--config` | profiling, both | Sniper configuration name (e.g., `gainestown`, `skylake`) |
| `--executable` | profiling, both | Application to simulate |
| `--results_dir` | extract_metrics, both | Directory containing Sniper simulation output |
| `--level` | extract_metrics, both | Memory level to analyze (`l1`, `l2`, `l3`, `dram`) |

### Optional ROI Control Arguments

These arguments enable fine-grained control over the simulation region using instruction counts:

| Flag | Default | Description |
|------|---------|-------------|
| `--roi_mode` | `none` | ROI control mode: `none` (full simulation) or `icount` (instruction count based) |
| `--fastforward` | 0 | Instructions to skip before warmup phase |
| `--warmup` | 0 | Instructions for cache warmup (no timing stats) |
| `--detailed` | 0 | Instructions to simulate in detailed mode (required if `roi_mode=icount`) |
| `--no_cache_warming` | false | Start with cold caches (disable cache warming during fast-forward) |

---

## Example Usage

### Basic: Run simulation and extract metrics

```bash
python3 main.py -p sniper -a both \
    --config gainestown \
    --level l3 \
    --results_dir . \
    --executable ./your_app
```

### With ROI Control (instruction-count based)

Simulate only a specific region of the application by skipping initialization (fast-forward), warming caches (warmup), and then collecting detailed statistics:

```bash
python3 main.py -p sniper -a both \
    --config gainestown \
    --roi_mode icount \
    --fastforward 100000 \
    --warmup 50000 \
    --detailed 100000 \
    --results_dir . \
    --level l3 \
    --executable ./your_app
```

This command:
1. **Fast-forwards** through the first 100,000 instructions (skipped, no simulation)
2. **Warms up** caches for 50,000 instructions (cache model active, no timing)
3. **Simulates in detail** for 100,000 instructions (full timing statistics)

### Extract metrics from previous simulation

```bash
python3 main.py -p sniper -a extract_metrics \
    --results_dir ./sim_output \
    --level dram
```

---

## Memory Level and Size Histogram

The `--level` flag determines which cache level statistics are extracted:

| Level | Access Granularity | Size Histogram |
|-------|-------------------|----------------|
| `l1` | Fine-grained (1, 2, 4, 8, 16, 32, 64 bytes) | Actual instruction access sizes |
| `l2` | Cache-line (64 bytes) | All accesses counted as 64B |
| `l3` | Cache-line (64 bytes) | All accesses counted as 64B |
| `dram` | Cache-line (64 bytes) | All accesses counted as 64B |

### Example Output (L1 level)
```json
{
  "read_size_histogram": {
    "1": 2, "2": 0, "4": 330126, "8": 79553,
    "16": 0, "32": 4, "64": 0, "other": 0
  },
  "write_size_histogram": {
    "1": 1, "2": 0, "4": 107182, "8": 25,
    "16": 0, "32": 1, "64": 0, "other": 2
  }
}
```

### Example Output (L3 level)
```json
{
  "read_size_histogram": {
    "1": 0, "2": 0, "4": 0, "8": 0,
    "16": 0, "32": 0, "64": 144, "other": 0
  },
  "write_size_histogram": {
    "1": 0, "2": 0, "4": 0, "8": 0,
    "16": 0, "32": 0, "64": 1702, "other": 0
  }
}
```

---

## Customizing Memory Modeling

Sniper supports memory hierarchy modeling by allowing users to define latency values in the config file. This enables simulation of custom memory technologies like DRAM, HBM, or NVM.

Modify your config file by setting the appropriate latency parameters:

```ini
[perf_model/l3_cache]
rw_enabled = true
read_access_time = 100
write_access_time = 50000
```

You may apply the same format to:

* `[perf_model/l1_dcache]`
* `[perf_model/l2_cache]`
* `[perf_model/dram]`

This feature lets you evaluate the impact of memory latency changes on workload performance.

### Built-in Configurations

Sniper includes several pre-built CPU configurations:

- `gainestown` - Intel Nehalem (default)
- `skylake` - Intel Skylake
- `haswell` - Intel Haswell

---

## Output Files

After simulation, the following files are generated:

| File | Description |
|------|-------------|
| `sim.out` | Human-readable simulation summary |
| `sim.cfg` | Configuration used for simulation |
| `sim.stats.sqlite3` | SQLite database with detailed statistics |
| `memsys_stats.out` | Extracted memory statistics (CSV format) |
| `memsyspatternconfig_*.json` | PatternConfig output for downstream analysis |
| `memsysmetadata_sniper.json` | System metadata (CPU, GPU, cache info) |

### PatternConfig Output

The final JSON output includes:

```json
{
  "exp_name": "SniperProfilers",
  "benchmark_name": "core_0",
  "read_freq": 2537892.14,
  "total_reads": 144,
  "write_freq": 29996475.15,
  "total_writes": 1702,
  "read_size": 64,
  "write_size": 64,
  "read_size_histogram": { ... },
  "write_size_histogram": { ... },
  "workingset_size": 0,
  "unit": {
    "read_freq": "count/s",
    "write_freq": "count/s",
    "total_reads": "count",
    "total_write": "count"
  }
}
```

---

## ROI Mode: Region of Interest Control

The `--roi_mode` option enables precise control over which portion of your application is simulated in detail. This is essential for:

- **Skipping initialization code** (malloc, file loading, setup)
- **Focusing on steady-state behavior** (main computation loops)
- **Reducing simulation time** while maintaining accuracy

### When to Use ROI Mode

| Scenario | Recommendation |
|----------|----------------|
| Quick full-application profile | `--roi_mode none` (default) |
| Skip initialization, profile main loop | `--roi_mode icount` with fastforward |
| Benchmark with warm caches | `--roi_mode icount` with warmup |
| Profile specific code region | `--roi_mode icount` with all three phases |

### ROI-Icount Parameters Explained

```
--roi_mode icount --fastforward F --warmup W --detailed D
```

| Phase | Instructions | What Happens | Statistics |
|-------|-------------|--------------|------------|
| **Fast-forward** | 0 to F | Instructions executed natively (no simulation) | None |
| **Warmup** | F to F+W | Cache model active, timing model disabled | Cache state only |
| **Detailed** | F+W to F+W+D | Full cycle-accurate simulation | Complete stats |

### Choosing Instruction Counts

**General guidelines:**

1. **Fast-forward (F)**: Set to skip initialization
   - Small programs: 10,000 - 100,000
   - Medium programs: 100,000 - 1,000,000
   - Large programs with heavy init: 1,000,000+

2. **Warmup (W)**: Set to fill caches before measurement
   - Typical: 50,000 - 500,000 instructions
   - Should be at least 2-3x your working set traversal

3. **Detailed (D)**: Set to capture representative behavior
   - Minimum: 100,000 for meaningful stats
   - Recommended: 1,000,000+ for accurate profiles

### Example: Profiling a Matrix Multiplication

```bash
# Profile only the main computation (skip setup)
python3 main.py -p sniper -a both \
    --config gainestown \
    --roi_mode icount \
    --fastforward 500000 \    # Skip matrix allocation/init
    --warmup 200000 \         # Warm caches with first iterations
    --detailed 1000000 \      # Profile steady-state behavior
    --results_dir . \
    --level l3 \
    --executable ./matmul 1024
```

### Example: Quick Test Run

```bash
# Fast test with minimal instructions
python3 main.py -p sniper -a both \
    --config gainestown \
    --roi_mode icount \
    --fastforward 10000 \
    --warmup 5000 \
    --detailed 50000 \
    --results_dir . \
    --level l1 \
    --executable ./test_app
```

### Cold Cache Analysis

To analyze behavior with empty caches (no warming during fast-forward):

```bash
python3 main.py -p sniper -a both \
    --config gainestown \
    --roi_mode icount \
    --fastforward 100000 \
    --warmup 0 \              # No warmup phase
    --detailed 100000 \
    --no_cache_warming \      # Caches start empty
    --results_dir . \
    --level l3 \
    --executable ./your_app
```

### How ROI-Icount Works Internally

When `--roi_mode icount` is enabled, Sniper uses the `roi-icount.py` script:

```
roi-icount:<fastforward>:<warmup>:<detailed>
```

The script registers periodic instruction callbacks to transition between phases:

```
Program Start
    │
    ▼
┌─────────────────┐
│   Fast-forward  │  Instructions: 0 → F
│   (FAST_FORWARD)│  No simulation overhead
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Warmup      │  Instructions: F → F+W
│   (CACHE_ONLY)  │  Cache model active, no timing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Detailed     │  Instructions: F+W → F+W+D
│    (DETAILED)   │  Full cycle-accurate simulation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      Done       │  Remaining instructions in FAST_FORWARD
│  (FAST_FORWARD) │  Statistics saved from detailed phase
└─────────────────┘
```

### Important Notes

1. **Application must execute enough instructions**: Your program must run at least `fastforward + warmup + detailed` instructions, otherwise statistics won't be captured.

2. **Periodic callback granularity**: Sniper checks instruction counts periodically (default: every 1M instructions). For small counts, you may see:
   ```
   [ROI-ICOUNT] Periodic instruction callback too short (1000000)
   ```
   This warning is informational - the simulation still works correctly.

3. **Statistics reflect only the detailed phase**: The extracted metrics represent behavior during the `detailed` instruction window only.

---

## Troubleshooting

* **Missing Sniper binary:**
  Ensure `run-sniper` exists in `profilers/sniper/snipersim/`. Run `make` in the sniper directory to build.

* **Missing stats script:**
  Confirm that `snipermem.py` is present in `snipersim/tools/`.

* **Invalid config block:**
  Check that `[perf_model/*]` sections in the config file are properly defined. Refer to the [official Sniper docs](https://snipersim.org) for more examples.

* **ROI stats not captured:**
  Ensure your application runs long enough to reach the specified instruction counts. Use a test program that executes sufficient instructions.

* **Periodic callback warning:**
  The default callback interval (1M instructions) may be too coarse for small instruction counts. Reduce `core/hook_periodic_ins/ins_global` in your config.

---

## License

This profiler integrates the [Sniper Multicore Simulator](https://snipersim.org), a high-speed, cycle-level architectural simulator developed by the HPC Group at Ghent University and contributors.

> Citation:
>
> Trevor E. Carlson, Wim Heirman, and Lieven Eeckhout.
> *Sniper: Exploring the Level of Abstraction for Scalable and Accurate Parallel Multi-Core Simulation.*
> In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'11)*.
> DOI: [10.1145/2063384.2063454](https://doi.org/10.1145/2063384.2063454)

> License: [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)
