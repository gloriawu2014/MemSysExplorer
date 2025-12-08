# Profiling with DynamoRIO

### Features

* Tracks memory access frequency and working set size.
* Separates read/write statistics.
* Configurable through runtime configuration files.
* Supports instruction threshold termination for controlled profiling.
* Protobuf-based output for trace and time-series data.
* HyperLogLog (HLL) approximate working set estimation.
* Windowed sampling with configurable sample sizes.
* Supports integration with MemSysExplorer for streamlined workflows.

## Setup Instructions

### 1. Environment Variables

```bash
export DYNAMORIO_HOME=/path/to/dynamorio
export APPS_HOME=/path/to/MemSysExplorer/apps
```

### 2. Build the DynamoRIO Client

```bash
cd $APPS_HOME/profilers/dynamorio/build
./build.sh
```

## Usage

### A. Standalone Mode

Run a target executable with `memcount` instrumentation:

```bash
$DYNAMORIO_HOME/bin64/drrun -c $APPS_HOME/profilers/dynamorio/build/libmemcount.so \
  -c $APPS_HOME/config/memcount_config.txt -- <executable>
```

**Note:** Use the `-c` flag after the library path to specify a configuration file. Without it, default settings will be used.

### B. Integrated with MemSysExplorer

```bash
cd $APPS_HOME
python3 main.py --profiler dynamorio --action both --executable /path/to/executable
```

With a custom configuration file:

```bash
python3 main.py --profiler dynamorio --action both \
  --config config/memcount_config.txt --executable /path/to/executable
```

## Configuration

The DynamoRIO memcount client can be configured through a text-based configuration file. An example is provided at `config/memcount_config.txt`.

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cache_line_size` | uint | 64 | Cache line size in bytes for address alignment |
| `hll_bits` | uint | 8 | HyperLogLog precision bits (4-16) |
| `sample_hll_bits` | uint | 8 | HLL precision for windowed sampling |
| `sample_window_refs` | uint | 2000 | Number of memory references per sampling window |
| `max_mem_refs` | uint | 8192 | Maximum buffered memory references before flush |
| `enable_trace` | bool | false | Enable detailed protobuf trace output |
| `wss_stat_tracking` | bool | true | Enable working set size statistics tracking |
| `wss_exact_tracking` | bool | true | Enable exact WSS tracking (memory intensive) |
| `wss_hll_tracking` | bool | true | Enable HLL-based approximate WSS tracking |
| `enable_instruction_threshold` | bool | false | Enable instruction count threshold termination |
| `instruction_threshold` | uint64 | 100000000 | Number of instructions before auto-termination |
| `pb_trace_output` | string | "memtrace" | Base name for protobuf trace output |
| `pb_timeseries_output` | string | "timeseries" | Base name for protobuf time-series output |

### Example Configuration File

```ini
# MemCount Configuration File
cache_line_size=64
hll_bits=8
sample_hll_bits=8
sample_window_refs=2000
max_mem_refs=8192

# Output Control
enable_trace=false
wss_stat_tracking=true

# WSS Tracking Methods
wss_exact_tracking=true
wss_hll_tracking=true

# Instruction Threshold Control
# Terminate profiling after N instructions (useful for limiting trace size)
enable_instruction_threshold=true
instruction_threshold=100000000

# Protobuf Output Files
pb_trace_output=memtrace
pb_timeseries_output=timeseries
```

### Instruction Threshold Feature

The instruction threshold feature allows you to automatically terminate profiling after a specified number of instructions have been executed. This is useful for:

* **Limiting trace file sizes** for long-running applications
* **Profiling initialization phases** by setting a low threshold
* **Controlled experiments** requiring consistent instruction counts
* **Testing and debugging** with reproducible cutoff points

When enabled and the threshold is reached, the profiler will:
1. Print a notification message with the exact instruction count
2. Flush all buffered data to output files
3. Print final statistics
4. Terminate the application gracefully

**Example output:**
```
=== Instruction threshold reached: 100000000 instructions ===
Terminating instrumentation and printing final stats...

Instrumentation results:
  saw 45123456 memory references
  number of reads: 32456789
  number of writes: 12666667
  working set size: 123456
```

## Output

The profiler generates multiple types of output depending on configuration:

### Terminal Output
* Total memory reads and writes
* Unique accessed addresses (Working Set Size)
* HLL-based approximate unique cache lines

### Protobuf Files
* **Trace file** (`memtrace_<pid>.pb`): Detailed per-access trace with timestamps, addresses, and read/write type
* **Time-series file** (`timeseries_<pid>.pb`): Windowed statistics including read/write counts, exact and approximate WSS per window

These can be analyzed using MemSysExplorer tools or custom protobuf parsers.

## License

This tool includes components built using [DynamoRIO](https://dynamorio.org/), which is licensed under the [DynamoRIO License](https://dynamorio.org/page_license.html).

