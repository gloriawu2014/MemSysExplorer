# MemSysExplorer Tools

Collection of analysis and utility tools for memory profiling data.

---

## Profiler Output Parsers

### 1. `trace_parser.py`
Parse binary protobuf memory traces into human-readable formats.

**Purpose:** Convert detailed per-access memory traces (`.pb` files) to JSON/CSV for analysis.

**Usage:**
```bash
# Show summary statistics
python tools/trace_parser.py memtrace_12345.pb --format summary

# Export to CSV (limit to first 10,000 events)
python tools/trace_parser.py memtrace_12345.pb --format csv --limit 10000 --output trace.csv

# Filter by thread ID
python tools/trace_parser.py memtrace_12345.pb --thread 10534 --format json --output thread_trace.json
```

**Output formats:** `summary` (default), `json`, `csv`

---

### 2. `timeseries_parser.py`
Parse binary protobuf time-series WSS metrics into human-readable formats.

**Purpose:** Convert windowed WSS samples (`.pb` files) to JSON/CSV for trend analysis.

**Usage:**
```bash
# Show summary statistics
python tools/timeseries_parser.py timeseries_12345.pb --format summary

# Export all samples to CSV
python tools/timeseries_parser.py timeseries_12345.pb --format csv --output wss_samples.csv

# Filter by thread ID
python tools/timeseries_parser.py timeseries_12345.pb --thread 10534 --format json
```

**Output formats:** `summary` (default), `json`, `csv`

---

## Memory Analysis Tools

### 3. `reuse_distance.py`
Calculate reuse distance for memory addresses from trace files.

**Purpose:** Analyze cache behavior by computing reuse distance (number of unique addresses between consecutive accesses to the same address).

**Usage:**
```bash
# Calculate reuse distance from CSV trace
python tools/reuse_distance.py trace.csv --output reuse_dist.csv

# Use windowed tracking (1000-address window)
python tools/reuse_distance.py trace.csv --window-size 1000 --output reuse_dist.csv
```

**Supports formats:** CSV (`timestamp,addr,op,size`) or legacy trace format

---

## Metadata Extraction Tools

These tools are used internally by `BaseMetadata.py` to collect system information.

### 4. `environment_capture.py`
Capture environment variables and system information.

**Purpose:** Python wrapper for C library that captures runtime environment details.

**Usage (programmatic):**
```python
from environment_capture import EnvironmentCapture

env = EnvironmentCapture()
print(f"OS: {env.os_name}")
print(f"User: {env.get_variable('USER')}")
print(f"All vars: {env.get_all_variables()}")
```

---

### 5. `makefile_parser.py`
Extract build metadata from Makefiles.

**Purpose:** Parse Makefiles to extract build configuration (targets, variables, compiler settings, versions).

**Usage (programmatic):**
```python
from makefile_parser import MakefileParser

parser = MakefileParser()
metadata = parser.parse_makefile('/path/to/Makefile')
print(metadata['variables'])
print(metadata['targets'])
```

---

### 6. `profiler_flag_parser.py`
Extract profiler-specific command flags from source code.

**Purpose:** Analyze profiler Python files to discover available flags and configuration options.

**Usage (programmatic):**
```python
from profiler_flag_parser import ProfilerFlagParser

parser = ProfilerFlagParser()
flags = parser.extract_flags('/path/to/profiler_dir')
print(flags['commands'])
print(flags['configuration_flags'])
```

---

## Quick Reference

| Tool | Input | Output | Use Case |
|------|-------|--------|----------|
| `trace_parser.py` | `memtrace_*.pb` | JSON/CSV/summary | Per-access memory trace analysis |
| `timeseries_parser.py` | `timeseries_*.pb` | JSON/CSV/summary | WSS trends over time |
| `reuse_distance.py` | `trace.csv` | `reuse_dist.csv` | Cache reuse distance analysis |
| `environment_capture.py` | - | Python dict | System environment metadata |
| `makefile_parser.py` | `Makefile` | Python dict | Build configuration metadata |
| `profiler_flag_parser.py` | Profiler dir | Python dict | Profiler flags metadata |

---

## Common Workflows

### Analyze Memory Behavior from Protobuf Output

```bash
# 1. Generate protobuf files from DynamoRIO
drrun -c libmemcount.so -config memcount_config.txt -- ./myapp

# 2. Parse memory traces
python tools/trace_parser.py memtrace_*.pb --format csv --limit 100000 --output trace.csv

# 3. Calculate reuse distance
python tools/reuse_distance.py trace.csv --output reuse_dist.csv

# 4. Analyze WSS over time
python tools/timeseries_parser.py timeseries_*.pb --format json --output wss_trends.json
```

### Generate Complete Metadata JSON

```python
from profilers.dynamorio.drio_Metadata import DrioMetadata
import json

# Create metadata (automatically includes protobuf file references)
metadata = DrioMetadata(output_dir=".")

# Save to JSON
with open("metadata.json", "w") as f:
    json.dump(metadata.as_dict(), f, indent=2)
```

---

## Installation Notes

### Protobuf Parsers
Python protobuf bindings are **automatically generated** when you build the common library:
```bash
make common  # Generates *_pb2.py files automatically
```

No manual `protoc` commands needed!

### Metadata Tools
No additional dependencies - uses standard Python libraries and common library integration.

---

## Getting Help

Run any tool with `-h` or `--help` for detailed usage:
```bash
python tools/trace_parser.py --help
python tools/timeseries_parser.py --help
python tools/reuse_distance.py --help
```
