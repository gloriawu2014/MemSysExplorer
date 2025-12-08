
# Benchmark Hooking and Profiling Integration (Development Workspace)

This repository serves as an active development environment for hooking standardized benchmark suites into the MemSysExplorer profiling and simulation framework. The goal is to create a clean, unified wrapper interface that enables seamless integration of various profilers and simulators (e.g., Sniper, NVBit, Perf, DynamoRIO) with benchmark workloads.

## Purpose

This directory is under **active development** and serves as a testing ground for integrating benchmark execution into the profiling tools supported by MemSysExplorer. The focus is on enabling dynamic command construction, dataset management, and profiler orchestration through a Python interface (`main.py`).

## Supported Benchmarks (Planned)

We are following a standardized structure to support and expand benchmark integration as shown below:

| Benchmark Dataset    | Status         | Profilers Integrated                            | Description                                                                 |
| -------------------- | -------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| **MLPerf Inference** | ❌ Not started  | Perf (HPC), Nsight Compute (HPC), NVBit (DBI)   | Industry-standard ML benchmarks for training and inference tasks.           |
| **SPEC2017**         | ⚠️ In Progress | Perf (HPC), Sniper (Simulator), DynamoRIO (DBI) | CPU-intensive benchmarks for compute, memory, and compiler efficiency.      |
| **CUDA SDK Samples** | ⚠️ In Progress | Nsight Compute (HPC), NVBit (DBI)               | GPU-accelerated CUDA examples, useful for low-level profiling and analysis. |

Current development is concentrated on **SPEC2017**, with functional wrappers and simulation flows being built for multiple workloads using `.cmd` input files. These commands were curated from:

* [SPECCPU2017Commands GitHub](https://github.com/donggyukim/Speckle-2017/tree/no-c-ext)
* [Tosiron Adegbija’s command reference PDF](https://tosiron.com/papers/2018/SPEC2017_commands.pdf)

The SPEC command lines were extracted using `specinvoke -n` and follow conventions reported in:

> A. Limaye, T. Adegbija, “A Workload Characterization of the SPEC CPU2017 Benchmark Suite,” ISPASS 2018.

## Layout Overview

```
.
├── commands/               # Contains .cmd files for each benchmark case
├── run_spec2017.sh         # Main automation script to prepare and simulate workloads
├── main.py                 # Python profiler wrapper (in MemSysExplorer)
├── spec_runs/              # Output directories per benchmark execution
├── config/                 # Configuration files for simulator setup
```

## Features

* Auto-detect and wrap benchmark commands
* Copies input/output data files and generates working directories
* Integrates with MemSysExplorer `main.py` interface
* Supports batch profiling of SPEC workloads across Sniper

## Status

This repo is experimental and not yet stable. Benchmarks and command interfaces may change as development progresses.

## Usage Example

Make sure you specify your SPEC2017 root path and the MemSys `APPSHOME` directory in the environment or script before running:

```bash
./run_spec2017.sh     # Automatically loops through selected SPEC2017 benchmarks
```

