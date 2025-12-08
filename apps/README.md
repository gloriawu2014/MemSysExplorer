# MemSysExplorer Application Interface

## Authors
- **Duc Anh Nguyen** (Tufts University)  
- **Mark Hempstead** (Tufts University)  
- **Lillian Pentecost** (Amherst College)
- **Olivia Fann** (Amherst College)
- **Tania Perehinets** (Smith College)
---

# Getting Started

This section provides a comprehensive guide to installing and setting up the **MemSysExplorer Application Profiler**, the top layer of the MemSysExplorer framework.

The profiler is designed to extract detailed memory access patterns and performance metrics from real applications, enabling subsequent analysis and modeling of memory technologies such as DRAM, HBM, and NVM. It supports a variety of workloads across both **CPU and GPU platforms**, and acts as the **entry point for generating structured profiling data** for downstream memory simulation and evaluation.

The overall profiling workflow is shown in Figure 1, which includes three categories of supported profilers:

* **Architecture-Independent Profilers:**
  Tools like **DynamoRIO** that use dynamic binary instrumentation (DBI) to analyze memory behavior independent of specific hardware.

* **Architecture-Dependent Profilers:**
  Tools such as **Linux perf** and **NVIDIA Nsight Compute**, which use hardware performance counters to collect platform-specific metrics.

* **Architectural Simulators:**
  Tools like **Sniper Simulator**, which model the entire memory hierarchy to provide cycle-level timing and bandwidth evaluation.

---

## Building the Documentation

To generate the full documentation for the **MemSysExplorer** framework, follow these steps:

### Prerequisites

Make sure you have the required Python packages installed:

* `sphinx`
* `sphinx_rtd_theme`
* Any optional extensions listed in `docs/requirements.txt`

You can install them with:

```bash
pip install -r requirements.txt
```

---

### Set the Environment Variable

Before building the documentation, set the `PYTHONPATH` environment variable to point to the root of the MemSysExplorer app directory (i.e., where `apps/` is located):

```bash
export PYTHONPATH=$(pwd)/apps  # Or the absolute path to your apps/ directory
```

This allows the build system to correctly locate all Python modules during documentation generation.

---

### Build Documentation

To generate the HTML documentation:

```bash
make docs
```

This will:

* Run `sphinx-apidoc` to auto-generate `.rst` files
* Build HTML documentation using Sphinx
* Output files to:

  ```bash
  docs/build/html/
  ```

You can view the docs locally by opening:

```bash
docs/build/html/index.html
```

---

### Clean & Rebuild

To perform a full clean and rebuild:

```bash
make clean_docs && make docs
```

This clears old generated files and regenerates everything from scratch.

---

### Troubleshooting

* Ensure `make` is installed (`sudo apt install make` on Linux)
* Make sure you have activated the correct Python virtual environment (if used)
* If you see import errors like `No module named 'profilers'`, verify that `APPS_HOME` is set and accessible


## Prerequisites

To run the MemSysExplorer Application Profiler, the following software dependencies must be installed:

* **CUDA:** Version 10.x or higher
* **CUDA Toolkit:** Version 10.x or higher
* **Python 3:** Version 3.x or higher
* **DynamoRIO:** Latest version
* **GCC:** Latest version

For profilers that rely on **Hardware Performance Counters (HPC)**—such as **Linux perf** and **NVIDIA Nsight Compute**—additional system configuration is required.

Specifically, you must **enable access to performance counters** at the system level to allow these tools to capture low-level memory statistics. On Linux systems, this often involves modifying `/proc/sys/kernel/perf_event_paranoid` or applying appropriate capabilities using `setcap` or `sudo` privileges.

To enable performance counters on NVIDIA GPUs, please follow the instructions provided at the following link:

[Enabling NVIDIA Performance Counters](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)

---

## Installing the Repository

To get started with MemSysExplorer, begin by cloning the repository from GitHub. This repository contains the full source code for the application profiler, including frontend tools, example workloads, and configuration scripts.

```bash
git clone -b apps_dev https://github.com/duca181/MemSysExplorer.git
```

After cloning, navigate into the repository directory:

```bash
cd MemSysExplorer/apps
```

The repository includes a top-level `Makefile` that supports building and configuring five available profilers:

* **dynamorio**
* **nvbit**
* **sniper**
* **perf**
* **nsight**

Among these, the following tools require compilation or setup:

* **dynamorio** – Build the custom DynamoRIO-based instrumentation frontend.
* **nvbit** – Build the NVBit-based GPU binary instrumentation layer.
* **sniper** – Compile and configure the Sniper architectural simulator.

The remaining two frontends rely on system-level availability:

* **perf** – Ensure Linux `perf` is installed and accessible via your system's performance counters (check `perf_event_paranoid` settings if needed).
* **ncu** – Requires the NVIDIA Nsight Compute CLI and proper GPU driver installation.

To build any of the build-required tools, simply run:

```bash
make <profiler-name>
```

Example:

```bash
make dynamorio
```

If you want to build everything, simply run:

```bash
make
```

### Profiler Registration with `built_profilers.json`

When you run a `make` command for any supported profiler, the `built_profilers.json` file is automatically created and updated to mark that profiler as **active**. This file controls which profilers are dynamically registered and available at runtime.

Example `built_profilers.json`:

```json
{
  "dynamorio": true,
  "nvbit": true,
  "sniper": true,
  "perf": true,
  "ncu": true
}
```

You may **manually set any value to `false`** if you wish to temporarily disable a profiler without deleting its build.
Only profilers marked as `true` will be recognized and used by the MemSysExplorer framework.

---

## Setting Up the Environment

MemSysExplorer provides shell-specific setup scripts to configure the required environment variables for each supported profiler. These scripts initialize toolchain paths, library paths, and compiler includes necessary for running and building different frontend tools.

Depending on your shell, use the appropriate setup script below.

### Using `tcsh` or `csh`

```tcsh
source setup/_setup.csh <option>
```

Available options:

* `1` or `dynamorio` – Sets up the GCC environment for building DynamoRIO-based tools.
* `2` or `cuda` – Sets up the CUDA and Nsight Compute environment.
* `3` or `sniper` – Sets up the environment for the Sniper simulator and Python integration.

Example:

```tcsh
source setup/setup.csh cuda
```

### Using `bash` or `sh`

```bash
source setup/setup.sh <option>
```

The available options are the same:

* `1` or `dynamorio`
* `2` or `cuda`
* `3` or `sniper`

Example:

```bash
source setup/setup.sh sniper
```

### Environment Summary

These scripts ensure that all necessary toolchains and runtime paths are correctly configured for MemSysExplorer's frontend profiling infrastructure. While you are free to configure your own environment manually, these scripts provide a reliable reference for the core variables required to run the profilers effectively.

---

## Example Usage

The MemSysExplorer profiling tool can be executed using the following general command structure:

```bash
python3 main.py --profiler <profiler_name> --action <action> --level <memory-level> --executable /path/to/your/executable
```

Additional arguments such as core count or configuration files may be required depending on the selected profiler.

### Main Flags

* `-p`, `--profiler` — Profiler backend to use: `dynamorio`, `perf`, `sniper`, `nvbit`, `ncu`
* `-a`, `--action` — Action to perform: `profiling`, `extract_metrics`, `both`
* `--level` — Memory level to profile: `l1`, `l2`, `l3`, `dram`
* `--config` — Optional config file for profilers like `sniper` and `ncu`
* `--executable` — Path to the application binary to be profiled

### Profiler-Specific Examples

**DynamoRIO:**

```bash
python3 main.py --profiler dynamorio --action profiling --executable /path/to/your/executable
```

Optionally, specify a configuration file to control profiling behavior (trace output, sampling windows, instruction thresholds, etc.):

```bash
python3 main.py --profiler dynamorio --action profiling --config config/memcount_config.txt --executable /path/to/your/executable
```

**Linux Perf:**

```bash
python3 main.py --profiler perf --action extract_metrics --level l3 --executable /path/to/your/executable
```

**Sniper Simulator:**

```bash
python3 main.py --profiler sniper --action both --level dram \
  --result_dir /path/to/output_dir --config config/sniper.cfg --executable /path/to/your/executable
```

**NVBit:**

```bash
python3 main.py --profiler nvbit --action profiling --executable /path/to/your/executable
```

**Nsight Compute (NCU):**

```bash
python3 main.py --profiler ncu --action extract_metrics --level l1 --executable /path/to/your/executable
```

### Supported Profilers

* `dynamorio` — Dynamic binary instrumentation for CPU
* `perf` — Linux hardware performance counters
* `sniper` — Architectural simulator for multicore systems
* `nvbit` — GPU binary instrumentation for CUDA applications
* `ncu` — NVIDIA Nsight Compute for GPU memory statistics

> **Note:** GPU frontends such as `nvbit` and `ncu` should only be used with GPU workloads. Attempting to profile CPU binaries with these tools will result in undefined behavior or runtime errors.

---

## Profiler-Specific Configurations

Each profiler may require additional specific arguments. Refer to the **Profilers** section for detailed information on the necessary flags and options for each profiler.

## References

1. L. Pentecost et al., "NVMExplorer: A Framework for Cross-Stack Comparisons of Embedded Non-Volatile Memories," *IEEE HPCA*, 2022.  
2. A. Hankin et al., "Evaluation of Non-Volatile Memory Based Last Level Cache Given Modern Use Case Behavior," *IEEE IISWC*, 2019.  
3. M. Lui et al., "Towards Cross-Framework Workload Analysis via Flexible Event-Driven Interfaces," *IEEE ISPASS*, 2018.  
4. N. Nethercote and J. Seward, "Valgrind: A Framework for Heavyweight Dynamic Binary Instrumentation," *ACM SIGPLAN Notices*, 2007.  
5. NVIDIA Corporation, "NVIDIA Nsight Systems," 2024.

---



