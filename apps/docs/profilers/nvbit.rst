NVBit Profiler Documentation
============================

The **NVBitProfiler** integrates NVIDIA's NVBit (NVIDIA Binary Instrumentation Tool) with MemSysExplorer to provide dynamic instrumentation of memory accesses at the GPU level. It collects memory access frequency and working set statistics during runtime by intercepting and analyzing memory instructions on CUDA binaries.

This lightweight profiler is suitable for evaluating the memory access behavior of GPU kernels without modifying source code.

.. important::

   **MemSysExplorer GitHub Repository**

   Refer to the codebase for the latest update: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps/profilers/nvbit

   To learn more about license terms and third-party attribution, refer to the :doc:`../licensing` page.


Overview
--------

NVBit leverages a custom shared object (`nvbit.so`) injected at runtime using the `LD_PRELOAD` mechanism. During execution, the instrumentation layer collects various memory-related statistics and writes them to a summary file.

The profiling workflow includes:

1. **Profiling (`profiling`)** – Runs the target GPU application with NVBit instrumentation enabled.
2. **Metric Extraction (`extract_metrics`)** – Parses the generated `global_summary.txt` file to extract relevant memory access statistics.

Required Arguments
------------------

The profiler defines required arguments for each stage as follows:

.. code-block:: python

    @classmethod
    def required_profiling_args(cls):
        return ["executable"]

    @classmethod
    def required_extract_args(cls, action):
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return []

Usage Example
-------------

- **Run NVBit profiling on a GPU application:**

  .. code-block:: bash

     python3 main.py --profiler nvbit --action profiling --executable ./your_gpu_app

- **Extract metrics from NVBit output:**

  .. code-block:: bash

     python3 main.py --profiler nvbit --action extract_metrics --report_file global_summary.txt

- **Do both profiling and extraction sequentially:**

  .. code-block:: bash

     python3 main.py --profiler nvbit --action both --executable ./your_gpu_app

Sample Output
-------------

The NVBit profiler in MemSysExplorer is based on sample instrumentation code provided by the NVBit SDK. We extend these scripts to extract memory traces and runtime statistics. The output consists of two main components:

- **Raw memory traces** in the following format:

  .. code-block:: text

     MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - 
     Kernel name %s - grid launch id %ld - grid size %d,%d,%d 
     - block size %d,%d,%d - nregs %d - shmem %d - cuda stream 
     id %ld

- **MemSysExplorer memory statistics**, exported in a structured format for downstream analysis.

Instrumentation Behavior
~~~~~~~~~~~~~~~~~~~~~~~~

Instrumentation is injected dynamically using `LD_PRELOAD`, which loads the following shared object during execution:

.. code-block:: text

   profilers/nvbit/lib/nvbit.so

This shared object hooks into CUDA kernels and records memory access events. Statistics are written to a text file (default: `global_summary.txt`), which includes aggregated memory reads, writes, and access sizes.

Environment Notes
~~~~~~~~~~~~~~~~~

- Ensure the `nvbit.so` library exists in `profilers/nvbit/lib/` and has execute permissions.
- The `LD_PRELOAD` environment variable is set automatically by the wrapper before program launch.
- Only CUDA-enabled applications with kernel launches are supported. Pure CPU binaries are ignored silently.

Additional Information
~~~~~~~~~~~~~~~~~~~~~~

- The base tracing scripts are derived from official NVBit examples.
- For more details and original implementation references, visit the NVBit official site:  
  https://nvbit.github.io/NVBit/

Troubleshooting
---------------

1. **Missing `nvbit.so`:**
   Ensure that `lib/nvbit.so` exists relative to the `NVBitProfilers` script.

2. **Permission Denied or Silent Exit:**
   Confirm that your application is CUDA-based and built for a supported GPU architecture.

3. **No Output Generated:**
   Make sure your application launches actual CUDA kernels. NVBit will not instrument empty or stub kernels.


