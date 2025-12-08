Perf Documentation
==============================

Perfs leverage **Hardware Performance Counters (HPC)** to collect low-level hardware metrics from CPUs. 
These profiler provide valuable insights into memory access patterns, kernel performance, and overall system efficiency. 

The profiling workflow in MemSysExplorer consists of two core actions, as provided by the main interface:

1. **Profiling (`profiling`)** – Captures runtime execution metrics by specifying the required executable.
2. **Metric Extraction (`extract_metrics`)** – Analyzes generated reports to extract memory and performance-related metrics.

When using the `both` action, profiling and metric extraction are performed sequentially.

.. important::

   **MemSysExplorer GitHub Repository**

   Refer to the codebase for the latest update: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps/profilers/perf

   To learn more about license terms and third-party attribution, refer to the :doc:`../licensing` page.

Required Arguments
------------------

To execute Nsight Profilers, specific arguments are required based on the chosen action. The necessary arguments are defined in the code as follows:
Perf can analyze three lavels of memory: l1, l2, and l3. The level of memory to be analyzed is specified using the `level` argument.

.. code-block:: python

    @classmethod
    def required_profiling_args(cls):
        """
        Return required arguments for the profiling method.
        """
        return ["executable", "level"]

    @classmethod
    def required_extract_args(cls, action):
        """
        Return required arguments for the extract_metrics method.
        """
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return []

Example Usage
-------------

Below are three examples of how to execute the profiling tool with different actions:

- **Profiling the application:**

  .. code-block:: bash

     python main.py --profiler perf --action profiling --level l1 --executable ./executable 

- **Extracting metrics from an existing report:**

  .. code-block:: bash

     python main.py --profiler perf --action extract_metrics --level l1 --report_file ./report_file.ncu-rep

- **Performing both profiling and metric extraction:**

  .. code-block:: bash

     python main.py --profiler perf --action both --level l1 --executable ./executable

.. note::  
        Perf is orgninally designed to work on Intel CPUs, so come metrics might be unavailable on other architectures.

Sample Output
-------------

This profiler generates output traces that follow the standardized format defined by the MemSysExplorer Application Interface.
     
Troubleshooting
---------------

If you encounter issues while running or extracting metrics using the Perf profiler, consider the following checks:

- **Ensure `perf` is installed and available** in your environment.

  You can verify this by running:

  .. code-block:: bash

     which perf
     perf --version

- **Check whether hardware performance counters are accessible**.

  On many Linux systems, user access to counters is restricted by default. You may need to reduce the kernel's perf event restriction level:

  .. code-block:: bash

     sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'

  Alternatively, configure access with:

  .. code-block:: bash

     sudo sysctl -w kernel.perf_event_paranoid=-1

- **Ensure you are running on a supported architecture.**
  MemSysExplorer’s `perf` integration is designed and tested primarily on **Intel CPUs**. Some counters may be missing or unsupported on AMD, ARM, or virtualized environments.

- **Check for compatibility with your Linux kernel version and `perf` version.**

  MemSysExplorer assumes compatibility with `perf` versions is above 6.x. Run:

  .. code-block:: bash

     uname -r
     perf --version

  to check kernel and `perf` versions.

If the profiler fails silently or skips metrics, it's likely due to unsupported or inaccessible counters. Consider testing a different memory level (`--level l1`, `l2`, or `l3`) or switching to another compatible platform.


