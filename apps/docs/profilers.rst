3. Profilers
============

The MemSysExplorer framework supports a variety of profiling tools that enable in-depth analysis of memory access patterns and performance characteristics. These tools help researchers and developers optimize memory systems by providing detailed insights into workload behavior.

Currently, the supported profiling tools fall into three main categories:

- **Dynamic Binary Instrumentation (DBI):** Enables instruction-level profiling without modifying the source code.
- **Hardware Performance Counter (HPC):** Provides low-level performance metrics collected directly from hardware.
- **Architectural Simulators:** Simulates system behavior to predict memory performance in different scenarios.

3.1 Available Profilers
------------------------

The following profiling tools are available and documented in detail:

.. toctree::
   :maxdepth: 2
   :caption: Profilers

   profilers/ncu
   profilers/dynamorio
   profilers/sniper
   profilers/perf
   profilers/nvbit

3.1.1 Common Profiler Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the profilers listed above, MemSysExplorer provides a **Common Profiler Library** containing shared utilities used across all profilers:

.. toctree::
   :maxdepth: 2

   profilers/common

3.2 Using the Profilers
------------------------

Each profiler provides unique insights based on workload execution. Depending on your needs, you can select the appropriate profiler to:

1. **Analyze memory access patterns** using DynamoRIO.
2. **Measure low-level GPU performance counters** with Nsight Compute.
3. **Simulate complex workload execution** via Sniper to predict system behavior.

To learn more about how to configure and use each profiler, refer to the specific profiler documentation linked above.


3.3 Getting Started with Profiling
----------------------------------

To begin profiling your application, follow these general steps:

1. **Set up your environment:** Ensure all dependencies, such as CUDA and the required libraries, are installed.
2. **Run profiling commands:** Use the provided command-line options to start profiling with the desired tool.
3. **Analyze results:** Utilize the generated reports and visualizations to gain insights into your application’s memory behavior.

For detailed usage instructions, please refer to the specific profiler documentation linked above.




