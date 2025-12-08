.. MemSysExplorer Application documentation master file, created by
   sphinx-quickstart on Thu Jan 23 10:56:36 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MemSysExplorer Application Interface
=========================================

Welcome to the **MemSysExplorer** Application Interface documentation!

**MemSysExplorer** is a comprehensive profiling and analysis framework designed to capture and understand memory behavior across diverse computing environments. It supports a wide range of workloads—including **CPU and GPU applications**—and is optimized for both real-system profiling and simulation-driven studies.

This documentation focuses on the **application layer** of MemSysExplorer, where users can **profile workloads**, extract **memory traces and performance metrics**, and prepare data for in-depth memory technology analysis. This layer serves as the **entry point for memory characterization**, offering a flexible frontend for data collection across diverse execution environments.

The collected memory statistics and traces from this layer are designed to be **ported into the lower layers** of MemSysExplorer, where they are used to **analyze and model memory technologies**—including DRAM, HBM, NVM, and hybrid memory systems—with respect to latency, bandwidth, energy, and other architectural constraints.

By leveraging a suite of modular profiling tools, MemSysExplorer enables researchers and system architects to:

- Understand memory access patterns, bandwidth usage, and spatial/temporal locality.
- Evaluate architectural trade-offs using real or simulated memory traces.
- Drive technology-aware simulations and modeling pipelines with accurate application-driven data.

This interface serves as the foundation for collecting structured profiling data that feeds the analytical backend of MemSysExplorer. The framework can be described as in Figure 1.

You can find the source code and contribute to the project on GitHub:

.. important::

   **MemSysExplorer GitHub Repository**

   Explore the codebase, raise issues, or contribute at: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps

   To learn more about license terms and third-party attribution, refer to the :doc:`licensing` page.

.. figure:: _static/Framework.png
   :alt: MemSysExplorer Application Profiling Framework
   :width: 600px
   :align: center

   *Figure 1: MemSysExplorer Application Profiling Framework*

This framework is designed to collect memory behavior data from applications using a **modular frontend architecture**. It consists of three primary types of profilers:

- **Binary Instrumentation**: Tools like DynamoRIO or NVBit dynamically insert analysis hooks into binaries to trace memory accesses.
- **Hardware Performance Counters**: Tools like `perf` and `Nsight Compute` collect aggregate statistics from CPU/GPU counters with minimal overhead.
- **Architectural Simulators**: Tools like Sniper simulate CPU memory systems at cycle-level detail for accurate performance modeling.

The frontends consume both **workload binaries** and **configuration files**, and produce three key types of output:

- **Memory Statistics**: Quantitative metrics like cache hit rates, memory throughput, or latency.
- **Memory Traces**: Fine-grained logs of memory access events for detailed offline analysis.
- **Metadata**: Descriptive tags and configuration data that contextualize profiling results.

.. note::

 The dotted line to **Memory Traces** indicates partial support—only some profilers generate detailed memory traces, depending on their instrumentation capabilities.

1. Additional Resources
-----------------------
For more details, check out the following sections:

- :doc:`metadata` – Information on profiling metadata and result structures.
- :doc:`profilers` – Overview of supported profilers and how to use them.
- :doc:`developing` – Guide for contributing to and extending the framework.
- :doc:`licensing` – Licensing and attribution information for integrated profiling tools.
- :doc:`packaging` – Structure and modules of the MemSysExplorer package.

2. Frontend Definition
----------------------

A **frontend** in MemSysExplorer is defined as the software interface responsible for capturing execution data (e.g., memory traces, cache statistics, hardware counters) from applications running on real hardware or in simulation. Frontends operate on the profiling layer and act as **data collection agents** for system-wide memory characterization.

Frontends can be classified into the following **profiler classes**:

- **Dynamic Instrumentation Profilers** – Insert analysis hooks into runtime application binaries.
- **Hardware Performance Counter Profilers** – Extract statistics from CPU/GPU hardware counters.
- **Architectural Simulators** – Provide synthetic, cycle-accurate profiling in controlled environments.

2.1 Dynamic Binary Instrumentation Profilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic instrumentation profilers insert analysis hooks at runtime to extract fine-grained information such as memory access instructions, operand values, and instruction sequences.

**DynamoRIO (CPU)**

- **What it does**: Instruments x86 binaries to record instruction-level memory access traces.
- **Use case**: Enables deep insight into read/write operations, accessed addresses, and control flow in CPU applications.
- **Output**: Instruction-level memory access traces, suitable for post-analysis or event-driven simulation.

**NVBit (GPU)**

- **What it does**: Dynamically instruments CUDA binaries to intercept memory accesses at the PTX or SASS level.
- **Use case**: Captures detailed per-thread or per-warp GPU memory behavior.
- **Output**: Memory access trace files per warp, thread, or kernel execution context.

2.2 Hardware Performance Counter Profilers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These profilers utilize hardware counters available in CPUs and GPUs to capture memory-related statistics without modifying or instrumenting application binaries.

**Perf (CPU)**

- **What it does**: Uses the Linux `perf_event` interface to record hardware events (e.g., cache misses, bandwidth, instruction cycles).
- **Use case**: Provides a low-overhead method for system-wide memory and performance monitoring.
- **Output**: Aggregated metrics including memory bandwidth, cache utilization, and CPU cycles.

**• Nsight Compute (GPU)**

- **What it does**: Captures kernel-level memory throughput, cache hit/miss statistics, warp execution stalls, and related hardware behaviors.
- **Use case**: Ideal for identifying GPU kernel inefficiencies and optimizing memory-bound workloads.
- **Output**: Structured reports with L2/DRAM access metrics and per-kernel analysis.

2.3 Architectural Simulators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulators model processor and memory hierarchy behavior at cycle-level or instruction-level granularity, enabling full-stack performance studies and architectural exploration.

**Sniper (CPU)**

- **What it does**: A multicore architectural simulator that models L1/L2/L3 cache hierarchies, DRAM access latencies, and inter-core interconnects.
- **Use case**: Allows simulation of realistic memory behavior under configurable hardware parameters.
- **Output**: Cycle-accurate memory access statistics across memory hierarchy levels (L1, L2, L3, DRAM).

3. Frontend Overview
----------------------
MemSysExplorer supports a modular and extensible **frontend architecture** for collecting memory traces and performance statistics from real applications and simulators. These frontends serve as the entry point for profiling workloads across both CPU and GPU platforms.

The current implementation provides stable support for several CPU profiling tools, including dynamic binary instrumentation, hardware performance counters, and architectural simulation. **GPU frontend support is actively under development**, with preliminary integration of NVBit and Nsight Compute for memory trace extraction and metric collection.

3.1 **CPU Frontend**
~~~~~~~~~~~~~~~~~~~~~
.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Frontend**
     - **Description**
   * - **DynamoRIO**
     - Low-level instruction tracing and memory access analysis.
   * - **Perf**
     - Linux-based tool for collecting hardware-level metrics.
   * - **Sniper**
     - Cycle-level simulator for evaluating CPU memory hierarchy and timing behavior.

3.2 **GPU Frontend**
~~~~~~~~~~~~~~~~~~~~~

Currently, **MemSysExplorer supports GPU profiling exclusively on NVIDIA platforms**. This is due to the reliance on NVIDIA-specific tooling (e.g., NVBit and Nsight Compute) that enables memory instrumentation and performance monitoring at the PTX/SASS or warp level. AMD's GPU can be supported in the future.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - **Frontend**
     - **Description**
   * - **NVBit**
     - NVIDIA’s dynamic instrumentation framework for injecting memory analysis at the PTX/SASS level.
   * - **Nsight Compute**
     - NVIDIA's performance analyzer providing warp-level, kernel-level, and memory hierarchy statistics.

4. Frontend Capabilities
------------------------
This section highlights the key capabilities of each supported frontend tool in MemSysExplorer. Each profiler offers different levels of support for **memory statistics collection**, **trace generation**, and **modeling integration**.

- **DynamoRIO** and **NVBit** serve as dynamic binary instrumentation tools capable of analyzing fine-grained memory operations.
- **Perf** and **Nsight Compute** provide hardware-level performance metrics and cache statistics via performance counters.
- **Sniper** acts as an architectural simulator capable of cycle-accurate memory modeling and latency-aware analysis.

.. note::

   DBI tools alone **cannot produce full memory statistics** such as:

   - Cache hit/miss rates
   - Memory-level latencies
   - Actual bandwidth utilization at L2/DRAM levels

        These tools require either:

        - **Hardware counter support** (e.g., via `perf` or Nsight Compute), or
        - **Cache simulation models** integrated into the analysis pipeline.

The table below summarizes which components are supported by each frontend, indicating their roles in driving application profiling and downstream memory technology modeling.

.. list-table:: **Profiler Tool Comparison**
   :widths: 15 35 25 25
   :header-rows: 1

   * - **Profiler Tool**
     - **Memory Statistics**
     - **Trace Collection**
     - **Memory Modeling**

   * - **DynamoRIO**
     - ✅ Uses memory instructions to  
       infer read/write operations.
     - ⚠️ Can be supported
     - ⚠️ Can be supported

   * - **NVBit**
     - ✅ Accumulates memory read/write  
       events to infer behavior.
     - ✅ Captures traces per warp,  
       context, or CTA
     - ❌ Not supported

   * - **Perf / Nsight Compute**
     - ✅ User-defined memory/cache  
       statistics collection
     - ❌ Not supported
     - ❌ Not supported

   * - **Sniper (Simulator)**
     - ✅ Targeted stats for  
       user-defined memory levels
     - ❌ Not supported
     - ✅ Customizable latency and  
       cache hierarchy modeling

5. Memory Statistics Level
--------------------------
Some frontend tools in MemSysExplorer are capable of capturing memory statistics at different levels of the memory hierarchy, such as L1, L2, L3 caches, and DRAM. These levels are essential for analyzing the spatial and temporal behavior of workloads as they traverse the memory subsystem.

Different tools offer **varying levels of support** depending on their architecture, access granularity, and integration with hardware or simulation backends.

**Why are different levels supported?**

- **Perf (CPU)** and **Nsight Compute (GPU)** has **partial support** for some levels of the memory hierarchy but not all. This is due to hardware and kernel dependencies — certain counters may not be available on all architectures or kernel versions. More comprehensive support is expected as newer processor models and `perf_event` capabilities expand.

- **Sniper Simulator** provides **full coverage** across all memory hierarchy levels (L1, L2, L3, DRAM) because it is an **architectural simulator**. Sniper can simulate and monitor each memory level in a controlled environment, offering deep visibility into cache hits, misses, latencies, and interconnect behavior.


.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 20

   * - **Profiler Tool**
     - **L1 Cache**
     - **L2 Cache**
     - **L3 Cache**
     - **DRAM**

   * - **Nsight Compute**
     - ❌ Not Supported
     - ✅ Supported
     - ❌ Not Supported
     - ✅ Supported

   * - **Perf**
     - ✅ Supported
     - ✅ Supported
     -    Partially Supported
     - ❌ Not Supported

   * - **Sniper Simulator**
     - ✅ Supported
     - ✅ Supported
     - ✅ Supported
     - ✅ Supported


.. note::
        - ✅ Supported: Fully supported with traceable statistics.
        - ❌ Not Supported: No visibility at this level with the given tool.
        - Partially Supported: Available on some hardware or kernel versions; limited or indirect support.


6. Getting Started
------------------
If you are new to MemSysExplorer, begin with the :doc:`getting_start` section to set up and start using the framework.

Thank you for using MemSysExplorer! For any questions, please refer to the respective sections or reach out to the project maintainers.
   
.. toctree::
   :maxdepth: 2
   :caption: Table of Contents

   getting_start
   metadata
   profilers
   developing
   licensing
   packaging
