Sniper Simulator Documentation
==============================

The **SniperProfiler** integrates the Sniper architectural simulator into MemSysExplorer to evaluate memory system behavior at cycle-level precision. It allows users to configure memory hierarchy models, run simulations with real applications, and extract detailed memory performance statistics.

Sniper is particularly useful for modeling the impact of emerging memory technologies by simulating read/write latencies at different levels of the memory hierarchy (L1, L2, L3, DRAM).

.. important::

   **MemSysExplorer GitHub Repository**

   Refer to the codebase for the latest update:: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps/profilers/sniper

   To learn more about license terms and third-party attribution, refer to the :doc:`../licensing` page.

Features
--------

- **Cycle-accurate simulation** of multi-core systems
- **Memory hierarchy modeling** (L1, L2, L3, DRAM) with configurable latencies
- **ROI-based simulation control** via instruction count (roi-icount)
- **Size histogram tracking** for memory access patterns
- **Per-core statistics** extraction and aggregation

Workflow Overview
-----------------

The Sniper integration in MemSysExplorer supports three actions:

1. **Profiling (`profiling`)** – Runs the Sniper simulation using the provided configuration file and application binary.
2. **Metric Extraction (`extract_metrics`)** – Processes the Sniper output to extract memory access statistics from the simulation.
3. **Both (`both`)** – Executes profiling and metric extraction in sequence.

Required Arguments
------------------

The following arguments are required for each mode:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Flag
     - Required For
     - Description
   * - ``-p sniper``
     - All
     - Select Sniper profiler
   * - ``-a <action>``
     - All
     - Action: ``profiling``, ``extract_metrics``, or ``both``
   * - ``--config``
     - profiling, both
     - Sniper configuration name (e.g., ``gainestown``, ``skylake``)
   * - ``--executable``
     - profiling, both
     - Binary to run during simulation
   * - ``--results_dir``
     - extract_metrics, both
     - Directory where Sniper output is stored
   * - ``--level``
     - extract_metrics, both
     - Memory level to analyze (``l1``, ``l2``, ``l3``, or ``dram``)

Optional ROI Control Arguments
------------------------------

These arguments enable fine-grained control over the simulation region using instruction counts:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Flag
     - Default
     - Description
   * - ``--roi_mode``
     - ``none``
     - ROI control mode: ``none`` (full simulation) or ``icount`` (instruction count based)
   * - ``--fastforward``
     - 0
     - Instructions to skip before warmup phase
   * - ``--warmup``
     - 0
     - Instructions for cache warmup (no timing stats)
   * - ``--detailed``
     - 0
     - Instructions to simulate in detailed mode (required if ``roi_mode=icount``)
   * - ``--no_cache_warming``
     - false
     - Start with cold caches (disable cache warming during fast-forward)

Example Usage
-------------

**Basic: Run simulation and collect stats:**

.. code-block:: bash

   python3 main.py -p sniper -a both \
       --config gainestown \
       --level l3 \
       --results_dir . \
       --executable ./your_app

**With ROI Control (instruction-count based):**

.. code-block:: bash

   python3 main.py -p sniper -a both \
       --config gainestown \
       --roi_mode icount \
       --fastforward 100000 \
       --warmup 50000 \
       --detailed 100000 \
       --results_dir . \
       --level l3 \
       --executable ./your_app

This command:

1. **Fast-forwards** through the first 100,000 instructions (skipped, no simulation)
2. **Warms up** caches for 50,000 instructions (cache model active, no timing)
3. **Simulates in detail** for 100,000 instructions (full timing statistics)

**Only extract metrics from prior run:**

.. code-block:: bash

   python3 main.py -p sniper -a extract_metrics \
       --results_dir ./sim_output \
       --level dram

ROI Mode: Region of Interest Control
------------------------------------

The ``--roi_mode`` option enables precise control over which portion of your application is simulated in detail. This is essential for:

- **Skipping initialization code** (malloc, file loading, setup)
- **Focusing on steady-state behavior** (main computation loops)
- **Reducing simulation time** while maintaining accuracy

When to Use ROI Mode
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Scenario
     - Recommendation
   * - Quick full-application profile
     - ``--roi_mode none`` (default)
   * - Skip initialization, profile main loop
     - ``--roi_mode icount`` with fastforward
   * - Benchmark with warm caches
     - ``--roi_mode icount`` with warmup
   * - Profile specific code region
     - ``--roi_mode icount`` with all three phases

ROI-Icount Parameters Explained
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   --roi_mode icount --fastforward F --warmup W --detailed D

.. list-table::
   :header-rows: 1
   :widths: 20 20 35 25

   * - Phase
     - Instructions
     - What Happens
     - Statistics
   * - **Fast-forward**
     - 0 to F
     - Instructions executed natively (no simulation)
     - None
   * - **Warmup**
     - F to F+W
     - Cache model active, timing model disabled
     - Cache state only
   * - **Detailed**
     - F+W to F+W+D
     - Full cycle-accurate simulation
     - Complete stats

Choosing Instruction Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Cold Cache Analysis
~~~~~~~~~~~~~~~~~~~

To analyze behavior with empty caches (no warming during fast-forward):

.. code-block:: bash

   python3 main.py -p sniper -a both \
       --config gainestown \
       --roi_mode icount \
       --fastforward 100000 \
       --warmup 0 \
       --detailed 100000 \
       --no_cache_warming \
       --results_dir . \
       --level l3 \
       --executable ./your_app

How ROI-Icount Works Internally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``--roi_mode icount`` is enabled, Sniper uses the ``roi-icount.py`` script:

.. code-block:: text

   roi-icount:<fastforward>:<warmup>:<detailed>

The script registers periodic instruction callbacks to transition between phases:

.. code-block:: text

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

Important Notes
~~~~~~~~~~~~~~~

1. **Application must execute enough instructions**: Your program must run at least ``fastforward + warmup + detailed`` instructions, otherwise statistics won't be captured.

2. **Periodic callback granularity**: Sniper checks instruction counts periodically (default: every 1M instructions). For small counts, you may see a warning - this is informational and the simulation still works correctly.

3. **Statistics reflect only the detailed phase**: The extracted metrics represent behavior during the ``detailed`` instruction window only.

Memory Level and Size Histogram
-------------------------------

The ``--level`` flag determines which cache level statistics are extracted:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Level
     - Access Granularity
     - Size Histogram
   * - ``l1``
     - Fine-grained (1, 2, 4, 8, 16, 32, 64 bytes)
     - Actual instruction access sizes
   * - ``l2``
     - Cache-line (64 bytes)
     - All accesses counted as 64B
   * - ``l3``
     - Cache-line (64 bytes)
     - All accesses counted as 64B
   * - ``dram``
     - Cache-line (64 bytes)
     - All accesses counted as 64B

**Example Output (L1 level):**

.. code-block:: json

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

**Example Output (L3 level):**

.. code-block:: json

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

Customizing Memory Modeling
---------------------------

Sniper allows you to adjust the memory latency at each cache level to simulate the behavior of different memory technologies (e.g., DRAM, NVM, SRAM).

To model a specific memory technology, modify the appropriate section of your Sniper configuration file by adding latency settings:

.. code-block:: ini

   [perf_model/l3_cache]
   rw_enabled = true
   read_access_time = 100
   write_access_time = 50000

You may also apply the same structure to:

- ``[perf_model/l1_dcache]``
- ``[perf_model/l2_cache]``
- ``[perf_model/dram]``

Built-in Configurations
~~~~~~~~~~~~~~~~~~~~~~~

Sniper includes several pre-built CPU configurations:

- ``gainestown`` - Intel Nehalem (default)
- ``skylake`` - Intel Skylake
- ``haswell`` - Intel Haswell

.. note::

   **Sniper is currently the only profiler in the MemSysExplorer ecosystem** that supports detailed memory modeling through configurable latency values. This makes it uniquely suited for architectural experiments involving emerging memory technologies.

Output Files
------------

After simulation, the following files are generated:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``sim.out``
     - Human-readable simulation summary
   * - ``sim.cfg``
     - Configuration used for simulation
   * - ``sim.stats.sqlite3``
     - SQLite database with detailed statistics
   * - ``memsys_stats.out``
     - Extracted memory statistics (CSV format)
   * - ``memsyspatternconfig_*.json``
     - PatternConfig output for downstream analysis
   * - ``memsysmetadata_sniper.json``
     - System metadata (CPU, GPU, cache info)

PatternConfig Output
~~~~~~~~~~~~~~~~~~~~

The final JSON output includes:

.. code-block:: json

   {
     "exp_name": "SniperProfilers",
     "benchmark_name": "core_0",
     "read_freq": 2537892.14,
     "total_reads": 144,
     "write_freq": 29996475.15,
     "total_writes": 1702,
     "read_size": 64,
     "write_size": 64,
     "read_size_histogram": { "...": "..." },
     "write_size_histogram": { "...": "..." },
     "workingset_size": 0,
     "unit": {
       "read_freq": "count/s",
       "write_freq": "count/s",
       "total_reads": "count",
       "total_write": "count"
     }
   }

Troubleshooting
---------------

- **Missing Sniper binary:**
  Make sure ``run-sniper`` is located in ``profilers/sniper/snipersim/`` or run ``make`` in the sniper directory to build.

- **Missing stats script:**
  Ensure ``snipermem.py`` exists in ``snipersim/tools/``.

- **Invalid config section:**
  Sniper requires correctly formatted ``[perf_model/*]`` blocks. Refer to the official `Sniper documentation <https://snipersim.org>`_ for details.

- **ROI stats not captured:**
  Ensure your application runs long enough to reach the specified instruction counts. Use a test program that executes sufficient instructions.

- **Periodic callback warning:**
  The default callback interval (1M instructions) may be too coarse for small instruction counts. Reduce ``core/hook_periodic_ins/ins_global`` in your config.
