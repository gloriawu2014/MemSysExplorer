DynamoRIO Documentation
==============================

**DynamoRIO** is a dynamic binary instrumentation framework that inserts analysis code at runtime, allowing fine-grained monitoring of program execution.
It provides valuable insights into instruction-level behavior, memory access patterns, and control flow, enabling detailed performance and security analysis. 

The profiling workflow in MemSysExplorer consists of two core actions, as provided by the main interface:

1. **Profiling (`profiling`)** – Captures runtime execution metrics by specifying the required executable.
2. **Metric Extraction (`extract_metrics`)** – Analyzes generated reports to extract memory and performance-related metrics.

When using the `both` action, profiling and metric extraction are performed sequentially.

.. important::

   **MemSysExplorer GitHub Repository**

   Refer to the codebase for the latest update: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps/profilers/dynamorio

   To learn more about license terms and third-party attribution, refer to the :doc:`../licensing` page.


Required Arguments
------------------

To execute DynamoRIO profilers, specific arguments are required based on the chosen action. The necessary arguments are defined in the code as follows:

.. code-block:: python

    @classmethod
    def required_profiling_args(cls):
        """
        Return required arguments for the profiling method.
        """
        return ["executable"]

    @classmethod
    def required_extract_args(cls, action):
        """
        Return required arguments for the extract_metrics method.
        """
        if action == "extract_metrics":
            return ["report_file"]
        else:
            return []

Configuration File Support
--------------------------

The DynamoRIO memcount client supports runtime configuration through a text-based configuration file. This allows fine-grained control over profiling behavior without recompiling the instrumentation client.

Configuration Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

The following parameters can be configured in ``config/memcount_config.txt``:

.. list-table::
   :widths: 25 10 15 50
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``cache_line_size``
     - uint
     - 64
     - Cache line size in bytes for address alignment
   * - ``hll_bits``
     - uint
     - 8
     - HyperLogLog precision bits (4-16)
   * - ``sample_hll_bits``
     - uint
     - 8
     - HLL precision for windowed sampling
   * - ``sample_window_refs``
     - uint
     - 2000
     - Number of memory references per sampling window
   * - ``max_mem_refs``
     - uint
     - 8192
     - Maximum buffered memory references before flush
   * - ``enable_trace``
     - bool
     - false
     - Enable detailed protobuf trace output
   * - ``wss_stat_tracking``
     - bool
     - true
     - Enable working set size statistics tracking
   * - ``wss_exact_tracking``
     - bool
     - true
     - Enable exact WSS tracking (memory intensive)
   * - ``wss_hll_tracking``
     - bool
     - true
     - Enable HLL-based approximate WSS tracking
   * - ``enable_instruction_threshold``
     - bool
     - false
     - Enable instruction count threshold termination
   * - ``instruction_threshold``
     - uint64
     - 100000000
     - Number of instructions before auto-termination
   * - ``pb_trace_output``
     - string
     - "memtrace"
     - Base name for protobuf trace output
   * - ``pb_timeseries_output``
     - string
     - "timeseries"
     - Base name for protobuf time-series output

Instruction Threshold Feature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instruction threshold feature automatically terminates profiling after a specified number of instructions. This is particularly useful for:

* **Limiting trace file sizes** for long-running applications
* **Profiling initialization phases** by setting a low threshold
* **Controlled experiments** requiring consistent instruction counts
* **Testing and debugging** with reproducible cutoff points

When the threshold is reached, the profiler will:

1. Print a notification message with the exact instruction count
2. Flush all buffered data to output files
3. Print final statistics
4. Terminate the application gracefully

Example configuration:

.. code-block:: ini

   enable_instruction_threshold=true
   instruction_threshold=100000000

Example Usage
-------------

Below are examples of how to execute the profiling tool with different actions:

- **Profiling the application:**

  .. code-block:: bash

     python main.py --profiler dynamorio --action profiling --executable ./executable

- **Profiling with custom configuration:**

  .. code-block:: bash

     python main.py --profiler dynamorio --action profiling --config config/memcount_config.txt --executable ./executable

- **Profiling with instruction threshold enabled:**

  Edit ``config/memcount_config.txt`` to enable the threshold:

  .. code-block:: ini

     enable_instruction_threshold=true
     instruction_threshold=100000000

  Then run:

  .. code-block:: bash

     python main.py --profiler dynamorio --action profiling --config config/memcount_config.txt --executable ./executable

- **Extracting metrics from an existing report:**

  .. code-block:: bash

     python main.py --profiler dynamorio --action extract_metrics --report_file ./report_file

- **Performing both profiling and metric extraction:**

  .. code-block:: bash

     python main.py --profiler dynamorio --action both --executable ./executable

Sample Output
-------------

This profiler generates output traces that follow the standardized format defined by the MemSysExplorer Application Interface.


Additional Notes
----------------

- The **DynamoRIO** must be correctly installed and accessible via the system `PATH` variable.

Troubleshooting
---------------

If you encounter issues when building the DynamoRIO profiler:

- **Ensure that the environment has been set up properly** using:

  .. code-block:: tcsh

     source setup/setup.csh dynamorio

  or

  .. code-block:: bash

     source setup/setup.sh dynamorio

- **Verify that the correct GCC version is installed and exported** in your environment. The profiler expects a compatible GCC version as configured in your setup script (e.g., GCC 11.2.0).

- **Check for missing compiler paths**: Make sure `PATH`, `LD_LIBRARY_PATH`, `LIBRARY_PATH`, and `C_INCLUDE_PATH` are configured to include your GCC installation directories.

If problems persist, rebuild the profiler after re-sourcing your environment.

