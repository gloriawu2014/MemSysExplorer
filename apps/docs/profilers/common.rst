Common Profiler Library Documentation
======================================

The **Common Profiler Library** provides a collection of reusable, high-performance utilities that are shared across all profilers in the MemSysExplorer framework. This library eliminates code duplication and ensures consistent data structures and algorithms are used throughout the profiling infrastructure.

.. important::

   **MemSysExplorer Common Library**

   Refer to the codebase for the latest update: https://github.com/duca181/MemSysExplorer/tree/apps_dev/apps/profilers/common

   To learn more about license terms and third-party attribution, refer to the :doc:`../licensing` page.

Overview
--------

The common library contains five core components that support memory profiling, cardinality estimation, hashing, metadata collection, and trace serialization:

1. **HyperLogLog (HLL)** – Probabilistic cardinality estimation for working set size
2. **MurmurHash3** – Fast non-cryptographic hash function
3. **Working Set Tree Search** – Exact working set tracking using tree-based data structures
4. **Memory Trace (Protobuf)** – Compact binary trace format for memory access events
5. **Environment Capture** – System and runtime metadata collection

Components
----------

HyperLogLog (HLL)
~~~~~~~~~~~~~~~~~

A probabilistic data structure that estimates the cardinality (number of unique elements) in large datasets with minimal memory overhead.

**Key Features:**

- **Memory Efficient**: Uses logarithmic space relative to dataset size
- **Configurable Precision**: Supports 4-16 precision bits
- **Fast Updates**: Constant-time element insertion
- **Merge Support**: Combine multiple HLL structures for distributed counting

**Use Cases:**

- Approximate working set size estimation
- Unique cache line counting
- Memory footprint analysis with low overhead

**Files:**

- ``include/hll.h``
- ``src/hll.c``

MurmurHash3
~~~~~~~~~~~

A fast, non-cryptographic hash function optimized for hash table lookups and distributed hashing.

**Key Features:**

- High-quality hash distribution
- Platform-independent output
- Optimized for x86/x64 architectures

**Files:**

- ``include/MurmurHash3.h``
- ``src/MurmurHash3.c``

**Dependencies:**

- Used internally by HyperLogLog for element hashing

Working Set Tree Search (ws_tsearch)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An exact working set tracking data structure based on balanced binary trees. This structure maintains precise counts of unique memory accesses and provides statistics for memory footprint analysis.

**Key Features:**

- **Exact Counting**: Tracks every unique access with perfect accuracy
- **Windowed Statistics**: Reset and track working sets over time windows
- **Efficient Lookup**: O(log n) insertion and search using GNU tsearch

**Tracked Metrics:**

- Total number of references
- Distinct unique addresses
- Single-access addresses (temporal locality indicator)

**Files:**

- ``include/ws_tsearch.h``
- ``src/ws_tsearch.c``

Memory Trace (Protobuf)
~~~~~~~~~~~~~~~~~~~~~~~

A Google Protocol Buffers-based serialization format for memory access traces. Provides compact binary encoding with structured schema for trace analysis.

**Key Features:**

- **Compact Format**: Binary protobuf serialization reduces file size
- **Structured Schema**: Well-defined message format for interoperability
- **Optional Dependency**: Gracefully degrades if protobuf is unavailable
- **Fast I/O**: Efficient buffered writes

**Trace Fields:**

- ``timestamp_us`` – Microsecond-resolution timestamp
- ``thread_id`` – Thread identifier
- ``address`` – Memory address accessed
- ``is_write`` – Read/write operation type
- ``cache_hit`` – Hit/miss status (if available)

**Files:**

- ``include/memory_trace.h``
- ``src/memory_trace.cpp``
- ``proto/memory_trace.proto``

Environment Capture
~~~~~~~~~~~~~~~~~~~

A standalone library for capturing system and runtime metadata including hostname, OS information, architecture, and environment variables.

**Key Features:**

- **Static Capture**: Captures environment at initialization time
- **Minimal Dependencies**: Uses only standard C library
- **Python Wrapper**: Integration with BaseMetadata.py for unified metadata collection
- **Comprehensive Coverage**: Captures all environment variables and system info

**Captured Metadata:**

- Hostname
- Operating system name and version
- System architecture (x86_64, ARM, etc.)
- Current working directory
- All environment variables (PATH, HOME, CUDA_HOME, etc.)

**Files:**

- ``include/environment_capture.h``
- ``src/environment_capture.c``
- Python wrapper: ``tools/environment_capture.py``

Usage in Profilers
------------------

To use the common library in your profiler implementation:

CMake Integration
~~~~~~~~~~~~~~~~~

Add the common library as a subdirectory in your ``CMakeLists.txt``:

.. code-block:: cmake

   # Add common library
   add_subdirectory(../common common)

   # Link against the library
   target_link_libraries(your_profiler profiler_common)

Header Inclusion
~~~~~~~~~~~~~~~~

Include the necessary headers in your C/C++ source:

.. code-block:: c

   #include "hll.h"                 // HyperLogLog cardinality estimation
   #include "MurmurHash3.h"         // Fast hashing
   #include "ws_tsearch.h"          // Exact working set tracking
   #include "memory_trace.h"        // Protobuf trace output (optional)
   #include "environment_capture.h" // System metadata capture

Example: Working Set Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   #include "ws_tsearch.h"
   #include "hll.h"

   int main() {
       // Exact working set tracking
       ws_ctx_t* ws = ws_create();
       ws_record(ws, 0x400000);  // Record cache line access
       ws_record(ws, 0x400040);

       ws_stats_t stats;
       ws_get_stats(ws, &stats);
       printf("Distinct accesses: %lu\n", stats.distinct);
       ws_destroy(ws);

       // Approximate working set tracking with HLL
       HLL hll;
       hll_init(&hll, 8);  // 8 precision bits

       uintptr_t addr = 0x400000;
       hll_add(&hll, &addr, sizeof(addr));

       double estimated_unique = hll_count(&hll);
       printf("Estimated unique lines: %.0f\n", estimated_unique);

       hll_destroy(&hll);
       return 0;
   }

Example: Memory Trace Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   #include "memory_trace.h"

   int main() {
       // Create trace writer
       memory_trace_writer_t* writer = memory_trace_create_writer();
       if (!writer) {
           fprintf(stderr, "Protobuf not available\n");
           return 1;
       }

       // Record memory events
       memory_trace_add_event(writer,
                              1000000,      // timestamp (us)
                              123,          // thread_id
                              0x400000,     // address
                              MEM_READ,     // read operation
                              CACHE_HIT);   // cache hit

       // Write to file
       memory_trace_write_to_file(writer, "trace.pb");
       memory_trace_destroy_writer(writer);

       return 0;
   }

Example: Environment Metadata Capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: c

   #include "environment_capture.h"

   int main() {
       system_environment_t* env = environment_capture_create();

       printf("Hostname: %s\n", env->hostname);
       printf("OS: %s\n", env->os_name);
       printf("Architecture: %s\n", env->arch);
       printf("Working Directory: %s\n", env->cwd);

       // Get specific environment variable
       const char* cuda_home = environment_capture_get_var(env, "CUDA_HOME");
       if (cuda_home) {
           printf("CUDA_HOME: %s\n", cuda_home);
       }

       environment_capture_destroy(env);
       return 0;
   }

Dependencies
------------

Core Dependencies
~~~~~~~~~~~~~~~~~

- **Standard C Library** (required)
- **GNU C Library** for tsearch/tfind/twalk/tdestroy (required on Linux)

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- **Google Protocol Buffers** (libprotobuf-dev, protobuf-compiler)

  Required for memory trace functionality. If not available, the library will compile without protobuf support.

Installing Protocol Buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install libprotobuf-dev protobuf-compiler

**CentOS/RHEL:**

.. code-block:: bash

   sudo yum install protobuf-devel protobuf-compiler

**macOS:**

.. code-block:: bash

   brew install protobuf

Build System
------------

The common library uses CMake and automatically detects available dependencies:

.. code-block:: bash

   cd profilers/common
   mkdir build && cd build
   cmake ..
   make

The build system will:

- Detect if Protocol Buffers is available
- Compile protobuf-dependent features only if found
- Provide graceful fallback for missing optional dependencies

Integration with Profilers
---------------------------

The following MemSysExplorer profilers use the common library:

- **DynamoRIO** – Uses HLL and ws_tsearch for working set estimation
- **NVBit** – Uses memory trace protobuf format for GPU memory access traces
- **Perf** – Uses environment capture for metadata collection
- **Sniper** – Uses working set tracking for cache simulation validation

Additional Notes
----------------

- The common library is designed to be **zero-dependency** for core functionality
- Protocol Buffers is **optional** and only required for trace serialization
- All data structures are **thread-safe** or explicitly documented as thread-unsafe
- The library follows **POSIX standards** for maximum portability

Troubleshooting
---------------

**Issue:** Protobuf not found during build

**Solution:** Install Protocol Buffers development libraries or build without protobuf support. The library will automatically disable protobuf-dependent features.

**Issue:** tsearch/tfind functions not available

**Solution:** Ensure you are building on a POSIX-compliant system with GNU C Library. For non-Linux systems, alternative implementations may be required.

**Issue:** Linker errors when using the library

**Solution:** Ensure ``profiler_common`` is properly linked in your CMakeLists.txt:

.. code-block:: cmake

   target_link_libraries(your_profiler profiler_common)
