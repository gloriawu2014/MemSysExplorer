2. Metadata Collection
======================

MemSysExplorer includes built-in support for collecting **system-level metadata** to accompany memory profiling outputs. This metadata provides detailed context about the hardware and software environment in which the profiling was performed, ensuring accurate interpretation and reproducibility of results.

.. note::

   The `BaseMetadata` class assumes a **Linux-based system**. Users on other platforms may encounter incomplete or missing metadata fields unless modified.

   The `BaseMetadata` implementation can be found in the repository here:
   `BaseMetadata.py <https://github.com/duca181/MemSysExplorer/blob/apps_dev/apps/profilers/BaseMetadata.py>`_

   In the future, we will provide **community metadata profiles** collected from different systems to help users compare and calibrate workload behaviors across architectures.

.. important::

   Every profiler must **enforce integration of `BaseMetadata`** or its subclass. Metadata collection is essential to ensure that workload traces are **reproducible** and properly **contextualized** based on their execution environment.

2.1 Collected Metadata Includes
-------------------------------

- **GPU Information**
  - Device name, driver version, and available GPU memory (`nvidia-smi`)

- **CPU Information**
  - Full `lscpu` dump, including architecture, core/thread counts, CPU family/model, etc.

- **Cache Hierarchy**
  - Sizes of L1 instruction/data, L2, and L3 caches from `/sys/devices/system/cpu/cpu0/cache`

- **Main Memory (DRAM)**
  - Total physical memory size in megabytes (`/proc/meminfo`)

- **Software Environment**
  - Operating system name and version
  - Installed compiler versions (e.g., GCC, Clang, AOCC)
  - BIOS and firmware information (`dmidecode`)
  - Filesystem type
  - Power policy and CPU governor

2.2 Class Structure
-------------------

The `BaseMetadata` class implements the following key methods:

- ``gpu_info()`` – Extracts GPU specifications
- ``cpu_info()`` – Parses CPU attributes from `lscpu`
- ``cache_info()`` – Returns cache sizes per level
- ``dram_info()`` – Measures DRAM size from system memory
- ``software_info()`` – Reports OS, kernel, compilers, BIOS, and policy info
- ``as_dict()`` – Converts all metadata to a single dictionary object
- ``__repr__()`` – Provides a human-readable summary string

2.3 Integration
---------------

Each profiler in MemSysExplorer (e.g., ``dynamorio``, ``perf``, ``sniper``, ``nvbit``, ``ncu``) may inherit from the ``BaseMetadata`` class or integrate its output into their reporting structures.

.. important::

   The use of ``BaseMetadata`` is **mandatory** across all profilers to ensure a **unified and reproducible profiling environment**. Reproducibility across profiling runs requires consistent capture of both hardware and software environment metadata.

To support consistent experimentation and collaboration, MemSysExplorer will include a **community-contributed database** of workload metadata and profiler outputs. This shared repository will facilitate reproducible research, cross-platform comparisons, and collaborative benchmarking across research groups.

