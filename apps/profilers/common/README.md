# Profiler Common Library

This library contains shared utilities that can be used across all profilers in the MemSysExplorer project.

## Components

### HyperLogLog (HLL)
- **Files**: `include/hll.h`, `src/hll.c`
- **Description**: Probabilistic data structure for estimating cardinality of large datasets
- **Dependencies**: MurmurHash3 for hashing

### MurmurHash3
- **Files**: `include/MurmurHash3.h`, `src/MurmurHash3.c`  
- **Description**: Fast non-cryptographic hash function
- **Usage**: Used by HyperLogLog implementation

### Working Set Tree Search (ws_tsearch)
- **Files**: `include/ws_tsearch.h`, `src/ws_tsearch.c`
- **Description**: Tree-based data structure for tracking working set statistics
- **Features**: Maintains counts of singles, distinct keys, and total events
- **Dependencies**: Uses GNU libc's tsearch/tfind/twalk/tdestroy functions

### Memory Trace (Protobuf)
- **Files**: `include/memory_trace.h`, `src/memory_trace.cpp`, `proto/memory_trace.proto`
- **Description**: Google Protocol Buffers-based memory trace format
- **Features**: Records memory operations with timestamp, thread_id, address, read/write, hit/miss
- **Dependencies**: Google Protocol Buffers (optional - graceful fallback if not available)
- **Format**: Binary protobuf serialization for compact, fast I/O

### Environment Capture (Standalone)
- **Files**: `include/environment_capture.h`, `src/environment_capture.c`
- **Description**: Standalone library for capturing system and environment metadata
- **Features**: Static capture of hostname, OS, architecture, working directory, all environment variables
- **Dependencies**: Standard C library only
- **Python Wrapper**: `tools/environment_capture.py` for BaseMetadata.py integration

## Usage

To use this library in your profiler:

1. Add the common library as a subdirectory in your CMakeLists.txt:
   ```cmake
   add_subdirectory(../common common)
   ```

2. Link against the library:
   ```cmake
   target_link_libraries(your_profiler profiler_common)
   ```

3. Include the headers:
   ```c
   #include "hll.h"
   #include "MurmurHash3.h"
   #include "ws_tsearch.h"
   #include "memory_trace.h"       // Only if protobuf is available
   #include "environment_capture.h" // Standalone environment capture
   ```

## Memory Trace Usage Example

```c
#include "memory_trace.h"

int main() {
    // Create a trace writer
    memory_trace_writer_t* writer = memory_trace_create_writer();
    if (!writer) {
        printf("Failed to create trace writer (protobuf not available?)\n");
        return 1;
    }
    
    // Add memory events
    memory_trace_add_event(writer, 1000000, 123, 0x400000, MEM_READ, CACHE_HIT);
    memory_trace_add_event(writer, 1000050, 123, 0x400040, MEM_WRITE, CACHE_MISS);
    
    // Write to file
    if (memory_trace_write_to_file(writer, "trace.pb") == 0) {
        printf("Trace written successfully\n");
    }
    
    // Cleanup
    memory_trace_destroy_writer(writer);
    return 0;
}
```

## Environment Capture Usage Example

### C Usage
```c
#include "environment_capture.h"

int main() {
    system_environment_t* env = environment_capture_create();
    
    printf("OS: %s\n", env->os_name);
    printf("User: %s\n", environment_capture_get_var(env, "USER"));
    printf("Total env vars: %zu\n", env->env_count);
    
    environment_capture_destroy(env);
    return 0;
}
```

### Python Usage (for BaseMetadata.py)
```python
from tools.environment_capture import EnvironmentCapture

env = EnvironmentCapture()
metadata = {
    'hostname': env.hostname,
    'os_name': env.os_name,
    'user': env.get_variable('USER'),
    'all_env': env.get_all_variables()
}
```

## Installing Protocol Buffers

To enable memory trace functionality:

### Ubuntu/Debian
```bash
sudo apt-get install libprotobuf-dev protobuf-compiler
```

### CentOS/RHEL
```bash
sudo yum install protobuf-devel protobuf-compiler
```