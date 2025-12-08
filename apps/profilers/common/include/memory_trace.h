#ifndef MEMORY_TRACE_H
#define MEMORY_TRACE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Memory operation types
typedef enum {
    MEM_READ = 0,
    MEM_WRITE = 1
} mem_op_t;

// Cache hit/miss status
typedef enum {
    CACHE_HIT = 0,
    CACHE_MISS = 1
} hit_miss_t;

// Opaque handle for memory trace writer
typedef struct memory_trace_writer memory_trace_writer_t;

// Create a new memory trace writer
// Returns NULL on failure
memory_trace_writer_t* memory_trace_create_writer(void);

// Add a memory event to the trace
// Returns 0 on success, -1 on failure
int memory_trace_add_event(memory_trace_writer_t* writer,
                          uint64_t timestamp,
                          uint32_t thread_id,
                          uint64_t address,
                          mem_op_t mem_op,
                          hit_miss_t hit_miss);

// Write the trace to a file (binary protobuf format)
// Returns 0 on success, -1 on failure
int memory_trace_write_to_file(memory_trace_writer_t* writer, const char* filename);

// Write the trace to a buffer
// Returns the size of the serialized data, or -1 on failure
// If buffer is NULL, returns the required buffer size
int memory_trace_write_to_buffer(memory_trace_writer_t* writer, 
                                void* buffer, 
                                size_t buffer_size);

// Get the number of events currently in the trace
size_t memory_trace_get_event_count(memory_trace_writer_t* writer);

// Clear all events from the trace (but keep the writer)
void memory_trace_clear(memory_trace_writer_t* writer);

// Destroy the memory trace writer and free resources
void memory_trace_destroy_writer(memory_trace_writer_t* writer);

#ifdef __cplusplus
}
#endif

#endif /* MEMORY_TRACE_H */