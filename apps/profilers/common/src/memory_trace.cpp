#ifdef HAVE_PROTOBUF

#include "memory_trace.h"
#include "memory_trace.pb.h"
#include <fstream>
#include <iostream>

struct memory_trace_writer {
    memory_trace::MemoryTrace trace;
};

extern "C" {

memory_trace_writer_t* memory_trace_create_writer(void) {
    try {
        return new memory_trace_writer_t();
    } catch (...) {
        return nullptr;
    }
}

int memory_trace_add_event(memory_trace_writer_t* writer,
                          uint64_t timestamp,
                          uint32_t thread_id,
                          uint64_t address,
                          mem_op_t mem_op,
                          hit_miss_t hit_miss) {
    if (!writer) return -1;
    
    try {
        auto* event = writer->trace.add_events();
        event->set_timestamp(timestamp);
        event->set_thread_id(thread_id);
        event->set_address(address);
        
        // Convert C enums to protobuf enums
        event->set_mem_op(mem_op == MEM_READ ? 
            memory_trace::READ : memory_trace::WRITE);
        event->set_hit_miss(hit_miss == CACHE_HIT ? 
            memory_trace::HIT : memory_trace::MISS);
        
        return 0;
    } catch (...) {
        return -1;
    }
}

int memory_trace_write_to_file(memory_trace_writer_t* writer, const char* filename) {
    if (!writer || !filename) return -1;
    
    try {
        std::ofstream output(filename, std::ios::binary);
        if (!output.is_open()) return -1;
        
        if (!writer->trace.SerializeToOstream(&output)) {
            return -1;
        }
        
        return 0;
    } catch (...) {
        return -1;
    }
}

int memory_trace_write_to_buffer(memory_trace_writer_t* writer, 
                                void* buffer, 
                                size_t buffer_size) {
    if (!writer) return -1;
    
    try {
        std::string serialized;
        if (!writer->trace.SerializeToString(&serialized)) {
            return -1;
        }
        
        // If buffer is NULL, return required size
        if (!buffer) {
            return static_cast<int>(serialized.size());
        }
        
        // Check if buffer is large enough
        if (buffer_size < serialized.size()) {
            return -1;
        }
        
        // Copy data to buffer
        std::memcpy(buffer, serialized.data(), serialized.size());
        return static_cast<int>(serialized.size());
        
    } catch (...) {
        return -1;
    }
}

size_t memory_trace_get_event_count(memory_trace_writer_t* writer) {
    if (!writer) return 0;
    return static_cast<size_t>(writer->trace.events_size());
}

void memory_trace_clear(memory_trace_writer_t* writer) {
    if (!writer) return;
    writer->trace.clear_events();
}

void memory_trace_destroy_writer(memory_trace_writer_t* writer) {
    delete writer;
}

} // extern "C"

#else // !HAVE_PROTOBUF

// Stub implementations when protobuf is not available
extern "C" {

memory_trace_writer_t* memory_trace_create_writer(void) {
    return nullptr;
}

int memory_trace_add_event(memory_trace_writer_t* writer,
                          uint64_t timestamp,
                          uint32_t thread_id,
                          uint64_t address,
                          mem_op_t mem_op,
                          hit_miss_t hit_miss) {
    (void)writer; (void)timestamp; (void)thread_id; (void)address; (void)mem_op; (void)hit_miss;
    return -1;
}

int memory_trace_write_to_file(memory_trace_writer_t* writer, const char* filename) {
    (void)writer; (void)filename;
    return -1;
}

int memory_trace_write_to_buffer(memory_trace_writer_t* writer, 
                                void* buffer, 
                                size_t buffer_size) {
    (void)writer; (void)buffer; (void)buffer_size;
    return -1;
}

size_t memory_trace_get_event_count(memory_trace_writer_t* writer) {
    (void)writer;
    return 0;
}

void memory_trace_clear(memory_trace_writer_t* writer) {
    (void)writer;
}

void memory_trace_destroy_writer(memory_trace_writer_t* writer) {
    (void)writer;
}

} // extern "C"

#endif // HAVE_PROTOBUF