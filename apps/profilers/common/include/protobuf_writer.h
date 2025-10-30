/*
 * Protobuf Writer Interface for Memory Profiling
 *
 * Provides simple C functions to write memory traces and time-series samples
 * using protobuf format for efficient storage and parsing.
 */

#ifndef PROTOBUF_WRITER_H
#define PROTOBUF_WRITER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef HAVE_PROTOBUF_C
#include "memory_trace.pb-c.h"
#include "timeseries_metrics.pb-c.h"
#endif

/* Opaque handle types */
typedef struct pb_trace_writer pb_trace_writer_t;
typedef struct pb_timeseries_writer pb_timeseries_writer_t;

/*
 * Memory Trace Writer (for enable_trace feature)
 * Writes detailed per-access memory traces
 */

/**
 * Create a new memory trace writer
 * @param filename Output .pb file path
 * @return Writer handle or NULL on error
 */
pb_trace_writer_t* pb_trace_writer_create(const char *filename);

/**
 * Write a single memory event to trace
 * @param writer Writer handle
 * @param timestamp Timestamp in microseconds
 * @param thread_id Thread ID
 * @param address Memory address
 * @param is_write true for write, false for read
 * @param size Access size in bytes
 */
void pb_trace_write_event(pb_trace_writer_t *writer,
                          uint64_t timestamp,
                          uint32_t thread_id,
                          uint64_t address,
                          bool is_write,
                          uint32_t size);

/**
 * Close and finalize memory trace writer
 * @param writer Writer handle
 */
void pb_trace_writer_close(pb_trace_writer_t *writer);


/*
 * Time-Series Metrics Writer (for WSS sampling feature)
 * Writes windowed WSS samples
 */

/**
 * Create a new time-series metrics writer
 * @param filename Output .pb file path
 * @param profiler Profiler name (e.g., "dynamorio")
 * @param pid Process ID
 * @param command Command line string
 * @param sample_window_refs Number of refs per window
 * @param cache_line_size Cache line size in bytes
 * @return Writer handle or NULL on error
 */
pb_timeseries_writer_t* pb_timeseries_writer_create(
    const char *filename,
    const char *profiler,
    uint32_t pid,
    const char *command,
    uint32_t sample_window_refs,
    uint32_t cache_line_size);

/**
 * Write a sample window
 * @param writer Writer handle
 * @param window_number Window index (0, 1, 2, ...)
 * @param thread_id Thread ID
 * @param read_count Number of reads in this window
 * @param write_count Number of writes in this window
 * @param total_refs Total references in this window (read_count + write_count)
 * @param wss_exact Exact working set size (from ws_tsearch)
 * @param wss_approx Approximate WSS (from HLL)
 * @param timestamp Sample timestamp in microseconds
 */
void pb_timeseries_write_sample(pb_timeseries_writer_t *writer,
                                uint64_t window_number,
                                uint32_t thread_id,
                                uint64_t read_count,
                                uint64_t write_count,
                                uint64_t total_refs,
                                uint64_t wss_exact,
                                double wss_approx,
                                uint64_t timestamp);

/**
 * Manually flush buffered samples to disk
 * @param writer Writer handle
 */
void pb_timeseries_flush(pb_timeseries_writer_t *writer);

/**
 * Manually flush buffered trace events to disk
 * @param writer Writer handle
 */
void pb_trace_flush(pb_trace_writer_t *writer);

/**
 * Set number of threads in metadata (call before closing)
 * @param writer Writer handle
 * @param num_threads Total thread count
 */
void pb_timeseries_set_num_threads(pb_timeseries_writer_t *writer,
                                   uint32_t num_threads);

/**
 * Close and finalize time-series writer
 * @param writer Writer handle
 */
void pb_timeseries_writer_close(pb_timeseries_writer_t *writer);

#endif /* PROTOBUF_WRITER_H */
