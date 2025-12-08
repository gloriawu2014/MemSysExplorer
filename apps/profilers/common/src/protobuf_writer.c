/*
 * Protobuf Writer Implementation
 */

#include "protobuf_writer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef HAVE_PROTOBUF_C

/* Memory Trace Writer Structure */
struct pb_trace_writer {
    FILE *file;
    size_t total_events_written;  /* Total events written to file */
};

/* Time-Series Writer Structure */
struct pb_timeseries_writer {
    FILE *file;
    Memsys__Timeseries__RunMetadata *metadata;  /* Keep metadata for writing with each sample batch */
    size_t total_samples_written;  /* Total samples written to file */
};

/* ========== Memory Trace Writer ========== */

pb_trace_writer_t* pb_trace_writer_create(const char *filename) {
    pb_trace_writer_t *writer = (pb_trace_writer_t*)malloc(sizeof(pb_trace_writer_t));
    if (!writer) return NULL;

    writer->file = fopen(filename, "wb");
    if (!writer->file) {
        free(writer);
        return NULL;
    }

    writer->total_events_written = 0;

    return writer;
}

void pb_trace_write_event(pb_trace_writer_t *writer,
                          uint64_t timestamp,
                          uint32_t thread_id,
                          uint64_t address,
                          bool is_write,
                          uint32_t size) {
    if (!writer) return;

    /* Create a single event */
    MemoryTrace__MemoryEvent event = MEMORY_TRACE__MEMORY_EVENT__INIT;
    event.timestamp = timestamp;
    event.thread_id = thread_id;
    event.address = address;
    event.mem_op = is_write ? MEMORY_TRACE__MEM_OP__WRITE : MEMORY_TRACE__MEM_OP__READ;
    event.hit_miss = MEMORY_TRACE__HIT_MISS__MISS;

    /* Create a wrapper trace with this single event */
    MemoryTrace__MemoryTrace trace = MEMORY_TRACE__MEMORY_TRACE__INIT;
    MemoryTrace__MemoryEvent *event_ptr = &event;
    trace.events = &event_ptr;
    trace.n_events = 1;

    /* Serialize and write immediately */
    size_t packed_size = memory_trace__memory_trace__get_packed_size(&trace);
    uint8_t *buffer = (uint8_t*)malloc(packed_size);
    memory_trace__memory_trace__pack(&trace, buffer);

    /* Write length-delimited: 4-byte size + data */
    uint32_t msg_size = (uint32_t)packed_size;
    fwrite(&msg_size, sizeof(uint32_t), 1, writer->file);
    fwrite(buffer, 1, packed_size, writer->file);
    fflush(writer->file);  /* Force to disk immediately */

    free(buffer);
    writer->total_events_written++;
}

void pb_trace_flush(pb_trace_writer_t *writer) {
    if (!writer) return;
    /* Ensure OS buffer is flushed to disk */
    fflush(writer->file);
}

void pb_trace_writer_close(pb_trace_writer_t *writer) {
    if (!writer) return;

    /* Final flush */
    fflush(writer->file);
    fclose(writer->file);
    free(writer);
}

/* ========== Time-Series Metrics Writer ========== */

pb_timeseries_writer_t* pb_timeseries_writer_create(
    const char *filename,
    const char *profiler,
    uint32_t pid,
    const char *command,
    uint32_t sample_window_refs,
    uint32_t cache_line_size) {

    pb_timeseries_writer_t *writer =
        (pb_timeseries_writer_t*)malloc(sizeof(pb_timeseries_writer_t));
    if (!writer) return NULL;

    writer->file = fopen(filename, "wb");
    if (!writer->file) {
        free(writer);
        return NULL;
    }

    /* Initialize and store metadata (will be written with each sample) */
    writer->metadata = (Memsys__Timeseries__RunMetadata*)
        malloc(sizeof(Memsys__Timeseries__RunMetadata));
    memsys__timeseries__run_metadata__init(writer->metadata);

    writer->metadata->profiler = strdup(profiler);
    writer->metadata->pid = pid;
    writer->metadata->command = strdup(command);
    writer->metadata->sample_window_refs = sample_window_refs;
    writer->metadata->cache_line_size = cache_line_size;
    writer->metadata->start_timestamp = 0;
    writer->metadata->num_threads = 0;

    writer->total_samples_written = 0;

    return writer;
}

void pb_timeseries_write_sample(pb_timeseries_writer_t *writer,
                                uint64_t window_number,
                                uint32_t thread_id,
                                uint64_t read_count,
                                uint64_t write_count,
                                uint64_t total_refs,
                                uint64_t wss_exact,
                                double wss_approx,
                                uint64_t timestamp,
                                uint64_t read_size_1, uint64_t read_size_2, uint64_t read_size_4, uint64_t read_size_8,
                                uint64_t read_size_16, uint64_t read_size_32, uint64_t read_size_64, uint64_t read_size_other,
                                uint64_t write_size_1, uint64_t write_size_2, uint64_t write_size_4, uint64_t write_size_8,
                                uint64_t write_size_16, uint64_t write_size_32, uint64_t write_size_64, uint64_t write_size_other) {
    if (!writer) return;

    /* Create a single sample */
    Memsys__Timeseries__SampleWindow sample = MEMSYS__TIMESERIES__SAMPLE_WINDOW__INIT;
    sample.window_number = window_number;
    sample.thread_id = thread_id;
    sample.read_count = read_count;
    sample.write_count = write_count;
    sample.total_refs = total_refs;
    sample.wss_exact = wss_exact;
    sample.wss_approx = wss_approx;
    sample.timestamp = timestamp;

    /* Set read size histograms */
    sample.read_size_1 = read_size_1;
    sample.read_size_2 = read_size_2;
    sample.read_size_4 = read_size_4;
    sample.read_size_8 = read_size_8;
    sample.read_size_16 = read_size_16;
    sample.read_size_32 = read_size_32;
    sample.read_size_64 = read_size_64;
    sample.read_size_other = read_size_other;

    /* Set write size histograms */
    sample.write_size_1 = write_size_1;
    sample.write_size_2 = write_size_2;
    sample.write_size_4 = write_size_4;
    sample.write_size_8 = write_size_8;
    sample.write_size_16 = write_size_16;
    sample.write_size_32 = write_size_32;
    sample.write_size_64 = write_size_64;
    sample.write_size_other = write_size_other;

    /* Create a wrapper TimeSeriesData with metadata and this single sample */
    Memsys__Timeseries__TimeSeriesData data = MEMSYS__TIMESERIES__TIME_SERIES_DATA__INIT;
    data.metadata = writer->metadata;
    Memsys__Timeseries__SampleWindow *sample_ptr = &sample;
    data.samples = &sample_ptr;
    data.n_samples = 1;

    /* Serialize and write immediately */
    size_t packed_size = memsys__timeseries__time_series_data__get_packed_size(&data);
    uint8_t *buffer = (uint8_t*)malloc(packed_size);
    memsys__timeseries__time_series_data__pack(&data, buffer);

    /* Write length-delimited: 4-byte size + data */
    uint32_t msg_size = (uint32_t)packed_size;
    fwrite(&msg_size, sizeof(uint32_t), 1, writer->file);
    fwrite(buffer, 1, packed_size, writer->file);
    fflush(writer->file);  /* Force to disk immediately */

    free(buffer);
    writer->total_samples_written++;
}

void pb_timeseries_flush(pb_timeseries_writer_t *writer) {
    if (!writer) return;
    /* Ensure OS buffer is flushed to disk */
    fflush(writer->file);
}

void pb_timeseries_set_num_threads(pb_timeseries_writer_t *writer,
                                   uint32_t num_threads) {
    if (!writer || !writer->metadata) return;
    writer->metadata->num_threads = num_threads;
}

void pb_timeseries_writer_close(pb_timeseries_writer_t *writer) {
    if (!writer) return;

    /* Final flush */
    fflush(writer->file);

    /* Cleanup metadata */
    free((void*)writer->metadata->profiler);
    free((void*)writer->metadata->command);
    free(writer->metadata);

    fclose(writer->file);
    free(writer);
}

#else /* !HAVE_PROTOBUF_C */

/* Stub implementations when protobuf-c is not available */

pb_trace_writer_t* pb_trace_writer_create(const char *filename) {
    fprintf(stderr, "Warning: Protobuf-c not available, trace output disabled\n");
    return NULL;
}

void pb_trace_write_event(pb_trace_writer_t *writer,
                          uint64_t timestamp, uint32_t thread_id,
                          uint64_t address, bool is_write, uint32_t size) {
    /* No-op */
}

void pb_trace_flush(pb_trace_writer_t *writer) {
    /* No-op */
}

void pb_trace_writer_close(pb_trace_writer_t *writer) {
    /* No-op */
}

pb_timeseries_writer_t* pb_timeseries_writer_create(
    const char *filename, const char *profiler, uint32_t pid,
    const char *command, uint32_t sample_window_refs,
    uint32_t cache_line_size) {
    fprintf(stderr, "Warning: Protobuf-c not available, time-series output disabled\n");
    return NULL;
}

void pb_timeseries_write_sample(pb_timeseries_writer_t *writer,
                                uint64_t window_number, uint64_t thread_id,
                                uint64_t read_count, uint64_t write_count,
                                uint64_t total_refs,
                                uint64_t wss_exact, double wss_approx,
                                uint64_t timestamp,
                                uint64_t read_size_1, uint64_t read_size_2, uint64_t read_size_4, uint64_t read_size_8,
                                uint64_t read_size_16, uint64_t read_size_32, uint64_t read_size_64, uint64_t read_size_other,
                                uint64_t write_size_1, uint64_t write_size_2, uint64_t write_size_4, uint64_t write_size_8,
                                uint64_t write_size_16, uint64_t write_size_32, uint64_t write_size_64, uint64_t write_size_other) {
    /* No-op */
}

void pb_timeseries_flush(pb_timeseries_writer_t *writer) {
    /* No-op */
}

void pb_timeseries_set_num_threads(pb_timeseries_writer_t *writer,
                                   uint32_t num_threads) {
    /* No-op */
}

void pb_timeseries_writer_close(pb_timeseries_writer_t *writer) {
    /* No-op */
}

#endif /* HAVE_PROTOBUF_C */
