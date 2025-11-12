/* ******************************************************************************
 * Copyright (c) 2011-2024 Google, Inc.  All rights reserved.
 * Copyright (c) 2010 Massachusetts Institute of Technology  All rights reserved.
 * ******************************************************************************/

/*
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of VMware, Inc. nor the names of its contributors may be
 *   used to endorse or promote products derived from this software without
 *   specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL VMWARE, INC. OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */

#include <stdio.h>
#include <string.h> /* for memset */
#include <stddef.h> /* for offsetof */
#include <math.h>
#include "dr_api.h"
#include "drmgr.h"
#include "drreg.h"
#include "drutil.h"
#include "drx.h"
#include "utils.h"
#include "ws_tsearch.h"
#include "hll.h"
#include "protobuf_writer.h"

/* Configuration structure */
typedef struct {
    /* Cache and memory parameters */
    uint cache_line_size;
    uint hll_bits;
    uint sample_hll_bits;
    uint sample_window_refs;
    uint max_mem_refs;

    /* Output control */
    bool enable_trace;
    bool wss_stat_tracking;

    /* WSS tracking method control */
    bool wss_exact_tracking;    /* Enable exact WSS tracking (memory intensive) */
    bool wss_hll_tracking;      /* Enable HLL-based WSS tracking (memory efficient) */

    /* Instruction threshold control */
    bool enable_instruction_threshold;  /* Enable instruction threshold termination */
    uint64 instruction_threshold;       /* Number of instructions before termination */

    /* Protobuf output file paths */
    char pb_trace_output[256];
    char pb_timeseries_output[256];
} memcount_config_t;

/* Global configuration with default values */
static memcount_config_t config = {
    .cache_line_size = 64,
    .hll_bits = 4,
    .sample_hll_bits = 4,
    .sample_window_refs = 1000,
    .max_mem_refs = 8192,
    .enable_trace = true,
    .wss_stat_tracking = true,
    .wss_exact_tracking = true,
    .wss_hll_tracking = true,
    .enable_instruction_threshold = false,
    .instruction_threshold = 100000000,  /* Default: 100M instructions */
    .pb_trace_output = "memtrace",
    .pb_timeseries_output = "timeseries"
};

/* Derived values calculated from config */
static uintptr_t cache_line_mask;
static size_t mem_buf_size;

static HLL global_hll;
static void *hll_mutex;

/* Each mem_ref_t includes the type of reference (read or write),
 * the address referenced, and the size of the reference.
 */ // FIXME: can this be shortened?
typedef struct _mem_ref_t {
    bool write;
    void *addr;
    size_t size;
    app_pc pc;
    uint64 timestamp;
} mem_ref_t;

/* Memory buffer size will be calculated from config at runtime */

/* thread private counter */
typedef struct {
    char *buf_ptr;
    char *buf_base;
    /* buf_end holds the negative value of real address of buffer end. */
    ptr_int_t buf_end;
    void *cache;
    uint64 num_refs;
    uint64 num_reads;
    uint64 num_writes;
    uint64 working_set;
    ws_ctx_t *ws;
    HLL hll;

    /*Sampling API*/
    ws_ctx_t *sample_ws;     /* exact WSS for the current window */
    HLL       sample_hll;    /* HLL WSS for the current window */
    uint64    sample_ref_count;
    uint64    sample_idx;    /* window number (0,1,2,...) */

    /* Per-window counters for protobuf output */
    uint64    sample_read_count;   /* reads in current window */
    uint64    sample_write_count;  /* writes in current window */

    /* Per-window size-specific counters */
    uint64 sample_read_size_1;
    uint64 sample_read_size_2;
    uint64 sample_read_size_4;
    uint64 sample_read_size_8;
    uint64 sample_read_size_16;
    uint64 sample_read_size_32;
    uint64 sample_read_size_64;
    uint64 sample_read_size_other;
    uint64 sample_write_size_1;
    uint64 sample_write_size_2;
    uint64 sample_write_size_4;
    uint64 sample_write_size_8;
    uint64 sample_write_size_16;
    uint64 sample_write_size_32;
    uint64 sample_write_size_64;
    uint64 sample_write_size_other;

    /* Size-specific counters for reads and writes (global per thread) */
    uint64 read_size_1;    /* 1-byte reads */
    uint64 read_size_2;    /* 2-byte reads */
    uint64 read_size_4;    /* 4-byte reads */
    uint64 read_size_8;    /* 8-byte reads */
    uint64 read_size_16;   /* 16-byte reads */
    uint64 read_size_32;   /* 32-byte reads */
    uint64 read_size_64;   /* 64-byte reads */
    uint64 read_size_other; /* other size reads */

    uint64 write_size_1;    /* 1-byte writes */
    uint64 write_size_2;    /* 2-byte writes */
    uint64 write_size_4;    /* 4-byte writes */
    uint64 write_size_8;    /* 8-byte writes */
    uint64 write_size_16;   /* 16-byte writes */
    uint64 write_size_32;   /* 32-byte writes */
    uint64 write_size_64;   /* 64-byte writes */
    uint64 write_size_other; /* other size writes */

} per_thread_t;

/* Cross-instrumentation-phase data. */
typedef struct {
    app_pc last_pc;
} instru_data_t;

static size_t page_size;
static client_id_t client_id;
static app_pc code_cache;
static void *mutex;            /* for multithread support */
static uint64 global_num_refs; /* keep a global memory reference count */
static uint64 global_num_reads;
static uint64 global_num_writes;
static uint64 global_working_set;
static int tls_index;

/* Global size-specific counters */
static uint64 global_read_size_1 = 0;
static uint64 global_read_size_2 = 0;
static uint64 global_read_size_4 = 0;
static uint64 global_read_size_8 = 0;
static uint64 global_read_size_16 = 0;
static uint64 global_read_size_32 = 0;
static uint64 global_read_size_64 = 0;
static uint64 global_read_size_other = 0;

static uint64 global_write_size_1 = 0;
static uint64 global_write_size_2 = 0;
static uint64 global_write_size_4 = 0;
static uint64 global_write_size_8 = 0;
static uint64 global_write_size_16 = 0;
static uint64 global_write_size_32 = 0;
static uint64 global_write_size_64 = 0;
static uint64 global_write_size_other = 0;

/* Instruction threshold tracking */
static uint64 global_instruction_count = 0;
static void *instr_count_mutex = NULL;
static volatile bool threshold_reached = false;

/* Execution time tracking */
static uint64 start_time_us = 0;
static uint64 end_time_us = 0;

/* Protobuf writers (global, single file for all threads) */
static pb_trace_writer_t *global_trace_writer = NULL;
static pb_timeseries_writer_t *global_timeseries_writer = NULL;
static void *trace_mutex = NULL;            /* mutex for trace writer */
static void *timeseries_mutex = NULL;       /* mutex for timeseries writer */
static uint32_t global_thread_count = 0;    /* track total threads */
static void *thread_count_mutex = NULL;

static void
event_exit(void);
static void
event_thread_init(void *drcontext);
static void
event_thread_exit(void *drcontext);
static void
check_instruction_threshold(uint64 num_instrs);
static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                 bool translating);
static dr_emit_flags_t
event_bb_analysis(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                  bool translating, void **user_data);
static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *instr,
                bool for_trace, bool translating, void *user_data);

static void
clean_call(void);
static void
memtrace(void *drcontext);
static void
code_cache_init(void);
static void
code_cache_exit(void);
static void
instrument_mem(void *drcontext, instrlist_t *ilist, instr_t *where, app_pc pc,
               instr_t *memref_instr, int pos, bool write);
static void
instrument_mem_direct(void *drcontext, instrlist_t *ilist, instr_t *where, app_pc pc,
                     instr_t *memref_instr, int pos, bool write);

//Sampling Helper
/* Get high-resolution timestamp */
static uint64 get_timestamp(void) {
    return dr_get_microseconds();
}

/* Check instruction threshold and terminate if reached */
static void check_instruction_threshold(uint64 num_instrs) {
    if (!config.enable_instruction_threshold || threshold_reached)
        return;

    dr_mutex_lock(instr_count_mutex);
    global_instruction_count += num_instrs;

    if (global_instruction_count >= config.instruction_threshold) {
        threshold_reached = true;
        dr_mutex_unlock(instr_count_mutex);

        dr_fprintf(STDERR, "\n=== Instruction threshold reached: %llu instructions ===\n",
                   global_instruction_count);
        dr_fprintf(STDERR, "Terminating instrumentation and printing final stats...\n\n");

        /* Trigger exit which will print final stats */
        dr_exit_process(0);
    } else {
        dr_mutex_unlock(instr_count_mutex);
    }
}

/* Direct trace function - called immediately for each memory access */
static void direct_trace_write(void *addr, bool write, size_t size, app_pc pc) {
    if (global_trace_writer && config.enable_trace) {
        void *drcontext = dr_get_current_drcontext();
        uint32_t thread_id = dr_get_thread_id(drcontext);
        uint64_t timestamp = get_timestamp();

        /* Thread-safe write to global protobuf file */
        dr_mutex_lock(trace_mutex);
        pb_trace_write_event(global_trace_writer,
                            timestamp,
                            thread_id,
                            (uint64_t)addr,
                            write,
                            (uint32_t)size);
        dr_mutex_unlock(trace_mutex);
    }
}

/* Parse a simple key=value configuration file */
static bool parse_config_file(const char *config_path) {
    file_t file;
    char buffer[2048];
    char *key, *value;
    int bytes_read;
    
    if (!config_path) return false;
    
    file = dr_open_file(config_path, DR_FILE_READ);
    if (file == INVALID_FILE) {
        dr_fprintf(STDERR, "Warning: Could not open config file %s, using defaults\n", config_path);
        return false;
    }
    
    /* Read entire file into buffer */
    bytes_read = dr_read_file(file, buffer, sizeof(buffer) - 1);
    dr_close_file(file);
    
    if (bytes_read <= 0) return false;
    
    buffer[bytes_read] = '\0';  /* Null terminate */
    
    /* Parse line by line */
    char *line = buffer;
    char *next_line;
    
    while (line && *line) {
        /* Find end of line */
        next_line = strchr(line, '\n');
        if (next_line) {
            *next_line = '\0';
            next_line++;
        }
        
        /* Remove carriage return if present */
        char *cr = strchr(line, '\r');
        if (cr) *cr = '\0';
        
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\0') {
            line = next_line;
            continue;
        }
        
        /* Parse key=value */
        key = line;
        value = strchr(line, '=');
        if (!value) {
            line = next_line;
            continue;
        }
        
        *value = '\0';  /* Terminate key string */
        value++;        /* Point to value */
        
        /* Trim whitespace */
        while (*key == ' ' || *key == '\t') key++;
        while (*value == ' ' || *value == '\t') value++;
        
        /* Parse configuration values */
        if (strcmp(key, "cache_line_size") == 0) {
            config.cache_line_size = (uint)atoi(value);
        } else if (strcmp(key, "hll_bits") == 0) {
            config.hll_bits = (uint)atoi(value);
        } else if (strcmp(key, "sample_hll_bits") == 0) {
            config.sample_hll_bits = (uint)atoi(value);
        } else if (strcmp(key, "sample_window_refs") == 0) {
            config.sample_window_refs = (uint)atoi(value);
        } else if (strcmp(key, "max_mem_refs") == 0) {
            config.max_mem_refs = (uint)atoi(value);
        } else if (strcmp(key, "enable_trace") == 0) {
            config.enable_trace = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        } else if (strcmp(key, "wss_stat_tracking") == 0) {
            config.wss_stat_tracking = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        } else if (strcmp(key, "wss_exact_tracking") == 0) {
            config.wss_exact_tracking = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        } else if (strcmp(key, "wss_hll_tracking") == 0) {
            config.wss_hll_tracking = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        } else if (strcmp(key, "enable_instruction_threshold") == 0) {
            config.enable_instruction_threshold = (strcmp(value, "true") == 0 || strcmp(value, "1") == 0);
        } else if (strcmp(key, "instruction_threshold") == 0) {
            config.instruction_threshold = (uint64)strtoull(value, NULL, 10);
        } else if (strcmp(key, "pb_trace_output") == 0) {
            strncpy(config.pb_trace_output, value, sizeof(config.pb_trace_output) - 1);
        } else if (strcmp(key, "pb_timeseries_output") == 0) {
            strncpy(config.pb_timeseries_output, value, sizeof(config.pb_timeseries_output) - 1);
        }
        
        line = next_line;
    }
    
    return true;
}

/* Initialize derived config values */
static void init_config_derived_values(void) {
    cache_line_mask = (~(uintptr_t)(config.cache_line_size - 1));
    mem_buf_size = sizeof(mem_ref_t) * config.max_mem_refs;
}

static void finalize_sample_window(per_thread_t *t) {
    if (!t) return;

    void *drcontext = dr_get_current_drcontext();
    uint32_t thread_id = dr_get_thread_id(drcontext);

    /* exact WSS (only if enabled) */
    ws_stats_t s = {0};
    if (config.wss_exact_tracking && t->sample_ws) {
        ws_get_stats(t->sample_ws, &s);
    }

    /* HLL WSS (approx, only if enabled) */
    double wss_est = 0.0;
    if (config.wss_hll_tracking) {
        wss_est = hll_count(&t->sample_hll);
    }

    /* Write to protobuf timeseries file (thread-safe) */
    if (global_timeseries_writer && config.wss_stat_tracking) {
        dr_mutex_lock(timeseries_mutex);
        pb_timeseries_write_sample(
            global_timeseries_writer,
            t->sample_idx,              // window_number
            thread_id,                  // thread_id
            t->sample_read_count,       // read_count
            t->sample_write_count,      // write_count
            t->sample_ref_count,        // total_refs
            s.distinct,                 // wss_exact
            wss_est,                    // wss_approx
            get_timestamp(),            // timestamp
            // Read size histograms
            t->sample_read_size_1, t->sample_read_size_2, t->sample_read_size_4, t->sample_read_size_8,
            t->sample_read_size_16, t->sample_read_size_32, t->sample_read_size_64, t->sample_read_size_other,
            // Write size histograms
            t->sample_write_size_1, t->sample_write_size_2, t->sample_write_size_4, t->sample_write_size_8,
            t->sample_write_size_16, t->sample_write_size_32, t->sample_write_size_64, t->sample_write_size_other
        );
        dr_mutex_unlock(timeseries_mutex);
    }

    /* reset for next window */
    if (config.wss_exact_tracking && t->sample_ws) {
        ws_reset(t->sample_ws);       /* destroys nodes & zeros stats */
    }
    if (config.wss_hll_tracking) {
        hll_reset(&t->sample_hll);    /* zero registers, keep alloc */
    }
    t->sample_ref_count = 0;
    t->sample_read_count = 0;
    t->sample_write_count = 0;

    /* Reset per-window size counters */
    t->sample_read_size_1 = 0;
    t->sample_read_size_2 = 0;
    t->sample_read_size_4 = 0;
    t->sample_read_size_8 = 0;
    t->sample_read_size_16 = 0;
    t->sample_read_size_32 = 0;
    t->sample_read_size_64 = 0;
    t->sample_read_size_other = 0;
    t->sample_write_size_1 = 0;
    t->sample_write_size_2 = 0;
    t->sample_write_size_4 = 0;
    t->sample_write_size_8 = 0;
    t->sample_write_size_16 = 0;
    t->sample_write_size_32 = 0;
    t->sample_write_size_64 = 0;
    t->sample_write_size_other = 0;

    t->sample_idx++;
}

DR_EXPORT void

dr_client_main(client_id_t id, int argc, const char *argv[])
{
   drreg_options_t ops = { sizeof(ops), 2, false};
    /* Specify priority relative to other instrumentation operations: */
    drmgr_priority_t priority = { sizeof(priority), /* size of struct */
                                  "memcount",       /* name of our operation */
                                  NULL, /* optional name of operation we should precede */
                                  NULL, /* optional name of operation we should follow */
                                  0 };  /* numeric priority */
    dr_set_client_name("Custom Client 'memcount'", NULL);
    
    /* Parse command line arguments */
    const char *config_file = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "-no_trace") == 0) {
            config.enable_trace = false;
        } else if (strcmp(argv[i], "-trace") == 0) {
            config.enable_trace = true;
        } else if (strcmp(argv[i], "-config") == 0 && i + 1 < argc) {
            config_file = argv[i + 1];
            i++; /* Skip the config file path argument */
        }
    }
    
    /* Load configuration file if provided */
    if (config_file) {
        if (parse_config_file(config_file)) {
            dr_fprintf(STDERR, "Config loaded: sample_window_refs=%u\n", config.sample_window_refs);
        } else {
            dr_fprintf(STDERR, "Failed to load config file: %s\n", config_file);
        }
    } else {
        dr_fprintf(STDERR, "No config file specified, using defaults: sample_window_refs=%u\n", config.sample_window_refs);
    }
    
    /* Initialize derived configuration values */
    init_config_derived_values();
        page_size = dr_page_size();
    drmgr_init();
    drutil_init();
    client_id = id;
    mutex = dr_mutex_create();
    dr_register_exit_event(event_exit);

    /* Record start time */
    start_time_us = dr_get_microseconds();

    hll_mutex = dr_mutex_create();
    DR_ASSERT(hll_init(&global_hll, config.hll_bits) == 0);

    /* Initialize instruction count mutex */
    instr_count_mutex = dr_mutex_create();

    /* Initialize protobuf writers (single file for all threads) */
    if (config.enable_trace) {
        char trace_filename[256];
        dr_snprintf(trace_filename, sizeof(trace_filename), "%s_%d.pb",
                   config.pb_trace_output, dr_get_process_id());
        global_trace_writer = pb_trace_writer_create(trace_filename);
        trace_mutex = dr_mutex_create();
        if (global_trace_writer) {
            dr_fprintf(STDERR, "Protobuf trace output enabled: %s\n", trace_filename);
        } else {
            dr_fprintf(STDERR, "Warning: Failed to create protobuf trace writer\n");
        }
    }

    if (config.wss_stat_tracking) {
        char timeseries_filename[256];
        const char *command;

        dr_snprintf(timeseries_filename, sizeof(timeseries_filename),
                   "%s_%d.pb", config.pb_timeseries_output, dr_get_process_id());

        /* Get command line (returns const char*) */
        command = dr_get_application_name();
        if (!command) {
            command = "unknown";
        }

        global_timeseries_writer = pb_timeseries_writer_create(
            timeseries_filename,
            "dynamorio",
            dr_get_process_id(),
            command,
            config.sample_window_refs,
            config.cache_line_size
        );

        timeseries_mutex = dr_mutex_create();
        thread_count_mutex = dr_mutex_create();

        if (global_timeseries_writer) {
            dr_fprintf(STDERR, "Protobuf time-series output enabled: %s\n", timeseries_filename);
        } else {
            dr_fprintf(STDERR, "Warning: Failed to create protobuf time-series writer\n");
        }
    }

    if (!drmgr_register_thread_init_event(event_thread_init) ||
        !drmgr_register_thread_exit_event(event_thread_exit) ||
        !drmgr_register_bb_app2app_event(event_bb_app2app, &priority) ||
        !drmgr_register_bb_instrumentation_event(event_bb_analysis, event_bb_insert,
                                                 &priority) ||
        drreg_init(&ops) != DRREG_SUCCESS || !drx_init()) {
        /* something is wrong: can't continue */
        DR_ASSERT(false);
        return;
    }
    tls_index = drmgr_register_tls_field();
    DR_ASSERT(tls_index != -1);

    code_cache_init();
    /* make it easy to tell, by looking at log file, which client executed */
    dr_log(NULL, DR_LOG_ALL, 1, "Client 'countcalls' initializing\n");
    /* also give notification to stderr */
    if (dr_is_notify_on()) {
#    ifdef WINDOWS
        /* ask for best-effort printing to cmd window.  must be called at init. */
        dr_enable_console_printing();
#    endif
        dr_fprintf(STDERR, "Client memcount is running\n");
    }
}

static void
event_exit()
{
    char msg[512];
    int len;
    double hll_est_lines = hll_count(&global_hll);

    /* Capture end time and calculate execution time */
    end_time_us = dr_get_microseconds();
    uint64 execution_time_us = end_time_us - start_time_us;
    double execution_time_ms = execution_time_us / 1000.0;
    double execution_time_sec = execution_time_us / 1000000.0;

    len = dr_snprintf(msg, sizeof(msg) / sizeof(msg[0]),
                        "Instrumentation results:\n"
                        "  saw %llu memory references\n"
                        "  number of reads: %llu\n"
                        "  number of writes: %llu\n"
                        "  working set size: %llu\n"
                        "  execution time (us): %llu\n"
                        "  execution time (ms): %.3f\n"
                        "  execution time (s): %.6f\n",
                        global_num_refs, global_num_reads, global_num_writes, global_working_set,
                        execution_time_us, execution_time_ms, execution_time_sec);
    DR_ASSERT (len > 0);
    NULL_TERMINATE_BUFFER(msg);
    DISPLAY_STRING(msg);

    /* Print size-specific read statistics */
    len = dr_snprintf(msg, sizeof(msg)/sizeof(msg[0]),
                    "Read size breakdown:\n"
                    "  1-byte reads: %llu\n"
                    "  2-byte reads: %llu\n"
                    "  4-byte reads: %llu\n"
                    "  8-byte reads: %llu\n"
                    "  16-byte reads: %llu\n"
                    "  32-byte reads: %llu\n"
                    "  64-byte reads: %llu\n"
                    "  other-size reads: %llu\n",
                    global_read_size_1, global_read_size_2, global_read_size_4, global_read_size_8,
                    global_read_size_16, global_read_size_32, global_read_size_64, global_read_size_other);
    DR_ASSERT(len > 0);
    NULL_TERMINATE_BUFFER(msg);
    DISPLAY_STRING(msg);

    /* Print size-specific write statistics */
    len = dr_snprintf(msg, sizeof(msg)/sizeof(msg[0]),
                    "Write size breakdown:\n"
                    "  1-byte writes: %llu\n"
                    "  2-byte writes: %llu\n"
                    "  4-byte writes: %llu\n"
                    "  8-byte writes: %llu\n"
                    "  16-byte writes: %llu\n"
                    "  32-byte writes: %llu\n"
                    "  64-byte writes: %llu\n"
                    "  other-size writes: %llu\n",
                    global_write_size_1, global_write_size_2, global_write_size_4, global_write_size_8,
                    global_write_size_16, global_write_size_32, global_write_size_64, global_write_size_other);
    DR_ASSERT(len > 0);
    NULL_TERMINATE_BUFFER(msg);
    DISPLAY_STRING(msg);

    /* Print HLL block */
    len = dr_snprintf(msg, sizeof(msg)/sizeof(msg[0]),
		    "Instrumentation results (HLL estimate):\n"
		    "  estimated unique lines: %llu\n",
		    (unsigned long long) hll_est_lines);
    NULL_TERMINATE_BUFFER(msg); DISPLAY_STRING(msg);

    /* Close protobuf writers */
    if (global_trace_writer) {
        pb_trace_writer_close(global_trace_writer);
        dr_fprintf(STDERR, "Protobuf trace file closed\n");
        global_trace_writer = NULL;
    }

    if (global_timeseries_writer) {
        /* Set thread count in metadata before closing */
        pb_timeseries_set_num_threads(global_timeseries_writer, global_thread_count);
        pb_timeseries_writer_close(global_timeseries_writer);
        dr_fprintf(STDERR, "Protobuf time-series file closed\n");
        global_timeseries_writer = NULL;
    }

    /* Destroy protobuf mutexes */
    if (trace_mutex) {
        dr_mutex_destroy(trace_mutex);
        trace_mutex = NULL;
    }
    if (timeseries_mutex) {
        dr_mutex_destroy(timeseries_mutex);
        timeseries_mutex = NULL;
    }
    if (thread_count_mutex) {
        dr_mutex_destroy(thread_count_mutex);
        thread_count_mutex = NULL;
    }

    code_cache_exit();

    if (!drmgr_unregister_tls_field(tls_index) ||
    !drmgr_unregister_thread_init_event(event_thread_init) ||
    !drmgr_unregister_thread_exit_event(event_thread_exit) ||
    !drmgr_unregister_bb_insertion_event(event_bb_insert) ||
    drreg_exit() != DRREG_SUCCESS)
    DR_ASSERT(false);

    dr_mutex_destroy(mutex);

    /* Destroy instruction count mutex */
    if (instr_count_mutex) {
        dr_mutex_destroy(instr_count_mutex);
        instr_count_mutex = NULL;
    }

    drutil_exit();
    drmgr_exit();
    drx_exit();
}

#ifdef WINDOWS
#    define IF_WINDOWS(x) x
#else
#    define IF_WINDOWS(x) /* nothing */
#endif

static void
event_thread_init(void *drcontext)
{
    per_thread_t *data;

    /* allocate thread private data */
    data = dr_thread_alloc(drcontext, sizeof(per_thread_t));
    drmgr_set_tls_field(drcontext, tls_index, data);
    data->buf_base = dr_thread_alloc(drcontext, mem_buf_size);
    data->buf_ptr = data->buf_base;
    /* set buf_end to be negative of address of buffer end for the lea later */
    data->buf_end = -(ptr_int_t)(data->buf_base + mem_buf_size);
    data->num_refs = 0;
    data->num_reads = 0;
    data->num_writes = 0;
    data->working_set = 0;
    if (config.wss_exact_tracking) {
        data->ws = ws_create();
    } else {
        data->ws = NULL;
    }
    if (config.wss_hll_tracking) {
        DR_ASSERT(hll_init(&data->hll, config.hll_bits) == 0);
    }

    /* Initialize size-specific counters */
    data->read_size_1 = 0;
    data->read_size_2 = 0;
    data->read_size_4 = 0;
    data->read_size_8 = 0;
    data->read_size_16 = 0;
    data->read_size_32 = 0;
    data->read_size_64 = 0;
    data->read_size_other = 0;
    data->write_size_1 = 0;
    data->write_size_2 = 0;
    data->write_size_4 = 0;
    data->write_size_8 = 0;
    data->write_size_16 = 0;
    data->write_size_32 = 0;
    data->write_size_64 = 0;
    data->write_size_other = 0;

    /* per-window sampling structures (independent of wss_stat_tracking) */
    if (config.wss_exact_tracking) {
        data->sample_ws = ws_create();
    } else {
        data->sample_ws = NULL;
    }

    if (config.wss_hll_tracking) {
        DR_ASSERT(hll_init(&data->sample_hll, config.sample_hll_bits) == 0);
    }

    /* per-window sampling counters (only needed if stat tracking is enabled) */
    if (config.wss_stat_tracking) {
        data->sample_ref_count = 0;
        data->sample_read_count = 0;
        data->sample_write_count = 0;
        data->sample_idx = 0;

        /* Initialize per-window size counters */
        data->sample_read_size_1 = 0;
        data->sample_read_size_2 = 0;
        data->sample_read_size_4 = 0;
        data->sample_read_size_8 = 0;
        data->sample_read_size_16 = 0;
        data->sample_read_size_32 = 0;
        data->sample_read_size_64 = 0;
        data->sample_read_size_other = 0;
        data->sample_write_size_1 = 0;
        data->sample_write_size_2 = 0;
        data->sample_write_size_4 = 0;
        data->sample_write_size_8 = 0;
        data->sample_write_size_16 = 0;
        data->sample_write_size_32 = 0;
        data->sample_write_size_64 = 0;
        data->sample_write_size_other = 0;
    } else {
        /* If stat tracking is disabled, still need to initialize counters */
        data->sample_ref_count = 0;
        data->sample_read_count = 0;
        data->sample_write_count = 0;
        data->sample_idx = 0;
    }

    /* Track total thread count for protobuf metadata */
    if (config.wss_stat_tracking && global_timeseries_writer) {
        dr_mutex_lock(thread_count_mutex);
        global_thread_count++;
        dr_mutex_unlock(thread_count_mutex);
    }

    dr_log(drcontext, DR_LOG_ALL, 1, "memcount: set up for thread " TIDFMT "\n",
           dr_get_thread_id(drcontext));
}

static void 
event_thread_exit(void *drcontext) 
{
    per_thread_t *data;

    memtrace(drcontext);
    data = drmgr_get_tls_field(drcontext, tls_index);

    /* flush last partial window (if any) */
    if (config.wss_stat_tracking && data->sample_ref_count > 0)
    	finalize_sample_window(data);

    /* destroy windowed structures (independent of wss_stat_tracking) */
    if (config.wss_exact_tracking && data->sample_ws) {
        ws_destroy(data->sample_ws);
    }
    if (config.wss_hll_tracking) {
        hll_destroy(&data->sample_hll);
    }

    ws_stats_t s = {0};
    if (config.wss_exact_tracking) {
        ws_get_stats(data->ws, &s);
        data->working_set = s.distinct;    /* #lines seen exactly once */
    } else if (config.wss_hll_tracking) {
        /* Use HLL estimate if exact tracking is disabled but HLL is enabled */
        data->working_set = (uint64)hll_count(&data->hll);
    } else {
        /* Both tracking methods disabled */
        data->working_set = 0;
    }
    if (data->ws) {
        ws_destroy(data->ws);
        data->ws = NULL;
    }

    dr_mutex_lock(mutex);
    global_num_refs += data->num_refs;
    global_num_reads += data->num_reads;
    global_num_writes += data->num_writes;
    global_working_set += data->working_set;

    /* Aggregate size-specific counters */
    global_read_size_1 += data->read_size_1;
    global_read_size_2 += data->read_size_2;
    global_read_size_4 += data->read_size_4;
    global_read_size_8 += data->read_size_8;
    global_read_size_16 += data->read_size_16;
    global_read_size_32 += data->read_size_32;
    global_read_size_64 += data->read_size_64;
    global_read_size_other += data->read_size_other;

    global_write_size_1 += data->write_size_1;
    global_write_size_2 += data->write_size_2;
    global_write_size_4 += data->write_size_4;
    global_write_size_8 += data->write_size_8;
    global_write_size_16 += data->write_size_16;
    global_write_size_32 += data->write_size_32;
    global_write_size_64 += data->write_size_64;
    global_write_size_other += data->write_size_other;

    dr_mutex_unlock(mutex);

    if (config.wss_hll_tracking) {
        dr_mutex_lock(hll_mutex);
        hll_merge(&global_hll, &data->hll);
        dr_mutex_unlock(hll_mutex);
        hll_destroy(&data->hll);
    }

    dr_thread_free(drcontext, data->buf_base, mem_buf_size);
    dr_thread_free(drcontext, data, sizeof(per_thread_t));
}

/* we transform string loops into regular loops so we can more easily
 * monitor every memory reference they make
 */
static dr_emit_flags_t
event_bb_app2app(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                 bool translating)
{
    if (!drutil_expand_rep_string(drcontext, bb)) {
        DR_ASSERT(false);
        /* in release build, carry on: we'll just miss per-iter refs */
    }
    if (!drx_expand_scatter_gather(drcontext, bb, NULL)) {
        DR_ASSERT(false);
    }
    return DR_EMIT_DEFAULT;
}

static dr_emit_flags_t
event_bb_analysis(void *drcontext, void *tag, instrlist_t *bb, bool for_trace,
                  bool translating, void **user_data)
{
    instru_data_t *data = (instru_data_t *)dr_thread_alloc(drcontext, sizeof(*data));
    data->last_pc = NULL;
    *user_data = (void *)data;
    return DR_EMIT_DEFAULT;
}

/* event_bb_insert calls instrument_mem to instrument every
 * application memory reference.
 */
static dr_emit_flags_t
event_bb_insert(void *drcontext, void *tag, instrlist_t *bb, instr_t *where,
                bool for_trace, bool translating, void *user_data)
{
    int i;
    instru_data_t *data = (instru_data_t *)user_data;
    /* Use the drmgr_orig_app_instr_* interface to properly handle our own use
     * of drutil_expand_rep_string() and drx_expand_scatter_gather() (as well
     * as another client/library emulating the instruction stream).
     */
    instr_t *instr_fetch = drmgr_orig_app_instr_for_fetch(drcontext);
    if (instr_fetch != NULL)
        data->last_pc = instr_get_app_pc(instr_fetch);
    app_pc last_pc = data->last_pc;

    /* Count instructions for threshold checking (do this once at the start of BB) */
    if (config.enable_instruction_threshold && !threshold_reached &&
        drmgr_is_first_instr(drcontext, where)) {
        /* Count number of app instructions in this basic block */
        uint64 bb_instr_count = 0;
        for (instr_t *instr = instrlist_first_app(bb); instr != NULL;
             instr = instr_get_next_app(instr)) {
            bb_instr_count++;
        }
        /* Insert clean call to check threshold at start of BB */
        dr_insert_clean_call(drcontext, bb, where,
                            (void *)check_instruction_threshold,
                            false, 1,
                            OPND_CREATE_INT64(bb_instr_count));
    }

    if (drmgr_is_last_instr(drcontext, where))
        dr_thread_free(drcontext, data, sizeof(*data));

    instr_t *instr_operands = drmgr_orig_app_instr_for_operands(drcontext);
    if (instr_operands == NULL ||
        (!instr_writes_memory(instr_operands) && !instr_reads_memory(instr_operands)))
        return DR_EMIT_DEFAULT;
    DR_ASSERT(instr_is_app(instr_operands));
    DR_ASSERT(last_pc != NULL);

    /* Use direct trace instrumentation for tracing, buffer-based for stats */
    if (config.enable_trace) {
        if (instr_reads_memory(instr_operands)) {
            for (i = 0; i < instr_num_srcs(instr_operands); i++) {
                if (opnd_is_memory_reference(instr_get_src(instr_operands, i))) {
                    instrument_mem_direct(drcontext, bb, where, last_pc, instr_operands, i, false);
                }
            }
        }
        if (instr_writes_memory(instr_operands)) {
            for (i = 0; i < instr_num_dsts(instr_operands); i++) {
                if (opnd_is_memory_reference(instr_get_dst(instr_operands, i))) {
                    instrument_mem_direct(drcontext, bb, where, last_pc, instr_operands, i, true);
                }
            }
        }
    }
    
    /* Use buffer-based instrumentation for WSS tracking */
    if (config.wss_exact_tracking || config.wss_hll_tracking || config.wss_stat_tracking) {
        if (instr_reads_memory(instr_operands)) {
            for (i = 0; i < instr_num_srcs(instr_operands); i++) {
                if (opnd_is_memory_reference(instr_get_src(instr_operands, i))) {
                    instrument_mem(drcontext, bb, where, last_pc, instr_operands, i, false);
                }
            }
        }
        if (instr_writes_memory(instr_operands)) {
            for (i = 0; i < instr_num_dsts(instr_operands); i++) {
                if (opnd_is_memory_reference(instr_get_dst(instr_operands, i))) {
                    instrument_mem(drcontext, bb, where, last_pc, instr_operands, i, true);
                }
            }
        }
    }
    return DR_EMIT_DEFAULT;
}

static void 
memtrace(void *drcontext)
{
    per_thread_t *data;
    int num_refs;
    int num_reads = 0;
    int num_writes = 0;
    int working_set = 0;
    mem_ref_t *mem_ref;

    data = drmgr_get_tls_field(drcontext, tls_index);
    mem_ref = (mem_ref_t *)data->buf_base;
    num_refs = (int)((mem_ref_t *)data->buf_ptr - mem_ref);

    for(int i = 0; i < num_refs; i++) {
        /* Note: Per-reference timestamps removed for performance.
         * Timestamps are captured at sampling window boundaries (finalize_sample_window)
         * and during trace events (direct_trace_write) if tracing is enabled. */

	uintptr_t key = ((uintptr_t)mem_ref->addr) & cache_line_mask;
        
	if (config.wss_exact_tracking) {
		ws_record(data->ws, key);
	}
	if (config.wss_hll_tracking) {
		hll_add(&data->hll, &key, sizeof(key));
	}

	/* WSS sampling only if enabled */
	if (config.wss_stat_tracking) {
		if (config.wss_exact_tracking && data->sample_ws) {
			ws_record(data->sample_ws, key);
		}
		if (config.wss_hll_tracking) {
			hll_add(&data->sample_hll, &key, sizeof key);
		}
	}

        /* Trace output is now handled directly via instrument_mem_direct() */

        if(mem_ref->write) {
            num_writes++;
            /* Track per-window write count for protobuf */
            if (config.wss_stat_tracking) {
                data->sample_write_count++;
            }
            /* Track size-specific write counts (global) */
            switch(mem_ref->size) {
                case 1:
                    data->write_size_1++;
                    if (config.wss_stat_tracking) data->sample_write_size_1++;
                    break;
                case 2:
                    data->write_size_2++;
                    if (config.wss_stat_tracking) data->sample_write_size_2++;
                    break;
                case 4:
                    data->write_size_4++;
                    if (config.wss_stat_tracking) data->sample_write_size_4++;
                    break;
                case 8:
                    data->write_size_8++;
                    if (config.wss_stat_tracking) data->sample_write_size_8++;
                    break;
                case 16:
                    data->write_size_16++;
                    if (config.wss_stat_tracking) data->sample_write_size_16++;
                    break;
                case 32:
                    data->write_size_32++;
                    if (config.wss_stat_tracking) data->sample_write_size_32++;
                    break;
                case 64:
                    data->write_size_64++;
                    if (config.wss_stat_tracking) data->sample_write_size_64++;
                    break;
                default:
                    data->write_size_other++;
                    if (config.wss_stat_tracking) data->sample_write_size_other++;
                    break;
            }
        } else {
            num_reads++;
            /* Track per-window read count for protobuf */
            if (config.wss_stat_tracking) {
                data->sample_read_count++;
            }
            /* Track size-specific read counts (global) */
            switch(mem_ref->size) {
                case 1:
                    data->read_size_1++;
                    if (config.wss_stat_tracking) data->sample_read_size_1++;
                    break;
                case 2:
                    data->read_size_2++;
                    if (config.wss_stat_tracking) data->sample_read_size_2++;
                    break;
                case 4:
                    data->read_size_4++;
                    if (config.wss_stat_tracking) data->sample_read_size_4++;
                    break;
                case 8:
                    data->read_size_8++;
                    if (config.wss_stat_tracking) data->sample_read_size_8++;
                    break;
                case 16:
                    data->read_size_16++;
                    if (config.wss_stat_tracking) data->sample_read_size_16++;
                    break;
                case 32:
                    data->read_size_32++;
                    if (config.wss_stat_tracking) data->sample_read_size_32++;
                    break;
                case 64:
                    data->read_size_64++;
                    if (config.wss_stat_tracking) data->sample_read_size_64++;
                    break;
                default:
                    data->read_size_other++;
                    if (config.wss_stat_tracking) data->sample_read_size_other++;
                    break;
            }
        }
        ++mem_ref;

	/* Sample window tracking only if WSS stats enabled */
	if (config.wss_stat_tracking) {
		data->sample_ref_count++;
		if (data->sample_ref_count == config.sample_window_refs) {
			finalize_sample_window(data);   // this should reset sample_ref_count to 0
		}
	}
    }

    memset(data->buf_base, 0, mem_buf_size);
    data->num_refs += num_refs;
    data->num_reads += num_reads;
    data->num_writes += num_writes;
    data->buf_ptr = data->buf_base;
}

/* clean_call dumps the memory reference info to the log file */ // FIXME: may not be necessary
static void
clean_call(void)
{
    void *drcontext = dr_get_current_drcontext();
    memtrace(drcontext);
}

static void
code_cache_init(void)
{
    void *drcontext;
    instrlist_t *ilist;
    instr_t *where;
    byte *end;

    drcontext = dr_get_current_drcontext();
    code_cache =
        dr_nonheap_alloc(page_size, DR_MEMPROT_READ | DR_MEMPROT_WRITE | DR_MEMPROT_EXEC);
    ilist = instrlist_create(drcontext);
    /* The lean procedure simply performs a clean call, and then jumps back
     * to the DR code cache.
     */
    where = INSTR_CREATE_jmp_ind(drcontext, opnd_create_reg(DR_REG_XCX));
    instrlist_meta_append(ilist, where);
    /* clean call */
    dr_insert_clean_call(drcontext, ilist, where, (void *)clean_call, false, 0);
    /* Encodes the instructions into memory and then cleans up. */
    end = instrlist_encode(drcontext, ilist, code_cache, false);
    DR_ASSERT((size_t)(end - code_cache) < page_size);
    instrlist_clear_and_destroy(drcontext, ilist);
    /* set the memory as just +rx now */
    dr_memory_protect(code_cache, page_size, DR_MEMPROT_READ | DR_MEMPROT_EXEC);
}

static void
code_cache_exit(void)
{
    dr_nonheap_free(code_cache, page_size);
}

/*
 * instrument_mem is called whenever a memory reference is identified.
 * It inserts code before the memory reference to to fill the memory buffer
 * and jump to our own code cache to call the clean_call when the buffer is full.
 */
static void // FIXME: can this be simplified?
instrument_mem(void *drcontext, instrlist_t *ilist, instr_t *where, app_pc pc,
               instr_t *memref_instr, int pos, bool write)
{
    instr_t *instr, *call, *restore;
    opnd_t ref, opnd1, opnd2;
    reg_id_t reg1, reg2;
    drvector_t allowed;

    /* Steal two scratch registers.
     * reg2 must be ECX or RCX for jecxz.
     */
    drreg_init_and_fill_vector(&allowed, false);
    drreg_set_vector_entry(&allowed, DR_REG_XCX, true);
    if (drreg_reserve_register(drcontext, ilist, where, &allowed, &reg2) !=
            DRREG_SUCCESS ||
        drreg_reserve_register(drcontext, ilist, where, NULL, &reg1) != DRREG_SUCCESS) {
        DR_ASSERT(false); /* cannot recover */
        drvector_delete(&allowed);
        return;
    }
    drvector_delete(&allowed);

    if (write)
        ref = instr_get_dst(memref_instr, pos);
    else
        ref = instr_get_src(memref_instr, pos);

    /* use drutil to get mem address */
    drutil_insert_get_mem_addr(drcontext, ilist, where, ref, reg1, reg2);

    /* The following assembly performs the following instructions
     * buf_ptr->write = write;
     * buf_ptr->addr  = addr;
     * buf_ptr->size  = size;
     * buf_ptr->pc    = pc;
     * buf_ptr++;
     * if (buf_ptr >= buf_end_ptr)
     *    clean_call();
     */
    drmgr_insert_read_tls_field(drcontext, tls_index, ilist, where, reg2);
    opnd1 = opnd_create_reg(reg2);
    opnd2 = OPND_CREATE_MEMPTR(reg2, offsetof(per_thread_t, buf_ptr));
    instr = INSTR_CREATE_mov_ld(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* Move write/read to write field */
    opnd1 = OPND_CREATE_MEM32(reg2, offsetof(mem_ref_t, write));
    opnd2 = OPND_CREATE_INT32(write);
    instr = INSTR_CREATE_mov_imm(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* Store address in memory ref */
    opnd1 = OPND_CREATE_MEMPTR(reg2, offsetof(mem_ref_t, addr));
    opnd2 = opnd_create_reg(reg1);
    instr = INSTR_CREATE_mov_st(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* Store size in memory ref */
    opnd1 = OPND_CREATE_MEMPTR(reg2, offsetof(mem_ref_t, size));
    /* drutil_opnd_mem_size_in_bytes handles OP_enter */
    opnd2 = OPND_CREATE_INT32(drutil_opnd_mem_size_in_bytes(ref, memref_instr));
    instr = INSTR_CREATE_mov_st(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* Store pc in memory ref */
    /* For 64-bit, we can't use a 64-bit immediate so we split pc into two halves.
     * We could alternatively load it into reg1 and then store reg1.
     * We use a convenience routine that does the two-step store for us.
     */
    opnd1 = OPND_CREATE_MEMPTR(reg2, offsetof(mem_ref_t, pc));
    instrlist_insert_mov_immed_ptrsz(drcontext, (ptr_int_t)pc, opnd1, ilist, where, NULL,
                                     NULL);

    /* Increment reg value by pointer size using lea instr */
    opnd1 = opnd_create_reg(reg2);
    opnd2 = opnd_create_base_disp(reg2, DR_REG_NULL, 0, sizeof(mem_ref_t), OPSZ_lea);
    instr = INSTR_CREATE_lea(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    drmgr_insert_read_tls_field(drcontext, tls_index, ilist, where, reg1);
    opnd1 = OPND_CREATE_MEMPTR(reg1, offsetof(per_thread_t, buf_ptr));
    opnd2 = opnd_create_reg(reg2);
    instr = INSTR_CREATE_mov_st(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* we use lea + jecxz trick for better performance
     * lea and jecxz won't disturb the eflags, so we won't insert
     * code to save and restore application's eflags.
     */
    /* lea [reg2 - buf_end] => reg2 */
    opnd1 = opnd_create_reg(reg1);
    opnd2 = OPND_CREATE_MEMPTR(reg1, offsetof(per_thread_t, buf_end));
    instr = INSTR_CREATE_mov_ld(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);
    opnd1 = opnd_create_reg(reg2);
    opnd2 = opnd_create_base_disp(reg1, reg2, 1, 0, OPSZ_lea);
    instr = INSTR_CREATE_lea(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);

    /* jecxz call */
    call = INSTR_CREATE_label(drcontext);
    opnd1 = opnd_create_instr(call);
    instr = INSTR_CREATE_jecxz(drcontext, opnd1);
    instrlist_meta_preinsert(ilist, where, instr);

    /* jump restore to skip clean call */
    restore = INSTR_CREATE_label(drcontext);
    opnd1 = opnd_create_instr(restore);
    instr = INSTR_CREATE_jmp(drcontext, opnd1);
    instrlist_meta_preinsert(ilist, where, instr);

    /* clean call */
    /* We jump to lean procedure which performs full context switch and
     * clean call invocation. This is to reduce the code cache size.
     */
    instrlist_meta_preinsert(ilist, where, call);
    /* mov restore DR_REG_XCX */
    opnd1 = opnd_create_reg(reg2);
    /* this is the return address for jumping back from lean procedure */
    opnd2 = opnd_create_instr(restore);
    /* We could use instrlist_insert_mov_instr_addr(), but with a register
     * destination we know we can use a 64-bit immediate.
     */
    instr = INSTR_CREATE_mov_imm(drcontext, opnd1, opnd2);
    instrlist_meta_preinsert(ilist, where, instr);
    /* jmp code_cache */
    opnd1 = opnd_create_pc(code_cache);
    instr = INSTR_CREATE_jmp(drcontext, opnd1);
    instrlist_meta_preinsert(ilist, where, instr);

    /* Restore scratch registers */
    instrlist_meta_preinsert(ilist, where, restore);
    if (drreg_unreserve_register(drcontext, ilist, where, reg1) != DRREG_SUCCESS ||
        drreg_unreserve_register(drcontext, ilist, where, reg2) != DRREG_SUCCESS)
        DR_ASSERT(false);
}

/* 
 * Direct instrumentation that calls our trace function immediately
 * instead of using the buffer system - much simpler and avoids buffer overflow
 */
static void
instrument_mem_direct(void *drcontext, instrlist_t *ilist, instr_t *where, app_pc pc,
                     instr_t *memref_instr, int pos, bool write)
{
    reg_id_t reg1;
    opnd_t ref;
    
    if (write)
        ref = instr_get_dst(memref_instr, pos);
    else
        ref = instr_get_src(memref_instr, pos);

    /* Reserve a register for address computation */
    if (drreg_reserve_register(drcontext, ilist, where, NULL, &reg1) != DRREG_SUCCESS) {
        DR_ASSERT(false);
        return;
    }

    /* Get memory address into register */
    if (!drutil_insert_get_mem_addr(drcontext, ilist, where, ref, reg1, DR_REG_NULL)) {
        DR_ASSERT(false);
        drreg_unreserve_register(drcontext, ilist, where, reg1);
        return;
    }

    /* Insert a clean call to our direct trace function */
    dr_insert_clean_call(drcontext, ilist, where, (void *)direct_trace_write, 
                        false, 4,
                        opnd_create_reg(reg1),                 /* addr in register */
                        OPND_CREATE_INT32(write),              /* write flag */
                        OPND_CREATE_INT32(drutil_opnd_mem_size_in_bytes(ref, memref_instr)), /* size */
                        OPND_CREATE_INTPTR(pc));               /* pc */

    /* Restore register */
    if (drreg_unreserve_register(drcontext, ilist, where, reg1) != DRREG_SUCCESS)
        DR_ASSERT(false);
}