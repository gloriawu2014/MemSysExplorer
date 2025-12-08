#ifndef ENVIRONMENT_CAPTURE_H
#define ENVIRONMENT_CAPTURE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// System environment information structure
typedef struct {
    char *hostname;
    char *os_name;
    char *os_version;
    char *architecture;
    char *working_directory;
    
    // Environment variables
    char **env_names;      // Array of environment variable names
    char **env_values;     // Array of environment variable values
    size_t env_count;      // Number of environment variables
} system_environment_t;

// Create and populate system environment structure
// Returns NULL on failure
system_environment_t* environment_capture_create(void);

// Free system environment structure and all allocated memory
void environment_capture_destroy(system_environment_t* env);

// Get specific environment variable value
// Returns NULL if not found
const char* environment_capture_get_var(const system_environment_t* env, const char* name);

// Print environment information (for debugging)
void environment_capture_print(const system_environment_t* env);

// Get current timestamp in nanoseconds
uint64_t environment_capture_timestamp_ns(void);

// Get current process ID
uint32_t environment_capture_process_id(void);

// Helper function to detect OS name
const char* environment_capture_detect_os(void);

// Helper function to detect architecture
const char* environment_capture_detect_arch(void);

#ifdef __cplusplus
}
#endif

#endif /* ENVIRONMENT_CAPTURE_H */