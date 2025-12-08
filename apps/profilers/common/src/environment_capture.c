#include "environment_capture.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <time.h>
#include <stdio.h>

extern char **environ;

// Helper function to duplicate a string
static char* safe_strdup(const char* str) {
    if (!str) return NULL;
    size_t len = strlen(str);
    char* dup = malloc(len + 1);
    if (dup) {
        memcpy(dup, str, len + 1);
    }
    return dup;
}

// Helper function to get hostname
static char* get_hostname(void) {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';  // Ensure null termination
        return safe_strdup(hostname);
    }
    return safe_strdup("unknown");
}

// Helper function to get current working directory
static char* get_working_directory(void) {
    char* cwd = getcwd(NULL, 0);  // Let getcwd allocate
    if (cwd) {
        return cwd;  // Already allocated by getcwd
    }
    return safe_strdup("unknown");
}

// Helper function to count environment variables
static size_t count_environment_variables(void) {
    size_t count = 0;
    if (environ) {
        for (char **env = environ; *env; env++) {
            count++;
        }
    }
    return count;
}

// Helper function to parse environment variable into name/value
static int parse_env_var(const char* env_str, char** name, char** value) {
    if (!env_str || !name || !value) return 0;
    
    char* equals = strchr(env_str, '=');
    if (!equals) return 0;
    
    size_t name_len = equals - env_str;
    *name = malloc(name_len + 1);
    if (!*name) return 0;
    
    memcpy(*name, env_str, name_len);
    (*name)[name_len] = '\0';
    
    *value = safe_strdup(equals + 1);
    if (!*value) {
        free(*name);
        *name = NULL;
        return 0;
    }
    
    return 1;
}

const char* environment_capture_detect_os(void) {
    static char os_name[256];
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        if (strcmp(sys_info.sysname, "Linux") == 0) {
            return "Linux";
        } else if (strcmp(sys_info.sysname, "Darwin") == 0) {
            return "macOS";
        } else if (strstr(sys_info.sysname, "CYGWIN") || strstr(sys_info.sysname, "MINGW")) {
            return "Windows";
        }
        // Copy to static buffer for other systems
        strncpy(os_name, sys_info.sysname, sizeof(os_name) - 1);
        os_name[sizeof(os_name) - 1] = '\0';
        return os_name;
    }
    return "unknown";
}

const char* environment_capture_detect_arch(void) {
    static char arch_name[256];
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        strncpy(arch_name, sys_info.machine, sizeof(arch_name) - 1);
        arch_name[sizeof(arch_name) - 1] = '\0';
        return arch_name;
    }
    return "unknown";
}

system_environment_t* environment_capture_create(void) {
    system_environment_t* env = calloc(1, sizeof(system_environment_t));
    if (!env) return NULL;
    
    // Capture basic system information
    env->hostname = get_hostname();
    env->os_name = safe_strdup(environment_capture_detect_os());
    env->architecture = safe_strdup(environment_capture_detect_arch());
    env->working_directory = get_working_directory();
    
    // Get OS version (Linux-specific for now)
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        env->os_version = safe_strdup(sys_info.release);
    } else {
        env->os_version = safe_strdup("unknown");
    }
    
    // Capture environment variables
    env->env_count = count_environment_variables();
    if (env->env_count > 0) {
        env->env_names = calloc(env->env_count, sizeof(char*));
        env->env_values = calloc(env->env_count, sizeof(char*));
        
        if (env->env_names && env->env_values) {
            size_t i = 0;
            for (char **environ_ptr = environ; *environ_ptr && i < env->env_count; environ_ptr++, i++) {
                if (!parse_env_var(*environ_ptr, &env->env_names[i], &env->env_values[i])) {
                    // If parsing fails, skip this entry
                    i--;
                    env->env_count--;
                }
            }
        }
    }
    
    return env;
}

void environment_capture_destroy(system_environment_t* env) {
    if (!env) return;
    
    free(env->hostname);
    free(env->os_name);
    free(env->os_version);
    free(env->architecture);
    free(env->working_directory);
    
    if (env->env_names) {
        for (size_t i = 0; i < env->env_count; i++) {
            free(env->env_names[i]);
        }
        free(env->env_names);
    }
    
    if (env->env_values) {
        for (size_t i = 0; i < env->env_count; i++) {
            free(env->env_values[i]);
        }
        free(env->env_values);
    }
    
    free(env);
}

const char* environment_capture_get_var(const system_environment_t* env, const char* name) {
    if (!env || !name || !env->env_names || !env->env_values) return NULL;
    
    for (size_t i = 0; i < env->env_count; i++) {
        if (env->env_names[i] && strcmp(env->env_names[i], name) == 0) {
            return env->env_values[i];
        }
    }
    return NULL;
}

void environment_capture_print(const system_environment_t* env) {
    if (!env) {
        printf("Environment: NULL\n");
        return;
    }
    
    printf("System Environment:\n");
    printf("  Hostname: %s\n", env->hostname ?: "unknown");
    printf("  OS Name: %s\n", env->os_name ?: "unknown");
    printf("  OS Version: %s\n", env->os_version ?: "unknown");
    printf("  Architecture: %s\n", env->architecture ?: "unknown");
    printf("  Working Directory: %s\n", env->working_directory ?: "unknown");
    printf("  Environment Variables: %zu\n", env->env_count);
    
    if (env->env_names && env->env_values) {
        for (size_t i = 0; i < env->env_count; i++) {
            if (env->env_names[i] && env->env_values[i]) {
                printf("    %s=%s\n", env->env_names[i], env->env_values[i]);
            }
        }
    }
}

uint64_t environment_capture_timestamp_ns(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

uint32_t environment_capture_process_id(void) {
    return (uint32_t)getpid();
}