# Common library for shared profiler utilities
cmake_minimum_required(VERSION 3.7)
project(profiler_common C CXX)

# Try to find protobuf and protobuf-c
# Note: The Makefile will build these from source if not on system
# and set CMAKE_PREFIX_PATH appropriately
find_package(Protobuf QUIET)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PROTOBUF_C QUIET libprotobuf-c)
endif()

# Always build the core utilities
set(CORE_SOURCES
    src/hll.c
    src/MurmurHash3.c
    src/ws_tsearch.c
    src/environment_capture.c
)

# Add protobuf-c sources if available (for C profilers like DynamoRIO)
set(PROTOBUF_C_SOURCES "")
if(PROTOBUF_C_FOUND)
    # Find protoc-c executable
    find_program(PROTOC_C_EXECUTABLE
        NAMES protoc-c
        HINTS ${CMAKE_PREFIX_PATH}/bin
        PATHS ENV PATH
    )

    if(NOT PROTOC_C_EXECUTABLE)
        message(FATAL_ERROR "protoc-c executable not found despite PROTOBUF_C being found")
    endif()

    # Custom command to generate C protobuf code
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.c
               ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.h
               ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.c
               ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.h
        COMMAND ${PROTOC_C_EXECUTABLE} --c_out=${CMAKE_CURRENT_BINARY_DIR}
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics.proto
        DEPENDS proto/memory_trace.proto proto/timeseries_metrics.proto
        COMMENT "Generating C protobuf code with ${PROTOC_C_EXECUTABLE}"
    )

    set(PROTOBUF_C_SOURCES
        src/protobuf_writer.c
        ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.c
        ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.c
    )

    message(STATUS "Protobuf-c found - including C protobuf writers")
else()
    message(WARNING "Protobuf-c not found - C protobuf support disabled")
    message(WARNING "Install protobuf-c or run 'make protobuf-c' from apps directory")

    # Still build protobuf_writer.c with stubs
    set(PROTOBUF_C_SOURCES src/protobuf_writer.c)
endif()

# Add Python protobuf sources if available (for tools)
set(PROTOBUF_PYTHON_SOURCES "")
set(PROTOBUF_PYTHON_FILES "")
if(Protobuf_FOUND)
    # Generate C++ files from proto (for C++ tools if needed)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
        proto/memory_trace.proto
    )

    set(PROTOBUF_PYTHON_SOURCES
        src/memory_trace.cpp
        ${PROTO_SRCS}
    )

    # Find protoc executable if not already set
    if(NOT Protobuf_PROTOC_EXECUTABLE)
        find_program(Protobuf_PROTOC_EXECUTABLE
            NAMES protoc
            HINTS ${CMAKE_PREFIX_PATH}/bin
            PATHS ENV PATH
        )
    endif()

    if(NOT Protobuf_PROTOC_EXECUTABLE)
        message(FATAL_ERROR "protoc executable not found despite Protobuf being found")
    endif()

    # Also generate Python protobuf files for Python tools
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace_pb2.py
               ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics_pb2.py
        COMMAND ${Protobuf_PROTOC_EXECUTABLE} --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics.proto
        DEPENDS proto/memory_trace.proto proto/timeseries_metrics.proto
        COMMENT "Generating Python protobuf code with ${Protobuf_PROTOC_EXECUTABLE}"
    )

    set(PROTOBUF_PYTHON_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace_pb2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics_pb2.py
    )

    message(STATUS "Protobuf found - including Python protobuf support")
else()
    message(WARNING "Protobuf not found - Python protobuf support disabled")
    message(WARNING "Install protobuf or run 'make protobuf' from apps directory")
endif()

# Create static library for common utilities
add_library(profiler_common STATIC
    ${CORE_SOURCES}
    ${PROTOBUF_C_SOURCES}
    ${PROTOBUF_PYTHON_SOURCES}
    ${PROTOBUF_PYTHON_FILES}  # Add Python files as dependencies to trigger generation
)

# Enable position-independent code (required for linking into shared libraries)
set_target_properties(profiler_common PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Ensure Python protobuf files are generated when building
if(PROTOBUF_PYTHON_FILES)
    add_custom_target(python_protobuf ALL DEPENDS ${PROTOBUF_PYTHON_FILES})
    add_dependencies(profiler_common python_protobuf)
endif()

# Set include directories
target_include_directories(profiler_common PUBLIC
    include
    ${CMAKE_CURRENT_BINARY_DIR}  # For generated protobuf headers
)

# Link libraries
if(PROTOBUF_C_FOUND)
    target_include_directories(profiler_common PUBLIC ${PROTOBUF_C_INCLUDE_DIRS})
    target_link_libraries(profiler_common ${PROTOBUF_C_LIBRARIES})
    target_compile_definitions(profiler_common PUBLIC HAVE_PROTOBUF_C=1)
endif()

if(Protobuf_FOUND)
    target_link_libraries(profiler_common ${Protobuf_LIBRARIES})
    target_include_directories(profiler_common PUBLIC ${Protobuf_INCLUDE_DIRS})
    target_compile_definitions(profiler_common PUBLIC HAVE_PROTOBUF=1)
endif()

# Link math library if on Unix (but not Apple)
if(UNIX AND NOT APPLE)
    target_link_libraries(profiler_common m)
endif()