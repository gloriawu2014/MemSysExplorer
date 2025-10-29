# Common library for shared profiler utilities
cmake_minimum_required(VERSION 3.14)
project(profiler_common C CXX)

include(FetchContent)

# Try to find protobuf first
find_package(Protobuf QUIET)

# If protobuf not found, build from source
if(NOT Protobuf_FOUND)
    message(STATUS "Protobuf not found on system - building from source")

    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG v21.12  # Use a stable version
        SOURCE_SUBDIR cmake
    )

    # Configure protobuf build options
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(protobuf_WITH_ZLIB OFF CACHE BOOL "" FORCE)
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(protobuf)

    # Set variables to match find_package behavior
    set(Protobuf_FOUND TRUE)
    set(Protobuf_LIBRARIES protobuf::libprotobuf)
    set(Protobuf_INCLUDE_DIRS ${protobuf_SOURCE_DIR}/src)
    set(Protobuf_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)

    # Create an alias for the protoc executable for custom commands
    add_executable(protoc_executable IMPORTED)
    set_target_properties(protoc_executable PROPERTIES
        IMPORTED_LOCATION ${Protobuf_PROTOC_EXECUTABLE}
    )

    message(STATUS "Built protobuf from source")
endif()

# Try to find protobuf-c
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PROTOBUF_C QUIET libprotobuf-c)
endif()

# If protobuf-c not found, build from source
if(NOT PROTOBUF_C_FOUND)
    message(STATUS "Protobuf-c not found on system - building from source")

    FetchContent_Declare(
        protobuf_c
        GIT_REPOSITORY https://github.com/protobuf-c/protobuf-c.git
        GIT_TAG v1.4.1  # Use a stable version
    )

    # Configure protobuf-c build options
    set(BUILD_PROTO3 ON CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(protobuf_c)

    # Set variables to match pkg-config behavior
    set(PROTOBUF_C_FOUND TRUE)
    set(PROTOBUF_C_LIBRARIES protobuf-c)
    set(PROTOBUF_C_INCLUDE_DIRS ${protobuf_c_SOURCE_DIR})

    # Find or use the built protoc-c compiler
    find_program(PROTOC_C_EXECUTABLE protoc-c
        HINTS ${protobuf_c_BINARY_DIR}
        PATHS ${protobuf_c_BINARY_DIR}/protoc-c
    )

    if(NOT PROTOC_C_EXECUTABLE)
        set(PROTOC_C_EXECUTABLE ${protobuf_c_BINARY_DIR}/protoc-c/protoc-gen-c)
    endif()

    message(STATUS "Built protobuf-c from source")
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
    # Determine protoc-c command
    if(PROTOC_C_EXECUTABLE)
        set(PROTOC_C_CMD ${PROTOC_C_EXECUTABLE})
    else()
        set(PROTOC_C_CMD protoc-c)
    endif()

    # Custom command to generate C protobuf code
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.c
               ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.h
               ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.c
               ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.h
        COMMAND ${PROTOC_C_CMD} --c_out=${CMAKE_CURRENT_BINARY_DIR}
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics.proto
        DEPENDS proto/memory_trace.proto proto/timeseries_metrics.proto
        COMMENT "Generating C protobuf code"
    )

    set(PROTOBUF_C_SOURCES
        src/protobuf_writer.c
        ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb-c.c
        ${CMAKE_CURRENT_BINARY_DIR}/timeseries_metrics.pb-c.c
    )

    message(STATUS "Protobuf-c available - including C protobuf writers")
endif()

# Add Python protobuf sources if available (for tools)
set(PROTOBUF_PYTHON_SOURCES "")
set(PROTOBUF_PYTHON_FILES "")
if(Protobuf_FOUND)
    # Determine protoc command
    if(TARGET protobuf::protoc)
        set(PROTOC_CMD $<TARGET_FILE:protobuf::protoc>)
    elseif(Protobuf_PROTOC_EXECUTABLE)
        set(PROTOC_CMD ${Protobuf_PROTOC_EXECUTABLE})
    else()
        set(PROTOC_CMD protoc)
    endif()

    # Generate C++ files from proto (for C++ tools if needed)
    if(TARGET protobuf::protoc)
        # Use the built-from-source protoc
        add_custom_command(
            OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb.cc
                   ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb.h
            COMMAND protobuf::protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                    --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                    ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace.proto
            DEPENDS proto/memory_trace.proto protobuf::protoc
            COMMENT "Generating C++ protobuf code"
        )
        set(PROTO_SRCS ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb.cc)
        set(PROTO_HDRS ${CMAKE_CURRENT_BINARY_DIR}/memory_trace.pb.h)
    else()
        # Use system protoc via CMake's protobuf_generate_cpp
        protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS
            proto/memory_trace.proto
        )
    endif()

    set(PROTOBUF_PYTHON_SOURCES
        src/memory_trace.cpp
        ${PROTO_SRCS}
    )

    # Also generate Python protobuf files for Python tools
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace_pb2.py
               ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics_pb2.py
        COMMAND ${PROTOC_CMD} --python_out=${CMAKE_CURRENT_SOURCE_DIR}/proto
                --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace.proto
                ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics.proto
        DEPENDS proto/memory_trace.proto proto/timeseries_metrics.proto
        COMMENT "Generating Python protobuf code"
    )

    set(PROTOBUF_PYTHON_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/memory_trace_pb2.py
        ${CMAKE_CURRENT_SOURCE_DIR}/proto/timeseries_metrics_pb2.py
    )

    message(STATUS "Protobuf available - including Python protobuf support")
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
    if(TARGET protobuf-c)
        # Built from source - link target directly
        target_link_libraries(profiler_common protobuf-c)
        target_include_directories(profiler_common PUBLIC
            ${protobuf_c_SOURCE_DIR}
            ${protobuf_c_BINARY_DIR}
        )
    else()
        # System installation - use pkg-config variables
        target_include_directories(profiler_common PUBLIC ${PROTOBUF_C_INCLUDE_DIRS})
        target_link_libraries(profiler_common ${PROTOBUF_C_LIBRARIES})
    endif()
    target_compile_definitions(profiler_common PUBLIC HAVE_PROTOBUF_C=1)
endif()

if(Protobuf_FOUND)
    if(TARGET protobuf::libprotobuf)
        # Built from source or modern CMake config - link target directly
        target_link_libraries(profiler_common protobuf::libprotobuf)
        target_include_directories(profiler_common PUBLIC ${protobuf_SOURCE_DIR}/src)
    else()
        # System installation with legacy variables
        target_link_libraries(profiler_common ${Protobuf_LIBRARIES})
        target_include_directories(profiler_common PUBLIC ${Protobuf_INCLUDE_DIRS})
    endif()
    target_compile_definitions(profiler_common PUBLIC HAVE_PROTOBUF=1)
endif()

# Link math library if on Unix (but not Apple)
if(UNIX AND NOT APPLE)
    target_link_libraries(profiler_common m)
endif()