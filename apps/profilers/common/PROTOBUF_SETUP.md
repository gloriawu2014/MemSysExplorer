# Protobuf-C Setup Guide

## Install Required Packages

To enable protobuf support in DynamoRIO profiler, install these packages:

```bash
sudo apt-get update
sudo apt-get install -y libprotobuf-c-dev protobuf-c-compiler
```

## Verify Installation

```bash
protoc-c --version  # Should show protobuf-c version
```

## Generate Protobuf Code

The build system automatically generates both C and Python protobuf code when you run `make common`.

**C code generated:**
- `memory_trace.pb-c.c` / `memory_trace.pb-c.h`
- `timeseries_metrics.pb-c.c` / `timeseries_metrics.pb-c.h`

**Python code generated:**
- `memory_trace_pb2.py`
- `timeseries_metrics_pb2.py`

You don't need to manually run `protoc` commands anymore!

## Rebuild Common Library

```bash
cd ../../..  # Back to apps directory
make clean_common
make common
```

## Rebuild DynamoRIO Client

```bash
cd profilers/dynamorio
make clean
make build_client
```
