#!/usr/bin/env bash

# Get the apps directory (3 levels up from this script)
APPS_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
THIRD_PARTY_DIR="$APPS_DIR/third_party"

# Set library paths for both configure and build
export PROTOBUF_PREFIX="$THIRD_PARTY_DIR/protobuf"
export PROTOBUF_C_PREFIX="$THIRD_PARTY_DIR/protobuf-c"

# Add library directories to linker search path
if [ -d "$PROTOBUF_PREFIX/lib" ]; then
    export LIBRARY_PATH="$PROTOBUF_PREFIX/lib:${LIBRARY_PATH}"
fi

if [ -d "$PROTOBUF_C_PREFIX/lib" ]; then
    export LIBRARY_PATH="$PROTOBUF_C_PREFIX/lib:${LIBRARY_PATH}"
fi

# Run cmake to configure
cmake -DDynamoRIO_DIR="$DYNAMORIO_HOME"/cmake ./client

# Build memcount
make memcount
