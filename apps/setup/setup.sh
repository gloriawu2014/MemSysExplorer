#!/bin/bash

# Set project home
#export APPS_HOME="$(pwd)"
export APPS_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GCC_VERSION="11.2.0"
export CUDA_VERSION="12.6"
export NCU_VERSION="2024.3.2"
export PYTHON_VERSION="3.9.2"

# Define paths - User specific
export GCC_PATH="/usr/sup/gcc-$GCC_VERSION"
export CUDA_HOME="/usr/local/cuda-$CUDA_VERSION"
export CUDA_PATH="/usr/local/cuda-$CUDA_VERSION"
export NCU_HOME="/opt/nvidia/nsight-compute/$NCU_VERSION"
export SNIPER_ROOT="$APPS_HOME/profilers/snipersim"
export PIN_HOME="$SNIPER_ROOT/pin_kit"
export PYTHON_HOME="/usr/sup/Python-$PYTHON_VERSION"

# Setup protobuf/protobuf-c if built locally in third_party
setup_protobuf() {
    local THIRD_PARTY_DIR="$APPS_HOME/third_party"
    local PROTOBUF_PREFIX="$THIRD_PARTY_DIR/protobuf"
    local PROTOBUF_C_PREFIX="$THIRD_PARTY_DIR/protobuf-c"

    # Check if locally-built protobuf exists
    if [[ -d "$PROTOBUF_PREFIX" ]]; then
        echo "Found locally-built protobuf at $PROTOBUF_PREFIX"

        # Add protobuf bin to PATH
        export PATH="$PROTOBUF_PREFIX/bin:$PATH"

        # Add protobuf libraries
        if [[ -z "$LD_LIBRARY_PATH" ]]; then
            export LD_LIBRARY_PATH="$PROTOBUF_PREFIX/lib"
        else
            export LD_LIBRARY_PATH="$PROTOBUF_PREFIX/lib:$LD_LIBRARY_PATH"
        fi

        # Add protobuf pkgconfig
        if [[ -z "$PKG_CONFIG_PATH" ]]; then
            export PKG_CONFIG_PATH="$PROTOBUF_PREFIX/lib/pkgconfig"
        else
            export PKG_CONFIG_PATH="$PROTOBUF_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
        fi

        # Add to CMAKE_PREFIX_PATH
        if [[ -z "$CMAKE_PREFIX_PATH" ]]; then
            export CMAKE_PREFIX_PATH="$PROTOBUF_PREFIX"
        else
            export CMAKE_PREFIX_PATH="$PROTOBUF_PREFIX:$CMAKE_PREFIX_PATH"
        fi
    fi

    # Check if locally-built protobuf-c exists
    if [[ -d "$PROTOBUF_C_PREFIX" ]]; then
        echo "Found locally-built protobuf-c at $PROTOBUF_C_PREFIX"

        # Add protobuf-c bin to PATH
        export PATH="$PROTOBUF_C_PREFIX/bin:$PATH"

        # Add protobuf-c libraries
        if [[ -z "$LD_LIBRARY_PATH" ]]; then
            export LD_LIBRARY_PATH="$PROTOBUF_C_PREFIX/lib"
        else
            export LD_LIBRARY_PATH="$PROTOBUF_C_PREFIX/lib:$LD_LIBRARY_PATH"
        fi

        # Add protobuf-c pkgconfig
        if [[ -z "$PKG_CONFIG_PATH" ]]; then
            export PKG_CONFIG_PATH="$PROTOBUF_C_PREFIX/lib/pkgconfig"
        else
            export PKG_CONFIG_PATH="$PROTOBUF_C_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
        fi

        # Add to CMAKE_PREFIX_PATH
        if [[ -z "$CMAKE_PREFIX_PATH" ]]; then
            export CMAKE_PREFIX_PATH="$PROTOBUF_C_PREFIX"
        else
            export CMAKE_PREFIX_PATH="$PROTOBUF_C_PREFIX:$CMAKE_PREFIX_PATH"
        fi
    fi
}

# Setup protobuf before environment-specific setup
setup_protobuf

# Function to display available options
show_options() {
    echo -e "Available environments:\n  1) DynamoRIO\n  2) CUDA\n  3) Sniper\n  Enter the corresponding number to set up the environment."
}

# Check if the user provided an argument
if [[ $# -eq 0 ]]; then
    echo "No environment specified. Please choose one:"
    show_options
    exit 1
fi

# Select the environment based on user input
case "$1" in
    1 | dynamorio)
        echo "Setting up DynamoRIO environment..."

        if [[ ! -d "$GCC_PATH" ]]; then
            echo "Error: GCC $GCC_VERSION not found at $GCC_PATH"
            exit 1
        fi

        export PATH="$GCC_PATH/bin:$PATH"

        # Set LD_LIBRARY_PATH
        if [[ -z "$LD_LIBRARY_PATH" ]]; then
            export LD_LIBRARY_PATH="$GCC_PATH/lib64"
        else
            export LD_LIBRARY_PATH="$GCC_PATH/lib64:$LD_LIBRARY_PATH"
        fi

        # Set LIBRARY_PATH
        if [[ -z "$LIBRARY_PATH" ]]; then
            export LIBRARY_PATH="$GCC_PATH/lib64"
        else
            export LIBRARY_PATH="$GCC_PATH/lib64:$LIBRARY_PATH"
        fi

        # Set C_INCLUDE_PATH
        if [[ -z "$C_INCLUDE_PATH" ]]; then
            export C_INCLUDE_PATH="$GCC_PATH/include"
        else
            export C_INCLUDE_PATH="$GCC_PATH/include:$C_INCLUDE_PATH"
        fi

        # Set CPLUS_INCLUDE_PATH
        if [[ -z "$CPLUS_INCLUDE_PATH" ]]; then
            export CPLUS_INCLUDE_PATH="$GCC_PATH/include/c++/$GCC_VERSION"
        else
            export CPLUS_INCLUDE_PATH="$GCC_PATH/include/c++/$GCC_VERSION:$CPLUS_INCLUDE_PATH"
        fi

        # Verifying GCC version
        echo "Testing GCC:"
        g++ --version

	# Confirming environment variables
        echo "----------------------------------------"
        echo "DynamoRIO Environment Variables Set:"
        env | grep -E 'GCC_PATH|PATH|LD_LIBRARY_PATH|LIBRARY_PATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH'
        echo "----------------------------------------"

        ;;

    2 | cuda)
        echo "Setting up CUDA environment..."
        export PATH="$CUDA_HOME/bin:$PATH"

        # Set LD_LIBRARY_PATH
        if [[ -z "$LD_LIBRARY_PATH" ]]; then
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64"
        else
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        fi

        # Nsight Compute Settings
        export NCU_PARSING="$NCU_HOME/extras/python"

        # Set PYTHONPATH
        if [[ -z "$PYTHONPATH" ]]; then
            export PYTHONPATH="$NCU_PARSING"
        else
            export PYTHONPATH="$NCU_PARSING:$PYTHONPATH"
        fi

	#Confirming Envinroment Variables
	echo "CUDA and NCU setup completed."
        echo "----------------------------------------"
        echo "CUDA Environment Variables Set:"
        env | grep -E 'CUDA_HOME|PATH|LD_LIBRARY_PATH|NCU_HOME|NCU_PARSING|PYTHONPATH'
        echo "----------------------------------------"
        
	;;
    
    3 | sniper)
        echo "Setting up Sniper environment..."
        export SNIPER_ROOT="$APPS_HOME/profilers/sniper/snipersim"
        export PIN_HOME="$SNIPER_ROOT/pin_kit"

        # Set LD_LIBRARY_PATH
        if [[ -z "$LD_LIBRARY_PATH" ]]; then
            export LD_LIBRARY_PATH="$PYTHON_HOME/lib"
        else
            export LD_LIBRARY_PATH="$PYTHON_HOME/lib:$LD_LIBRARY_PATH"
        fi

        # Set C_INCLUDE_PATH
        if [[ -z "$C_INCLUDE_PATH" ]]; then
            export C_INCLUDE_PATH="$PYTHON_HOME/include/python$PYTHON_VERSION"
        else
            export C_INCLUDE_PATH="$PYTHON_HOME/include/python$PYTHON_VERSION:$C_INCLUDE_PATH"
        fi

        # Set CPLUS_INCLUDE_PATH
        if [[ -z "$CPLUS_INCLUDE_PATH" ]]; then
            export CPLUS_INCLUDE_PATH="$PYTHON_HOME/include/python$PYTHON_VERSION"
        else
            export CPLUS_INCLUDE_PATH="$PYTHON_HOME/include/python$PYTHON_VERSION:$CPLUS_INCLUDE_PATH"
        fi

	#Confirming 
        echo "Sniper environment setup completed."
        echo "----------------------------------------"
        echo "Sniper Environment Variables Set:"
        env | grep -E 'SNIPER_ROOT|PIN_HOME|PATH|LD_LIBRARY_PATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH'
        echo "----------------------------------------"	

        ;;

    *)
        echo "Invalid selection. Please choose one of the following:"
        show_options
        exit 1
        ;;
esac

