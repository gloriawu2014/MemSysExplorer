#!/bin/tcsh

set APPS_HOME = $PWD
set GCC_VERSION = '11.2.0'
set CUDA_VERSION = '12.9'
set NCU_VERSION = "2023.2.2"
set PYTHON_VERSION = "3.9.2"

# Define paths - User specific -> Please include this your library
setenv GCC_PATH "/usr/sup/gcc-$GCC_VERSION"
setenv CUDA_HOME "/usr/local/cuda-$CUDA_VERSION"
setenv NCU_HOME "/opt/nvidia/nsight-compute/$NCU_VERSION"
setenv SNIPER_ROOT "$APPS_HOME/profilers/snipersim"
#setenv PIN_HOME "$SNIPER_ROOT/pin_kit"
setenv PYTHON_HOME "/usr/sup/Python-$PYTHON_VERSION"
setenv PIN_HOME "/r/tcal/work/dnguye/lib/pin"

# Setup protobuf/protobuf-c if built locally in third_party
# Check if locally-built protobuf exists
if (-d "$APPS_HOME/third_party/protobuf") then
    echo "Found locally-built protobuf at $APPS_HOME/third_party/protobuf"

    # Add protobuf bin to PATH
    setenv PATH "$APPS_HOME/third_party/protobuf/bin":$PATH

    # Add protobuf libraries
    if (! $?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH "$APPS_HOME/third_party/protobuf/lib"
    else
        setenv LD_LIBRARY_PATH "$APPS_HOME/third_party/protobuf/lib":$LD_LIBRARY_PATH
    endif

    # Add protobuf pkgconfig
    if (! $?PKG_CONFIG_PATH) then
        setenv PKG_CONFIG_PATH "$APPS_HOME/third_party/protobuf/lib/pkgconfig"
    else
        setenv PKG_CONFIG_PATH "$APPS_HOME/third_party/protobuf/lib/pkgconfig":$PKG_CONFIG_PATH
    endif

    # Add to CMAKE_PREFIX_PATH
    if (! $?CMAKE_PREFIX_PATH) then
        setenv CMAKE_PREFIX_PATH "$APPS_HOME/third_party/protobuf"
    else
        setenv CMAKE_PREFIX_PATH "$APPS_HOME/third_party/protobuf":$CMAKE_PREFIX_PATH
    endif
endif

# Check if locally-built protobuf-c exists
if (-d "$APPS_HOME/third_party/protobuf-c") then
    echo "Found locally-built protobuf-c at $APPS_HOME/third_party/protobuf-c"

    # Add protobuf-c bin to PATH
    setenv PATH "$APPS_HOME/third_party/protobuf-c/bin":$PATH

    # Add protobuf-c libraries
    if (! $?LD_LIBRARY_PATH) then
        setenv LD_LIBRARY_PATH "$APPS_HOME/third_party/protobuf-c/lib"
    else
        setenv LD_LIBRARY_PATH "$APPS_HOME/third_party/protobuf-c/lib":$LD_LIBRARY_PATH
    endif

    # Add protobuf-c pkgconfig
    if (! $?PKG_CONFIG_PATH) then
        setenv PKG_CONFIG_PATH "$APPS_HOME/third_party/protobuf-c/lib/pkgconfig"
    else
        setenv PKG_CONFIG_PATH "$APPS_HOME/third_party/protobuf-c/lib/pkgconfig":$PKG_CONFIG_PATH
    endif

    # Add to CMAKE_PREFIX_PATH
    if (! $?CMAKE_PREFIX_PATH) then
        setenv CMAKE_PREFIX_PATH "$APPS_HOME/third_party/protobuf-c"
    else
        setenv CMAKE_PREFIX_PATH "$APPS_HOME/third_party/protobuf-c":$CMAKE_PREFIX_PATH
    endif
endif

#setenv PATH "/usr/lib64/ccache:$PATH"
#echo "Testing Environmental Path"
#echo $GCC_PATH $NCU_HOME $SNIPER_ROOT $PIN_HOME $PYTHON_HOME

#Display available sourcing options
alias show_options 'echo "Available environments:\n  1) DynamoRIO\n  2) CUDA\n  3) Sniper\n  Enter the corresponding number to set up the environment."'

# Check if the user provided an argument
if ($#argv == 0) then
    echo "No environment specified. Please choose one:"
    show_options
    exit 1
endif

# Select the environment based on user input
switch ($1)
    case "1":
    case "dynamorio":
        echo "Setting up DynamoRIO environment..."
        if (! -d "$GCC_PATH") then
            echo "Error: GCC $GCC_VERSION not found at $GCC_PATH"
            exit 1
        endif
        
	setenv PATH "$GCC_PATH/bin:$PATH"
	
	#Set LD_LIBRARY_PATH
	if (! $?LD_LIBRARY_PATH) then
    		setenv LD_LIBRARY_PATH "$GCC_PATH/lib64"
	else
    		setenv LD_LIBRARY_PATH "$GCC_PATH/lib64:$LD_LIBRARY_PATH"
	endif

	#Set LIBRARY_PATH
	if (! $?LIBRARY_PATH) then
    		setenv LIBRARY_PATH "$GCC_PATH/lib64"
	else
    		setenv LIBRARY_PATH "$GCC_PATH/lib64:$LIBRARY_PATH"
	endif

	#Set C_INCLUDE_PATH
	if (! $?C_INCLUDE_PATH) then
    		setenv C_INCLUDE_PATH "$GCC_PATH/include"
	else
    		setenv C_INCLUDE_PATH "$GCC_PATH/include:$C_INCLUDE_PATH"
	endif

	#Set CPLUS_INCLUDE_PATH
	if (! $?CPLUS_INCLUDE_PATH) then
    		setenv CPLUS_INCLUDE_PATH "$GCC_PATH/include/c++/11.2.0"
	else
    		setenv CPLUS_INCLUDE_PATH "$GCC_PATH/include/c++/11.2.0:$CPLUS_INCLUDE_PATH"
	endif

	#Verifying GCC Version
	echo "Testing gcc"
	g++ version

	# Confirming environment variables
	echo "----------------------------------------"
	echo "DynamoRIO Environment Variables Set:"
	env | grep -E 'GCC_PATH|PATH|LD_LIBRARY_PATH|LIBRARY_PATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH'
	echo "----------------------------------------"
	
	breaksw
   case "2":
   case "cuda":
	setenv PATH "$CUDA_HOME/bin:$PATH"
        
	#Set LD_LIBRARY_PATH
        if (! $?LD_LIBRARY_PATH) then
                setenv LD_LIBRARY_PATH "$CUDA_HOME/lib64"
        else
                setenv LD_LIBRARY_PATH "$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
        endif

	#Nsight Compute Settings
	setenv NCU_PARSING "$NCU_HOME/extras/python"

	if (! $?PYTHON_PATH) then
    		setenv PYTHONPATH "$NCU_PARSING"
	else
    		setenv PYTHONPATH "$NCU_PARSING":"$PYTHONPATH"
	endif

	echo "CUDA and NCU Setup Completed."
	echo "----------------------------------------"
	echo "CUDA Environment Variables Set:"
	env | grep -E 'CUDA_HOME|PATH|LD_LIBRARY_PATH|NCU_HOME|NCU_PARSING|PYTHONPATH'
	echo "----------------------------------------"
		
	breaksw

   case "3":
   case "sniper":
	setenv SNIPER_ROOT "$APPS_HOME/profilers/sniper/snipersim"
	setenv PIN_HOME "$SNIPER_ROOT/pin_kit"
	setenv LD_RUN_PATH "/usr/sup/lib64:/usr/sup/lib"
	setenv PATH "/usr/share/Modules/bin:$PATH" 

	#Set LD_LIBRARY_PATH
        if (! $?LD_LIBRARY_PATH) then
                setenv LD_LIBRARY_PATH "$PYTHON_HOME/lib"
        else
                setenv LD_LIBRARY_PATH "$PYTHON_HOME/lib:$LD_LIBRARY_PATH"
        endif

	if (! $?C_INCLUDE_PATH) then
    		setenv C_INCLUDE_PATH "$PYTHON_HOME/include/python3.9"
	else
    		setenv C_INCLUDE_PATH "$PYTHON_HOME/include/python3.9:${C_INCLUDE_PATH}"
	endif

	if (! $?CPLUS_INCLUDE_PATH) then
    		setenv CPLUS_INCLUDE_PATH "$PYTHON_HOME/include/python3.9"
	else
    		setenv CPLUS_INCLUDE_PATH "$PYTHON_HOME/include/python3.9:${CPLUS_INCLUDE_PATH}"
	endif

	echo "Sniper Setup Completed."
	echo "----------------------------------------"
	echo "Sniper Environment Variables Set:"
	env | grep -E 'SNIPER_ROOT|PIN_HOME|PATH|LD_LIBRARY_PATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH'
	echo "----------------------------------------"
	breaksw

    default:
        echo "Invalid selection. Please choose one of the following:"
        show_options
        exit 1
endsw


