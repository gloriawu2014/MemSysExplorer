# Setting Application Build Settings
APPS_HOME ?= $(shell readlink -f "$(CURDIR)")
PATCH_DIR := $(APPS_HOME)/patches
ENV_DIR := $(APPS_HOME)/tools
JSON_FILE := $(APPS_HOME)/built_profilers.json

# Define third-party library directories
THIRD_PARTY_DIR := $(APPS_HOME)/third_party
PROTOBUF_PREFIX := $(THIRD_PARTY_DIR)/protobuf
PROTOBUF_C_PREFIX := $(THIRD_PARTY_DIR)/protobuf-c

# Define profiler directories
COMMON_DIR := $(APPS_HOME)/profilers/common
DYNAMORIO_DIR := $(APPS_HOME)/profilers/dynamorio
SNIPER_DIR := $(APPS_HOME)/profilers/sniper
NCU_DIR := $(APPS_HOME)/profilers/ncu
NVBIT_DIR := $(APPS_HOME)/profilers/nvbit
PERF_DIR := $(APPS_HOME)/profilers/perf
DOCS_DIR := $(APPS_HOME)/docs

.PHONY: all common dynamorio sniper nvbit perf ncu docs clean clean_common clean_dynamorio clean_sniper clean_nvbit clean_perf clean_docs clean_pycache clean_builds protobuf protobuf-c clean_protobuf

# Default target: build everything
all: common dynamorio sniper nvbit perf ncu docs
	@echo "All profilers and documentation built successfully."

# Initialize JSON file if it doesn't exist
init_json:
	@if [ ! -f "$(JSON_FILE)" ]; then \
		echo '{ "dynamorio": false, "ncu": false, "nvbit": false, "sniper": false, "perf": false }' > "$(JSON_FILE)"; \
	fi

# Update JSON file when a profiler is built
update_json:
	@jq --arg key "$(PROFILER)" '.[$$key] = true' "$(JSON_FILE)" > "$(JSON_FILE).tmp" && mv "$(JSON_FILE).tmp" "$(JSON_FILE)"

# Build protobuf from source (if not on system)
protobuf:
	@echo "        Building Protobuf from source        "
	@if ! pkg-config --exists protobuf 2>/dev/null; then \
		echo "Protobuf not found on system, building from source..."; \
		mkdir -p $(THIRD_PARTY_DIR)/build/protobuf; \
		if [ ! -d "$(THIRD_PARTY_DIR)/protobuf-src" ]; then \
			cd $(THIRD_PARTY_DIR) && git clone --depth 1 --branch v21.12 https://github.com/protocolbuffers/protobuf.git protobuf-src; \
		fi; \
		cd $(THIRD_PARTY_DIR)/build/protobuf && \
		cmake $(THIRD_PARTY_DIR)/protobuf-src/cmake \
			-DCMAKE_INSTALL_PREFIX=$(PROTOBUF_PREFIX) \
			-Dprotobuf_BUILD_TESTS=OFF \
			-Dprotobuf_BUILD_EXAMPLES=OFF \
			-Dprotobuf_WITH_ZLIB=OFF \
			-DCMAKE_BUILD_TYPE=Release && \
		make -j$$(nproc) && \
		make install; \
		echo "Protobuf built and installed to $(PROTOBUF_PREFIX)"; \
	else \
		echo "Protobuf found on system, skipping build"; \
	fi

# Build protobuf-c from source (if not on system)
protobuf-c: protobuf
	@echo "      Building Protobuf-C from source      "
	@if ! pkg-config --exists libprotobuf-c 2>/dev/null; then \
		echo "Protobuf-C not found on system, building from source..."; \
		mkdir -p $(THIRD_PARTY_DIR)/build/protobuf-c; \
		if [ ! -d "$(THIRD_PARTY_DIR)/protobuf-c-src" ]; then \
			cd $(THIRD_PARTY_DIR) && git clone --depth 1 --branch v1.4.1 https://github.com/protobuf-c/protobuf-c.git protobuf-c-src; \
		fi; \
		cd $(THIRD_PARTY_DIR)/build/protobuf-c && \
		PKG_CONFIG_PATH=$(PROTOBUF_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH \
		cmake $(THIRD_PARTY_DIR)/protobuf-c-src/build-cmake \
			-DCMAKE_INSTALL_PREFIX=$(PROTOBUF_C_PREFIX) \
			-DCMAKE_PREFIX_PATH=$(PROTOBUF_PREFIX) \
			-DProtobuf_DIR=$(PROTOBUF_PREFIX)/lib/cmake/protobuf \
			-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
			-DCMAKE_C_FLAGS="-fPIC" \
			-DCMAKE_CXX_FLAGS="-fPIC" \
			-DBUILD_PROTO3=ON \
			-DBUILD_TESTS=OFF \
			-DCMAKE_BUILD_TYPE=Release && \
		make -j$$(nproc) && \
		make install; \
		echo "Protobuf-C built and installed to $(PROTOBUF_C_PREFIX)"; \
	else \
		echo "Protobuf-C found on system, skipping build"; \
	fi

# Build common library
common: protobuf protobuf-c
	@echo "        Building Common Library          "
	@mkdir -p $(COMMON_DIR)/build
	@cd $(COMMON_DIR)/build && \
		PATH=$(PROTOBUF_C_PREFIX)/bin:$(PROTOBUF_PREFIX)/bin:$$PATH \
		PKG_CONFIG_PATH=$(PROTOBUF_C_PREFIX)/lib/pkgconfig:$(PROTOBUF_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH \
		CMAKE_PREFIX_PATH=$(PROTOBUF_C_PREFIX):$(PROTOBUF_PREFIX):$$CMAKE_PREFIX_PATH \
		cmake -DProtobuf_DIR=$(PROTOBUF_PREFIX)/lib/cmake/protobuf .. && \
		PATH=$(PROTOBUF_C_PREFIX)/bin:$(PROTOBUF_PREFIX)/bin:$$PATH make
	@echo "Common library built successfully."

dynamorio: init_json common
	@echo "            Building DynamoRIO             "
	cd $(DYNAMORIO_DIR) && make PATCH_DIR=$(PATCH_DIR)
	@echo "DynamoRIO build completed."
	@make update_json PROFILER=dynamorio

ncu: init_json
	@echo "               Checking CUDA               "
	cd $(NCU_DIR) && make check_all APPS_HOME=$(APPS_HOME)
	@echo "Nsight Profilers check completed."
	@make update_json PROFILER=ncu

sniper: init_json common
	@echo "              Building Sniper              "
	cd $(SNIPER_DIR) && make PATCH_DIR=$(PATCH_DIR)
	@echo "Sniper build completed."
	@make update_json PROFILER=sniper

nvbit: init_json common
	@echo "              Building NVBit               "
	cd $(NVBIT_DIR) && make PATCH_DIR=$(PATCH_DIR)
	@echo "NVBit build completed."
	@make update_json PROFILER=nvbit

perf: init_json
	@echo "              Building Perf                "
	@echo "Perf is completed."
	@make update_json PROFILER=perf

docs:
	@echo "        Building Project Documentation      "
	cd $(APPS_HOME) && sphinx-apidoc -o $(DOCS_DIR)/source/ ./ --force --no-toc -e && cd $(DOCS_DIR) && make html
	@echo "Documentation built at $(DOCS_DIR)/_build/html"

# Clean all profilers and docs
clean: clean_common clean_dynamorio clean_sniper clean_nvbit clean_perf clean_docs clean_pycache clean_builds clean_protobuf
	@rm -f $(JSON_FILE)
	@echo "         Cleaning Completed                "

# Clean protobuf builds
clean_protobuf:
	@echo "Cleaning Protobuf builds..."
	@rm -rf $(THIRD_PARTY_DIR)/build
	@echo "Protobuf build directories cleaned (source kept for faster rebuild)."

# Clean individual profilers
clean_common:
	@echo "Cleaning Common Library..."
	@rm -rf $(COMMON_DIR)/build
	@echo "Common library cleaned."

clean_dynamorio:
	@echo "Cleaning DynamoRIO..."
	cd $(DYNAMORIO_DIR) && make clean
	@echo "DynamoRIO cleaned."

clean_sniper:
	@echo "Cleaning Sniper..."
	cd $(SNIPER_DIR) && make clean
	@echo "Sniper cleaned."

clean_nvbit:
	@echo "Cleaning NVBit..."
	cd $(NVBIT_DIR) && make clean
	@echo "NVBit cleaned."

clean_perf:
	@echo "Cleaning Perf..."
	@echo "Perf working directory is cleared."

# Clean __pycache__ directories
clean_pycache:
	@echo "      Removing caching directories         "
	find $(APPS_HOME) -type d -name "__pycache__" -exec rm -rf {} +
	@echo "All __pycache__ directories removed."

clean_docs:
	@echo "Cleaning documentation build directory..."
	rm -rf $(DOCS_DIR)/build && rm -rf $(DOCS_DIR)/source/*.rst
	@echo "Documentation build directory removed."

# Clean build directories
clean_builds:
	@echo "Cleaning build directories..."
	@echo "Removing temp build directories..."
	find $(APPS_HOME)/temp -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing common library build directories..."
	rm -rf $(APPS_HOME)/profilers/common/build
	@echo "Removing generated protobuf files..."
	find $(APPS_HOME)/tools -name "*_pb2.py" -delete 2>/dev/null || true
	find $(APPS_HOME)/tools -name "*.pyc" -delete 2>/dev/null || true
	@echo "Build directories cleaned."
