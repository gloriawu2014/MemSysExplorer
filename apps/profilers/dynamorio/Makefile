# Version and Git repository for DynamoRIO
DYNAMORIO_VERSION ?= 11.3.0
DYNAMORIO_REPO=https://github.com/DynamoRIO/dynamorio.git

# Directories
INSTALL_DIR ?= $(shell readlink -f "$(CURDIR)/dynamorio_install")
DYNAMORIO_HOME ?= $(INSTALL_DIR)/DynamoRIO-${DYNAMORIO_VERSION}-1
DYNAMORIO_BUILD=$(DYNAMORIO_HOME)/build

# Patch directory (comes from the top-level Makefile)
DYNAMORIO_PATCH=$(PATCH_DIR)/dynamorio.patch

.PHONY: all install_dynamorio apply_patch build_dynamorio clean

all: install_dynamorio apply_patch build_dynamorio build_client

install_dynamorio:
	@if [ -d "$(DYNAMORIO_HOME)" ] && [ -f "$(DYNAMORIO_HOME)/CMakeLists.txt" ]; then \
		echo "DynamoRIO is already installed at $(DYNAMORIO_HOME). Skipping installation."; \
	else \
		echo "Cloning DynamoRIO ${DYNAMORIO_VERSION}..."; \
		mkdir -p $(INSTALL_DIR); \
		cd $(INSTALL_DIR) && git clone --branch release_${DYNAMORIO_VERSION}-1 --recurse-submodules -j4 $(DYNAMORIO_REPO) $(DYNAMORIO_HOME); \
		echo "DynamoRIO cloned successfully into $(DYNAMORIO_HOME)"; \
	fi

apply_patch: install_dynamorio
	@if [ -f "$(DYNAMORIO_PATCH)" ]; then \
		echo "Applying patch to DynamoRIO..."; \
		patch -N -r -p1 -d $(DYNAMORIO_HOME) < $(DYNAMORIO_PATCH) || echo "Warning: Patch failed!"; \
	else \
		echo "No patch file found for DynamoRIO."; \
	fi

build_dynamorio: apply_patch
	@echo "Building DynamoRIO..."
	mkdir -p $(DYNAMORIO_BUILD)
	cd $(DYNAMORIO_BUILD) && cmake -DBUILD_DOCS=OFF ..
	cd $(DYNAMORIO_BUILD) && make -j$(nproc)
	@echo "DynamoRIO built successfully. Binaries are in $(DYNAMORIO_BUILD)"

build_client:
	@echo "Building the DynamoRIO client..."
	@if [ ! -f "$(CURDIR)/build/build.sh" ]; then \
		echo "Error: build.sh not found in $(CURDIR)/build. Ensure the script is present."; \
		exit 1; \
	fi

	# Setup protobuf paths for building the client
	$(eval APPS_HOME := $(shell cd ../../ && pwd))
	$(eval THIRD_PARTY_DIR := $(APPS_HOME)/third_party)
	$(eval PROTOBUF_PREFIX := $(THIRD_PARTY_DIR)/protobuf)
	$(eval PROTOBUF_C_PREFIX := $(THIRD_PARTY_DIR)/protobuf-c)

	cd $(CURDIR)/build && \
		export DYNAMORIO_HOME=$(DYNAMORIO_BUILD)/cmake && \
		export PATH=$(PROTOBUF_C_PREFIX)/bin:$(PROTOBUF_PREFIX)/bin:$$PATH && \
		export LD_LIBRARY_PATH=$(PROTOBUF_C_PREFIX)/lib:$(PROTOBUF_PREFIX)/lib:$$LD_LIBRARY_PATH && \
		export LIBRARY_PATH=$(PROTOBUF_C_PREFIX)/lib:$(PROTOBUF_PREFIX)/lib:$$LIBRARY_PATH && \
		export PKG_CONFIG_PATH=$(PROTOBUF_C_PREFIX)/lib/pkgconfig:$(PROTOBUF_PREFIX)/lib/pkgconfig:$$PKG_CONFIG_PATH && \
		export CMAKE_PREFIX_PATH=$(PROTOBUF_C_PREFIX):$(PROTOBUF_PREFIX):$(DYNAMORIO_BUILD)/cmake:$$CMAKE_PREFIX_PATH && \
		sh build.sh

	@echo "Client built successfully."

clean:
	@echo "Removing DynamoRIO installation..."
	rm -rf $(INSTALL_DIR)
	@echo "Cleaning DynamoRIO build directory..."
	@find $(CURDIR)/build -mindepth 1 -maxdepth 1 ! -name 'build.sh' ! -name 'client' -exec rm -rf {} +
	@echo "Cleanup completed."

