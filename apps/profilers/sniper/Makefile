# SniperSim Repository Information
SNIPERSIM_REPO=https://github.com/snipersim/snipersim.git
SNIPERSIM_COMMIT=090ac23cf7eed63a684093c1c252bf0efe5f320a
SNIPERSIM_DIR=$(CURDIR)/snipersim
PIN_URL=https://software.intel.com/sites/landingpage/pintool/downloads/pin-3.30-98830-g1d7b601b3-gcc-linux.tar.gz

# Default patch directory - can be overridden with PATCH_DIR=...
PATCH_DIR ?= $(CURDIR)/../../patches
SNIPERSIM_PATCH=$(PATCH_DIR)/tracing_sniper.patch

.PHONY: all install_snipersim checkout_version build_snipersim clean

all: install_snipersim checkout_version build_snipersim

install_snipersim:
	@if [ -d "$(SNIPERSIM_DIR)" ] && [ -d "$(SNIPERSIM_DIR)/.git" ]; then \
		echo "SniperSim is already cloned in $(SNIPERSIM_DIR). Skipping clone."; \
	else \
		echo "Cloning SniperSim repository..."; \
		git clone $(SNIPERSIM_REPO) $(SNIPERSIM_DIR); \
		echo "SniperSim cloned successfully into $(SNIPERSIM_DIR)"; \
	fi
	@if [ ! -d "$(SNIPERSIM_DIR)/pin_kit" ]; then \
		echo "Downloading and installing Pin..."; \
		wget -O /tmp/pin.tar.gz $(PIN_URL); \
		tar -xzf /tmp/pin.tar.gz -C /tmp; \
		mv /tmp/pin-* $(SNIPERSIM_DIR)/pin_kit; \
		echo "Pin installed successfully at $(SNIPERSIM_DIR)/pin_kit"; \
	else \
		echo "Pin already installed in $(SNIPERSIM_DIR)/pin_kit. Skipping."; \
	fi


checkout_version: install_snipersim
	@echo "Checking out SniperSim version 8.1 at commit $(SNIPERSIM_COMMIT)..."
	cd $(SNIPERSIM_DIR) && git fetch --all
	cd $(SNIPERSIM_DIR) && git checkout $(SNIPERSIM_COMMIT)
	@echo "Checked out SniperSim commit $(SNIPERSIM_COMMIT) successfully."

build_snipersim: checkout_version
	@echo "Building SniperSim..."
	@if [ -f "$(SNIPERSIM_PATCH)" ]; then \
                echo "Applying patch to SniperSim..."; \
		patch -p1 -d $(SNIPERSIM_DIR) < $(SNIPERSIM_PATCH) || echo "Warning: Patch failed!"; \
        else \
                echo "No patch file found for SniperSim."; \
        fi

	cd $(SNIPERSIM_DIR) && make USE_PIN=1 -j$(nproc)
	cp $(CURDIR)/snipermem.py $(SNIPERSIM_DIR)/tools/
	@echo "SniperSim built successfully."

clean:
	@echo "Removing SniperSim installation..."
	rm -rf $(SNIPERSIM_DIR)
	@echo "Cleanup completed."

