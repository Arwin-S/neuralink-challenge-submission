# Makefile for setting up and building the project

# Define variables
BUILD_DIR = build
CMAKE = cmake
MAKE = make

.PHONY: all setup build clean

# Default target
all: setup build

# Setup target: create build directory and run cmake
setup:
	@echo "Setting up the build directory and running cmake..."
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) ..

# Build target: build the project using make
build:
	@echo "Building the project..."
	@$(MAKE) -C $(BUILD_DIR)

# Clean target: remove the build directory
clean:
	@echo "Cleaning up..."
	@rm -rf $(BUILD_DIR)

# Help target: display available targets
help:
	@echo "Usage:"
	@echo "  make         - Setup and build the project"
	@echo "  make setup   - Setup the build directory and run cmake"
	@echo "  make build   - Build the project"
	@echo "  make clean   - Remove the build directory"
