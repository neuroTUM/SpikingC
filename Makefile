# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -Werror -pedantic -std=c++17
DEBUG_FLAGS = -g -O0 -fsanitize=address -fsanitize=undefined -fsanitize=leak
RELEASE_FLAGS = -O3
LINKER_FLAGS = -lm

# Paths
DEBUG_PATH = target/debug
RELEASE_PATH = target/release
SRC_PATH = src

# Source files
SRCS = $(wildcard $(SRC_PATH)/*.cpp)

# Default target path
TARGET_PATH = $(DEBUG_PATH)

# Targets
TARGET = main
# Dynamic object files path. Placeholder will be replaced in specific targets.
OBJS = $(SRCS:$(SRC_PATH)/%.cpp=$(TARGET_PATH)/%.o)

.PHONY: clean help debug release run prepare_debug prepare_release

# Default to debug build
debug: TARGET_PATH = $(DEBUG_PATH)
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Release broken for now lmao, consider using CMake
#release: TARGET_PATH = $(RELEASE_PATH)
#release: CXXFLAGS += $(RELEASE_FLAGS)
#release: $(TARGET)

run: debug
	./$(TARGET_PATH)/$(TARGET)

help:
	@echo "Available targets:"
	@echo "  debug    - Build with debug flags"
	@echo "  release  - Build with optimization flags"
	@echo "  run      - Build in debug mode and run"
	@echo "  clean    - Delete build directories"

# Generic rule for transforming .cpp files into .o files within TARGET_PATH
$(TARGET_PATH)/%.o: $(SRC_PATH)/%.cpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Linking
$(TARGET): $(OBJS)
	mkdir -p $(TARGET_PATH)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET_PATH)/$(TARGET) $(LINKER_FLAGS)

# Clean up
clean:
	rm -rf target