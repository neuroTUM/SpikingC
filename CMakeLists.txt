cmake_minimum_required(VERSION 3.10)
project(SpikingC C) # Specify the project as a C project

set(CMAKE_C_STANDARD 99)

# flags for all builds
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -pedantic")

# flags for debug builds (in addition to cmake defaults)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0 -fsanitize=address -fsanitize=undefined -fsanitize=leak")

# flags for release builds (in addition to cmake defaults)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)

# flags for release builds (in addition to cmake defaults)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/target)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# output binaries to a different folder depending on the build type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# Include the "include" directory for header files
include_directories(include)

# Add source files
file(GLOB SOURCES "src/*.c")

# Create the executable
add_executable(SpikingC ${SOURCES})

# Link against the math library (and any other libraries needed)
target_link_libraries(SpikingC m)
