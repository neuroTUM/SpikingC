cmake_minimum_required(VERSION 3.10)
project(SpikingC C) # Specify the project as a C project

# Set the C standard you're using - for example, C99
set(CMAKE_C_STANDARD 99)

# flags for all builds
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall -Wextra -Werror -pedantic ")

# flags for debug builds (in addition to cmake defaults)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

# flags for release builds (in addition to cmake defaults)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/target)

# Set default build type to debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS " Build type: ${CMAKE_BUILD_TYPE} ")

# Output binaries to a different folder depending on the build type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# Include the "include" directory
include_directories(include)

# List your C source files here
set(SOURCES
  src/LIF.c
  src/Linear.c
  src/Model.c
  src/Utility.c
  src/main.c
)

message(STATUS "Sources found: ${SOURCES}")

# Create the executable
add_executable(SpikingC ${SOURCES})

# Link libraries, if you have any. Example: linking math library
target_link_libraries(SpikingC m)
