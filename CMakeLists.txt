cmake_minimum_required(VERSION 3.10)
project(SpikingC C) # Specify the project as a C project

# Set the C standard you're using - for example, C99
set(CMAKE_C_STANDARD 99)

# flags for all builds
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -pedantic")

# flags for debug builds (in addition to cmake defaults)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g")

# flags for release builds (in addition to cmake defaults)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Include the "include" directory for header files
include_directories(include)

# Add source files
file(GLOB SOURCES "src/*.c")

# Create the executable
add_executable(SpikingC ${SOURCES})

# Link against the math library (and any other libraries needed)
target_link_libraries(SpikingC m)
