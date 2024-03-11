# to build a debug build, run "cmake ." and then "make"
# to build a release build, run "cmake -DCMAKE_BUILD_TYPE=Release ." and then "make"
# all build types: Debug, Release, RelWithDebInfo, MinSizeRel

cmake_minimum_required(VERSION 3.10)
project(SpikingCpp)

set(CMAKE_CXX_STANDARD 17)

# flags for all builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -pedantic")

# flags for debug builds (in addition to cmake defaults)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address -fsanitize=undefined -fsanitize=leak")

# set(CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/..)

# flags for release builds (in addition to cmake defaults)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/target)

# set default build type to debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# output binaries to a different folder depending on the build type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# include include/ directory
include_directories(include)

file(GLOB SOURCES "src/*.cpp")

add_executable(SpikingCpp ${SOURCES})

target_link_libraries(SpikingCpp m)