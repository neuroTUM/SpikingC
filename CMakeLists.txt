# to build a debug build, run "cmake ." and then "make"
# to build a release build, run "cmake -DCMAKE_BUILD_TYPE=Release ." and then "make"
# all build types: Debug, Release, RelWithDebInfo, MinSizeRel

cmake_minimum_required(VERSION 3.10)
project(SpikingCpp)

set(CMAKE_CXX_STANDARD 17)

# flags for all builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=leak -g -O0 -Wall -Wextra -Werror -pedantic ")

# flags for debug builds (in addition to cmake defaults)
set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g -O0 -fsanitize=undefined -fsanitize=leak ")

# flags for release builds (in addition to cmake defaults)
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/target)

# set default build type to debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS " Build type: ${CMAKE_BUILD_TYPE} ")

# output binaries to a different folder depending on the build type
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# include include/ directory
include_directories(include)

set(SOURCES
  src/main.cpp
)

#file(GLOB SOURCES "src/*.cpp ")
message(STATUS "Sources found: ${SOURCES}")


add_executable(SpikingCpp ${SOURCES})

target_link_libraries(SpikingCpp m)