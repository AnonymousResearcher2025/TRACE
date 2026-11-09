cmake_minimum_required(VERSION 3.16)
project(trace_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Native tuning: x86 uses -march=native (hnswlib SIMD kernels);
# aarch64 (Graviton) uses -mcpu=native.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
check_cxx_compiler_flag("-mcpu=native" HAS_MCPU_NATIVE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  if(HAS_MCPU_NATIVE)
    set(TRACE_ARCH_FLAG "-mcpu=native")
  endif()
elseif(HAS_MARCH_NATIVE)
  set(TRACE_ARCH_FLAG "-march=native")
endif()

find_library(AIO_LIB aio REQUIRED)

pybind11_add_module(_trace
  cpp/src/bindings.cpp
  cpp/src/hnsw_executor.cpp
  cpp/src/disk_executor.cpp
  cpp/src/trace_engine.cpp
  cpp/src/baselines.cpp
  cpp/src/vamana.cpp)
target_include_directories(_trace PRIVATE cpp/include)
target_include_directories(_trace SYSTEM PRIVATE third_party)
# Bound-critical arithmetic relies on IEEE-754 binary32 semantics:
# never enable -ffast-math, and do not contract a*b+c into FMA.
target_compile_options(_trace PRIVATE -O3 ${TRACE_ARCH_FLAG}
  -fno-fast-math -ffp-contract=off -fno-math-errno -Wall -Wextra
  -Wno-unused-parameter -Wno-sign-compare)
target_link_libraries(_trace PRIVATE ${AIO_LIB} pthread)
set_target_properties(_trace PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/trace_harness)
