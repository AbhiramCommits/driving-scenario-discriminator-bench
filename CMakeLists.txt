cmake_minimum_required(VERSION 3.22)
project(dsdbench VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_subdirectory(src/sim)

# The pybind11 extension is only built when driven by scikit-build-core.
if(SKBUILD)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_sim src/bindings/sim_module.cpp)
  target_link_libraries(_sim PRIVATE dsd_sim)
  install(TARGETS _sim LIBRARY DESTINATION dsdbench)
endif()

option(DSD_BUILD_TESTS "Build C++ unit tests" OFF)
if(DSD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
