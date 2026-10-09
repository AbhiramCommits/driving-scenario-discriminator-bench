add_library(
  dsd_sim
  STATIC
  batch_rollout.cpp
  kinematic_bicycle.cpp
  noise_model.cpp
  pure_pursuit.cpp
  rollout.cpp
)

target_include_directories(dsd_sim PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(NOT MSVC)
  target_compile_options(dsd_sim PRIVATE -Wall -Wextra)
endif()

# Linked into the _sim Python extension (a shared object), so it must be PIC.
set_target_properties(dsd_sim PROPERTIES POSITION_INDEPENDENT_CODE ON)
