include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(
  dsd_sim_tests
  test_determinism.cpp
  test_kinematic_bicycle.cpp
  test_pure_pursuit.cpp
)
target_link_libraries(dsd_sim_tests PRIVATE dsd_sim Catch2::Catch2WithMain)

include(CTest)
add_test(NAME dsd_sim_tests COMMAND dsd_sim_tests)
