#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dsd_sim/pure_pursuit.hpp"
#include "dsd_sim/types.hpp"

using namespace dsd;

namespace {

Path offset_path(double y_offset) {
    Path path;
    for (int i = 0; i <= 100; ++i) {
        path.points.push_back(Vec2{static_cast<double>(i), y_offset});
    }
    return path;
}

} // namespace

TEST_CASE("Pure pursuit steers toward an offset path", "[pure_pursuit]") {
    PurePursuitController controller;
    controller.reset(0);
    State state;
    state.speed = 5.0; // at origin, heading 0

    const double left = controller.compute_steer(state, offset_path(2.0), 2.7);
    REQUIRE(left > 0.0);

    const double right = controller.compute_steer(state, offset_path(-2.0), 2.7);
    REQUIRE(right < 0.0);

    const double straight = controller.compute_steer(state, offset_path(0.0), 2.7);
    REQUIRE(straight == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("Actuation latency delays the command", "[pure_pursuit]") {
    ControllerErrorModel error_model;
    error_model.latency_steps = 2;
    PurePursuitController controller(3.0, 0.5, error_model);
    controller.reset(0);
    State state;
    state.speed = 5.0;
    const Path path = offset_path(2.0);

    const double step0 = controller.compute_steer(state, path, 2.7);
    const double step1 = controller.compute_steer(state, path, 2.7);
    const double step2 = controller.compute_steer(state, path, 2.7);
    // The first two outputs are the zero-filled latency buffer; the third
    // releases the command computed on the first call.
    REQUIRE(step0 == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(step1 == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(step2 > 0.0);
}
