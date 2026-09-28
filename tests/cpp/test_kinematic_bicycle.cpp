#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "dsd_sim/kinematic_bicycle.hpp"
#include "dsd_sim/rollout.hpp"

using namespace dsd;

namespace {

Path straight_path(int n = 200) {
    Path path;
    for (int i = 0; i < n; ++i) {
        path.points.push_back(Vec2{static_cast<double>(i), 0.0});
    }
    return path;
}

} // namespace

TEST_CASE("Straight-line constant-velocity rollout matches closed form", "[rollout]") {
    SimConfig config;
    config.dt = 0.1;
    // All noise/bias/latency defaults are zero.

    const double v0 = 10.0;
    State initial;
    initial.speed = v0;

    const int horizon = 100;
    const Trajectory traj = rollout(straight_path(), initial, horizon, config, 7);

    REQUIRE(traj.horizon == horizon);
    REQUIRE(traj.data.size() == static_cast<std::size_t>(horizon + 1) * 6);
    for (int i = 0; i <= horizon; ++i) {
        const double *row = traj.data.data() + static_cast<std::size_t>(i) * 6;
        const double t = static_cast<double>(i) * config.dt;
        REQUIRE(row[0] == Catch::Approx(t).margin(1e-12));
        REQUIRE(row[1] == Catch::Approx(initial.x + v0 * t).margin(1e-9));
        REQUIRE(row[2] == Catch::Approx(initial.y).margin(1e-9));
        REQUIRE(row[3] == Catch::Approx(initial.heading).margin(1e-9));
        REQUIRE(row[4] == Catch::Approx(v0).margin(1e-12));
        REQUIRE(row[5] == Catch::Approx(0.0).margin(1e-12));
    }
}

TEST_CASE("Steering command saturates at max_steer", "[kinematic_bicycle]") {
    KinematicBicycleModel model(2.7, 0.6, 3.0, 5.0, 0.1);
    State state;
    state.speed = 5.0;

    Control over_right;
    over_right.steer = 1.5;
    State next = model.step(state, over_right);
    REQUIRE(next.steer_angle == Catch::Approx(0.6).margin(1e-12));

    Control over_left;
    over_left.steer = -1.5;
    next = model.step(state, over_left);
    REQUIRE(next.steer_angle == Catch::Approx(-0.6).margin(1e-12));
}

TEST_CASE("Acceleration clamps to max_accel and max_decel", "[kinematic_bicycle]") {
    KinematicBicycleModel model(2.7, 0.6, 2.0, 4.0, 0.1);
    State state;
    state.speed = 10.0;

    Control hard_accel;
    hard_accel.accel = 10.0;
    State next = model.step(state, hard_accel);
    REQUIRE(next.speed == Catch::Approx(10.2).margin(1e-12));

    Control hard_brake;
    hard_brake.accel = -10.0;
    next = model.step(state, hard_brake);
    REQUIRE(next.speed == Catch::Approx(9.6).margin(1e-12));
}
