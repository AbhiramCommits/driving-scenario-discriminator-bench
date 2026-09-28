#pragma once

#include <cstdint>
#include <vector>

namespace dsd {

struct Vec2 {
    double x = 0.0;
    double y = 0.0;
};

// Vehicle state with a rear-axle reference point.
struct State {
    double x = 0.0;
    double y = 0.0;
    double heading = 0.0;     // yaw angle [rad]
    double speed = 0.0;       // longitudinal speed [m/s]
    double steer_angle = 0.0; // front wheel steer angle [rad]

    State() = default;
    State(double x_, double y_, double heading_, double speed_, double steer_angle_)
        : x(x_), y(y_), heading(heading_), speed(speed_), steer_angle(steer_angle_) {}
};

struct Control {
    double steer = 0.0; // commanded steer angle [rad]
    double accel = 0.0; // commanded longitudinal acceleration [m/s^2]

    Control() = default;
    Control(double steer_, double accel_) : steer(steer_), accel(accel_) {}
};

// Polyline reference path, one point per row.
struct Path {
    std::vector<Vec2> points;
};

// Injectable error model applied on top of the pure-pursuit command:
// 1. proportional steering bias: cmd *= (1 + steer_bias)
// 2. Gaussian steering noise: cmd += N(0, steer_noise_std)
// 3. N-step actuation latency: commands are delayed through a ring buffer
struct ControllerErrorModel {
    double steer_bias = 0.0;
    double steer_noise_std = 0.0;
    int latency_steps = 0;

    ControllerErrorModel() = default;
    ControllerErrorModel(double steer_bias_, double steer_noise_std_, int latency_steps_)
        : steer_bias(steer_bias_), steer_noise_std(steer_noise_std_),
          latency_steps(latency_steps_) {}
};

struct SimConfig {
    double dt = 0.1;                // integration timestep [s]
    double wheelbase = 2.7;         // [m]
    double max_steer = 0.6;         // steer angle saturation [rad]
    double max_accel = 3.0;         // [m/s^2]
    double max_decel = 5.0;         // [m/s^2]
    double accel = 0.0;             // constant longitudinal acceleration command [m/s^2]
    double lookahead_gain = 3.0;    // pure-pursuit lookahead = gain * speed
    double min_lookahead = 0.5;     // pure-pursuit lookahead floor [m]
    double pos_noise_std = 0.0;     // per-axis position observation noise stddev [m]
    double heading_noise_std = 0.0; // heading observation noise stddev [rad]
    int latency_steps = 0;          // actuation latency [steps]
    double steer_bias = 0.0;        // proportional steering bias
    double steer_noise_std = 0.0;   // Gaussian steering noise stddev [rad]
};

} // namespace dsd
