#include "dsd_sim/kinematic_bicycle.hpp"

#include <algorithm>
#include <cmath>

namespace dsd {

KinematicBicycleModel::KinematicBicycleModel(double wheelbase, double max_steer, double max_accel,
                                             double max_decel, double dt)
    : wheelbase_(wheelbase), max_steer_(max_steer), max_accel_(max_accel), max_decel_(max_decel),
      dt_(dt) {}

State KinematicBicycleModel::step(const State &state, const Control &control) const {
    State next = state;
    const double steer = std::clamp(control.steer, -max_steer_, max_steer_);
    const double accel = control.accel >= 0.0 ? std::min(control.accel, max_accel_)
                                              : std::max(control.accel, -max_decel_);

    next.x += state.speed * std::cos(state.heading) * dt_;
    next.y += state.speed * std::sin(state.heading) * dt_;
    next.heading += state.speed / wheelbase_ * std::tan(steer) * dt_;
    next.speed += accel * dt_;
    if (next.speed < 0.0) {
        next.speed = 0.0;
    }
    next.steer_angle = steer;
    return next;
}

} // namespace dsd
