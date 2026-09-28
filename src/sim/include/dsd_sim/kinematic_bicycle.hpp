#pragma once

#include "dsd_sim/types.hpp"

namespace dsd {

// Kinematic bicycle model with rear-axle reference point:
//   x'      = v * cos(heading)
//   y'      = v * sin(heading)
//   heading'= v / wheelbase * tan(steer)
//   v'      = accel
// integrated with explicit Euler at dt. The steer command saturates at
// +-max_steer and the acceleration saturates at max_accel / -max_decel.
class KinematicBicycleModel {
  public:
    KinematicBicycleModel(double wheelbase, double max_steer, double max_accel, double max_decel,
                          double dt);

    State step(const State &state, const Control &control) const;

    double wheelbase() const {
        return wheelbase_;
    }
    double max_steer() const {
        return max_steer_;
    }
    double max_accel() const {
        return max_accel_;
    }
    double max_decel() const {
        return max_decel_;
    }
    double dt() const {
        return dt_;
    }

  private:
    double wheelbase_;
    double max_steer_;
    double max_accel_;
    double max_decel_;
    double dt_;
};

} // namespace dsd
