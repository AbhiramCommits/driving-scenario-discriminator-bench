#pragma once

#include <cstdint>
#include <random>

#include "dsd_sim/types.hpp"

namespace dsd {

// Gaussian position/heading observation noise, deterministic given a seed.
// Uses std::mt19937_64, reseeded per rollout so every scenario reproduces
// exactly, independent of execution order or thread count.
class NoiseModel {
  public:
    NoiseModel(double pos_std = 0.0, double heading_std = 0.0);

    void reset(std::uint64_t seed);

    // Returns a noisy copy of the true state (independent Gaussian draws per
    // axis of position and per heading).
    State observe(const State &true_state);

  private:
    double pos_std_;
    double heading_std_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};
};

} // namespace dsd
