#pragma once

#include <cstdint>
#include <vector>

#include "dsd_sim/types.hpp"

namespace dsd {

// A recorded trajectory: (horizon + 1) rows of (t, x, y, heading, speed,
// yaw_rate) stored row-major in a contiguous float64 buffer. x/y/heading are
// the (noisy) observations; speed and yaw_rate come from the true dynamics.
struct Trajectory {
    int horizon = 0;
    std::vector<double> data; // size == (horizon + 1) * 6
};

// Simulates `horizon` control steps (t = 0 .. horizon * dt). All RNG streams
// are derived from `seed` via splitmix64, so identical inputs give bit-identical
// outputs.
Trajectory rollout(const Path &path, const State &initial_state, int horizon,
                   const SimConfig &config, std::uint64_t seed);

} // namespace dsd
