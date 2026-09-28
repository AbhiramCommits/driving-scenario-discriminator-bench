#pragma once

#include <cstdint>
#include <deque>
#include <random>

#include "dsd_sim/types.hpp"

namespace dsd {

// Pure-pursuit controller over a polyline reference path with a configurable
// lookahead gain (lookahead = max(gain * speed, min_lookahead)) and an injectable
// error model (proportional bias, Gaussian steering noise, actuation latency).
class PurePursuitController {
  public:
    PurePursuitController(double lookahead_gain = 3.0, double min_lookahead = 0.5,
                          ControllerErrorModel error_model = {});

    // Resets per-rollout state: reseeds the steering-noise RNG and clears the
    // latency ring buffer (filled with zeros). Called once per rollout so results
    // depend only on the seed, not on call history.
    void reset(std::uint64_t seed);

    // Returns the commanded steer angle [rad] for the observed state and path.
    double compute_steer(const State &state, const Path &path, double wheelbase);

  private:
    double lookahead_gain_;
    double min_lookahead_;
    ControllerErrorModel error_model_;
    std::mt19937_64 rng_;
    std::normal_distribution<double> gauss_{0.0, 1.0};
    std::deque<double> latency_buffer_;
};

} // namespace dsd
