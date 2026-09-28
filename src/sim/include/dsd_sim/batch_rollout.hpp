#pragma once

#include <cstdint>
#include <vector>

#include "dsd_sim/rng.hpp"
#include "dsd_sim/rollout.hpp"
#include "dsd_sim/types.hpp"

namespace dsd {

constexpr std::uint64_t kDefaultSeedBase = 0x243f6a8885a308d3ULL;

// Per-scenario seed derived only from the scenario id (and the base seed), never
// from scheduling order, so batch results are identical for any thread count.
inline std::uint64_t scenario_seed(std::uint64_t seed_base, std::uint64_t scenario_id) {
    return splitmix64(seed_base ^ scenario_id * 0x9e3779b97f4a7c15ULL);
}

// Runs `rollout` for every scenario on a std::thread pool. Results are returned
// in scenario order and are bit-identical regardless of n_threads.
std::vector<Trajectory> batch_rollout(const std::vector<SimConfig> &configs,
                                      const std::vector<Path> &paths,
                                      const std::vector<State> &initial_states,
                                      const std::vector<int> &horizons, int n_threads,
                                      std::uint64_t seed_base = kDefaultSeedBase);

} // namespace dsd
