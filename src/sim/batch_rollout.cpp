#include "dsd_sim/batch_rollout.hpp"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <thread>

namespace dsd {

std::vector<Trajectory> batch_rollout(const std::vector<SimConfig> &configs,
                                      const std::vector<Path> &paths,
                                      const std::vector<State> &initial_states,
                                      const std::vector<int> &horizons, int n_threads,
                                      std::uint64_t seed_base) {
    const std::size_t n = configs.size();
    if (paths.size() != n || initial_states.size() != n || horizons.size() != n) {
        throw std::invalid_argument("batch_rollout: input vector size mismatch");
    }

    std::vector<Trajectory> results(n);
    const int workers = std::max(1, std::min(n_threads, static_cast<int>(n)));
    std::atomic<std::size_t> next{0};

    auto worker = [&]() {
        for (;;) {
            const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) {
                break;
            }
            // Seed depends only on the scenario id: deterministic for any thread count.
            const std::uint64_t seed = scenario_seed(seed_base, static_cast<std::uint64_t>(i));
            results[i] = rollout(paths[i], initial_states[i], horizons[i], configs[i], seed);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(workers - 1));
    for (int t = 0; t < workers - 1; ++t) {
        pool.emplace_back(worker);
    }
    worker();
    for (auto &th : pool) {
        th.join();
    }
    return results;
}

} // namespace dsd
