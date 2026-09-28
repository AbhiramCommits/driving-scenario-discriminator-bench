#include <catch2/catch_test_macros.hpp>

#include <cmath>

#include "dsd_sim/batch_rollout.hpp"
#include "dsd_sim/rollout.hpp"

using namespace dsd;

namespace {

Path curved_path(double phase = 0.0) {
    Path path;
    for (int i = 0; i <= 300; ++i) {
        const double x = static_cast<double>(i) * 0.2;
        path.points.push_back(Vec2{x, 2.0 * std::sin(0.05 * x + phase)});
    }
    return path;
}

SimConfig noisy_config() {
    SimConfig config;
    config.dt = 0.05;
    config.pos_noise_std = 0.1;
    config.heading_noise_std = 0.02;
    config.steer_noise_std = 0.01;
    config.steer_bias = 0.05;
    config.latency_steps = 3;
    return config;
}

} // namespace

TEST_CASE("Zero-noise rollout is bit-identical across repeated runs", "[determinism]") {
    const Path path = curved_path();
    SimConfig config;
    State initial;
    initial.x = 0.0;
    initial.y = 0.5;
    initial.speed = 8.0;
    const int horizon = 100;

    const Trajectory a = rollout(path, initial, horizon, config, 99);
    const Trajectory b = rollout(path, initial, horizon, config, 99);
    REQUIRE(a.data == b.data);
}

TEST_CASE("Seeded noisy rollout is bit-identical across repeated runs", "[determinism]") {
    const Path path = curved_path();
    const SimConfig config = noisy_config();
    State initial;
    initial.x = 0.0;
    initial.y = 0.5;
    initial.speed = 8.0;
    const int horizon = 100;

    const Trajectory a = rollout(path, initial, horizon, config, 123);
    const Trajectory b = rollout(path, initial, horizon, config, 123);
    REQUIRE(a.data == b.data);
}

TEST_CASE("Batch rollout is identical for 1 vs 8 threads", "[determinism]") {
    const int n = 16;
    std::vector<SimConfig> configs(n);
    std::vector<Path> paths(n);
    std::vector<State> states(n);
    std::vector<int> horizons(n);
    for (int i = 0; i < n; ++i) {
        configs[i] = (i % 2 == 0) ? SimConfig{} : noisy_config();
        configs[i].dt = 0.05 + 0.005 * i;
        paths[i] = curved_path(0.1 * i);
        states[i] = State{0.0, 0.5 * i, 0.0, 5.0 + i, 0.0};
        horizons[i] = 40 + i;
    }

    const auto single = batch_rollout(configs, paths, states, horizons, 1);
    const auto multi = batch_rollout(configs, paths, states, horizons, 8);

    REQUIRE(single.size() == static_cast<std::size_t>(n));
    REQUIRE(multi.size() == static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        REQUIRE(single[i].data == multi[i].data);
    }
}

TEST_CASE("Batch rollout matches per-scenario rollouts with derived seeds", "[determinism]") {
    constexpr std::uint64_t seed_base = kDefaultSeedBase;
    const int n = 6;
    std::vector<SimConfig> configs(n);
    std::vector<Path> paths(n);
    std::vector<State> states(n);
    std::vector<int> horizons(n);
    for (int i = 0; i < n; ++i) {
        configs[i] = noisy_config();
        paths[i] = curved_path(0.2 * i);
        states[i] = State{0.0, 0.25 * i, 0.1 * i, 6.0, 0.0};
        horizons[i] = 25 + i;
    }

    const auto batch = batch_rollout(configs, paths, states, horizons, 2, seed_base);
    REQUIRE(batch.size() == static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Trajectory single = rollout(paths[i], states[i], horizons[i], configs[i],
                                          scenario_seed(seed_base, static_cast<std::uint64_t>(i)));
        REQUIRE(batch[i].data == single.data);
    }
}
