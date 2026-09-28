#include "dsd_sim/rollout.hpp"

#include <cmath>
#include <stdexcept>

#include "dsd_sim/kinematic_bicycle.hpp"
#include "dsd_sim/noise_model.hpp"
#include "dsd_sim/pure_pursuit.hpp"
#include "dsd_sim/rng.hpp"

namespace dsd {

namespace {
constexpr std::uint64_t kControllerSeedSalt = 0xa5a5a5a5a5a5a5a5ULL;
constexpr std::uint64_t kNoiseSeedSalt = 0x5a5a5a5a5a5a5a5aULL;
} // namespace

Trajectory rollout(const Path &path, const State &initial_state, int horizon,
                   const SimConfig &config, std::uint64_t seed) {
    if (horizon < 0) {
        throw std::invalid_argument("rollout: horizon must be >= 0");
    }

    Trajectory traj;
    traj.horizon = horizon;
    traj.data.assign(static_cast<std::size_t>(horizon + 1) * 6, 0.0);

    KinematicBicycleModel model(config.wheelbase, config.max_steer, config.max_accel,
                                config.max_decel, config.dt);
    PurePursuitController controller(
        config.lookahead_gain, config.min_lookahead,
        {config.steer_bias, config.steer_noise_std, config.latency_steps});
    NoiseModel noise(config.pos_noise_std, config.heading_noise_std);

    controller.reset(splitmix64(seed ^ kControllerSeedSalt));
    noise.reset(splitmix64(seed ^ kNoiseSeedSalt));

    State state = initial_state;
    for (int i = 0; i <= horizon; ++i) {
        // The controller and the recorded trajectory share the same (noisy)
        // observation of the state, as a real logged trajectory would.
        const State observed = noise.observe(state);
        const double yaw_rate = state.speed / config.wheelbase * std::tan(state.steer_angle);

        double *row = traj.data.data() + static_cast<std::size_t>(i) * 6;
        row[0] = static_cast<double>(i) * config.dt;
        row[1] = observed.x;
        row[2] = observed.y;
        row[3] = observed.heading;
        row[4] = state.speed;
        row[5] = yaw_rate;

        if (i < horizon) {
            const double steer = controller.compute_steer(observed, path, config.wheelbase);
            state = model.step(state, Control{steer, config.accel});
        }
    }
    return traj;
}

} // namespace dsd
