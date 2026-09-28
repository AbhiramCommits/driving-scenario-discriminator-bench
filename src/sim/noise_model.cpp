#include "dsd_sim/noise_model.hpp"

namespace dsd {

NoiseModel::NoiseModel(double pos_std, double heading_std)
    : pos_std_(pos_std), heading_std_(heading_std) {
    reset(0);
}

void NoiseModel::reset(std::uint64_t seed) {
    rng_.seed(seed);
}

State NoiseModel::observe(const State &true_state) {
    State observed = true_state;
    if (pos_std_ > 0.0) {
        observed.x += pos_std_ * gauss_(rng_);
        observed.y += pos_std_ * gauss_(rng_);
    }
    if (heading_std_ > 0.0) {
        observed.heading += heading_std_ * gauss_(rng_);
    }
    return observed;
}

} // namespace dsd
