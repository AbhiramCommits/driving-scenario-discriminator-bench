#include "dsd_sim/pure_pursuit.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace dsd {

PurePursuitController::PurePursuitController(double lookahead_gain, double min_lookahead,
                                             ControllerErrorModel error_model)
    : lookahead_gain_(lookahead_gain), min_lookahead_(min_lookahead), error_model_(error_model) {
    reset(0);
}

void PurePursuitController::reset(std::uint64_t seed) {
    rng_.seed(seed);
    latency_buffer_.clear();
    latency_buffer_.assign(static_cast<std::size_t>(std::max(error_model_.latency_steps, 0)), 0.0);
}

double PurePursuitController::compute_steer(const State &state, const Path &path,
                                            double wheelbase) {
    if (path.points.size() < 2) {
        return 0.0;
    }

    const double lookahead = std::max(lookahead_gain_ * state.speed, min_lookahead_);

    // Find the segment closest to the vehicle: on paths that double back
    // (e.g. U-turns) the lookahead circle intersects both ahead and behind
    // the vehicle, and searching from the path start would target points
    // behind it, sending the vehicle into a spin.
    std::size_t start_idx = 0;
    {
        double best_d2 = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i + 1 < path.points.size(); ++i) {
            const Vec2 a = path.points[i];
            const Vec2 b = path.points[i + 1];
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < 1e-12) {
                continue;
            }
            const double t =
                std::clamp(((state.x - a.x) * dx + (state.y - a.y) * dy) / d2, 0.0, 1.0);
            const Vec2 p{a.x + t * dx, a.y + t * dy};
            const double dd2 =
                (p.x - state.x) * (p.x - state.x) + (p.y - state.y) * (p.y - state.y);
            if (dd2 < best_d2) {
                best_d2 = dd2;
                start_idx = i;
            }
        }
    }

    // Find the first intersection of the circle (center = rear axle, radius =
    // lookahead) with the polyline, walking forward from the closest segment.
    Vec2 target = path.points.front();
    bool found = false;
    for (std::size_t i = start_idx; i + 1 < path.points.size(); ++i) {
        const Vec2 a = path.points[i];
        const Vec2 b = path.points[i + 1];
        const double dx = b.x - a.x;
        const double dy = b.y - a.y;
        const double fx = a.x - state.x;
        const double fy = a.y - state.y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < 1e-12) {
            continue;
        }
        // Solve |a + t*d - p|^2 = lookahead^2.
        const double A = d2;
        const double B = 2.0 * (fx * dx + fy * dy);
        const double C = fx * fx + fy * fy - lookahead * lookahead;
        const double disc = B * B - 4.0 * A * C;
        if (disc < 0.0) {
            continue;
        }
        const double sqrt_disc = std::sqrt(disc);
        const double t1 = (-B - sqrt_disc) / (2.0 * A);
        const double t2 = (-B + sqrt_disc) / (2.0 * A);
        const double t = (t1 >= 0.0 && t1 <= 1.0) ? t1 : (t2 >= 0.0 && t2 <= 1.0 ? t2 : -1.0);
        if (t < 0.0) {
            continue;
        }
        target = Vec2{a.x + t * dx, a.y + t * dy};
        found = true;
        break;
    }

    if (!found) {
        // No circle/path intersection (path too close, before the start, or beyond
        // the end): fall back to the closest point on the polyline.
        double best_d2 = std::numeric_limits<double>::max();
        for (std::size_t i = 0; i + 1 < path.points.size(); ++i) {
            const Vec2 a = path.points[i];
            const Vec2 b = path.points[i + 1];
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < 1e-12) {
                continue;
            }
            const double t =
                std::clamp(((state.x - a.x) * dx + (state.y - a.y) * dy) / d2, 0.0, 1.0);
            const Vec2 p{a.x + t * dx, a.y + t * dy};
            const double dd2 =
                (p.x - state.x) * (p.x - state.x) + (p.y - state.y) * (p.y - state.y);
            if (dd2 < best_d2) {
                best_d2 = dd2;
                target = p;
            }
        }
    }

    const double alpha = std::atan2(target.y - state.y, target.x - state.x) - state.heading;
    double steer = std::atan2(2.0 * wheelbase * std::sin(alpha), lookahead);

    // Error model: proportional bias, then Gaussian steering noise.
    steer *= 1.0 + error_model_.steer_bias;
    if (error_model_.steer_noise_std > 0.0) {
        steer += error_model_.steer_noise_std * gauss_(rng_);
    }

    // N-step actuation latency ring buffer.
    if (error_model_.latency_steps > 0) {
        latency_buffer_.push_back(steer);
        steer = latency_buffer_.front();
        latency_buffer_.pop_front();
    }
    return steer;
}

} // namespace dsd
