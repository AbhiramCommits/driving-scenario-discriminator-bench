#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>

#include "dsd_sim/batch_rollout.hpp"
#include "dsd_sim/kinematic_bicycle.hpp"
#include "dsd_sim/noise_model.hpp"
#include "dsd_sim/pure_pursuit.hpp"
#include "dsd_sim/rollout.hpp"

namespace py = pybind11;
using namespace dsd;

namespace {

Path path_from_array(const py::array_t<double, py::array::c_style | py::array::forcecast> &arr) {
    if (arr.ndim() != 2 || arr.shape(1) != 2) {
        throw std::invalid_argument("path must be an (N, 2) float64 array");
    }
    Path path;
    const auto *ptr = arr.data();
    const std::size_t n = static_cast<std::size_t>(arr.shape(0));
    path.points.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        path.points.push_back(Vec2{ptr[2 * i], ptr[2 * i + 1]});
    }
    return path;
}

py::array_t<double> trajectory_to_array(const Trajectory &traj) {
    py::array_t<double> arr(std::vector<py::ssize_t>{static_cast<py::ssize_t>(traj.horizon + 1),
                                                     static_cast<py::ssize_t>(6)});
    std::memcpy(arr.mutable_data(), traj.data.data(), traj.data.size() * sizeof(double));
    return arr;
}

} // namespace

PYBIND11_MODULE(_sim, m) {
    m.doc() = "C++17 driving-scenario simulator core for dsdbench";
    m.attr("__version__") = "0.1.0";

    py::class_<State>(m, "State")
        .def(py::init<double, double, double, double, double>(), py::arg("x") = 0.0,
             py::arg("y") = 0.0, py::arg("heading") = 0.0, py::arg("speed") = 0.0,
             py::arg("steer_angle") = 0.0)
        .def_readwrite("x", &State::x)
        .def_readwrite("y", &State::y)
        .def_readwrite("heading", &State::heading)
        .def_readwrite("speed", &State::speed)
        .def_readwrite("steer_angle", &State::steer_angle)
        .def("__repr__", [](const State &s) {
            return "State(x=" + std::to_string(s.x) + ", y=" + std::to_string(s.y) +
                   ", heading=" + std::to_string(s.heading) + ", speed=" + std::to_string(s.speed) +
                   ", steer_angle=" + std::to_string(s.steer_angle) + ")";
        });

    py::class_<Control>(m, "Control")
        .def(py::init<double, double>(), py::arg("steer") = 0.0, py::arg("accel") = 0.0)
        .def_readwrite("steer", &Control::steer)
        .def_readwrite("accel", &Control::accel);

    py::class_<ControllerErrorModel>(m, "ControllerErrorModel")
        .def(py::init<double, double, int>(), py::arg("steer_bias") = 0.0,
             py::arg("steer_noise_std") = 0.0, py::arg("latency_steps") = 0)
        .def_readwrite("steer_bias", &ControllerErrorModel::steer_bias)
        .def_readwrite("steer_noise_std", &ControllerErrorModel::steer_noise_std)
        .def_readwrite("latency_steps", &ControllerErrorModel::latency_steps);

    py::class_<SimConfig>(m, "SimConfig")
        .def(py::init<>())
        .def_readwrite("dt", &SimConfig::dt)
        .def_readwrite("wheelbase", &SimConfig::wheelbase)
        .def_readwrite("max_steer", &SimConfig::max_steer)
        .def_readwrite("max_accel", &SimConfig::max_accel)
        .def_readwrite("max_decel", &SimConfig::max_decel)
        .def_readwrite("accel", &SimConfig::accel)
        .def_readwrite("lookahead_gain", &SimConfig::lookahead_gain)
        .def_readwrite("min_lookahead", &SimConfig::min_lookahead)
        .def_readwrite("pos_noise_std", &SimConfig::pos_noise_std)
        .def_readwrite("heading_noise_std", &SimConfig::heading_noise_std)
        .def_readwrite("latency_steps", &SimConfig::latency_steps)
        .def_readwrite("steer_bias", &SimConfig::steer_bias)
        .def_readwrite("steer_noise_std", &SimConfig::steer_noise_std);

    py::class_<KinematicBicycleModel>(m, "KinematicBicycleModel")
        .def(py::init<double, double, double, double, double>(), py::arg("wheelbase"),
             py::arg("max_steer"), py::arg("max_accel"), py::arg("max_decel"), py::arg("dt"))
        .def("step", &KinematicBicycleModel::step, py::arg("state"), py::arg("control"))
        .def_property_readonly("wheelbase", &KinematicBicycleModel::wheelbase)
        .def_property_readonly("max_steer", &KinematicBicycleModel::max_steer)
        .def_property_readonly("max_accel", &KinematicBicycleModel::max_accel)
        .def_property_readonly("max_decel", &KinematicBicycleModel::max_decel)
        .def_property_readonly("dt", &KinematicBicycleModel::dt);

    py::class_<PurePursuitController>(m, "PurePursuitController")
        .def(py::init<double, double, ControllerErrorModel>(), py::arg("lookahead_gain") = 3.0,
             py::arg("min_lookahead") = 0.5, py::arg("error_model") = ControllerErrorModel{})
        .def("reset", &PurePursuitController::reset, py::arg("seed"))
        .def(
            "compute_steer",
            [](PurePursuitController &controller, const State &state,
               const py::array_t<double, py::array::c_style | py::array::forcecast> &path,
               double wheelbase) {
                return controller.compute_steer(state, path_from_array(path), wheelbase);
            },
            py::arg("state"), py::arg("path"), py::arg("wheelbase"));

    py::class_<NoiseModel>(m, "NoiseModel")
        .def(py::init<double, double>(), py::arg("pos_std") = 0.0, py::arg("heading_std") = 0.0)
        .def("reset", &NoiseModel::reset, py::arg("seed"))
        .def("observe", &NoiseModel::observe, py::arg("state"));

    m.def(
        "rollout",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast> &path,
           const State &initial_state, int horizon, const SimConfig &config, std::uint64_t seed) {
            const Path cpp_path = path_from_array(path);
            Trajectory traj;
            {
                py::gil_scoped_release release;
                traj = dsd::rollout(cpp_path, initial_state, horizon, config, seed);
            }
            return trajectory_to_array(traj);
        },
        py::arg("path"), py::arg("initial_state"), py::arg("horizon"), py::arg("config"),
        py::arg("seed") = 0);

    m.def(
        "batch_rollout",
        [](const std::vector<SimConfig> &configs,
           const std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> &paths,
           const std::vector<State> &initial_states, const std::vector<int> &horizons,
           int n_threads, std::uint64_t seed_base) {
            if (paths.size() != configs.size() || initial_states.size() != configs.size() ||
                horizons.size() != configs.size()) {
                throw std::invalid_argument("batch_rollout: input list size mismatch");
            }
            std::vector<Path> cpp_paths;
            cpp_paths.reserve(paths.size());
            for (const auto &p : paths) {
                cpp_paths.push_back(path_from_array(p));
            }

            // Pre-allocate all output arrays while holding the GIL.
            std::vector<py::array_t<double>> arrays;
            arrays.reserve(horizons.size());
            for (const int h : horizons) {
                arrays.emplace_back(std::vector<py::ssize_t>{static_cast<py::ssize_t>(h + 1),
                                                             static_cast<py::ssize_t>(6)});
            }

            std::vector<Trajectory> results;
            {
                py::gil_scoped_release release;
                results = dsd::batch_rollout(configs, cpp_paths, initial_states, horizons,
                                             n_threads, seed_base);
            }

            py::list out;
            for (std::size_t i = 0; i < results.size(); ++i) {
                std::memcpy(arrays[i].mutable_data(), results[i].data.data(),
                            results[i].data.size() * sizeof(double));
                out.append(arrays[i]);
            }
            return out;
        },
        py::arg("configs"), py::arg("paths"), py::arg("initial_states"), py::arg("horizons"),
        py::arg("n_threads") = 1, py::arg("seed_base") = kDefaultSeedBase);
}
