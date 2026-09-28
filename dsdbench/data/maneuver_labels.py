"""Rule-based per-segment maneuver labeling.

Each segment is classified into one of five maneuver classes using physical
thresholds computed from the segment trajectory:

* ``stop_and_go`` -- the agent comes to a (near) stop and accelerates again
* ``unprotected_left`` -- sustained left (positive, CCW) heading change
* ``cut_in`` -- lateral displacement peaking in the central part of the window
* ``merge`` -- lateral displacement peaking near either window edge
* ``lane_keep`` -- everything else

Thresholds (defaults from ``maneuver_config.yaml``):

=========================  =====  ==========================================
parameter                  default  meaning
=========================  =====  ==========================================
stop_speed_ms              0.5    speed at/below which the agent counts as
                                  stopped
min_speed_range_ms         5.0    speed range required to separate a genuine
                                  stop-and-go cycle from a parked/slow agent
min_heading_change_rad     1.0    net positive heading change required for a
                                  left turn (right turns are out of scope for
                                  this label set and fall through to
                                  lane_keep)
min_mean_speed_ms          1.0    minimum mean speed for a left turn
                                  (excludes near-stationary pivoting)
min_lateral_m              1.6    peak lateral displacement relative to the
                                  initial-heading frame required for a
                                  lateral maneuver (cut_in / merge)
max_heading_change_rad     0.6    maximum net |heading change| allowed for a
                                  lateral maneuver (turns are labeled first)
edge_fraction              0.25   fraction of the window at each end; a
                                  lateral peak inside the central
                                  1 - 2*edge_fraction is a cut_in, a peak in
                                  the edge regions a merge
=========================  =====  ==========================================

Labeling order: stop_and_go, unprotected_left, cut_in/merge (lateral shifts),
lane_keep. All thresholds are configurable via a YAML file; partial files are
deep-merged over the defaults in ``maneuver_config.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike

DEFAULT_CONFIG = Path(__file__).with_name("maneuver_config.yaml")

MANEUVERS = ("cut_in", "merge", "unprotected_left", "lane_keep", "stop_and_go")


class ManeuverLabeler:
    """Labels one segment trajectory with a maneuver class (see module docstring)."""

    def __init__(
        self, config_path: str | Path | None = None, overrides: dict[str, Any] | None = None
    ) -> None:
        self._config = _load_config(config_path, overrides)

    def label(
        self,
        t: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        heading: ArrayLike,
        speed: ArrayLike,
        yaw_rate: ArrayLike,
    ) -> str:
        t_arr = np.asarray(t, dtype=np.float64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        heading_arr = np.unwrap(np.asarray(heading, dtype=np.float64))
        speed_arr = np.asarray(speed, dtype=np.float64)
        if t_arr.size < 2:
            raise ValueError("maneuver labeling needs at least 2 samples")

        h0 = float(heading_arr[0])
        # Signed lateral displacement in the initial-heading frame.
        lateral = -np.sin(h0) * (x_arr - x_arr[0]) + np.cos(h0) * (y_arr - y_arr[0])
        heading_change = float(heading_arr[-1] - heading_arr[0])
        v_min = float(speed_arr.min())
        v_max = float(speed_arr.max())

        stop_cfg = self._config["stop_and_go"]
        left_cfg = self._config["unprotected_left"]
        lat_cfg = self._config["lateral_shift"]

        if v_min <= stop_cfg["stop_speed_ms"] and (v_max - v_min) >= stop_cfg["min_speed_range_ms"]:
            return "stop_and_go"

        if (
            heading_change >= left_cfg["min_heading_change_rad"]
            and float(speed_arr.mean()) >= left_cfg["min_mean_speed_ms"]
        ):
            return "unprotected_left"

        peak = int(np.argmax(np.abs(lateral)))
        frac = (
            float((t_arr[peak] - t_arr[0]) / (t_arr[-1] - t_arr[0]))
            if t_arr[-1] > t_arr[0]
            else 0.0
        )
        if (
            abs(float(lateral[peak])) >= lat_cfg["min_lateral_m"]
            and abs(heading_change) <= lat_cfg["max_heading_change_rad"]
        ):
            edge = float(lat_cfg["edge_fraction"])
            if edge < frac < 1.0 - edge:
                return "cut_in"
            return "merge"

        return "lane_keep"


def _load_config(
    config_path: str | Path | None, overrides: dict[str, Any] | None
) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if config_path is not None:
        extra = yaml.safe_load(Path(config_path).read_text()) or {}
        _deep_merge(cfg, extra)
    if overrides:
        _deep_merge(cfg, overrides)
    required = {
        "stop_and_go": {"stop_speed_ms", "min_speed_range_ms"},
        "unprotected_left": {"min_heading_change_rad", "min_mean_speed_ms"},
        "lateral_shift": {"min_lateral_m", "max_heading_change_rad", "edge_fraction"},
    }
    for section, keys in required.items():
        missing = keys - set(cfg.get(section, {}))
        if missing:
            raise ValueError(f"maneuver config section {section!r} missing keys: {sorted(missing)}")
    return cfg


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
