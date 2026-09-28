"""Per-segment scalar feature extraction for the tree model.

Produces 30 float64 features per segment. All features are computed from the
six raw channels (t, x, y, heading, speed, yaw_rate) sampled at ~10 Hz.

Speed profile (4)
    speed_mean, speed_std, speed_max, speed_min -- longitudinal speed stats.
Acceleration (3)
    accel_mean, accel_std, accel_max -- from d(speed)/dt; mean/max use the
    absolute value, std is of the signed series.
Jerk (4)
    jerk_mean, jerk_std, jerk_max, jerk_p95 -- from d^2(speed)/dt^2; mean/max/p95
    use the absolute value.
Lateral acceleration (5)
    lat_accel_mean, lat_accel_std, lat_accel_max, lat_accel_p95, lat_accel_rms --
    a_lat = speed * yaw_rate; mean/max/p95 use the absolute value.
Yaw-rate spectral energy (4)
    yaw_psd_0_2hz, yaw_psd_2_5hz -- Welch PSD energy integrated over 0-2 Hz and
    2-5 Hz; yaw_psd_total -- total energy; yaw_psd_ratio -- low/high band ratio
    (clipped to [0, 100], 100 if there is low but no high energy).
Curvature (3)
    curvature_mean, curvature_p95 -- |yaw_rate / speed| stats (speed < 0.5 m/s
    counts as straight); curv_speed_corr -- Pearson correlation between
    curvature magnitude and speed (0 when either is constant).
Heading (4)
    heading_rate_mean, heading_rate_std, heading_rate_max -- d(heading)/dt
    stats (mean/max use the absolute value); heading_smoothness --
    1 / (1 + std(heading rate)), in (0, 1], higher is smoother.
Time-to-collision proxies (3)
    ttc_stop_mean, ttc_stop_min -- speed / 6 m/s^2 (typical emergency
    deceleration); ttc_path_end_min -- remaining arc length to the end of the
    segment path divided by speed (speed floored at 0.1 m/s, capped at 10 s).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import trapezoid
from scipy.signal import welch

TTC_DECEL_MS2 = 6.0
TTC_PATH_END_CAP_S = 10.0
MIN_SPEED_FOR_CURVATURE = 0.5
MIN_SPEED_FOR_TTC = 0.1

FEATURE_NAMES: tuple[str, ...] = (
    "speed_mean",
    "speed_std",
    "speed_max",
    "speed_min",
    "accel_mean",
    "accel_std",
    "accel_max",
    "jerk_mean",
    "jerk_std",
    "jerk_max",
    "jerk_p95",
    "lat_accel_mean",
    "lat_accel_std",
    "lat_accel_max",
    "lat_accel_p95",
    "lat_accel_rms",
    "yaw_psd_0_2hz",
    "yaw_psd_2_5hz",
    "yaw_psd_total",
    "yaw_psd_ratio",
    "curvature_mean",
    "curvature_p95",
    "curv_speed_corr",
    "heading_rate_mean",
    "heading_rate_std",
    "heading_rate_max",
    "heading_smoothness",
    "ttc_stop_mean",
    "ttc_stop_min",
    "ttc_path_end_min",
)


def extract_features(
    t: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    heading: ArrayLike,
    speed: ArrayLike,
    yaw_rate: ArrayLike,
    *,
    fs: float = 10.0,
) -> dict[str, float]:
    """Extract the 30 scalar features for one segment; all values are finite."""
    t_arr = np.asarray(t, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    heading_arr = np.asarray(heading, dtype=np.float64)
    speed_arr = np.asarray(speed, dtype=np.float64)
    yaw_rate_arr = np.asarray(yaw_rate, dtype=np.float64)
    if speed_arr.size < 4:
        raise ValueError("feature extraction needs at least 4 samples")

    dt = float(np.median(np.diff(t_arr)))

    feats: dict[str, float] = {}
    with np.errstate(all="ignore"):
        accel = np.gradient(speed_arr, dt)
        jerk = np.gradient(accel, dt)
        lateral_accel = speed_arr * yaw_rate_arr

        feats["speed_mean"] = float(np.mean(speed_arr))
        feats["speed_std"] = float(np.std(speed_arr))
        feats["speed_max"] = float(np.max(speed_arr))
        feats["speed_min"] = float(np.min(speed_arr))

        feats["accel_mean"] = float(np.mean(np.abs(accel)))
        feats["accel_std"] = float(np.std(accel))
        feats["accel_max"] = float(np.max(np.abs(accel)))

        feats["jerk_mean"] = float(np.mean(np.abs(jerk)))
        feats["jerk_std"] = float(np.std(jerk))
        feats["jerk_max"] = float(np.max(np.abs(jerk)))
        feats["jerk_p95"] = float(np.percentile(np.abs(jerk), 95))

        feats["lat_accel_mean"] = float(np.mean(np.abs(lateral_accel)))
        feats["lat_accel_std"] = float(np.std(lateral_accel))
        feats["lat_accel_max"] = float(np.max(np.abs(lateral_accel)))
        feats["lat_accel_p95"] = float(np.percentile(np.abs(lateral_accel), 95))
        feats["lat_accel_rms"] = float(np.sqrt(np.mean(lateral_accel**2)))

        freqs, psd = welch(
            yaw_rate_arr, fs=fs, nperseg=min(yaw_rate_arr.size, 64), detrend="constant"
        )
        low_mask = (freqs >= 0.0) & (freqs <= 2.0)
        high_mask = (freqs > 2.0) & (freqs <= 5.0)
        energy_low = float(trapezoid(psd[low_mask], freqs[low_mask]))
        energy_high = float(trapezoid(psd[high_mask], freqs[high_mask]))
        feats["yaw_psd_0_2hz"] = energy_low
        feats["yaw_psd_2_5hz"] = energy_high
        feats["yaw_psd_total"] = float(trapezoid(psd, freqs))
        if energy_high > 1e-12:
            feats["yaw_psd_ratio"] = float(np.clip(energy_low / energy_high, 0.0, 100.0))
        else:
            feats["yaw_psd_ratio"] = 100.0 if energy_low > 1e-12 else 0.0

        curvature = np.where(
            speed_arr > MIN_SPEED_FOR_CURVATURE, np.abs(yaw_rate_arr / speed_arr), 0.0
        )
        feats["curvature_mean"] = float(np.mean(curvature))
        feats["curvature_p95"] = float(np.percentile(curvature, 95))
        if curvature.std() > 1e-12 and speed_arr.std() > 1e-12:
            feats["curv_speed_corr"] = float(np.corrcoef(curvature, speed_arr)[0, 1])
        else:
            feats["curv_speed_corr"] = 0.0

        heading_rate = np.gradient(np.unwrap(heading_arr), dt)
        feats["heading_rate_mean"] = float(np.mean(np.abs(heading_rate)))
        feats["heading_rate_std"] = float(np.std(heading_rate))
        feats["heading_rate_max"] = float(np.max(np.abs(heading_rate)))
        feats["heading_smoothness"] = 1.0 / (1.0 + float(np.std(heading_rate)))

        ttc_stop = speed_arr / TTC_DECEL_MS2
        feats["ttc_stop_mean"] = float(np.mean(ttc_stop))
        feats["ttc_stop_min"] = float(np.min(ttc_stop))

        arc = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x_arr), np.diff(y_arr)))))
        remaining = float(arc[-1]) - arc
        effective_speed = np.where(speed_arr > MIN_SPEED_FOR_TTC, speed_arr, MIN_SPEED_FOR_TTC)
        ttc_path_end = np.clip(remaining / effective_speed, 0.0, TTC_PATH_END_CAP_S)
        feats["ttc_path_end_min"] = float(np.min(ttc_path_end))

    for name in FEATURE_NAMES:
        value = feats.get(name, 0.0)
        feats[name] = float(value) if np.isfinite(value) else 0.0
    return {name: feats[name] for name in FEATURE_NAMES}
