-- =============================================================================
-- dsdbench benchmark dataset queries
-- Run against the DuckDB database produced by `dsdbench.data.pipeline`
-- (tables: segments, labels, features).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Class balance: segment counts per maneuver class and source (real/sim)
-- -----------------------------------------------------------------------------
SELECT source,
       maneuver,
       count(*)                                           AS n_segments,
       round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
FROM labels
GROUP BY source, maneuver
ORDER BY source, maneuver;

-- -----------------------------------------------------------------------------
-- 2. Per-slice (train/val/test) feature distribution summaries
-- -----------------------------------------------------------------------------
SELECT l.split,
       count(*)                            AS n_segments,
       round(avg(f.speed_mean), 3)         AS mean_speed_mean,
       round(stddev(f.speed_max), 3)       AS sd_speed_max,
       round(avg(f.accel_max), 3)          AS mean_accel_max,
       round(avg(f.jerk_p95), 3)           AS mean_jerk_p95,
       round(avg(f.lat_accel_rms), 3)      AS mean_lat_accel_rms,
       round(avg(f.yaw_psd_0_2hz), 6)      AS mean_yaw_energy_0_2hz,
       round(avg(f.yaw_psd_2_5hz), 6)      AS mean_yaw_energy_2_5hz,
       round(avg(f.curv_speed_corr), 3)    AS mean_curv_speed_corr,
       round(avg(f.heading_smoothness), 3) AS mean_heading_smoothness,
       round(avg(f.ttc_stop_min), 3)       AS mean_min_ttc_stop
FROM labels l
JOIN features f USING (segment_id)
GROUP BY l.split
ORDER BY l.split;

-- -----------------------------------------------------------------------------
-- 3. Train/val/test split with grouped splitting by scene id.
--    Splits are assigned per scene by dsdbench.data.pipeline.assign_splits
--    (deterministic md5(scene_id) buckets, 70/15/15), so every segment of a
--    scene lands in exactly one split and no scene leaks across splits.
-- -----------------------------------------------------------------------------

-- 3a. Scene -> split assignment (one row per scene)
CREATE OR REPLACE TABLE split_assignment AS
SELECT scene_id,
       min(split) AS split,
       count(*)    AS n_segments
FROM labels
GROUP BY scene_id
ORDER BY scene_id;

-- 3b. Join that produces the split tables: trajectory rows joined to their
--     label (grouped by scene via split_assignment, so real/sim counterparts
--     of the same scene always share a split).
CREATE OR REPLACE TABLE train_set AS
SELECT s.segment_id, s.scene_id, s.source, l.maneuver,
       s.t, s.x, s.y, s.heading, s.speed, s.yaw_rate
FROM segments s
JOIN labels l USING (segment_id)
WHERE l.split = 'train';

CREATE OR REPLACE TABLE val_set AS
SELECT s.segment_id, s.scene_id, s.source, l.maneuver,
       s.t, s.x, s.y, s.heading, s.speed, s.yaw_rate
FROM segments s
JOIN labels l USING (segment_id)
WHERE l.split = 'val';

CREATE OR REPLACE TABLE test_set AS
SELECT s.segment_id, s.scene_id, s.source, l.maneuver,
       s.t, s.x, s.y, s.heading, s.speed, s.yaw_rate
FROM segments s
JOIN labels l USING (segment_id)
WHERE l.split = 'test';

-- 3c. Per-split segment-level table (labels + all features, one row per segment)
CREATE OR REPLACE TABLE split_features AS
SELECT l.split, l.scene_id, l.segment_id, l.source, l.maneuver,
       f.* EXCLUDE (segment_id, scene_id, source)
FROM labels l
JOIN features f USING (segment_id);

-- 3d. Grouped-splitting invariant check: must return zero rows.
SELECT scene_id, count(DISTINCT split) AS n_splits
FROM labels
GROUP BY scene_id
HAVING n_splits > 1;

-- =============================================================================
-- SWEEP-SECTION: queries over the sweep_results table written by
-- experiments/run_benchmark.py (one row per config x model, scope is
-- 'overall' for headline metrics or 'slice' for maneuver slices).
-- Executed by run_benchmark against results.duckdb and skipped by
-- dsdbench.data.pipeline (the sweep table does not exist at pipeline time).
-- Note: this banner intentionally contains no semicolons so the statement
-- splitter keeps it intact.
-- =============================================================================

-- Which simulator knob most degrades realism? For each knob and level, the
-- mean overall AUC across models and the other knob dimensions. A higher
-- mean AUC means simulated segments are easier to detect, i.e. less
-- realistic.
SELECT 'noise' AS knob,
       knob_noise AS level,
       round(avg(auc), 4) AS mean_auc,
       count(*) AS n_evals
FROM sweep_results
WHERE scope = 'overall'
GROUP BY 2
UNION ALL
SELECT 'latency',
       CAST(knob_latency AS VARCHAR),
       round(avg(auc), 4),
       count(*)
FROM sweep_results
WHERE scope = 'overall'
GROUP BY 2
UNION ALL
SELECT 'bias',
       CAST(knob_bias AS VARCHAR),
       round(avg(auc), 4),
       count(*)
FROM sweep_results
WHERE scope = 'overall'
GROUP BY 2
ORDER BY mean_auc DESC;

-- Per-model knob degradation (interaction view).
SELECT model,
       knob_noise,
       knob_latency,
       knob_bias,
       round(avg(auc), 4) AS mean_auc
FROM sweep_results
WHERE scope = 'overall'
GROUP BY 1, 2, 3, 4
ORDER BY model, mean_auc DESC;

-- Which maneuver slice is hardest to simulate faithfully? The slice where
-- discriminators achieve the highest AUC, averaged over configurations and
-- models.
SELECT slice_name,
       round(avg(auc), 4) AS mean_auc,
       round(avg(ci_low), 4) AS mean_ci_low,
       count(*) AS n_evals
FROM sweep_results
WHERE scope = 'slice'
GROUP BY slice_name
ORDER BY mean_auc DESC;
