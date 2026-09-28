"""Tests for the benchmark sweep and report generation."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest

from dsdbench.data.pipeline import QUERIES_SQL, pipeline_queries, sweep_queries

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_query_sections_are_disjoint_and_complete():
    sql_text = QUERIES_SQL.read_text()
    before = pipeline_queries(sql_text)
    after = sweep_queries(sql_text)
    assert before, "pipeline section must not be empty"
    assert after, "sweep section must not be empty"
    assert "sweep_results" in "\n".join(after)
    assert all("sweep_results" not in stmt for stmt in before)


def test_pipeline_section_still_runs_on_dataset_db(tmp_path: Path):
    # The dataset pipeline executes the pre-sweep queries only; make sure the
    # section splitter is consistent with what the pipeline itself runs.
    from dsdbench.data.pipeline import run_pipeline

    run_pipeline(
        tmp_path / "bench",
        synthetic_fallback=True,
        n_scenes=8,
        agents_per_scene=2,
        duration_s=12.0,
        seed=5,
    )
    con = duckdb.connect(str(tmp_path / "bench" / "benchmark.duckdb"))
    for stmt in pipeline_queries(QUERIES_SQL.read_text()):
        con.execute(stmt)
    con.close()


def test_benchmark_sweep_quick(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    from experiments.run_benchmark import main as bench_main

    out = tmp_path / "experiments"
    rc = bench_main(
        [
            "--out-dir",
            str(out),
            "--quick",
            "--n-scenes",
            "16",
            "--epochs",
            "1",
            "--figures-dir",
            str(tmp_path / "figures"),
            "--baseline-path",
            str(tmp_path / "baseline.json"),
        ],
    )
    assert rc == 0

    results = pq.read_table(out / "results.parquet").to_pydict()
    assert set(results["model"]) == {"cnn"}
    assert "overall" in results["scope"]
    assert "slice" in results["scope"]
    overall = [i for i, s in enumerate(results["scope"]) if s == "overall"]
    assert len(overall) == 1
    auc = results["auc"][overall[0]]
    assert np.isfinite(auc) and 0.0 <= auc <= 1.0

    # DuckDB table exists and the sweep queries execute against it.
    con = duckdb.connect(str(out / "results.duckdb"), read_only=True)
    for stmt in sweep_queries(QUERIES_SQL.read_text()):
        con.execute(stmt)
    n_rows = con.execute("SELECT count(*) FROM sweep_results").fetchall()[0][0]
    assert n_rows > 0
    con.close()

    # Baseline JSON written with the expected schema.
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["schema"] == "dsdbench-slice-baseline-v1"
    assert (tmp_path / "figures").is_dir()


def test_write_readme_render_roundtrip(tmp_path: Path):
    # Placeholder rendering must fill every token and produce the tables.
    from experiments.write_readme import build_replacements, render

    rows = [
        {
            "scope": "overall",
            "model": "cnn",
            "knob_noise": "mid",
            "knob_latency": 2,
            "knob_bias": 0.0,
            "auc": 0.8,
            "ci_low": 0.75,
            "ci_high": 0.85,
            "ece": 0.05,
            "ece_after_isotonic": 0.02,
        },
        {
            "scope": "slice",
            "model": "cnn",
            "knob_noise": "mid",
            "knob_latency": 2,
            "knob_bias": 0.0,
            "slice_name": "maneuver:cut_in",
            "auc": 0.9,
            "ci_low": 0.8,
            "ci_high": 0.95,
        },
    ]
    throughput = {
        "rollouts_per_sec_n_threads_1": 100.0,
        "rollouts_per_sec_n_threads_8": 400.0,
        "speedup_8_vs_1": 4.0,
    }
    template = (
        "AUC:\n{{AUC_TABLE}}\n{{ECE_TABLE}}\n{{SLICE_TABLE}}\n"
        "{{KNOB_TABLE}}\n{{KNOB_WORST}}\n{{MANEUVER_HARDEST}}\n"
        "{{POWER_MIN_N}}\n{{TPUT_TABLE}}\n{{TPUT_SPEEDUP}}\n"
    )
    out = render(template, build_replacements(rows, throughput))
    assert "cnn" in out and "cut_in" in out
    assert "{{" not in out
