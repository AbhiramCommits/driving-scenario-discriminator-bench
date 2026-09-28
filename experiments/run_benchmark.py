"""Benchmark sweep: simulator realism knobs x discriminator models.

Sweeps the simulator realism knobs -- 3 observation-noise levels x 3 actuation
latency settings x 2 controller-bias settings -- rebuilds the synthetic
fallback dataset per configuration, trains all three discriminators (gbt, cnn,
transformer) on each, and evaluates every run with the statistical harness.
Outputs a single ``results.parquet`` (one row per config x model, scope
'overall' or 'slice') plus a DuckDB database with the ``sweep_results`` table
and the sweep queries of ``dsdbench/data/queries.sql``. Also regenerates
``docs/figures/``, ``artifacts/baseline.json`` (from the representative
mid-knob CNN run) and, with ``--write-readme``, README.md.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import dsdbench.models.gbt  # noqa: F401  (registers "gbt"; lightgbm stays lazy)
from dsdbench.data.pipeline import QUERIES_SQL, run_pipeline, sweep_queries
from dsdbench.eval.report import (
    EvalArtifacts,
    plot_forest,
    plot_power_curve,
    plot_reliability,
    plot_roc_slices,
)
from dsdbench.evaluate import run_evaluation, write_outputs
from dsdbench.models.registry import create

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

NOISE_LEVELS = ("low", "mid", "high")
LATENCY_SETTINGS = (0, 2, 4)
BIAS_SETTINGS = (0.0, 0.2)
MODELS = ("gbt", "cnn", "transformer")

# Explicit sampling ranges per observation-noise level (log-uniform within
# each range). "low" matches the reference controller of the synthetic real
# side exactly, "mid" is the default simulator range, and "high" is heavily
# degraded, so the sweep spans the full detectable range of the knob.
NOISE_RANGES = {
    "low": {
        "pos_noise_std": (0.002, 0.002),
        "heading_noise_std": (0.0003, 0.0003),
        "steer_noise_std": (0.0003, 0.0003),
    },
    "mid": {
        "pos_noise_std": (1e-3, 0.15),
        "heading_noise_std": (1e-4, 0.03),
        "steer_noise_std": (1e-4, 0.05),
    },
    "high": {
        "pos_noise_std": (1e-2, 0.5),
        "heading_noise_std": (1e-3, 0.1),
        "steer_noise_std": (1e-3, 0.1),
    },
}

EVAL_CONFIG = {
    "n_resamples": 300,
    "n_permutations": 100,
    "n_bins": 15,
    "n_jobs": 1,
    "alpha": 0.05,
    "seed": 0,
    "regression_threshold": 0.05,
}
QUICK_EVAL_CONFIG = {
    "n_resamples": 60,
    "n_permutations": 30,
    "n_bins": 15,
    "n_jobs": 1,
    "alpha": 0.05,
    "seed": 0,
    "regression_threshold": 0.05,
}

CNN_CONFIG = {"epochs": 2, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "seed": 0}
TRANSFORMER_CONFIG = {"epochs": 2, "batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4, "seed": 0}
GBT_CONFIG = {
    "n_estimators": 150,
    "n_cv_folds": 3,
    "learning_rate": 0.05,
    "early_stopping_rounds": 30,
    "shap_sample_size": 500,
    "seed": 0,
}

REPRESENTATIVE = ("mid", 2, 0.0, "cnn")  # used for ROC/reliability/forest figures


def config_id(noise_level: str, latency: int, bias: float) -> str:
    return f"noise_{noise_level}_lat_{latency}_bias_{bias:g}"


def knob_overrides(
    noise_level: str, latency: int, bias: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Knob ranges/fixed dicts for one sweep cell.

    ``accel`` is pinned to zero so the sweep isolates the swept knobs (the
    simulator's longitudinal control is a constant acceleration, so a nonzero
    sampled accel would dominate the sim-real gap), and the secondary knobs
    (wheelbase, lookahead) are pinned to the reference controller's values so
    the swept knobs are the only systematic sim-real difference (see README
    Limitations).
    """
    ranges = dict(NOISE_RANGES[noise_level])
    fixed: dict[str, float | int] = {
        "latency_steps": latency,
        "steer_bias": bias,
        "accel": 0.0,
        "wheelbase": 2.7,
        "lookahead_gain": 0.9,
        "min_lookahead": 0.3,
    }
    return ranges, fixed


def _model_config(model: str) -> dict[str, Any]:
    if model == "gbt":
        return dict(GBT_CONFIG)
    if model == "cnn":
        return dict(CNN_CONFIG)
    return dict(TRANSFORMER_CONFIG)


def _row_from_summary(
    config_id_: str,
    noise_level: str,
    latency: int,
    bias: float,
    model: str,
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten an evaluation summary into overall + maneuver-slice rows."""
    rows: list[dict[str, Any]] = []
    m = summary["metrics"]
    base = {
        "config_id": config_id_,
        "knob_noise": noise_level,
        "knob_latency": int(latency),
        "knob_bias": float(bias),
        "model": model,
        "split_used": summary["split_used"],
    }
    rows.append(
        {
            **base,
            "scope": "overall",
            "slice_name": None,
            "n": m["n"],
            "n_pos": m["n_pos"],
            "n_neg": m["n_neg"],
            "auc": m["auc_roc"],
            "ci_low": m["auc_roc_ci"]["low"],
            "ci_high": m["auc_roc_ci"]["high"],
            "auc_pr": m["auc_pr"],
            "brier": m["brier"],
            "ece": m["ece"],
            "ece_ci_low": m["ece_ci"]["low"],
            "ece_ci_high": m["ece_ci"]["high"],
            "mce": m["mce"],
            "ece_after_platt": (m["calibration"] or {}).get("ece_after_platt"),
            "ece_after_isotonic": (m["calibration"] or {}).get("ece_after_isotonic"),
            "p_perm": m["permutation"]["p_value"],
        }
    )
    for result in summary["slices"]:
        if not result["name"].startswith("maneuver:"):
            continue
        rows.append(
            {
                **base,
                "scope": "slice",
                "slice_name": result["name"],
                "n": result["n"],
                "n_pos": result["n_pos"],
                "n_neg": result["n_neg"],
                "auc": result["auc"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "auc_pr": None,
                "brier": None,
                "ece": None,
                "ece_ci_low": None,
                "ece_ci_high": None,
                "mce": None,
                "ece_after_platt": None,
                "ece_after_isotonic": None,
                "p_perm": None,
            }
        )
    return rows


def run_sweep(
    out_dir: str | Path,
    *,
    n_scenes: int,
    agents_per_scene: int,
    duration_s: float,
    epochs: int,
    models: Sequence[str],
    noise_levels: Sequence[str],
    latency_settings: Sequence[int],
    bias_settings: Sequence[float],
    quick: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full sweep; returns (result rows, representative artifacts)."""
    out = Path(out_dir)
    runs_root = out / "runs"
    rows: list[dict[str, Any]] = []
    representative: dict[str, Any] = {}

    # Import temporal (torch) only when a temporal model is actually trained,
    # so gbt subprocesses that re-import this module never load torch.
    if any(model in ("cnn", "transformer") for model in models):
        import dsdbench.models.temporal  # noqa: F401  (registers "cnn", "transformer")

    for noise_level in noise_levels:
        for latency in latency_settings:
            for bias in bias_settings:
                cid = config_id(noise_level, latency, bias)
                data_dir = runs_root / cid / "data"
                ranges, fixed = knob_overrides(noise_level, latency, bias)
                t0 = time.monotonic()
                run_pipeline(
                    data_dir,
                    synthetic_fallback=True,
                    n_scenes=n_scenes,
                    agents_per_scene=agents_per_scene,
                    duration_s=duration_s,
                    seed=0,
                    knob_ranges=ranges,
                    knob_fixed=fixed,
                )
                logger.info("[%s] dataset built in %.1fs", cid, time.monotonic() - t0)

                for model in models:
                    run_dir = runs_root / cid / model
                    run_dir.mkdir(parents=True, exist_ok=True)
                    train_config = {
                        "data_dir": str(data_dir),
                        "arch": model,
                        **_model_config(model),
                    }
                    if model != "gbt":
                        train_config["epochs"] = epochs
                    (run_dir / "config.yaml").write_text(
                        yaml.safe_dump(train_config, sort_keys=False)
                    )
                    t0 = time.monotonic()
                    trainer = create(model, train_config)
                    trainer.train(data_dir, train_config, run_dir)
                    train_s = time.monotonic() - t0

                    eval_config = dict(QUICK_EVAL_CONFIG if quick else EVAL_CONFIG)
                    summary = run_evaluation(run_dir, eval_config)
                    rows.extend(_row_from_summary(cid, noise_level, latency, bias, model, summary))
                    if (noise_level, latency, bias, model) == REPRESENTATIVE:
                        representative = {
                            "run_dir": run_dir,
                            "summary": dict(summary),
                            "eval_config": eval_config,
                        }
                        write_outputs(run_dir, summary, eval_config)
                    logger.info(
                        "[%s/%s] train %.1fs (auc %.3f)",
                        cid,
                        model,
                        train_s,
                        summary["metrics"]["auc_roc"],
                    )
    return rows, representative


def write_results(out_dir: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write results.parquet and a DuckDB database with the sweep queries."""
    out = Path(out_dir)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out / "results.parquet")

    db_path = out / "results.duckdb"
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE OR REPLACE TABLE sweep_results AS SELECT * FROM read_parquet(?)",
        [str((out / "results.parquet").resolve())],
    )
    for stmt in sweep_queries(QUERIES_SQL.read_text()):
        con.execute(stmt)
    con.close()
    return out / "results.parquet"


def make_figures(
    out_dir: str | Path,
    rows: list[dict[str, Any]],
    representative: dict[str, Any],
    figures_dir: str | Path | None = None,
) -> None:
    """Regenerate all benchmark figures from the sweep results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir) if figures_dir else REPO_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Knob degradation: mean AUC per knob level, per model.
    by_model: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    overall = [r for r in rows if r["scope"] == "overall"]
    for r in overall:
        by_model[r["model"]]["noise"].append((r["knob_noise"], r["auc"]))
        by_model[r["model"]]["latency"].append((str(r["knob_latency"]), r["auc"]))
        by_model[r["model"]]["bias"].append((f"{r['knob_bias']:g}", r["auc"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, knob in zip(axes, ("noise", "latency", "bias"), strict=True):
        levels = sorted({level for model in by_model for level, _ in by_model[model][knob]})
        x = np.arange(len(levels))
        width = 0.25
        for i, model in enumerate(MODELS):
            means = [
                float(np.mean([v for level, v in by_model[model][knob] if level == lvl]))
                for lvl in levels
            ]
            ax.bar(x + (i - 1) * width, means, width, label=model)
        ax.set_xticks(x, levels)
        ax.set_xlabel(knob)
        ax.set_ylabel("mean AUC-ROC")
        ax.set_title(f"Discriminability vs {knob} knob")
        ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Which simulator knob most degrades realism? (higher AUC = easier to detect)")
    fig.tight_layout()
    fig.savefig(figures_dir / "knob_degradation.png", dpi=150)
    plt.close(fig)

    # 2. Maneuver slices: mean AUC per maneuver, per model.
    slice_rows = [r for r in rows if r["scope"] == "slice"]
    maneuvers = sorted({r["slice_name"] for r in slice_rows})
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(maneuvers))
    width = 0.25
    for i, model in enumerate(MODELS):
        means = [
            float(
                np.mean(
                    [r["auc"] for r in slice_rows if r["slice_name"] == m and r["model"] == model]
                )
            )
            for m in maneuvers
        ]
        ax.bar(x + (i - 1) * width, means, width, label=model)
    ax.set_xticks(x, [m.split(":", 1)[1] for m in maneuvers], fontsize="small")
    ax.set_ylabel("mean AUC-ROC (across sweep configurations)")
    ax.set_title("Which maneuver slice is hardest to simulate faithfully?")
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / "maneuver_slices.png", dpi=150)
    plt.close(fig)

    # 3. Representative-run figures (ROC per slice, reliability, forest).
    if representative:
        summary = representative["summary"]
        artifacts = summary.get("_artifacts")
        if isinstance(artifacts, EvalArtifacts):
            fig, ax = plt.subplots()
            plot_roc_slices(artifacts, ax=ax)
            fig.tight_layout()
            fig.savefig(figures_dir / "roc_slices.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots()
            plot_reliability(artifacts, ax=ax)
            fig.tight_layout()
            fig.savefig(figures_dir / "reliability.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots()
            plot_forest(summary["slices"], ax=ax)
            fig.tight_layout()
            fig.savefig(figures_dir / "forest.png", dpi=150)
            plt.close(fig)

    # 4. Power curve (incl. the AUC = 0.55 headline target).
    fig, ax = plt.subplots()
    plot_power_curve(aucs=(0.55, 0.6, 0.7, 0.8), ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "power_curve.png", dpi=150)
    plt.close(fig)


def write_baseline(representative: dict[str, Any], baseline_path: str | Path | None = None) -> None:
    """Copy the representative run's slice baseline to the baseline JSON path."""
    run_dir = representative.get("run_dir")
    if run_dir is None:
        raise ValueError("no representative run available for the baseline")
    source = Path(run_dir) / "eval" / "baseline.json"
    if not source.exists():
        raise FileNotFoundError(f"{source} missing; run write_outputs on the representative run")
    target = Path(baseline_path) if baseline_path else REPO_ROOT / "artifacts" / "baseline.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target)
    logger.info("baseline written to %s", target)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.run_benchmark",
        description="Sweep simulator realism knobs, train all discriminators, "
        "and regenerate results + figures + baseline + README.",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--n-scenes", type=int, default=16)
    parser.add_argument("--agents-per-scene", type=int, default=4)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="1 config x cnn only (test/dev mode).")
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the sweep; regenerate figures/baseline/readme only.",
    )
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Override the figures output directory (default docs/figures).",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=None,
        help="Override the baseline JSON path (default artifacts/baseline.json).",
    )
    parser.add_argument(
        "--write-readme", action="store_true", help="Regenerate README.md from README.template.md."
    )
    parser.add_argument(
        "--write-throughput", action="store_true", help="Re-measure batch_rollout throughput."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.quick:
        noise_levels = [NOISE_LEVELS[1]]
        latency_settings = [LATENCY_SETTINGS[1]]
        bias_settings = [BIAS_SETTINGS[0]]
        models = ["cnn"]
    else:
        noise_levels = list(NOISE_LEVELS)
        latency_settings = list(LATENCY_SETTINGS)
        bias_settings = list(BIAS_SETTINGS)
        models = list(MODELS)

    if args.skip_sweep:
        out = Path(args.out_dir)
        rows = pq.read_table(out / "results.parquet").to_pylist()
        cid = config_id(*REPRESENTATIVE[:3])
        rep_run = out / "runs" / cid / REPRESENTATIVE[3]
        if not rep_run.exists():
            raise FileNotFoundError(f"representative run {rep_run} missing; rerun the sweep")
        summary = run_evaluation(rep_run, dict(EVAL_CONFIG))
        representative = {
            "run_dir": rep_run,
            "summary": dict(summary),
            "eval_config": dict(EVAL_CONFIG),
        }
        write_outputs(rep_run, summary, dict(EVAL_CONFIG))
    else:
        start = time.monotonic()
        rows, representative = run_sweep(
            args.out_dir,
            n_scenes=args.n_scenes,
            agents_per_scene=args.agents_per_scene,
            duration_s=args.duration_s,
            epochs=args.epochs,
            models=models,
            noise_levels=noise_levels,
            latency_settings=latency_settings,
            bias_settings=bias_settings,
            quick=args.quick,
        )
        logger.info(
            "sweep finished in %.1f min (%d rows)", (time.monotonic() - start) / 60.0, len(rows)
        )
        write_results(args.out_dir, rows)

    if not args.no_figures:
        make_figures(args.out_dir, rows, representative, figures_dir=args.figures_dir)
    write_baseline(representative, baseline_path=args.baseline_path)

    if args.write_throughput:
        from experiments.benchmark_throughput import write_throughput

        write_throughput(Path(args.out_dir) / "throughput.json")

    if args.write_readme:
        from experiments.write_readme import main as write_readme_main

        return write_readme_main(["--results", str(Path(args.out_dir) / "results.parquet")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
