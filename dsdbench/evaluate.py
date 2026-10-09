"""Evaluation CLI: ``python -m dsdbench.evaluate --run <run_dir>``.

Scores the discriminator checkpoint stored in a training run directory:
overall metrics with scene-level BCa bootstrap CIs, a permutation test
against chance, recalibration (fit on val, applied to test), per-slice
metrics with Benjamini-Hochberg FDR control, and an optional regression
check against a stored baseline JSON (exit code 1 when flagged, so it can
gate CI). Writes ``report.md``, matplotlib figures, ``metrics.json`` and a
fresh ``baseline.json`` into ``<run_dir>/eval/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

from dsdbench.eval import bootstrap as bootstrap_mod
from dsdbench.eval import metrics as metrics_mod
from dsdbench.eval.calibration import RecalibrationResult, recalibrate
from dsdbench.eval.permutation import permutation_test
from dsdbench.eval.report import EvalArtifacts, build_report, save_figures
from dsdbench.eval.slices import (
    DEFAULT_MIN_SLICE_N,
    SliceConfig,
    SliceReport,
    baseline_from_report,
    build_slices,
    check_regression,
    run_slices,
)

logger = logging.getLogger(__name__)

DEFAULTS = {
    "n_resamples": 2000,
    "n_permutations": 1000,
    "n_bins": 15,
    "n_jobs": -1,
    "alpha": 0.05,
    "regression_threshold": 0.05,
    "min_slice_n": DEFAULT_MIN_SLICE_N,
    "seed": 0,
}


def load_run(
    run_dir: str | Path,
) -> tuple[dict[str, list[Any]], dict[str, Any], dict[str, list[Any]]]:
    """Load predictions + train config + dataset labels for a run directory."""
    run_dir = Path(run_dir)
    predictions = pq.read_table(run_dir / "predictions.parquet").to_pydict()
    config = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    data_dir = config.get("data_dir")
    if data_dir is None:
        raise ValueError("run config.yaml has no data_dir; cannot resolve dataset labels")
    labels = pq.read_table(Path(data_dir) / "labels.parquet").to_pydict()
    return predictions, config, labels


def run_evaluation(
    run_dir: str | Path,
    eval_config: dict[str, Any],
    *,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the full evaluation summary for a trained run (deterministic).

    The returned dict is JSON-serializable except for two private keys used by
    ``write_outputs``: ``_artifacts`` (figure inputs) and ``_slice_report``
    (the SliceReport used to serialize the baseline).
    """
    cfg = {**DEFAULTS, **eval_config}
    predictions, _train_config, labels = load_run(run_dir)

    y = np.asarray([1.0 if s == "sim" else 0.0 for s in predictions["source"]], dtype=np.float64)
    probs = np.asarray(predictions["prob_sim"], dtype=np.float64)
    scenes = np.asarray(predictions["scene_id"], dtype=object)
    split = np.asarray(predictions["split"], dtype=object)

    split_used = "train"
    for candidate in ("test", "val", "train"):
        if (split == candidate).any():
            split_used = candidate
            break
    eval_mask = split == split_used
    val_mask = split == "val"
    # Slice analysis (and its figures) use the evaluation split only.
    eval_idx = np.flatnonzero(eval_mask)
    predictions_eval = {key: [value[i] for i in eval_idx] for key, value in predictions.items()}

    y_eval, p_eval, c_eval = y[eval_mask], probs[eval_mask], scenes[eval_mask]

    metrics: dict[str, Any] = {
        "n": int(y_eval.size),
        "n_pos": int(np.sum(y_eval == 1.0)),
        "n_neg": int(np.sum(y_eval == 0.0)),
        "n_bins": int(cfg["n_bins"]),
        "auc_roc": metrics_mod.auc_roc(y_eval, p_eval),
        "auc_pr": metrics_mod.auc_pr(y_eval, p_eval),
        "brier": metrics_mod.brier_score(y_eval, p_eval),
        "ece": metrics_mod.expected_calibration_error(y_eval, p_eval, n_bins=cfg["n_bins"]),
        "mce": metrics_mod.maximum_calibration_error(y_eval, p_eval, n_bins=cfg["n_bins"]),
        "youden": metrics_mod.youden_accuracy(y_eval, p_eval).to_dict(),
    }
    auc_ci = bootstrap_mod.bootstrap_auc(
        y_eval,
        p_eval,
        c_eval,
        n_resamples=cfg["n_resamples"],
        seed=cfg["seed"],
        alpha=cfg["alpha"],
        n_jobs=cfg["n_jobs"],
    )
    metrics["auc_roc_ci"] = {"low": auc_ci.ci_low, "high": auc_ci.ci_high}
    ece_ci = bootstrap_mod.bootstrap_ece(
        y_eval,
        p_eval,
        c_eval,
        n_bins=cfg["n_bins"],
        n_resamples=cfg["n_resamples"],
        seed=cfg["seed"],
        alpha=cfg["alpha"],
        n_jobs=cfg["n_jobs"],
    )
    metrics["ece_ci"] = {"low": ece_ci.ci_low, "high": ece_ci.ci_high}
    metrics["permutation"] = permutation_test(
        y_eval,
        p_eval,
        bootstrap_mod.auc_statistic,
        n_permutations=cfg["n_permutations"],
        seed=cfg["seed"],
        n_jobs=cfg["n_jobs"],
    ).to_dict()

    recal: RecalibrationResult | None
    if val_mask.any():
        recal = recalibrate(y[val_mask], probs[val_mask], y_eval, p_eval, n_bins=cfg["n_bins"])
        metrics["calibration"] = recal.to_dict()
    else:
        recal = None
        metrics["calibration"] = None

    slice_config = SliceConfig(
        alpha=cfg["alpha"],
        n_bins=cfg["n_bins"],
        n_resamples=cfg["n_resamples"],
        n_permutations=cfg["n_permutations"],
        seed=cfg["seed"],
        n_jobs=cfg["n_jobs"],
    )
    slice_report = run_slices(predictions_eval, labels, slice_config)

    regression: dict[str, Any] | None = None
    if baseline is not None:
        regression = check_regression(
            slice_report,
            baseline,
            threshold=cfg["regression_threshold"],
            min_slice_n=cfg.get("min_slice_n", DEFAULT_MIN_SLICE_N),
        ).to_dict()

    summary: dict[str, Any] = {
        "split_used": split_used,
        "metrics": metrics,
        "slices": [r.to_dict() for r in slice_report.results],
        "skipped_slices": slice_report.skipped,
        "regression": regression,
        "_slice_report": slice_report,
    }
    if recal is not None:
        masks = build_slices(predictions_eval, labels)
        names = {r.name for r in slice_report.results}
        slice_data = {
            name: (y_eval[mask], p_eval[mask]) for name, mask in masks.items() if name in names
        }
        summary["_artifacts"] = EvalArtifacts(
            y_eval=y_eval,
            probs_eval=p_eval,
            slice_data=slice_data,
            recalibration=recal,
            n_bins=cfg["n_bins"],
            split_used=split_used,
            n_bootstrap=cfg["n_resamples"],
        )
    return summary


def write_outputs(
    run_dir: str | Path, summary: dict[str, Any], eval_config: dict[str, Any]
) -> Path:
    """Write metrics.json, baseline.json, config.json, report.md and figures."""
    out_dir = Path(run_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = summary.pop("_artifacts", None)
    slice_report: SliceReport = summary.pop("_slice_report")
    if artifacts is not None:
        save_figures(artifacts, summary, out_dir)
    (out_dir / "report.md").write_text(build_report(summary, run_dir))
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out_dir / "config.json").write_text(json.dumps(eval_config, indent=2, sort_keys=True))
    baseline = baseline_from_report(slice_report)
    (out_dir / "baseline.json").write_text(json.dumps(baseline, indent=2, sort_keys=True))
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsdbench.evaluate",
        description="Statistically evaluate a dsdbench discriminator run.",
    )
    parser.add_argument("--run", required=True, type=Path, help="Training run directory.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Stored baseline JSON to check for regression (exit 1 if flagged).",
    )
    parser.add_argument("--n-resamples", type=int, default=DEFAULTS["n_resamples"])
    parser.add_argument("--n-permutations", type=int, default=DEFAULTS["n_permutations"])
    parser.add_argument("--n-bins", type=int, default=DEFAULTS["n_bins"])
    parser.add_argument("--n-jobs", type=int, default=DEFAULTS["n_jobs"])
    parser.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULTS["regression_threshold"],
        help="AUC drop below baseline that flags a regression.",
    )
    parser.add_argument(
        "--min-slice-n",
        type=int,
        default=DEFAULTS["min_slice_n"],
        help="Slices with fewer segments (baseline or current) are not regression-checked.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    eval_config: dict[str, Any] = {
        "n_resamples": args.n_resamples,
        "n_permutations": args.n_permutations,
        "n_bins": args.n_bins,
        "n_jobs": args.n_jobs,
        "alpha": args.alpha,
        "regression_threshold": args.threshold,
        "min_slice_n": args.min_slice_n,
        "seed": args.seed,
    }
    baseline = None
    if args.baseline is not None:
        baseline = json.loads(Path(args.baseline).read_text())

    summary = run_evaluation(args.run, eval_config, baseline=baseline)
    out_dir = write_outputs(args.run, summary, eval_config)
    logger.info("evaluation written to %s", out_dir)

    regression = summary.get("regression")
    if regression is not None and regression["flag"]:
        logger.error(
            "REGRESSION FLAGGED: %d violation(s):\n%s",
            len(regression["violations"]),
            json.dumps(regression["violations"], indent=2),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
