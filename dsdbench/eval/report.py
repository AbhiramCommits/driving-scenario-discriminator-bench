"""Report generation: markdown report plus matplotlib figures.

Produces, for one evaluation:

* ``roc_slices.png`` -- ROC curves for the overall test set and every slice;
* ``reliability.png`` -- reliability diagram before and after recalibration
  (equal-mass bins, Naeini et al. 2015);
* ``forest.png`` -- AUC with scene-level BCa CIs per slice (forest plot);
* ``power_curve.png`` -- achieved power vs. sample size (Hanley & McNeil 1982).

References are those of the modules whose numbers are visualized
(``dsdbench.eval.metrics``, ``bootstrap``, ``power``, ``calibration``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsdbench.eval.calibration import RecalibrationResult
from dsdbench.eval.power import plot_power_curve

_FIGURE_NAMES = ("roc_slices.png", "reliability.png", "forest.png", "power_curve.png")


@dataclass
class EvalArtifacts:
    """Raw evaluation objects needed for figures (not JSON-serializable)."""

    y_eval: np.ndarray
    probs_eval: np.ndarray
    slice_data: dict[str, tuple[np.ndarray, np.ndarray]]  # name -> (y, probs)
    recalibration: RecalibrationResult
    n_bins: int
    split_used: str
    n_bootstrap: int


def _roc_curve(y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y, probs)
    return fpr, tpr


def plot_roc_slices(artifacts: EvalArtifacts, ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    fpr, tpr = _roc_curve(artifacts.y_eval, artifacts.probs_eval)
    ax.plot(fpr, tpr, color="black", linewidth=2, label=f"overall ({artifacts.split_used})")
    for name, (y_s, p_s) in sorted(artifacts.slice_data.items()):
        fpr_s, tpr_s = _roc_curve(y_s, p_s)
        ax.plot(fpr_s, tpr_s, linewidth=1, alpha=0.8, label=name)
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC per slice")
    ax.legend(fontsize="small", loc="lower right")
    ax.grid(alpha=0.3)
    return ax


def plot_reliability(artifacts: EvalArtifacts, ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    from dsdbench.eval.metrics import _calibration_bins

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    rec = artifacts.recalibration
    curves = {
        "before": (artifacts.probs_eval, rec.ece_before),
        "after Platt": (rec.probs_platt, rec.ece_after_platt),
        "after isotonic": (rec.probs_isotonic, rec.ece_after_isotonic),
    }
    for label, (probs, ece_value) in curves.items():
        _, mean_pred, frac_pos = _calibration_bins(artifacts.y_eval, probs, artifacts.n_bins)
        ax.plot(
            mean_pred, frac_pos, marker="o", markersize=4, label=f"{label} (ECE={ece_value:.4f})"
        )
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean predicted probability (equal-mass bin)")
    ax.set_ylabel("Fraction of simulated segments")
    ax.set_title("Reliability diagram before/after recalibration")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.3)
    return ax


def plot_forest(slice_results: list[dict[str, Any]], ax: Any = None) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(slice_results))))
    ordered = sorted(slice_results, key=lambda r: r["auc"])
    names = [r["name"] for r in ordered]
    aucs = np.asarray([r["auc"] for r in ordered])
    # With few scenes the BCa endpoints can cross the estimate; keep bars sane.
    lows = np.minimum(np.asarray([r["ci_low"] for r in ordered]), aucs)
    highs = np.maximum(np.asarray([r["ci_high"] for r in ordered]), aucs)
    ys = np.arange(len(ordered))
    xerr = np.vstack([aucs - lows, highs - aucs])
    ax.errorbar(aucs, ys, xerr=xerr, fmt="o", capsize=3, color="tab:blue")
    ax.axvline(0.5, color="0.5", linestyle="--", linewidth=1, label="chance")
    ax.set_yticks(ys, names, fontsize="small")
    ax.set_xlabel("AUC-ROC with 95% BCa CI (scene-level bootstrap)")
    ax.set_title("Discrimination by slice")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize="small", loc="lower right")
    return ax


def save_figures(artifacts: EvalArtifacts, summary: dict[str, Any], out_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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

    fig, ax = plt.subplots()
    plot_power_curve(ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "power_curve.png", dpi=150)
    plt.close(fig)


def build_report(summary: dict[str, Any], run_dir: str | Path) -> str:
    """Render the evaluation summary into a markdown report."""
    lines: list[str] = []
    lines.append("# dsdbench evaluation report")
    lines.append("")
    lines.append(f"- run: `{run_dir}`")
    lines.append(f"- split used for evaluation: `{summary['split_used']}`")
    lines.append("")
    lines.append("## Overall metrics")
    lines.append("")
    m = summary["metrics"]
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in ("n", "n_pos", "n_neg"):
        lines.append(f"| {key} | {m[key]} |")
    lines.append(f"| auc_roc | {m['auc_roc']:.4f} |")
    lines.append(
        f"| auc_roc 95% BCa CI | [{m['auc_roc_ci']['low']:.4f}, {m['auc_roc_ci']['high']:.4f}] |"
    )
    lines.append(f"| auc_pr | {m['auc_pr']:.4f} |")
    lines.append(
        f"| accuracy @ Youden threshold ({m['youden']['threshold']:.3f}) | "
        f"{m['youden']['accuracy']:.4f} |"
    )
    lines.append(f"| brier | {m['brier']:.4f} |")
    lines.append(f"| ece (n_bins={m['n_bins']}) | {m['ece']:.4f} |")
    lines.append(f"| ece 95% BCa CI | [{m['ece_ci']['low']:.4f}, {m['ece_ci']['high']:.4f}] |")
    lines.append(f"| mce | {m['mce']:.4f} |")
    lines.append(
        f"| permutation p (vs chance, n={m['permutation']['n_permutations']}) | "
        f"{m['permutation']['p_value']:.4f} |"
    )
    lines.append("")
    lines.append("## Calibration (fitted on val, applied to test)")
    lines.append("")
    cal = m["calibration"]
    lines.append("| method | ECE | MCE |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| before | {cal['ece_before']:.4f} | {cal['mce_before']:.4f} |")
    lines.append(f"| after Platt | {cal['ece_after_platt']:.4f} | {cal['mce_after_platt']:.4f} |")
    lines.append(
        f"| after isotonic | {cal['ece_after_isotonic']:.4f} | {cal['mce_after_isotonic']:.4f} |"
    )
    lines.append("")
    lines.append("## Slices (Benjamini-Hochberg FDR across slice tests)")
    lines.append("")
    lines.append("| slice | n | AUC | 95% CI | p | p_adj |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for result in sorted(summary["slices"], key=lambda r: r["name"]):
        lines.append(
            f"| {result['name']} | {result['n']} | {result['auc']:.4f} | "
            f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}] | "
            f"{result['p_perm']:.4f} | {result['p_adj']:.4f} |"
        )
    if summary.get("skipped_slices"):
        lines.append("")
        lines.append(f"skipped (fewer than 2 per class): {', '.join(summary['skipped_slices'])}")
    lines.append("")
    lines.append("## Regression check")
    lines.append("")
    reg = summary.get("regression")
    if reg is None:
        lines.append("no baseline provided; run with `--baseline` to gate CI.")
    elif reg["flag"]:
        lines.append(f"**REGRESSION FLAGGED** (threshold={reg['threshold']}):")
        for violation in reg["violations"]:
            lines.append(f"- {violation}")
    else:
        lines.append("no regression detected against the stored baseline.")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for name in _FIGURE_NAMES:
        lines.append(f"![{name}](figures/{name})")
    lines.append("")
    return "\n".join(lines)
