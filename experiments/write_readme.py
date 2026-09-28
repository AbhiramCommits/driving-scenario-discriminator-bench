"""Regenerate README.md from README.template.md using committed artifacts.

Every number in the README results section is computed here (never typed by
hand) from ``experiments/results.parquet``, the measured throughput JSON, and
the power analysis. Run via ``make reproduce`` (or ``python -m
experiments.write_readme``).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dsdbench.eval.power import required_samples

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "README.template.md"
RESULTS = REPO_ROOT / "experiments" / "results.parquet"
THROUGHPUT = REPO_ROOT / "experiments" / "throughput.json"

MID_CONFIG = ("mid", 2, 0.0)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["scope"] == "overall"]


def _slice_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r["scope"] == "slice"]


def auc_table(rows: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    lines = ["| model | mean AUC-ROC | mean 95% BCa CI |"]
    lines.append("| --- | --- | --- |")
    for model in models:
        subset = [r for r in _overall_rows(rows) if r["model"] == model]
        mean_auc = _mean([r["auc"] for r in subset])
        mean_low = _mean([r["ci_low"] for r in subset])
        mean_high = _mean([r["ci_high"] for r in subset])
        lines.append(f"| {model} | {mean_auc:.3f} | [{mean_low:.3f}, {mean_high:.3f}] |")
    return "\n".join(lines)


def ece_table(rows: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    lines = ["| model | ECE before | ECE after isotonic |"]
    lines.append("| --- | --- | --- |")
    for model in models:
        subset = [
            r
            for r in _overall_rows(rows)
            if r["model"] == model
            and (r["knob_noise"], r["knob_latency"], r["knob_bias"]) == MID_CONFIG
        ]
        if not subset:
            raise ValueError(f"no mid-config row for model {model}")
        row = subset[0]
        lines.append(f"| {model} | {row['ece']:.4f} | {row['ece_after_isotonic']:.4f} |")
    return "\n".join(lines)


def slice_table(rows: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    maneuvers = sorted({r["slice_name"] for r in _slice_rows(rows)})
    lines = ["| maneuver | " + " | ".join(models) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in models) + " |")
    for maneuver in maneuvers:
        cells = []
        for model in models:
            subset = [
                r for r in _slice_rows(rows) if r["slice_name"] == maneuver and r["model"] == model
            ]
            cells.append(f"{_mean([r['auc'] for r in subset]):.3f}")
        lines.append(f"| {maneuver.split(':', 1)[1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def knob_table(rows: list[dict[str, Any]]) -> str:
    """Mean overall AUC per knob level, averaged over models and other knobs."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in _overall_rows(rows):
        grouped[("noise", str(r["knob_noise"]))].append(r["auc"])
        grouped[("latency", str(r["knob_latency"]))].append(r["auc"])
        grouped[("bias", f"{r['knob_bias']:g}")].append(r["auc"])
    lines = ["| knob | level | mean AUC |"]
    lines.append("| --- | --- | --- |")
    for (knob, level), values in sorted(grouped.items()):
        lines.append(f"| {knob} | {level} | {_mean(values):.3f} |")
    return "\n".join(lines)


def worst_knob(rows: list[dict[str, Any]]) -> str:
    """The knob level with the largest mean-AUC difference vs. its best level."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in _overall_rows(rows):
        grouped["noise"][str(r["knob_noise"])].append(r["auc"])
        grouped["latency"][str(r["knob_latency"])].append(r["auc"])
        grouped["bias"][f"{r['knob_bias']:g}"].append(r["auc"])
    worst_name, worst_delta = "", -1.0
    for knob, levels in grouped.items():
        means = {level: _mean(v) for level, v in levels.items()}
        spread = max(means.values()) - min(means.values())
        if spread > worst_delta:
            worst_delta = spread
            worst_name = knob
    return f"{worst_name} (mean-AUC spread {worst_delta:.3f} across its levels)"


def hardest_maneuver(rows: list[dict[str, Any]]) -> str:
    by_maneuver: dict[str, list[float]] = defaultdict(list)
    for r in _slice_rows(rows):
        by_maneuver[r["slice_name"]].append(r["auc"])
    hardest = max(by_maneuver, key=lambda m: _mean(by_maneuver[m]))
    return (
        f"{hardest.split(':', 1)[1]} (mean AUC {_mean(by_maneuver[hardest]):.3f} "
        f"across configurations and models)"
    )


def render(template_text: str, replacements: dict[str, str]) -> str:
    out = template_text
    for key, value in replacements.items():
        token = "{{" + key + "}}"
        if token not in out:
            raise ValueError(f"template has no placeholder {token}")
        out = out.replace(token, value)
    leftovers = [line for line in out.splitlines() if "{{" in line]
    if leftovers:
        raise ValueError(f"unfilled placeholders: {leftovers}")
    return out


def build_replacements(rows: list[dict[str, Any]], throughput: dict[str, Any]) -> dict[str, str]:
    models = tuple(sorted({r["model"] for r in rows}))
    power_n = required_samples(0.55, alpha=0.05, power=0.8)
    return {
        "AUC_TABLE": auc_table(rows, models),
        "ECE_TABLE": ece_table(rows, models),
        "SLICE_TABLE": slice_table(rows, models),
        "KNOB_TABLE": knob_table(rows),
        "KNOB_WORST": worst_knob(rows),
        "MANEUVER_HARDEST": hardest_maneuver(rows),
        "POWER_MIN_N": str(power_n),
        "TPUT_TABLE": (
            "| threads | rollouts/s |\n"
            "| --- | --- |\n"
            f"| 1 | {throughput['rollouts_per_sec_n_threads_1']:.1f} |\n"
            f"| 8 | {throughput['rollouts_per_sec_n_threads_8']:.1f} |"
        ),
        "TPUT_SPEEDUP": f"{throughput['speedup_8_vs_1']:.1f}x",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.write_readme",
        description="Regenerate README.md from README.template.md and the "
        "committed benchmark artifacts.",
    )
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--throughput", type=Path, default=THROUGHPUT)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "README.md")
    args = parser.parse_args(argv)

    rows = pq.read_table(args.results).to_pylist()
    throughput = json.loads(args.throughput.read_text())
    rendered = render(args.template.read_text(), build_replacements(rows, throughput))
    args.out.write_text(rendered)
    print(f"wrote {args.out} ({len(rows)} result rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
