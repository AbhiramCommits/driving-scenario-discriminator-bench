"""Fixed-seed replay: ``python -m dsdbench.replay --run <run_dir>``.

Re-executes the entire simulate -> feature -> train -> eval pipeline from the
artifacts stored in a run directory (train ``config.yaml``, the dataset's
``manifest.json``, and the eval ``config.json``) with all seeds fixed, then
asserts that every reported metric matches the original run to within 1e-6.
Exits non-zero with a loud diff on any mismatch.

Because every stochastic component is seeded deterministically -- synthetic
ingest RNG, per-scenario rollout seeds (independent of thread count),
LightGBM/PyTorch seeds, and per-resample bootstrap/permutation seeds -- the
only expected differences are wall-clock timings (excluded from the
comparison). Determinism guarantees rest on:

* the reproducibility machinery of the simulator (see ``dsdbench._sim``);
* fixed-seed training (see ``dsdbench.models``);
* seeded bootstrap/permutation resampling (see ``dsdbench.eval``).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

import dsdbench.models.gbt  # noqa: F401  (registers "gbt")
import dsdbench.models.temporal  # noqa: F401  (registers "cnn", "transformer")
from dsdbench.data.pipeline import run_pipeline
from dsdbench.evaluate import run_evaluation, write_outputs
from dsdbench.models.registry import create

logger = logging.getLogger(__name__)

TOLERANCE = 1e-6
_EXCLUDED_KEYS = {"elapsed_s"}


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten nested dicts/lists into dotted float paths."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, path))
        elif isinstance(value, list):
            flat[f"{path}#len"] = len(value)
            for i, item in enumerate(value):
                flat[f"{path}[{i}]"] = item
        else:
            flat[path] = value
    return flat


def compare_metrics(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """Compare two metrics dicts to within TOLERANCE; return mismatch messages."""
    exp = _flatten(expected)
    act = _flatten(actual)
    problems: list[str] = []
    for key in sorted(set(exp) | set(act)):
        base = key.split("[")[0].split(".")[0]
        if base in _EXCLUDED_KEYS:
            continue
        if key not in act:
            problems.append(f"{key}: missing in replay ({exp[key]!r})")
            continue
        if key not in exp:
            problems.append(f"{key}: extra in replay ({act[key]!r})")
            continue
        e_value, a_value = exp[key], act[key]
        if isinstance(e_value, float) and isinstance(a_value, float):
            if abs(e_value - a_value) > TOLERANCE:
                problems.append(
                    f"{key}: {e_value} vs {a_value} (diff {abs(e_value - a_value):.3e})"
                )
        elif e_value != a_value:
            problems.append(f"{key}: {e_value!r} vs {a_value!r}")
    return problems


def _load_pipeline_params(data_dir: str) -> dict[str, Any]:
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    if "pipeline_params" not in manifest:
        raise ValueError(
            f"{data_dir}/manifest.json has no pipeline_params (built by an older "
            "dsdbench version); rebuild the dataset with dsdbench.data.pipeline"
        )
    return dict(manifest["pipeline_params"])


def run_replay(run_dir: str | Path, tmp_root: str | Path | None = None) -> list[str]:
    """Replay a run end-to-end and return the list of metric mismatches."""
    run_dir = Path(run_dir)
    train_config: dict[str, Any] = yaml.safe_load((run_dir / "config.yaml").read_text()) or {}
    model = train_config.get("arch") or run_dir.parent.name
    data_dir = train_config.get("data_dir")
    if not data_dir:
        raise ValueError("run config.yaml has no data_dir")

    tmp = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="dsdbench-replay-"))
    replay_data = tmp / "data"
    replay_artifacts = tmp / "artifacts"

    # 1. Simulate + features: rebuild the exact same dataset from the manifest.
    params = _load_pipeline_params(str(data_dir))
    logger.info("replaying pipeline with params: %s", params)
    run_pipeline(replay_data, **params)

    # 2. Train with the same config and seeds into a fresh run directory.
    replay_config = {
        **train_config,
        "data_dir": str(replay_data),
        "artifacts_dir": str(replay_artifacts),
        "epochs": train_config.get("epochs", 20),
    }
    replay_run_dir = replay_artifacts / model / "replay"
    replay_run_dir.mkdir(parents=True, exist_ok=True)
    (replay_run_dir / "config.yaml").write_text(yaml.safe_dump(replay_config, sort_keys=False))
    trainer = create(model, replay_config)
    trainer.train(replay_data, replay_config, replay_run_dir)

    # 3. Evaluate with the same eval settings as the original run.
    eval_config_path = run_dir / "eval" / "config.json"
    eval_config: dict[str, Any] = {}
    if eval_config_path.exists():
        eval_config = json.loads(eval_config_path.read_text())
    summary = run_evaluation(replay_run_dir, eval_config)
    write_outputs(replay_run_dir, summary, eval_config)

    # 4. Compare reported metrics.
    problems: list[str] = []
    problems += compare_metrics(
        json.loads((run_dir / "metrics.json").read_text()),
        json.loads((replay_run_dir / "metrics.json").read_text()),
    )
    if (run_dir / "eval" / "metrics.json").exists():
        problems += compare_metrics(
            json.loads((run_dir / "eval" / "metrics.json").read_text()),
            json.loads((replay_run_dir / "eval" / "metrics.json").read_text()),
        )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsdbench.replay",
        description="Replay a run end-to-end and verify metric reproducibility.",
    )
    parser.add_argument("--run", required=True, type=Path, help="Training run directory.")
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=None,
        help="Scratch directory for the replay (default: system temp).",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true", help="Do not delete the replay scratch directory."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    tmp_root = args.tmp_root
    cleanup: Path | None = None
    if tmp_root is None:
        cleanup = Path(tempfile.mkdtemp(prefix="dsdbench-replay-"))
        tmp_root = cleanup

    try:
        problems = run_replay(args.run, tmp_root)
    except Exception as exc:  # noqa: BLE001 - report loudly and fail
        logger.error("replay failed: %s", exc)
        return 2

    if problems:
        logger.error(
            "REPLAY MISMATCH: %d metric(s) differ by more than %g", len(problems), TOLERANCE
        )
        for message in problems:
            logger.error("  - %s", message)
        return 1
    logger.info("replay PASS: all reported metrics match within %g", TOLERANCE)
    if cleanup is not None and not args.keep_tmp:
        shutil.rmtree(cleanup, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
