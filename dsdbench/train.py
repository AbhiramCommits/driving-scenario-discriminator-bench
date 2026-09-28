"""CLI entrypoint: ``python -m dsdbench.train --model {gbt,cnn,transformer}``.

Trains the requested discriminator on a built benchmark dataset (see
``dsdbench.data.pipeline``) and writes a versioned run directory under
``<artifacts_dir>/<model>/<version>/`` containing:

* ``config.yaml`` -- the resolved run config
* ``metrics.json`` -- CV/train/val/test metrics
* ``predictions.parquet`` -- per-segment predicted probabilities
* ``checkpoint.pt`` (temporal models) or ``model.txt`` + ``gbt_shap.json`` (gbt)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

# Import order matters on macOS: gbt (LightGBM) must load before temporal
# (torch) so both OpenMP runtimes coexist (see dsdbench.models.gbt).
import dsdbench.models.gbt  # noqa: F401  (registers "gbt")
import dsdbench.models.temporal  # noqa: F401  (registers "cnn", "transformer")
from dsdbench.models.registry import available, create


def build_run_dir(artifacts_dir: str | Path, model: str, version: str | None) -> Path:
    version = version or time.strftime("v%Y%m%d-%H%M%S")
    return Path(artifacts_dir) / model / version


def load_config(config_path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text())
    return dict(raw) if raw else {}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dsdbench.train",
        description="Train a dsdbench discriminator model.",
    )
    parser.add_argument("--model", required=True, help="Model name in the registry.")
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to a config YAML (e.g. configs/gbt.yaml)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Override data_dir from the config."
    )
    parser.add_argument(
        "--artifacts-dir", type=Path, default=None, help="Override artifacts_dir from the config."
    )
    parser.add_argument("--version", default=None, help="Run version tag (default: timestamp).")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs (temporal models)."
    )
    args = parser.parse_args(argv)

    if args.model not in available():
        parser.error(f"unknown model {args.model!r}; available: {available()}")

    config = load_config(args.config)
    if args.data_dir is not None:
        config["data_dir"] = str(args.data_dir)
    if args.artifacts_dir is not None:
        config["artifacts_dir"] = str(args.artifacts_dir)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    config.setdefault("arch", args.model)
    if "data_dir" not in config:
        parser.error("config must define data_dir (or pass --data-dir)")

    run_dir = build_run_dir(config.get("artifacts_dir", "artifacts"), args.model, args.version)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    trainer = create(args.model, config)
    metrics = trainer.train(Path(config["data_dir"]), config, run_dir)

    shap_path = run_dir / "gbt_shap.json"
    if shap_path.exists():
        # Also surface the latest SHAP importance at <artifacts_dir>/gbt_shap.json.
        shutil.copy(shap_path, run_dir.parents[1] / "gbt_shap.json")

    print(json.dumps({"model": args.model, "run_dir": str(run_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
