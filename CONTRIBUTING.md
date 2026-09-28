# Contributing

Thanks for contributing to dsdbench. The repo is a hybrid C++17/Python project;
please keep both halves honest.

## Setup

```sh
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only, optional
pip install -e ".[dev,ml,eval]"
pre-commit install
```

`[dev]` = pytest/ruff/mypy/pre-commit, `[ml]` = torch/lightgbm/shap/scikit-learn,
`[eval]` = joblib/matplotlib. nuScenes ingestion additionally needs
`pip install "dsdbench[nuscenes]"` and a dataroot.

## Checks before submitting

```sh
pre-commit run --all-files        # ruff, ruff-format, mypy (strict on dsdbench), clang-format
pytest -q                         # Python suite (bindings, data, models, eval, benchmark)
pytest --cov=dsdbench --cov-fail-under=80
cmake -S . -B build -DDSD_BUILD_TESTS=ON && cmake --build build -j
ctest --test-dir build            # C++ unit tests (Catch2)
python scripts/profile_determinism.py
```

CI runs all of the above plus the smoke benchmark and the regression gate
against `artifacts/baseline.json`.

## Ground rules

- **Determinism is a feature.** Every stochastic component is seeded
  (simulator rollouts, training, bootstrap, permutations). If you add a
  source of randomness, it must be seeded and survive
  `scripts/profile_determinism.py` and `dsdbench.replay`.
- **Numbers in the README are generated.** Update `README.template.md`, not
  `README.md`; regenerate with `make reproduce` (or
  `python -m experiments.write_readme` for the tables only).
- **Statistics cite their method.** Every function in `dsdbench/eval/` names
  its method and primary citation in the docstring; keep that up.
- **Regressions gate CI.** If a change alters simulator realism, update the
  committed baseline (`artifacts/baseline.json`) via a documented re-run
  rather than deleting slices.
- Keep the C++ core C++17-clean, warning-free with `-Wall -Wextra`, and
  clang-formatted.
