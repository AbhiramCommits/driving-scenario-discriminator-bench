.PHONY: reproduce reproduce-quick test

# Regenerate every number, table, and figure in the README from scratch:
# sweep the realism knobs, retrain all discriminators, re-evaluate, measure
# throughput, regenerate docs/figures + artifacts/baseline.json + README.md.
reproduce:
	python -m experiments.run_benchmark --write-readme --write-throughput

# Fast sanity version (1 config x CNN, no README rewrite).
reproduce-quick:
	python -m experiments.run_benchmark --quick

test:
	pytest -q
