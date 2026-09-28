"""Data layer for the dsdbench benchmark.

Builds the benchmark dataset end-to-end: ingest real trajectories (nuScenes or
a synthetic fallback), generate matched simulated counterparts, label maneuvers,
extract scalar features, and persist everything as Parquet + DuckDB. See
``dsdbench.data.pipeline`` for the entry point.
"""
