"""Statistical evaluation harness for real-vs-simulated trajectory discriminators.

Scoring a binary discriminator (label: 0 = real, 1 = simulated) with
rigorous, reproducible statistics: point metrics, scene-clustered BCa
bootstrap confidence intervals, permutation tests, AUC power analysis,
recalibration, slice analysis with FDR control, and regression gating.
"""

from dsdbench.eval import (  # noqa: F401
    bootstrap,
    calibration,
    metrics,
    permutation,
    power,
    slices,
)
