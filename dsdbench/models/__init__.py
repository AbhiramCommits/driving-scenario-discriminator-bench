"""ML discriminators for the dsdbench benchmark.

Two families behind a common registry (``dsdbench.models.registry``):

* ``gbt`` -- LightGBM on the tabular segment features, with grouped CV by
  scene, early stopping on the val split, and SHAP feature importance.
* ``cnn`` / ``transformer`` -- PyTorch temporal models on the raw (T, 6)
  trajectory tensors (TemporalCNN, TrajTransformer).

Run any model via ``python -m dsdbench.train --model <name> --config
configs/<name>.yaml``.
"""
