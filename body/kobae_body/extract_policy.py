"""Extract a flybody DMPO policy (TF SavedModel) into a numpy .npz so it can run without TensorFlow.

Architecture (Acme LayerNormMLP + MultivariateNormalDiagHead, Sonnet creation order):
  Linear1 (b,w) -> LayerNorm (offset, scale) -> tanh -> Linear2..4 (b,w) with tanh -> mean head (b,w), [std head unused]
Usage: python -m kobae_body.extract_policy data/flybody/policies/flight body/policies/flight.npz
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


def main(src: str, dst: str):
    reader = tf.train.load_checkpoint(str(Path(src) / "variables" / "variables"))
    v = {}
    for k in reader.get_variable_to_shape_map():
        if k == "_CHECKPOINTABLE_OBJECT_GRAPH":
            continue
        v[int(k.split("/")[1])] = reader.get_tensor(k)
    for i in sorted(v):
        print(f"  v{i:>2} {str(v[i].shape):16s}")
    layers = []
    # pairs of (b, w) in creation order, skipping the layernorm pair
    idx = sorted(v)
    i = 0
    ln = None
    while i < len(idx):
        a, b = v[idx[i]], v[idx[i + 1]]
        if a.ndim == 1 and b.ndim == 2:            # Linear: b then w
            layers.append((b, a)); i += 2
        elif a.ndim == 1 and b.ndim == 1:          # LayerNorm: offset then scale
            ln = (a, b); i += 2
        else:
            raise RuntimeError(f"unexpected pair at {idx[i]}: {a.shape} {b.shape}")
    hidden = [l for l in layers if l[0].shape[1] == layers[0][0].shape[1]]
    heads = [l for l in layers if l[0].shape[1] != layers[0][0].shape[1]]
    mean_w, mean_b = heads[0]
    out = {"ln_offset": ln[0], "ln_scale": ln[1], "mean_w": mean_w, "mean_b": mean_b, "n_hidden": len(hidden)}
    for k, (w, b) in enumerate(hidden):
        out[f"w{k}"] = w; out[f"b{k}"] = b
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    print(f"obs {hidden[0][0].shape[0]} -> {len(hidden)}x{hidden[0][0].shape[1]} -> act {mean_w.shape[1]}  saved {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
