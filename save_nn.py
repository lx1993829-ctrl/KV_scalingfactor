#!/usr/bin/env python3
"""
Compute per-token nearest-neighbour distances and save them for plotting.

The collision counts in analyze_kv_meta.py summarise the left tail of this
distribution.  Saving the full distance vector lets you show the thing the
counts hide: whether separability rests on a wide gap or a narrow one, and
whether a model that fails does so because tokens are genuinely degenerate
(mass at exactly zero) or because the metadata has too little resolution
(mass piled just above zero, discretised by the bf16 grid of the underlying
activations).

Output npz, one per (model, quantization mode, feature set)
    nn_dist      [n] float64   distance to nearest other token
    nn_index     [n] int64     which token that was, as a vocabulary id
    token_id     [n] int64     vocabulary id of each row
    norm         [n] float64   L2 norm of each token's feature vector
    median_norm  scalar        for normalising across models
    feature, mode, model, dims  metadata

Usage
    python3 save_nn.py --path /mnt/f/kvmeta_int8.npz --feature v
    python3 save_nn.py --path /mnt/f/kvmeta_int4.npz --feature kv --tag int4
    for f in /mnt/f/kvmeta_*.npz; do python3 save_nn.py --path $f --feature v; done
"""

import os
import argparse
import numpy as np


def nn_search(feat, block):
    """
    Nearest neighbour excluding self, via  ||a-b||^2 = |a|^2 + |b|^2 - 2ab.

    The expansion loses precision near zero, so anything landing in the bottom
    of the range is recomputed by direct subtraction.  That keeps the left tail
    (the part the figure is about) trustworthy while the bulk keeps the speed
    of a matmul.
    """
    f = np.ascontiguousarray(feat, dtype=np.float64)
    n = f.shape[0]
    sq = (f * f).sum(1)
    nn_d = np.empty(n)
    nn_i = np.empty(n, dtype=np.int64)

    for i in range(0, n, block):
        j = min(i + block, n)
        d2 = sq[i:j, None] + sq[None, :] - 2.0 * (f[i:j] @ f.T)
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(j - i), np.arange(i, j)] = np.inf
        c = d2.argmin(1)
        nn_i[i:j] = c
        nn_d[i:j] = np.sqrt(d2[np.arange(j - i), c])
        del d2
        if (i // block) % 20 == 0:
            print(f"    {i}/{n}", end="\r")

    thresh = np.percentile(nn_d, 5.0)
    small = np.nonzero(nn_d <= max(thresh, 1e-5))[0]
    print(f"\n    recomputing {len(small)} small distances exactly")
    for r in small:
        nn_d[r] = np.linalg.norm(f[r] - f[nn_i[r]])
    return nn_d, nn_i


def build(d, feature):
    """Assemble the feature matrix. zp is standardised in, never raw."""
    asym = "k_zp" in d
    ks, vs = d["k_scale"], d["v_scale"]
    if feature == "k":
        parts = [ks] + ([d["k_zp"]] if asym else [])
    elif feature == "v":
        parts = [vs] + ([d["v_zp"]] if asym else [])
    else:
        parts = [ks, vs] + ([d["k_zp"], d["v_zp"]] if asym else [])
    x = np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1)
    if asym:
        # scale ~1e-2 against zp 0-15: without standardising, zp is the metric
        mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
        sd[sd == 0] = 1.0
        x = (x - mu) / sd
    return x.astype(np.float64), asym


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--feature", default="v", choices=["k", "v", "kv"])
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None,
                    help="short label for the figure legend")
    args = ap.parse_args()

    d = np.load(args.path, allow_pickle=False)
    keep = d["status"] == 1
    ids = np.nonzero(keep)[0]
    sub = {k: d[k][keep] for k in d.files
           if k in ("k_scale", "v_scale", "k_zp", "v_zp")}

    model = str(d["model"]) if "model" in d else "?"
    mode = str(d["kv_dtype"]) if "kv_dtype" in d else "?"
    tag = args.tag or f"{model.split('/')[-1]}_{mode.split('_')[0]}"

    x, asym = build(sub, args.feature)
    print(f"{model}  {mode}  feature={args.feature}  "
          f"{x.shape[1]} dims  {x.shape[0]} tokens"
          + ("  (standardised, includes zp)" if asym else ""))

    nn_d, nn_i = nn_search(x, args.block)
    norms = np.linalg.norm(x, axis=1)
    med = float(np.median(norms))

    out = args.out or args.path.replace(".npz", f"_nn_{args.feature}.npz")
    np.savez(out,
             nn_dist=nn_d, nn_index=ids[nn_i], token_id=ids,
             norm=norms, median_norm=med,
             feature=args.feature, mode=mode, model=model,
             dims=x.shape[1], tag=tag)

    zero = int((nn_d == 0.0).sum())
    q = np.percentile(nn_d, [0.1, 1, 5, 50, 95])
    print(f"  exact zeros      {zero}  ({100.0*zero/len(nn_d):.4f}%)")
    print(f"  median norm      {med:.6e}")
    print(f"  nn/norm  .1/1/5/50/95  " +
          "  ".join(f"{v/med:.4f}" for v in q))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
