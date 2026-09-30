#!/usr/bin/env python3
"""
How many layers of KV metadata are needed to identify every token?

Reads the npz from profile_layers.py and reports collision counts as layers
are accumulated: layer 0 alone, layers 0-1, layers 0-2, and so on.  Also
reports each layer individually, so you can see whether deeper layers are more
or less discriminative on their own.

Valid at position 0 only.  A single token request attends to itself, so no
other token's information enters the residual stream and every layer stays a
function of the token alone.  From position 1 onward this breaks.

Usage
    python3 analyze_layers.py --path /mnt/f/kvlayers_Qwen25-15B_int8.npz
    python3 analyze_layers.py --path ... --feature v      # V only
    python3 analyze_layers.py --path ... --block 256      # lower RAM
"""

import argparse
import numpy as np


def collisions(feat, block, tol=1e-12):
    """Exact nearest neighbour count, via Gram expansion over row blocks."""
    f = np.ascontiguousarray(feat, dtype=np.float64)
    n = f.shape[0]
    sq = (f * f).sum(1)
    nn = np.empty(n)
    idx = np.empty(n, dtype=np.int64)
    for i in range(0, n, block):
        j = min(i + block, n)
        d2 = sq[i:j, None] + sq[None, :] - 2.0 * (f[i:j] @ f.T)
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(j - i), np.arange(i, j)] = np.inf
        c = d2.argmin(1)
        idx[i:j] = c
        nn[i:j] = np.sqrt(d2[np.arange(j - i), c])
        del d2
    for r in np.nonzero(nn < 1e-5)[0]:
        nn[r] = np.linalg.norm(f[r] - f[idx[r]])
    return int((nn <= tol).sum()), nn


def build(d, feature, mode_asym, lo, hi):
    """Stack layers [lo, hi) into a flat feature matrix."""
    parts = []
    if feature in ("k", "kv"):
        parts.append(d["k_scale"][:, lo:hi, :])
        if mode_asym:
            parts.append(d["k_zp"][:, lo:hi, :])
    if feature in ("v", "kv"):
        parts.append(d["v_scale"][:, lo:hi, :])
        if mode_asym:
            parts.append(d["v_zp"][:, lo:hi, :])
    x = np.concatenate([p.reshape(p.shape[0], -1) for p in parts], axis=1)
    return x


def standardise(x):
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--feature", default="kv", choices=["k", "v", "kv"])
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--max-layers", type=int, default=0,
                    help="stop the cumulative sweep after this many layers")
    ap.add_argument("--per-layer", action="store_true",
                    help="also score each layer on its own")
    args = ap.parse_args()

    d = np.load(args.path, allow_pickle=False)
    keep = d["status"] == 1
    n = int(keep.sum())
    layers = d["layers"]
    nkv = int(d["num_kv_heads"]) if "num_kv_heads" in d else d["k_scale"].shape[2]
    asym = "k_zp" in d
    model = str(d["model"]) if "model" in d else "?"

    sub = {k: d[k][keep] for k in ("k_scale", "v_scale")}
    if asym:
        sub["k_zp"], sub["v_zp"] = d["k_zp"][keep], d["v_zp"][keep]

    per_layer_dims = nkv * (2 if args.feature == "kv" else 1) * (2 if asym else 1)
    print(f"model    {model}")
    print(f"tokens   {n}   kv_heads {nkv}   layers {len(layers)}")
    print(f"feature  {args.feature}{'  (+zp)' if asym else ''}   "
          f"{per_layer_dims} dims per layer")
    print(f"\nlog2(vocab) = {np.log2(n):.1f} bits needed to index the vocabulary\n")

    top = args.max_layers or len(layers)

    print("cumulative: layers 0..L-1 stacked")
    print(f"  {'L':>3} {'dims':>6} {'collisions':>12} {'unique %':>10}")
    first_zero = None
    for L in range(1, top + 1):
        x = build(sub, args.feature, asym, 0, L)
        c, _ = collisions(standardise(x.astype(np.float64)), args.block)
        pct = 100.0 * (n - c) / n
        print(f"  {L:>3} {x.shape[1]:>6} {c:>12} {pct:>9.4f}%")
        if c == 0 and first_zero is None:
            first_zero = L
            break

    if first_zero:
        print(f"\n=> {first_zero} layers suffice for 100% unique identification "
              f"({first_zero * per_layer_dims} dims)")
    else:
        print(f"\n=> still colliding after {top} layers")

    if args.per_layer:
        print("\nindividual layers")
        print(f"  {'layer':>5} {'collisions':>12} {'unique %':>10}")
        for L in range(min(top, len(layers))):
            x = build(sub, args.feature, asym, L, L + 1)
            c, _ = collisions(standardise(x.astype(np.float64)), args.block)
            print(f"  {L:>5} {c:>12} {100.0*(n-c)/n:>9.4f}%")

    print("\nNote: this holds at position 0 only. A single token request attends")
    print("to itself, so no other token enters the residual stream and every")
    print("layer stays a function of the token alone. At position >= 1 the")
    print("deeper layers see the prefix and the static table no longer applies.")


if __name__ == "__main__":
    main()
