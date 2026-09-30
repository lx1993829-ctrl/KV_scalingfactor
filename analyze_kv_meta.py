#!/usr/bin/env python3
"""
Separability analysis for vLLM per-token-head KV metadata.

Reads an npz from profile_kv_meta.py and reports how uniquely each vocabulary
token is identified by the metadata vLLM stores for it.

CPU only.  Peak RAM is about  block * n_tokens * 8 bytes  (default ~0.5 GB at
128k tokens), because the nearest-neighbour search uses a Gram-matrix
expansion over small row blocks rather than a full broadcast.

Feature sets
    symmetric modes (int8, fp8)
        K scale        8 dims
        V scale        8 dims
        K+V raw       16 dims, reported only to show K dominating V
        K+V z         16 dims, per-dimension standardised
        K+V log       16 dims, log10 then standardised

    asymmetric mode (int4)
        K scale        8 dims
        V scale        8 dims
        V scale+zp    16 dims   does the zero point add anything over scale
        K+V scale     16 dims, standardised
        K+V scale+zp  32 dims, standardised

Raw concatenation of scale and zp is never reported: zp is an integer 0-15
while scale is order 1e-2, so Euclidean distance on the concatenation would be
pure zp.  Everything mixing the two is standardised first.

Usage
    python3 analyze_kv_meta.py --path /mnt/f/kvmeta_int4.npz
    python3 analyze_kv_meta.py --path /mnt/f/kvmeta_fp8.npz --block 256
    python3 analyze_kv_meta.py --path a.npz --compare b.npz
"""

import argparse
import numpy as np


def nn_search(feat, block, exact_tol, near_tol):
    """
    Exact nearest neighbour excluding self, via  ||a-b||^2 = |a|^2 + |b|^2 - 2ab.

    That expansion loses precision near zero, so any pair landing below a
    generous multiple of near_tol is recomputed by direct subtraction before
    being reported.  Collision counts stay trustworthy; the bulk distances keep
    the speed of a matmul.
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
        col = d2.argmin(1)
        nn_i[i:j] = col
        nn_d[i:j] = np.sqrt(d2[np.arange(j - i), col])
        del d2

    for r in np.nonzero(nn_d < near_tol * 10.0)[0]:
        nn_d[r] = np.linalg.norm(f[r] - f[nn_i[r]])

    return nn_d, nn_i, int((nn_d <= exact_tol).sum()), int((nn_d <= near_tol).sum())


def byte_duplicates(feat):
    v = np.ascontiguousarray(feat).view(
        np.dtype((np.void, feat.dtype.itemsize * feat.shape[1]))).ravel()
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    return int((cnt[inv] > 1).sum())


def standardise(x):
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return (x - mu) / sd


def report(name, feat, idx, args, results, special_cut=None):
    f = feat.astype(np.float64)
    n, d = f.shape
    ndup = byte_duplicates(np.ascontiguousarray(feat, dtype=np.float32))
    nn_d, nn_i, exact, near = nn_search(f, args.block, args.exact_tol, args.near_tol)

    typical = np.median(np.linalg.norm(f, axis=1))
    med = np.median(nn_d)
    margin = med / typical if typical > 0 else float("nan")

    print(f"\n=== {name}   ({d} dims, {n} tokens) ===")
    print(f"  bit-identical rows   {ndup}")
    print(f"  exact collisions     {exact}   (<= {args.exact_tol:g})")
    print(f"  near collisions      {near}   (<= {args.near_tol:g})")
    print(f"  unique rate          {100.0 * (n - exact) / n:.4f}%")
    print(f"  median NN distance   {med:.6e}")
    print(f"  median vector norm   {typical:.6e}")
    print(f"  separation margin    {margin:.4f}")
    q = np.percentile(nn_d, [0.1, 1, 5, 50])
    print(f"  NN pct .1/1/5/50     {q[0]:.3e} {q[1]:.3e} {q[2]:.3e} {q[3]:.3e}")

    # how many collisions involve only ordinary vocabulary?
    if special_cut is not None:
        coll = np.nonzero(nn_d <= args.exact_tol)[0]
        both_real = sum(1 for r in coll
                        if idx[r] < special_cut and idx[nn_i[r]] < special_cut)
        print(f"  of {exact} collisions, {both_real} have both ids < "
              f"{special_cut} (ordinary vocabulary)")

    worst = np.argsort(nn_d)[:6]
    print("  closest pairs:")
    for r in worst:
        print(f"     {idx[r]:>7d} <-> {idx[nn_i[r]]:>7d}   d={nn_d[r]:.6e}")

    results[name] = {"dims": d, "exact": exact, "margin": margin, "n": n}


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {"k_scale": d["k_scale"], "v_scale": d["v_scale"],
           "status": d["status"]}
    out["asym"] = "k_zp" in d
    if out["asym"]:
        out["k_zp"], out["v_zp"] = d["k_zp"], d["v_zp"]
    out["mode"] = str(d["kv_dtype"]) if "kv_dtype" in d else "unknown"
    out["qmax"] = float(d["quant_max"]) if "quant_max" in d else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--compare", default=None,
                    help="second npz; prints a side-by-side summary at the end")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--exact-tol", type=float, default=1e-12)
    ap.add_argument("--near-tol", type=float, default=1e-6)
    ap.add_argument("--special-cut", type=int, default=128000,
                    help="ids at or above this are treated as special tokens")
    args = ap.parse_args()

    def run(path):
        D = load(path)
        keep = D["status"] == 1
        idx = np.nonzero(keep)[0]
        K, V = D["k_scale"][keep], D["v_scale"][keep]
        qm = D["qmax"]

        print(f"\n{'='*70}\nfile   {path}\nmode   {D['mode']}  "
              f"QUANT_MAX={qm}  {'asymmetric' if D['asym'] else 'symmetric'}")
        print(f"rows   {len(D['status'])}   clean {int(keep.sum())}   "
              f"ambiguous {int((D['status']==2).sum())}   "
              f"missing {int((D['status']==0).sum())}")
        print(f"K scale range [{K.min():.3e}, {K.max():.3e}]   "
              f"V scale range [{V.min():.3e}, {V.max():.3e}]")

        res = {}
        report("K scale", K, idx, args, res, args.special_cut)
        report("V scale", V, idx, args, res, args.special_cut)

        if D["asym"]:
            KZ, VZ = D["k_zp"][keep], D["v_zp"][keep]
            print(f"\nzp ranges: K [{KZ.min():.0f}, {KZ.max():.0f}]   "
                  f"V [{VZ.min():.0f}, {VZ.max():.0f}]")
            # scale and zp differ by orders of magnitude, so standardise
            report("V scale+zp", standardise(np.concatenate([V, VZ], 1).astype(np.float64)),
                   idx, args, res, args.special_cut)
            report("K+V scale", standardise(np.concatenate([K, V], 1).astype(np.float64)),
                   idx, args, res, args.special_cut)
            report("K+V scale+zp",
                   standardise(np.concatenate([K, V, KZ, VZ], 1).astype(np.float64)),
                   idx, args, res, args.special_cut)
        else:
            KV = np.concatenate([K, V], 1)
            report("K+V raw (K dominates, not a valid combined result)",
                   KV, idx, args, res, args.special_cut)
            report("K+V standardised", standardise(KV.astype(np.float64)),
                   idx, args, res, args.special_cut)
            report("K+V log10 standardised",
                   standardise(np.log10(np.clip(KV.astype(np.float64), 1e-12, None))),
                   idx, args, res, args.special_cut)
        return D["mode"], res

    mode_a, res_a = run(args.path)
    if args.compare:
        mode_b, res_b = run(args.compare)
        print(f"\n{'='*70}\nside by side\n")
        print(f"  {'feature set':<34}{mode_a:>18}{mode_b:>18}")
        for k in sorted(set(res_a) | set(res_b)):
            a = res_a.get(k)
            b = res_b.get(k)
            fa = f"{a['exact']} ({a['dims']}d)" if a else "-"
            fb = f"{b['exact']} ({b['dims']}d)" if b else "-"
            print(f"  {k:<34}{fa:>18}{fb:>18}")
        print("\n  numbers are exact collisions; fewer is a stronger leak")

    print("\nReading the numbers:")
    print("  Nearest-neighbour uniqueness is the separability measure. A Top-1")
    print("  test against this same table would be circular and read 100%,")
    print("  since re-profiling a token reproduces its metadata bit for bit.")
    print("  Separation margin is median NN distance over median vector norm:")
    print("  larger means identity survives more measurement noise.")


if __name__ == "__main__":
    main()
