#!/usr/bin/env python3
"""
Plot nearest-neighbour distance distributions from save_nn.py output.

Three figures, each answering a question the collision counts cannot.

  --mode overlay   several models or quantization modes on one axis, distances
                   normalised by each one's median vector norm so magnitudes
                   are comparable.  Shows whether separability rests on a wide
                   gap or a narrow one.
  --mode single    one file, split by token id range.  Shows whether the mass
                   at zero is concentrated in the reserved / undertrained tail
                   of the vocabulary rather than in ordinary text.
  --mode comb      one file, linear axis over the smallest distances.  If the
                   metadata is resolution limited the distances land on a
                   discrete grid inherited from the bf16 spacing of the
                   underlying activations, and that comb is visible here.

Zeros cannot go on a log axis, so they are drawn as a separate labelled bar at
the left rather than dropped.

Usage
    python3 plot_nn.py --files a_nn_v.npz b_nn_v.npz --out fig.png
    python3 plot_nn.py --files x_nn_v.npz --mode single --special-cut 128000
    python3 plot_nn.py --files x_nn_v.npz --mode comb
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    d = np.load(path, allow_pickle=False)
    return {
        "nn": d["nn_dist"], "tid": d["token_id"],
        "med": float(d["median_norm"]),
        "tag": str(d["tag"]) if "tag" in d else path,
        "mode": str(d["mode"]) if "mode" in d else "?",
        "model": str(d["model"]) if "model" in d else "?",
        "dims": int(d["dims"]) if "dims" in d else -1,
    }


def overlay(files, out, bins, title):
    fig, (axz, ax) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 7]})

    data = [load(f) for f in files]
    lo, hi = np.inf, -np.inf
    for d in data:
        r = d["nn"] / d["med"]
        pos = r[r > 0]
        if len(pos):
            lo = min(lo, pos.min())
            hi = max(hi, pos.max())
    edges = np.logspace(np.log10(lo * 0.8), np.log10(hi * 1.2), bins)

    width = 0.8 / max(len(data), 1)
    for i, d in enumerate(data):
        r = d["nn"] / d["med"]
        n = len(r)
        nz = int((r == 0).sum())
        label = f"{d['tag']}  ({d['dims']}d)"
        axz.bar(i * width, max(nz, 0.5), width=width * 0.9)
        ax.hist(r[r > 0], bins=edges, histtype="step", linewidth=1.6,
                label=f"{label}  zeros={nz}")

    axz.set_xticks([])
    axz.set_yscale("log")
    axz.set_ylabel("number of tokens")
    axz.set_title("collisions\n(distance = 0)", fontsize=9)
    axz.spines[["top", "right"]].set_visible(False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nearest-neighbour distance / median vector norm")
    ax.set_ylabel("number of tokens")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")


def single(path, out, bins, cut, title):
    d = load(path)
    r = d["nn"] / d["med"]
    ordinary = d["tid"] < cut
    fig, ax = plt.subplots(figsize=(8, 4.2))

    pos = r[r > 0]
    edges = np.logspace(np.log10(pos.min() * 0.8),
                        np.log10(pos.max() * 1.2), bins)
    for mask, lab, c in ((ordinary, f"id < {cut}", "#2b6cb0"),
                         (~ordinary, f"id >= {cut}", "#c05621")):
        sel = r[mask]
        nz = int((sel == 0).sum())
        ax.hist(sel[sel > 0], bins=edges, histtype="stepfilled", alpha=0.55,
                color=c, label=f"{lab}   n={mask.sum()}  zeros={nz}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nearest-neighbour distance / median vector norm")
    ax.set_ylabel("count")
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title or f"{d['model']}  {d['mode']}  ({d['dims']}d)")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")

    nz_o = int((r[ordinary] == 0).sum())
    nz_s = int((r[~ordinary] == 0).sum())
    print(f"  zeros: ordinary {nz_o}, special {nz_s}")



def allmode(path, out, bins, title):
    """Every token, one colour, zeros included as a bar at the left."""
    d = load(path)
    r = d["nn"] / d["med"]
    n = len(r)
    nz = int((r == 0).sum())
    pos = r[r > 0]

    fig, (axz, ax) = plt.subplots(
        1, 2, figsize=(9, 4.2), gridspec_kw={"width_ratios": [1, 8]},
        sharey=True)

    axz.bar([0], [max(nz, 0.5)], width=0.6, color="#2b6cb0")
    axz.set_xticks([0]); axz.set_xticklabels(["0"])
    axz.set_ylabel("count")
    axz.set_yscale("log")
    axz.set_title(f"zeros\n{nz}", fontsize=9)
    axz.spines[["top", "right"]].set_visible(False)

    edges = np.logspace(np.log10(pos.min() * 0.8),
                        np.log10(pos.max() * 1.2), bins)
    ax.hist(pos, bins=edges, color="#2b6cb0")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nearest-neighbour distance / median vector norm")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title or f"{d['tag']}   {d['dims']}d   n={n}")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")

    q = np.percentile(pos, [0, 1, 25, 50, 75, 99, 100])
    print(f"  tokens {n}, zeros {nz} ({100.0*nz/n:.4f}%)")
    print("  nonzero min/1/25/50/75/99/max:  " +
          "  ".join(f"{x:.4e}" for x in q))


def comb(path, out, bins, title):
    """Linear histogram over the smallest distances, to expose discretisation."""
    d = load(path)
    r = d["nn"]
    pos = r[r > 0]
    if len(pos) == 0:
        print("all distances are zero; nothing to plot")
        return
    top = np.percentile(pos, 5.0)
    sel = pos[pos <= top]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(sel, bins=bins, color="#2b6cb0")
    ax.set_xlabel("nearest-neighbour distance (raw units)")
    ax.set_ylabel("count")
    ax.set_title(title or
                 f"{d['model']}  {d['mode']}  bottom 5% of distances")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
    print("  evenly spaced spikes here mean the metadata is resolution")
    print("  limited: absmax values land on the bf16 grid of the activations,")
    print("  so distances are multiples of that step rather than continuous.")

    u = np.unique(np.round(sel, 12))
    print(f"  {len(u)} distinct values among {len(sel)} smallest distances")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--mode", default="overlay",
                    choices=["overlay", "single", "comb", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--special-cut", type=int, default=128000)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    out = args.out or f"nn_{args.mode}.png"
    if args.mode == "overlay":
        overlay(args.files, out, args.bins, args.title)
    elif args.mode == "all":
        allmode(args.files[0], out, args.bins, args.title)
    elif args.mode == "single":
        single(args.files[0], out, args.bins, args.special_cut, args.title)
    else:
        comb(args.files[0], out, args.bins, args.title)


if __name__ == "__main__":
    main()
