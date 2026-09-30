#!/usr/bin/env python3
"""
Layer sweep of two invariance properties, across models and quantization modes.

For a given token, measures its per-head K and V quantization metadata
    across positions 0,1,2,4,8,16          (position dependence)
    across four prompts at a fixed index   (context dependence)
at every requested layer, and reports whether each is bit-invariant.

Quantization modes
    int8_per_token_head   symmetric, scale = absmax/127, 8 dims per tensor
    fp8_per_token_head    symmetric, scale = absmax/448, 8 dims per tensor
    int4_per_token_head   asymmetric, scale = (max-min)/15 with the 4-bit
                          zero point packed into the low mantissa bits of the
                          same fp32 field, so 16 dims per tensor

For int4 the stored word is  (float_bits & ~0xF) | (zp & 0xF).  This script
unpacks it back into (scale, zp) and compares the pair, because comparing the
packed float alone lets the 4-bit zp nibble dominate the change test.

Expected result, unchanged by quantization mode, because the invariances come
from where the projections sit rather than how the output is encoded
    layer 0     V invariant on both axes.  K context-invariant but position
                sensitive in a minority of heads: RoPE is applied after k_proj
                and absmax is not rotation invariant, while V never enters the
                QK dot product so RoPE never touches it.
    layer >= 1  everything fails.  Layer 0's attention output is in the
                residual stream, so the projection input is no longer a
                function of the token alone.

Usage
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1

    python3 layer_sweep3.py --kv-dtype int8_per_token_head --csv int8.csv
    python3 layer_sweep3.py --kv-dtype fp8_per_token_head  --csv fp8.csv
    python3 layer_sweep3.py --kv-dtype int4_per_token_head --csv int4.csv
    python3 layer_sweep3.py --model Qwen/Qwen2.5-1.5B --words "Hello,password"
"""

import os
import sys
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

MAX_LEN = 64
POSITIONS = [0, 1, 2, 4, 8, 16]

CONTEXT_TEXTS = {
    "A": "The cat sat on the mat today",
    "B": "I like to eat fresh bread here",
    "C": "Large language models are very fast now",
    "D": ". . . . . . .",
}
FILLER_TEXT = "."

QUANT_MAX = {
    "int8_per_token_head": 127.0,
    "fp8_per_token_head": 448.0,
    "int4_per_token_head": 15.0,
}


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------

def _find_impls(worker):
    model = worker.model_runner.model
    impls = []
    for _, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(impl, "_k_scale_cache"):
            impls.append(impl)
    return impls


def rpc_info(worker):
    impls = _find_impls(worker)
    if not impls:
        return {"n_layers": 0, "error": "no impl exposes _k_scale_cache; "
                                        "is kv_cache_dtype per-token-head?"}
    ksc = impls[0]._k_scale_cache
    if ksc is None:
        return {"n_layers": len(impls),
                "error": "scale caches unset; run a forward pass first"}
    kv = worker.model_runner.kv_caches[0]
    return {
        "n_layers": len(impls),
        "kv_shape": list(kv.shape),
        "kv_dtype": str(kv.dtype),
        "scale_shape": list(ksc.shape),
        "num_blocks": ksc.shape[0],
        "block_size": ksc.shape[1],
        "num_kv_heads": ksc.shape[2],
    }


def rpc_changed_multi(worker, layers):
    """
    Slots whose metadata moved since the last call, for every requested layer.
    All layers share one slot mapping, derived from the first requested layer.
    Raw fp32 words are returned; unpacking happens on the host so the worker
    stays mode agnostic.
    """
    import torch

    impls = _find_impls(worker)
    torch.cuda.synchronize()

    snap = {}
    for L in layers:
        impl = impls[L]
        snap[L] = (impl._k_scale_cache.clone(), impl._v_scale_cache.clone())

    block_size = impls[layers[0]]._k_scale_cache.shape[1]
    prev = getattr(worker, "_sweep_prev", None)
    if prev is None:
        worker._sweep_prev = snap
        return {"first": True, "block_size": block_size, "per_layer": {}}

    L0 = layers[0]
    k_now, v_now = snap[L0]
    k_old, v_old = prev[L0]
    changed = ((k_now != k_old) | (v_now != v_old)).any(dim=-1).nonzero().tolist()
    changed.sort(key=lambda bs: bs[0] * block_size + bs[1])

    per_layer = {}
    for L in layers:
        kn, vn = snap[L]
        per_layer[L] = [
            {"lin": b * block_size + s,
             "k": kn[b, s].tolist(),
             "v": vn[b, s].tolist()}
            for b, s in changed
        ]

    worker._sweep_prev = snap
    return {"first": False, "block_size": block_size, "per_layer": per_layer}


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def unpack(words, kv_dtype):
    """
    Turn the raw fp32 words into the feature vector actually stored.

    int8 / fp8 : the word is the scale, 8 dims.
    int4       : the word is (scale_bits & ~0xF) | (zp & 0xF).  Split it into
                 scale and zp and concatenate, 16 dims.  Comparing the packed
                 word alone would let the zp nibble mask a scale change and
                 vice versa.
    """
    a = np.asarray(words, dtype=np.float32)
    if kv_dtype != "int4_per_token_head":
        return a, None

    bits = a.view(np.int32)
    zp = (bits & 0xF).astype(np.float32)
    scale = (bits & ~0xF).astype(np.int32).view(np.float32)
    return scale, zp


def feature(words, kv_dtype):
    scale, zp = unpack(words, kv_dtype)
    return scale if zp is None else np.concatenate([scale, zp])


def rpc(llm, fn, *args):
    return llm.collective_rpc(fn, timeout=180, args=args)[0]


def build_engine(model, kv_dtype, util, blocks):
    from vllm import LLM
    return LLM(
        model=model,
        kv_cache_dtype=kv_dtype,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=MAX_LEN,
        max_num_seqs=1,
        gpu_memory_utilization=util,
        num_gpu_blocks_override=blocks,
    )


def make_feeder(llm):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    try:
        from vllm import TokensPrompt

        def feed(ids):
            llm.generate(TokensPrompt(prompt_token_ids=list(ids)),
                         sampling_params=sp, use_tqdm=False)
        return feed
    except Exception:
        pass

    def feed(ids):
        llm.generate(sampling_params=sp, prompt_token_ids=[list(ids)],
                     use_tqdm=False)
    return feed


def encode_one(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"{text!r} encoded to nothing")
    return ids[0]


def measure(llm, feed, layers, prompt, target_index, kv_dtype, retries=8):
    """
    Feature vectors for the token at target_index, per layer, or None.

    A prompt of N tokens must produce exactly N changed slots at contiguous
    linear indices.  Anything else means a write was invisible to the diff
    (its bytes matched the slot's previous contents), so the index mapping
    cannot be trusted and the run is discarded.
    """
    n = len(prompt)
    for _ in range(retries):
        feed(prompt)
        r = rpc(llm, rpc_changed_multi, layers)
        pl = r.get("per_layer") or {}
        if not pl:
            continue
        slots = pl[layers[0]]
        if len(slots) != n:
            continue
        lins = [s["lin"] for s in slots]
        if lins != list(range(lins[0], lins[0] + n)):
            continue
        return {L: (feature(pl[L][target_index]["k"], kv_dtype),
                    feature(pl[L][target_index]["v"], kv_dtype))
                for L in layers}
    return None


def analyse(rows):
    if len(rows) < 2:
        return None
    k0, v0 = rows[0][1], rows[0][2]
    k_inv = all(np.array_equal(k, k0) for _, k, _ in rows)
    v_inv = all(np.array_equal(v, v0) for _, _, v in rows)
    dk = max(np.abs(k - k0).max() / max(abs(k0.max()), 1e-12) for _, k, _ in rows)
    dv = max(np.abs(v - v0).max() / max(abs(v0.max()), 1e-12) for _, _, v in rows)
    nd = len(k0)
    k_moved = [h for h in range(nd) if any(r[1][h] != k0[h] for r in rows)]
    v_moved = [h for h in range(nd) if any(r[2][h] != v0[h] for r in rows)]
    return {"k_inv": k_inv, "v_inv": v_inv, "dk": dk, "dv": dv,
            "k_moved": k_moved, "v_moved": v_moved,
            "n": len(rows), "n_dims": nd}


def sweep_token(llm, feed, layers, token, filler, ctx_ids, ctx_index, kv_dtype):
    pos_rows = {L: [] for L in layers}
    for pos in POSITIONS:
        got = measure(llm, feed, layers, [filler] * pos + [token], pos, kv_dtype)
        if got is None:
            print(f"    pos {pos}: slot mapping unresolved, skipped")
            continue
        for L in layers:
            pos_rows[L].append((f"pos{pos}",) + got[L])

    ctx_rows = {L: [] for L in layers}
    p = ctx_index
    for label, pre in ctx_ids.items():
        prompt = list(pre[:p]) + [token] + [filler, filler]
        got = measure(llm, feed, layers, prompt, p, kv_dtype)
        if got is None:
            print(f"    ctx {label}: slot mapping unresolved, skipped")
            continue
        for L in layers:
            ctx_rows[L].append((label,) + got[L])

    return {L: {"pos": analyse(pos_rows[L]), "ctx": analyse(ctx_rows[L])}
            for L in layers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--kv-dtype", default="int8_per_token_head",
                    choices=sorted(QUANT_MAX.keys()))
    ap.add_argument("--layers", default="0,1,8")
    ap.add_argument("--words", default="Hello,password,London")
    ap.add_argument("--ctx-index", type=int, default=4)
    ap.add_argument("--gpu-util", type=float, default=0.78)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        if var not in os.environ:
            print(f"warning: {var} not set")

    layers = [int(x) for x in args.layers.split(",")]
    is_int4 = args.kv_dtype == "int4_per_token_head"

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"model      {args.model}")
    print(f"kv dtype   {args.kv_dtype}  (QUANT_MAX={QUANT_MAX[args.kv_dtype]})")
    if is_int4:
        print("           asymmetric: scale=(max-min)/15, 4-bit zp packed into "
              "the low mantissa bits; features are [scale | zp], 16 dims")
    print(f"vocab      {len(tok)}")

    words = [w.strip() for w in args.words.split(",") if w.strip()]
    targets = []
    for w in words:
        try:
            targets.append((w, encode_one(tok, w)))
        except ValueError as e:
            print(f"  skipping {w!r}: {e}")
    filler = encode_one(tok, FILLER_TEXT)
    ctx_ids = {k: tok.encode(v, add_special_tokens=False)
               for k, v in CONTEXT_TEXTS.items()}

    short = [k for k, v in ctx_ids.items() if len(v) < args.ctx_index]
    if short:
        print(f"  contexts too short for index {args.ctx_index}: {short}")
        return 1

    print(f"filler     {filler} = {tok.decode([filler])!r}")
    print(f"targets    " + ", ".join(f"{t}={w!r}" for w, t in targets))

    llm = build_engine(args.model, args.kv_dtype, args.gpu_util, args.blocks)
    feed = make_feeder(llm)
    feed([filler])
    rpc(llm, rpc_changed_multi, layers)

    info = rpc(llm, rpc_info)
    if info.get("error"):
        print(f"\nFATAL: {info['error']}")
        return 1
    print(f"\nlayers {info['n_layers']}  kv_shape {info['kv_shape']} "
          f"({info['kv_dtype']})  scale_shape {info['scale_shape']}  "
          f"kv_heads {info['num_kv_heads']}  block_size {info['block_size']}")
    # int8 and fp8 are both one byte per element, so kv_shape alone cannot
    # tell them apart.  int4 packs two values per byte and should differ.
    print(f"NOTE: confirm the engine log above says kv_cache_dtype="
          f"{args.kv_dtype}; shape alone does not distinguish int8 from fp8.")

    bad = [L for L in layers if L >= info["n_layers"]]
    if bad:
        print(f"requested layers out of range: {bad}")
        return 1

    rows = []
    for word, tid in targets:
        print(f"\n### {tid} = {word!r}")
        res = sweep_token(llm, feed, layers, tid, filler, ctx_ids,
                          args.ctx_index, args.kv_dtype)
        print(f"  {'layer':>5}  {'axis':<4} {'K inv':>6} {'V inv':>6} "
              f"{'K max dev':>11} {'V max dev':>11}  {'K dims moved':<22}"
              f"{'V dims moved'}")
        for L in layers:
            for axis in ("pos", "ctx"):
                r = res[L][axis]
                if r is None:
                    print(f"  {L:>5}  {axis:<4}   insufficient samples")
                    continue
                km = ",".join(map(str, r["k_moved"])) or "-"
                vm = ",".join(map(str, r["v_moved"])) or "-"
                print(f"  {L:>5}  {axis:<4} {str(r['k_inv']):>6} "
                      f"{str(r['v_inv']):>6} {r['dk']:>11.3e} "
                      f"{r['dv']:>11.3e}  "
                      f"{len(r['k_moved'])}/{r['n_dims']} [{km}]".ljust(22)
                      + f" {len(r['v_moved'])}/{r['n_dims']} [{vm}]")
                rows.append({"model": args.model, "kv_dtype": args.kv_dtype,
                             "word": word, "token": tid, "layer": L,
                             "axis": axis,
                             "k_invariant": r["k_inv"],
                             "v_invariant": r["v_inv"],
                             "k_max_dev": r["dk"], "v_max_dev": r["dv"],
                             "k_dims_moved": len(r["k_moved"]),
                             "v_dims_moved": len(r["v_moved"]),
                             "k_moved_idx": " ".join(map(str, r["k_moved"])),
                             "v_moved_idx": " ".join(map(str, r["v_moved"])),
                             "n_dims": r["n_dims"], "n_samples": r["n"]})

    print("\nsummary across targets")
    for L in layers:
        for axis in ("pos", "ctx"):
            sub = [r for r in rows if r["layer"] == L and r["axis"] == axis]
            if not sub:
                continue
            ki = sum(r["k_invariant"] for r in sub)
            vi = sum(r["v_invariant"] for r in sub)
            print(f"  layer {L:>2} {axis}: K invariant {ki}/{len(sub)}, "
                  f"V invariant {vi}/{len(sub)}")

    # is the unstable K dimension the same one for every token?
    pos0 = [r for r in rows if r["layer"] == 0 and r["axis"] == "pos"]
    idx_sets = {r["k_moved_idx"] for r in pos0}
    if pos0:
        if len(idx_sets) == 1:
            print(f"\nlayer 0 position instability hits the same K dims for "
                  f"every token tested: [{pos0[0]['k_moved_idx']}] — this is a "
                  f"property of the weight geometry, not of the token")
        else:
            print(f"\nlayer 0 position instability hits different K dims per "
                  f"token: {sorted(idx_sets)} — token dependent, so a fixed "
                  f"'drop the unstable heads' rule will not work")

    if args.csv and rows:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
