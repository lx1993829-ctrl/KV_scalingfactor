#!/usr/bin/env python3
"""
Multi-layer KV metadata profiler, position 0.

Why every layer is usable at position 0
---------------------------------------
A single token request attends only to itself, so the attention output is its
own V and nothing from any other token enters the residual stream.  The hidden
state at every layer therefore stays a deterministic function of the token
alone, and the metadata written at layer L is as context free as layer 0's.
Stacking layers multiplies the available dimensions by the layer count, which
matters for models whose per layer feature count is too small on its own
(Qwen2.5-1.5B has 2 KV heads, giving 4 dims at one layer but 4*n_layers when
stacked).

This holds only at position 0.  From position 1 onward the deeper layers see
the prefix through attention and the context freeness is lost.

Output npz
    k_scale, v_scale   [vocab, n_layers, num_kv_heads] float32
    k_zp, v_zp         same shape, int4 only
    status             [vocab] int8   1 ok, 2 ambiguous, 0 not captured
    layers             [n_layers] int32

Usage
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1

    python3 profile_layers.py --model Qwen/Qwen2.5-1.5B --probe
    python3 profile_layers.py --model Qwen/Qwen2.5-1.5B
    python3 profile_layers.py --model Qwen/Qwen2.5-1.5B --layers 0,1,2,3 --smoke 2000
"""

import os
import sys
import time
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

FLOOR = 1e-6
QUANT_MAX = {
    "int8_per_token_head": 127.0,
    "fp8_per_token_head": 448.0,
    "int4_per_token_head": 15.0,
}
ASYMMETRIC = {"int4_per_token_head"}


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
        return {"error": "no impl exposes _k_scale_cache"}
    ksc = impls[0]._k_scale_cache
    if ksc is None:
        return {"error": "scale caches unset; run a forward pass first"}
    kv = worker.model_runner.kv_caches[0]
    return {"n_layers": len(impls), "kv_shape": list(kv.shape),
            "kv_dtype": str(kv.dtype), "scale_shape": list(ksc.shape),
            "block_size": ksc.shape[1], "num_kv_heads": ksc.shape[2]}


def rpc_step_multi(worker, layers):
    """
    Diff every requested layer against the worker held snapshot and return the
    slot written since the last call, for all layers at once.

    All layers share one slot mapping, so the changed slot is located from the
    first requested layer and the rest are read at the same index.  Nothing is
    written into cache memory: the metadata caches are strided views aliasing
    live K/V bytes.
    """
    import torch

    impls = _find_impls(worker)
    torch.cuda.synchronize()
    snap = {L: (impls[L]._k_scale_cache.clone(),
                impls[L]._v_scale_cache.clone()) for L in layers}

    prev = getattr(worker, "_ml_prev", None)
    if prev is None:
        worker._ml_prev = snap
        return {"n_changed": -1}

    L0 = layers[0]
    k_now, v_now = snap[L0]
    k_old, v_old = prev[L0]
    changed = ((k_now != k_old) | (v_now != v_old)).any(dim=-1).nonzero()
    n = int(changed.shape[0])

    res = {"n_changed": n, "k": None, "v": None}
    if n >= 1:
        b, s = changed[0].tolist()
        res["k"] = [snap[L][0][b, s].tolist() for L in layers]
        res["v"] = [snap[L][1][b, s].tolist() for L in layers]

    worker._ml_prev = snap
    return res


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def unpack(arr, kv_dtype):
    """arr: [..., heads] float32 -> (scale, zp).  zp None for symmetric modes."""
    a = np.asarray(arr, dtype=np.float32)
    if kv_dtype not in ASYMMETRIC:
        return a, None
    bits = a.view(np.int32)
    zp = (bits & 0xF).astype(np.float32)
    scale = (bits & ~0xF).astype(np.int32).view(np.float32)
    return scale, zp


def rpc(llm, fn, *args):
    return llm.collective_rpc(fn, timeout=180, args=args)[0]


def build_engine(model, kv_dtype, util, blocks, max_len):
    from vllm import LLM
    return LLM(model=model, kv_cache_dtype=kv_dtype, dtype="bfloat16",
               enforce_eager=True, enable_prefix_caching=False,
               max_model_len=max_len, max_num_seqs=1,
               gpu_memory_utilization=util, num_gpu_blocks_override=blocks)


def make_feeder(llm):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    try:
        from vllm import TokensPrompt

        def feed(ids):
            llm.generate(TokensPrompt(prompt_token_ids=list(ids)),
                         sampling_params=sp, use_tqdm=False)
        feed([1])
        return feed
    except Exception:
        pass

    def feed(ids):
        llm.generate(sampling_params=sp, prompt_token_ids=[list(ids)],
                     use_tqdm=False)
    feed([1])
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--kv-dtype", default="int8_per_token_head",
                    choices=sorted(QUANT_MAX.keys()))
    ap.add_argument("--layers", default="all",
                    help="'all' or a comma separated list")
    ap.add_argument("--out", default=None)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.78)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=32)
    args = ap.parse_args()

    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        if var not in os.environ:
            print(f"warning: {var} not set")

    mode = args.kv_dtype
    tag = args.model.split("/")[-1].replace(".", "")
    out_path = args.out or f"/mnt/f/kvlayers_{tag}_{mode.split('_')[0]}.npz"
    ckpt_path = out_path.replace(".npz", "_ckpt.npz")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = len(tok)
    print(f"model  {args.model}")
    print(f"mode   {mode}   vocab {vocab}")
    print(f"out    {out_path}")

    llm = build_engine(args.model, mode, args.gpu_util, args.blocks, args.max_len)
    feed = make_feeder(llm)
    feed([0])

    info = rpc(llm, rpc_info)
    if info.get("error"):
        print(f"FATAL: {info['error']}")
        return 1
    n_layers_total = info["n_layers"]
    nkv = info["num_kv_heads"]
    layers = (list(range(n_layers_total)) if args.layers == "all"
              else [int(x) for x in args.layers.split(",")])
    bad = [L for L in layers if L >= n_layers_total]
    if bad:
        print(f"layers out of range: {bad} (model has {n_layers_total})")
        return 1

    print(f"\nlayers {n_layers_total}, capturing {len(layers)}: {layers}")
    print(f"kv_shape {info['kv_shape']} ({info['kv_dtype']})  "
          f"kv_heads {nkv}  block_size {info['block_size']}")
    print(f"feature budget: {len(layers)} layers x {nkv} heads x 2 (K,V) = "
          f"{len(layers)*nkv*2} dims"
          + (f" x2 for scale+zp = {len(layers)*nkv*4}" if mode in ASYMMETRIC else ""))
    if args.probe:
        return 0

    rpc(llm, rpc_step_multi, layers)     # baseline

    nL = len(layers)
    k_scale = np.zeros((vocab, nL, nkv), dtype=np.float32)
    v_scale = np.zeros((vocab, nL, nkv), dtype=np.float32)
    k_zp = np.zeros((vocab, nL, nkv), dtype=np.float32)
    v_zp = np.zeros((vocab, nL, nkv), dtype=np.float32)
    status = np.zeros(vocab, dtype=np.int8)

    limit = args.smoke or vocab
    bad_rows, t0 = [], time.time()
    for tid in range(limit):
        try:
            feed([tid])
            r = rpc(llm, rpc_step_multi, layers)
            n = r["n_changed"]
            if n >= 1:
                ks, kz = unpack(r["k"], mode)
                vs, vz = unpack(r["v"], mode)
                k_scale[tid], v_scale[tid] = ks, vs
                if kz is not None:
                    k_zp[tid], v_zp[tid] = kz, vz
                status[tid] = 1 if n == 1 else 2
            elif len(bad_rows) < 20:
                bad_rows.append((tid, "no write detected"))
        except Exception as e:
            if len(bad_rows) < 20:
                bad_rows.append((tid, repr(e)[:120]))

        if (tid + 1) % 500 == 0:
            el = time.time() - t0
            sp = (tid + 1) / el
            np.savez(ckpt_path, k_scale=k_scale, v_scale=v_scale,
                     k_zp=k_zp, v_zp=v_zp, status=status,
                     layers=np.array(layers), done=tid + 1)
            print(f"  {tid+1}/{limit}  {sp:.1f} tok/s  el={el/60:.1f}m  "
                  f"eta={(limit-tid-1)/sp/60:.1f}m  ok={int((status==1).sum())}")

    save = dict(k_scale=k_scale, v_scale=v_scale, status=status,
                layers=np.array(layers, dtype=np.int32),
                quant_max=QUANT_MAX[mode], model=args.model,
                kv_dtype=mode, num_kv_heads=nkv)
    if mode in ASYMMETRIC:
        save["k_zp"], save["v_zp"] = k_zp, v_zp
    np.savez(out_path, **save)

    print(f"\nsaved {out_path}")
    print(f"  clean {int((status==1).sum())}/{limit}   "
          f"ambiguous {int((status==2).sum())}")
    print(f"  floor hits: K {int((k_scale==FLOOR).sum())}  "
          f"V {int((v_scale==FLOOR).sum())}")
    if bad_rows:
        print("first failures:")
        for tid, why in bad_rows:
            print(f"  {tid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
