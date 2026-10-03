#!/usr/bin/env python3
"""
vLLM per-token-head KV metadata profiler  (v6, multi-mode)

Extracts the quantization metadata vLLM stores for every token in the
vocabulary, one token per request at position 0.

Modes
    int8_per_token_head   symmetric   scale = max(absmax/127, 1e-6)
    fp8_per_token_head    symmetric   scale = max(absmax/448, 1e-6)
    int4_per_token_head   asymmetric  scale = max((max-min)/15, 1e-6)
                                      zp    = clamp(round(-min/scale), 0, 15)
                                      stored as (scale_bits & ~0xF) | zp

For the symmetric modes  absmax = scale * QUANT_MAX  exactly, so one stored
number per (token, head).  For int4 the single fp32 word carries both the
scale and the 4-bit zero point, and from the pair you recover both min and
max of the head vector, i.e. two independent order statistics instead of one.

Output npz
    k_scale, v_scale        [vocab, num_kv_heads] float32
    k_zp, v_zp              [vocab, num_kv_heads] float32   (int4 only)
    status                  [vocab] int8   1 ok, 2 ambiguous, 0 not captured

Required env
    VLLM_USE_FLASHINFER_SAMPLER=0
    VLLM_WSL2_ENABLE_PIN_MEMORY=1
    VLLM_ALLOW_INSECURE_SERIALIZATION=1

Usage
    python3 profile_kv_meta.py --kv-dtype int4_per_token_head
    python3 profile_kv_meta.py --kv-dtype fp8_per_token_head --smoke 200
    python3 profile_kv_meta.py --kv-dtype int8_per_token_head --probe
"""

import os
import sys
import time
import argparse
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

LAYER = 0
FLOOR = 1e-6
QUANT_MAX = {
    "int8_per_token_head": 127.0,
    "fp8_per_token_head": 448.0,
    "int4_per_token_head": 15.0,
}
ASYMMETRIC = {"int4_per_token_head"}


# --------------------------------------------------------------------------
# worker side.  Pickled to the EngineCore subprocess, so module level only and
# no reliance on module globals; everything arrives through args.
# --------------------------------------------------------------------------

def _find_impls(worker):
    model = worker.model_runner.model
    impls = []
    for _, mod in model.named_modules():
        impl = getattr(mod, "impl", None)
        if impl is not None and hasattr(impl, "_k_scale_cache"):
            impls.append(impl)
    return impls


def rpc_probe(worker, layer):
    import torch

    impls = _find_impls(worker)
    if not impls:
        return {"error": "no impl exposes _k_scale_cache"}
    impl = impls[layer]
    ksc, vsc = impl._k_scale_cache, impl._v_scale_cache
    if ksc is None:
        return {"error": "scale caches unset; run a forward pass first"}

    kv = worker.model_runner.kv_caches[layer]
    out = {
        "n_layers": len(impls),
        "impl_type": type(impl).__name__,
        "kv_shape": list(kv.shape),
        "kv_dtype": str(kv.dtype),
        "kv_stride": list(kv.stride()),
        "scale_shape": list(ksc.shape),
        "scale_stride": list(ksc.stride()),
        "num_kv_heads": ksc.shape[2],
        "block_size": ksc.shape[1],
    }
    touched = (ksc != 1.0).any(dim=-1).nonzero()
    out["n_touched"] = int(touched.shape[0])
    if touched.shape[0]:
        b, s = touched[0].tolist()
        out["probe_slot"] = [b, s]
        out["k_raw"] = ksc[b, s].tolist()
        out["v_raw"] = vsc[b, s].tolist()
        # int8/fp8 store one value per element, so the extreme quantized
        # element must sit exactly at QUANT_MAX.  int4 packs two values per
        # byte, so the same check needs unpacking and is done host side.
        out["k_int_absmax"] = [
            int(kv[b, h, s, : kv.shape[-1] // 2].abs().max().item())
            if kv.dtype != torch.uint8 else -1
            for h in range(ksc.shape[2])
        ]
    return out


def rpc_step(worker, layer):
    """
    Diff the metadata caches against the snapshot held on the worker, return
    the slot written for the current request, then re-baseline.

    Between two consecutive single-token requests TWO kinds of slot can differ:
    the slot freshly written for the new token, and the slot of the PREVIOUS
    request being evicted back to the 1.0 sentinel (max_num_seqs=1, no prefix
    caching).  A naive "count every changed slot" sees both and brands every
    token ambiguous, and worse, changed[0] may land on the evicted slot, so the
    recorded scale is the wrong token's.  This is exactly the Gemma/int4 failure
    mode: the int4 packed word makes the eviction register as a change where the
    symmetric modes' reset-to-1.0 did not.

    Fix: among changed slots keep only those whose NEW value is still
    non-sentinel.  An eviction returns to 1.0 in every head and drops out, so
    what remains is the genuine write(s).  n_changed counts those; n_raw keeps
    the unfiltered count so the host can see the eviction traffic.

    Nothing is ever written into cache memory: the caches are strided views
    aliasing live K/V bytes, and writing into them while kernels are in flight
    corrupts the run.
    """
    import torch

    impl = _find_impls(worker)[layer]
    ksc, vsc = impl._k_scale_cache, impl._v_scale_cache

    torch.cuda.synchronize()
    k_now, v_now = ksc.clone(), vsc.clone()

    prev = getattr(worker, "_meta_prev", None)
    if prev is None:
        worker._meta_prev = (k_now, v_now)
        return {"n_changed": -1}

    k_prev, v_prev = prev
    changed_slot = ((k_now != k_prev) | (v_now != v_prev)).any(dim=-1)
    n_raw = int(changed_slot.sum())

    # a genuine write leaves the slot non-sentinel; an eviction resets to 1.0
    active = (k_now != 1.0).any(dim=-1) | (v_now != 1.0).any(dim=-1)
    sel = (changed_slot & active).nonzero()
    n = int(sel.shape[0])

    res = {"n_changed": n, "n_raw": n_raw, "k": None, "v": None, "slot": None}
    if n >= 1:
        b, s = sel[0].tolist()
        res["slot"] = [b, s]
        res["k"] = k_now[b, s].tolist()
        res["v"] = v_now[b, s].tolist()

    worker._meta_prev = (k_now, v_now)
    return res


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------

def unpack(words, kv_dtype):
    """
    Raw fp32 words -> (scale, zp).  zp is None for the symmetric modes.

    int4 packs the 4-bit zero point into the low mantissa bits of the scale:
        stored = (scale_bits & ~0xF) | (zp & 0xF)
    Clearing those 4 bits costs about 1e-6 relative error on the scale, far
    below any matching tolerance, so the packing loses the attacker nothing.
    """
    a = np.asarray(words, dtype=np.float32)
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
    return LLM(
        model=model,
        kv_cache_dtype=kv_dtype,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=max_len,
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
        feed([1])
        return feed
    except Exception:
        pass

    def feed(ids):
        llm.generate(sampling_params=sp, prompt_token_ids=[list(ids)],
                     use_tqdm=False)
    feed([1])
    return feed


def show_probe(info, kv_dtype):
    print("--- layout ---")
    for k in ("impl_type", "n_layers", "kv_shape", "kv_dtype", "kv_stride",
              "scale_shape", "scale_stride", "num_kv_heads", "block_size",
              "n_touched", "probe_slot"):
        if k in info:
            print(f"  {k}: {info[k]}")

    qm = QUANT_MAX[kv_dtype]
    k_scale, k_zp = unpack(info["k_raw"], kv_dtype)
    v_scale, v_zp = unpack(info["v_raw"], kv_dtype)

    print(f"--- metadata at probe slot ({kv_dtype}) ---")
    for h in range(len(k_scale)):
        if k_zp is None:
            print(f"  head {h}: k_scale={k_scale[h]:.8f} absmax={k_scale[h]*qm:8.5f}"
                  f"   v_scale={v_scale[h]:.8f} absmax={v_scale[h]*qm:8.5f}")
        else:
            k_min = -k_zp[h] * k_scale[h]
            k_max = k_min + qm * k_scale[h]
            v_min = -v_zp[h] * v_scale[h]
            v_max = v_min + qm * v_scale[h]
            print(f"  head {h}: k_scale={k_scale[h]:.8f} zp={int(k_zp[h]):2d} "
                  f"-> [{k_min:8.5f}, {k_max:8.5f}]   "
                  f"v_scale={v_scale[h]:.8f} zp={int(v_zp[h]):2d} "
                  f"-> [{v_min:8.5f}, {v_max:8.5f}]")

    if kv_dtype not in ASYMMETRIC:
        ok = all(x == 127 for x in info.get("k_int_absmax", []))
        if kv_dtype == "int8_per_token_head":
            print(f"  int8 absmax==127 contract: "
                  f"{'HOLDS' if ok else 'VIOLATED ' + str(info['k_int_absmax'])}")
    else:
        print("  int4 packs two values per byte; the extreme-element check "
              "needs nibble unpacking and is skipped here")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--kv-dtype", default="int8_per_token_head",
                    choices=sorted(QUANT_MAX.keys()))
    ap.add_argument("--out", default=None,
                    help="default /mnt/f/kvmeta_<mode>.npz")
    ap.add_argument("--probe", action="store_true",
                    help="layout report only, no profiling")
    ap.add_argument("--smoke", type=int, default=0,
                    help="profile only the first N tokens")
    ap.add_argument("--gpu-util", type=float, default=0.78)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=32)
    ap.add_argument("--resume", action="store_true",
                    help="load the checkpoint and skip tokens already captured")
    args = ap.parse_args()

    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        if var not in os.environ:
            print(f"warning: {var} not set")

    mode = args.kv_dtype
    out_path = args.out or f"/mnt/f/kvmeta_{mode.split('_')[0]}.npz"
    ckpt_path = out_path.replace(".npz", "_ckpt.npz")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    vocab = len(tok)          # 128256 for Llama-3.2, not tok.vocab_size
    print(f"model    {args.model}")
    print(f"mode     {mode}  QUANT_MAX={QUANT_MAX[mode]}  "
          f"{'asymmetric, scale+zp' if mode in ASYMMETRIC else 'symmetric, scale only'}")
    print(f"vocab    {vocab}  (tokenizer.vocab_size={tok.vocab_size})")
    print(f"out      {out_path}")

    llm = build_engine(args.model, mode, args.gpu_util, args.blocks, args.max_len)
    feed = make_feeder(llm)
    feed([9906 % vocab])

    info = rpc(llm, rpc_probe, LAYER)
    if info.get("error"):
        print(f"FATAL: {info['error']}")
        return 1
    show_probe(info, mode)
    if args.probe:
        return 0

    nkv = info["num_kv_heads"]
    rpc(llm, rpc_step, LAYER)          # establish the diff baseline

    k_scale = np.zeros((vocab, nkv), dtype=np.float32)
    v_scale = np.zeros((vocab, nkv), dtype=np.float32)
    k_zp = np.zeros((vocab, nkv), dtype=np.float32)
    v_zp = np.zeros((vocab, nkv), dtype=np.float32)
    status = np.zeros(vocab, dtype=np.int8)

    if args.resume and os.path.exists(ckpt_path):
        d = np.load(ckpt_path)
        k_scale, v_scale = d["k_scale"], d["v_scale"]
        if "k_zp" in d:
            k_zp, v_zp = d["k_zp"], d["v_zp"]
        status = d["status"]
        print(f"resumed: {int((status != 0).sum())} tokens already captured")

    limit = args.smoke or vocab
    bad, t0 = [], time.time()
    for tid in range(limit):
        if args.resume and status[tid] != 0:
            continue
        try:
            feed([tid])
            r = rpc(llm, rpc_step, LAYER)
            n = r["n_changed"]
            if n >= 1:
                ks, kz = unpack(r["k"], mode)
                vs, vz = unpack(r["v"], mode)
                k_scale[tid], v_scale[tid] = ks, vs
                if kz is not None:
                    k_zp[tid], v_zp[tid] = kz, vz
                status[tid] = 1 if n == 1 else 2
                if n > 1 and len(bad) < 20:
                    bad.append((tid, f"n_changed={n} (raw={r.get('n_raw','?')})"))
            elif len(bad) < 20:
                bad.append((tid, f"no active write (raw={r.get('n_raw','?')})"))
        except Exception as e:
            if len(bad) < 20:
                bad.append((tid, repr(e)[:120]))

        if (tid + 1) % 500 == 0:
            el = time.time() - t0
            sp = (tid + 1) / el
            np.savez(ckpt_path, k_scale=k_scale, v_scale=v_scale,
                     k_zp=k_zp, v_zp=v_zp, status=status, done=tid + 1)
            print(f"  {tid+1}/{limit}  {sp:.1f} tok/s  el={el/60:.1f}m  "
                  f"eta={(limit-tid-1)/sp/60:.1f}m  "
                  f"ok={int((status==1).sum())}  amb={int((status==2).sum())}")

    save = dict(k_scale=k_scale, v_scale=v_scale, status=status,
                quant_max=QUANT_MAX[mode], model=args.model,
                kv_dtype=mode, layer=LAYER,
                asymmetric=mode in ASYMMETRIC)
    if mode in ASYMMETRIC:
        save["k_zp"], save["v_zp"] = k_zp, v_zp
    np.savez(out_path, **save)

    print(f"\nsaved {out_path}")
    print(f"  clean     {int((status==1).sum())}/{limit}")
    print(f"  ambiguous {int((status==2).sum())}")
    print(f"  missing   {int((status==0).sum()) - (vocab - limit)}")
    print(f"  floor hits (scale=={FLOOR:g}): "
          f"K {int((k_scale==FLOOR).sum())}  V {int((v_scale==FLOOR).sum())}")
    if mode in ASYMMETRIC:
        ok = status == 1
        if ok.any():
            print(f"  zp range: K [{k_zp[ok].min():.0f}, {k_zp[ok].max():.0f}]  "
                  f"V [{v_zp[ok].min():.0f}, {v_zp[ok].max():.0f}]")
        else:
            print("  zp range: n/a (no clean tokens captured)")
    if bad:
        print("first failures:")
        for tid, why in bad:
            print(f"  {tid}: {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
