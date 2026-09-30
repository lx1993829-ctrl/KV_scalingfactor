#!/usr/bin/env python3
"""
Per-channel (transposed) quantization grouping as a defense.

Current scheme groups along head_dim: for one (token, head) take the max over
that token's 64 values and store it.  The stored number therefore describes a
single token, which is what makes it a fingerprint.

Transposed scheme groups along the token axis: for one (channel, head) take
the max over G consecutive tokens and store it.  Each stored number is now a
max over G different tokens, so no single token determines it.

Dequantization is unaffected: every element belongs to exactly one group either
way, and is multiplied by that group's scale.  Attention does not care which
axis the grouping ran along.

What this script measures
-------------------------
1. Baseline.  Per-token-head scales over the whole vocabulary, and the
   collision count.  This should reproduce the numbers from the vLLM profiles,
   which validates that computing layer 0 analytically matches what the engine
   stores.

2. Leakage under transposed grouping.  A block's signature is the elementwise
   max of |V| over its G tokens.  A candidate token can only be in the block if
   it is dominated by the signature in every channel, since the max cannot be
   smaller than any member.  Counting survivors of that dominance test gives a
   direct measure of how much the signature narrows the vocabulary: G survivors
   means full recovery, tens of thousands means the channel is closed.

3. Accuracy.  Relative quantization error under both groupings, so the defense
   can be priced rather than assumed free.

Layer 0 at position 0 is computed in closed form: attention over a single token
is the identity, so V = v_proj(RMSNorm(emb(t))) and K = k_proj(RMSNorm(emb(t)))
with RoPE at position 0 also the identity.  No vLLM, no KV cache, no engine.

Usage
    python3 transposed_eval.py --model meta-llama/Llama-3.2-1B
    python3 transposed_eval.py --model meta-llama/Llama-3.2-1B --groups 2,4,8,16
    python3 transposed_eval.py --model ... --tensor k --blocks 200
    python3 transposed_eval.py --model ... --save /mnt/f/layer0_vectors.npz
"""

import argparse
import numpy as np
import torch


# --------------------------------------------------------------------------
# layer 0 in closed form
# --------------------------------------------------------------------------

def layer0_vectors(model_name, tensor, device, chunk, dtype=torch.float32):
    """
    Return [vocab, n_kv_heads, head_dim] of layer-0 K or V at position 0.

    At position 0 a single-token request attends only to itself, so nothing
    from any other token is in the residual stream and the projection input is
    the token embedding after the input layernorm.  RoPE at position 0 is the
    identity, so K needs no rotation either.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()

    base = model.model if hasattr(model, "model") else model.transformer
    emb = base.embed_tokens if hasattr(base, "embed_tokens") else base.wte
    layer0 = base.layers[0] if hasattr(base, "layers") else base.h[0]
    ln = (layer0.input_layernorm if hasattr(layer0, "input_layernorm")
          else layer0.ln_1)
    attn = layer0.self_attn if hasattr(layer0, "self_attn") else layer0.attn
    proj = attn.v_proj if tensor == "v" else attn.k_proj

    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    vocab = emb.weight.shape[0]

    emb, ln, proj = emb.to(device), ln.to(device), proj.to(device)
    out = np.empty((vocab, n_kv, hd), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, vocab, chunk):
            j = min(i + chunk, vocab)
            ids = torch.arange(i, j, device=device)
            h = emb(ids)
            h = ln(h)
            x = proj(h).to(dtype)
            out[i:j] = x.reshape(j - i, n_kv, hd).cpu().numpy()
            if (i // chunk) % 10 == 0:
                print(f"    {i}/{vocab}", end="\r")
    print(f"    {vocab}/{vocab}   done")

    del model
    torch.cuda.empty_cache()
    return out, tok, n_kv, hd


# --------------------------------------------------------------------------
# groupings
# --------------------------------------------------------------------------

def per_token_scales(x, qmax):
    """Current scheme: max over head_dim, one scale per (token, head)."""
    return np.abs(x).max(axis=2) / qmax           # [vocab, heads]


def collisions(f):
    f = np.ascontiguousarray(f, dtype=np.float32)
    w = f.view(np.dtype((np.void, f.dtype.itemsize * f.shape[1]))).ravel()
    _, inv, cnt = np.unique(w, return_inverse=True, return_counts=True)
    return int((cnt[inv] > 1).sum())


def quant_error(x, scales, axis, qmax):
    """
    Relative error from quantising with the given scales.

    axis 2 means the scale is shared along head_dim (per token, current
    scheme); axis 0 means it is shared along tokens (transposed scheme).
    """
    s = np.expand_dims(scales, axis)
    s = np.maximum(s, 1e-12)
    q = np.clip(np.round(x / s), -qmax, qmax)
    recon = q * s
    num = np.linalg.norm((x - recon).reshape(x.shape[0], -1), axis=1)
    den = np.linalg.norm(x.reshape(x.shape[0], -1), axis=1)
    return num / np.maximum(den, 1e-12)


def dominance_survivors(absx, sig, chunk=8192):
    """
    How many vocabulary tokens could be in a block with this signature?

    The signature is an elementwise max over the block's tokens, so any member
    must satisfy |v[c]| <= sig[c] in every channel.  Tokens violating that in
    any channel are excluded.  The survivor count is an upper bound on what the
    attacker can narrow the block to from the stored scales alone.
    """
    n = absx.shape[0]
    total = 0
    for i in range(0, n, chunk):
        blk = absx[i:i + chunk]
        total += int((blk <= sig[None, :]).all(axis=1).sum())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--tensor", default="v", choices=["k", "v"])
    ap.add_argument("--groups", default="2,4,8,16,32",
                    help="token-group sizes G for the transposed scheme")
    ap.add_argument("--qmax", type=float, default=127.0)
    ap.add_argument("--blocks", type=int, default=100,
                    help="random blocks sampled for the dominance test")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--save", default=None,
                    help="cache the layer-0 vectors so reruns skip the model")
    ap.add_argument("--load", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.load:
        d = np.load(args.load)
        x, n_kv, hd = d["x"], int(d["n_kv"]), int(d["hd"])
        print(f"loaded {args.load}: {x.shape}")
    else:
        print(f"computing layer-0 {args.tensor.upper()} for {args.model}")
        x, tok, n_kv, hd = layer0_vectors(
            args.model, args.tensor, args.device, args.chunk)
        if args.save:
            np.savez(args.save, x=x, n_kv=n_kv, hd=hd,
                     model=args.model, tensor=args.tensor)
            print(f"saved {args.save}")

    vocab = x.shape[0]
    print(f"\nvocab {vocab}   kv_heads {n_kv}   head_dim {hd}")
    print(f"tensor {args.tensor.upper()}   QUANT_MAX {args.qmax}")

    # ---- 1. baseline: current per-token-head scheme ----------------------
    s_tok = per_token_scales(x, args.qmax)
    c = collisions(s_tok)
    err_tok = quant_error(x, s_tok, 2, args.qmax)
    print(f"\n=== current scheme: max over head_dim, per (token, head) ===")
    print(f"  scales stored per token   {n_kv}")
    print(f"  collisions                {c}  ({100.0*(vocab-c)/vocab:.4f}% unique)")
    print(f"  median relative error     {np.median(err_tok)*100:.4f}%")
    print(f"  p99 relative error        {np.percentile(err_tok,99)*100:.4f}%")
    print(f"  -> the stored value is a function of one token, so it is a "
          f"lookup key")

    # ---- 2. transposed scheme --------------------------------------------
    absx = np.abs(x).reshape(vocab, n_kv * hd)
    rng = np.random.default_rng(args.seed)

    print(f"\n=== transposed scheme: max over G tokens, per (channel, head) ===")
    print(f"  {'G':>4} {'scales/token':>13} {'med err %':>11} {'p99 err %':>11} "
          f"{'survivors med':>14} {'survivors p05':>14} {'ratio to G':>11}")

    for G in [int(g) for g in args.groups.split(",")]:
        # accuracy: scale shared across G tokens, per channel
        n_blocks = vocab // G
        xb = x[:n_blocks * G].reshape(n_blocks, G, n_kv, hd)
        s_ch = np.abs(xb).max(axis=1) / args.qmax        # [blocks, heads, hd]
        s_full = np.repeat(s_ch[:, None], G, axis=1)     # broadcast back
        q = np.clip(np.round(xb / np.maximum(s_full, 1e-12)),
                    -args.qmax, args.qmax)
        recon = q * s_full
        num = np.linalg.norm((xb - recon).reshape(n_blocks * G, -1), axis=1)
        den = np.linalg.norm(xb.reshape(n_blocks * G, -1), axis=1)
        err = num / np.maximum(den, 1e-12)

        # leakage: dominance survivors for randomly sampled blocks
        surv = []
        for _ in range(args.blocks):
            idx = rng.choice(vocab, size=G, replace=False)
            sig = absx[idx].max(axis=0)
            surv.append(dominance_survivors(absx, sig))
        surv = np.array(surv)

        print(f"  {G:>4} {n_kv*hd/G:>13.1f} {np.median(err)*100:>10.4f}% "
              f"{np.percentile(err,99)*100:>10.4f}% "
              f"{int(np.median(surv)):>14} {int(np.percentile(surv,5)):>14} "
              f"{np.median(surv)/G:>11.1f}")

    print(f"\nReading the survivor columns: a block's stored scales are an")
    print(f"elementwise max over its G tokens, so any member must be dominated")
    print(f"by the signature in every channel. 'survivors' counts how many of")
    print(f"the {vocab} vocabulary tokens pass that test. A value near G means")
    print(f"the block is still essentially recoverable; a value in the")
    print(f"thousands means the stored scales no longer single out members.")
    print(f"'survivors p05' is the unlucky case, the 5th percentile over")
    print(f"sampled blocks, which is the number a defense has to survive.")
    print(f"\nThe error columns price the defense. Per-channel grouping is")
    print(f"known to suit K, whose outliers concentrate in particular channels,")
    print(f"and to suit V less well, since V errors accumulate along the token")
    print(f"axis in the attention-weighted sum. Compare both tensors before")
    print(f"drawing a conclusion.")


if __name__ == "__main__":
    main()
