#!/usr/bin/env python3
"""
Run the full grid: every model x every quantization mode, then V and K+V
nearest-neighbour distances and every plot.

Skips work that already exists, so it is safe to re-run after an interruption
or a Ctrl-C.  Profiling dominates the runtime; the distance and plotting
stages take seconds.

    stage 1  profile      one vLLM run per (model, mode)      ~3.5 h total
    stage 2  distances    V and K+V per profile               ~2 min each
    stage 3  plots        per-file, per-model, per-mode        seconds

Environment must be set before running:
    export VLLM_USE_FLASHINFER_SAMPLER=0
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VLLM_ALLOW_INSECURE_SERIALIZATION=1

Usage
    python3 run_grid.py                  # everything that is missing
    python3 run_grid.py --stage profile  # only the long part
    python3 run_grid.py --stage plot     # only regenerate figures
    python3 run_grid.py --dry-run        # show what would run
    python3 run_grid.py --models llama,qwen --modes int4
    python3 run_grid.py --models llama8b,qwen7b,mistral7b   # 8B class
"""

import os
import sys
import time
import argparse
import subprocess

OUT = "/mnt/f"

MODELS = {
    # --- 1B class ---
    "llama":  ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
    "qwen":   ("Qwen/Qwen2.5-1.5B",       "Qwen2.5-1.5B"),
    "gpt2":   ("openai-community/gpt2",   "GPT-2"),
    "pythia": ("EleutherAI/pythia-1.4b",  "Pythia-1.4B"),
    "opt":    ("facebook/opt-1.3b",       "OPT-1.3B"),
    # --- 8B class (need H200 or similar 80GB GPU) ---
    "llama8b":   ("meta-llama/Llama-3.1-8B",  "Llama-3.1-8B"),
    "qwen7b":    ("Qwen/Qwen2.5-7B",          "Qwen2.5-7B"),
    "mistral7b": ("mistralai/Mistral-7B-v0.3", "Mistral-7B-v0.3"),
    "gemma9b":   ("google/gemma-2-9b",         "Gemma-2-9B"),
}

# models requiring higher gpu-util and more blocks due to size
LARGE_MODELS = {"llama8b", "qwen7b", "mistral7b", "gemma9b"}

MODES = {
    "int8": "int8_per_token_head",
    "fp8":  "fp8_per_token_head",
    "int4": "int4_per_token_head",
}

# profiles that already exist under a different naming scheme
LEGACY = {
    ("llama", "int8"): f"{OUT}/kvmeta_int8.npz",
    ("llama", "fp8"):  f"{OUT}/kvmeta_fp8_full.npz",
    ("llama", "int4"): f"{OUT}/kvmeta_int4.npz",
    ("qwen", "int8"):   f"{OUT}/kvmeta_qwen_int8.npz",
    ("gpt2", "int8"):   f"{OUT}/kvmeta_gpt2_int8.npz",
    ("pythia", "int8"): f"{OUT}/kvmeta_pythia_int8.npz",
    ("opt", "int8"):    f"{OUT}/kvmeta_opt_int8.npz",
}


def profile_path(mk, mode):
    legacy = LEGACY.get((mk, mode))
    if legacy and os.path.exists(legacy):
        return legacy
    return f"{OUT}/kvmeta_{mk}_{mode}.npz"


def nn_path(mk, mode, feat):
    return profile_path(mk, mode).replace(".npz", f"_nn_{feat}.npz")


def profile_ok(path):
    """
    A profile counts as present only if it actually captured usable tokens.

    An all-ambiguous run (e.g. the Gemma/int4 eviction-aliasing failure) still
    writes an npz, so a bare os.path.exists would skip it forever on re-run and
    never pick up the fix.  Here we load the tiny status array and require that
    at least half the captured tokens are clean (status==1); otherwise the file
    is treated as missing and re-profiled.  Wherever OUT points, no path needs
    to be deleted by hand.

    We only force a redo when we can POSITIVELY prove the file is bad.  A file
    that exists but lacks a status array (older format) or cannot be read here
    is assumed present, so a false negative never triggers a 90-minute reprofile
    of a good profile.
    """
    if not os.path.exists(path):
        return False
    try:
        import numpy as np
        d = np.load(path)
        if "status" not in d.files:
            return True          # older format, no status array -> assume valid
        status = d["status"]
    except Exception:
        return True              # exists but unreadable here -> don't redo blindly
    captured = int((status != 0).sum())
    clean = int((status == 1).sum())
    if captured == 0:
        return False
    return clean >= 0.5 * captured


def run(cmd, dry):
    print("  $ " + " ".join(cmd))
    if dry:
        return True
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  FAILED (exit {r.returncode}) after {time.time()-t0:.0f}s")
        return False
    print(f"  ok, {time.time()-t0:.0f}s")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--stage", default="all",
                    choices=["all", "profile", "nn", "plot"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="redo steps whose output already exists")
    args = ap.parse_args()

    mks = [m.strip() for m in args.models.split(",") if m.strip() in MODELS]
    mds = [m.strip() for m in args.modes.split(",") if m.strip() in MODES]
    dry = args.dry_run

    for var in ("VLLM_USE_FLASHINFER_SAMPLER", "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        if var not in os.environ and args.stage in ("all", "profile"):
            print(f"warning: {var} not set")

    ok, failed = [], []

    # ---- stage 1: profile -------------------------------------------------
    if args.stage in ("all", "profile"):
        # Decide what to (re)profile.  A profile is redone when it is missing,
        # when --force is given, or when it exists but is INVALID (an all-
        # ambiguous run such as the Gemma/int4 failure still writes an npz).
        # For an invalid profile we announce it and remove both the bad npz and
        # any stale checkpoint, so the redo is a clean overwrite and the log
        # shows exactly what happened -- nobody has to hunt for a path to delete.
        todo = []
        for mk in mks:
            for md in mds:
                p = profile_path(mk, md)
                if args.force:
                    todo.append((mk, md))
                    continue
                if profile_ok(p):
                    continue
                if os.path.exists(p):
                    try:
                        import numpy as np
                        st = np.load(p)["status"]
                        print(f"  INVALID profile {mk}/{md}: clean "
                              f"{int((st==1).sum())}/{int((st!=0).sum())} "
                              f"-> re-profiling (will overwrite)")
                    except Exception:
                        print(f"  INVALID profile {mk}/{md}: unreadable "
                              f"-> re-profiling (will overwrite)")
                    if not dry:
                        for stale in (p, p.replace(".npz", "_ckpt.npz")):
                            try:
                                os.remove(stale)
                                print(f"    removed stale {stale}")
                            except OSError:
                                pass
                todo.append((mk, md))
        print(f"\n=== stage 1: profile, {len(todo)} runs "
              f"({len(mks)*len(mds) - len(todo)} already valid) ===")
        for mk, md in todo:
            hf, _ = MODELS[mk]
            path = f"{OUT}/kvmeta_{mk}_{md}.npz"
            print(f"\n[{mk} / {md}]")
            cmd = ["python3", "profile_kv_meta.py",
                   "--model", hf, "--kv-dtype", MODES[md],
                   "--out", path]
            if mk in LARGE_MODELS:
                cmd += ["--gpu-util", "0.90", "--blocks", "128"]
            good = run(cmd, dry)
            (ok if good else failed).append(f"profile {mk}/{md}")

    # ---- stage 2: distances ----------------------------------------------
    if args.stage in ("all", "nn"):
        print("\n=== stage 2: nearest-neighbour distances ===")
        for mk in mks:
            for md in mds:
                p = profile_path(mk, md)
                if not os.path.exists(p):
                    print(f"  skip {mk}/{md}: no profile")
                    continue
                for feat in ("v", "kv"):
                    if not args.force and os.path.exists(nn_path(mk, md, feat)):
                        continue
                    tag = f"{MODELS[mk][1]}_{md}_{feat.upper()}"
                    print(f"\n[{mk} / {md} / {feat}]")
                    good = run(["python3", "save_nn.py", "--path", p,
                                "--feature", feat, "--tag", tag], dry)
                    (ok if good else failed).append(f"nn {mk}/{md}/{feat}")

    # ---- stage 3: plots ---------------------------------------------------
    if args.stage in ("all", "plot"):
        print("\n=== stage 3: plots ===")

        # one per (model, mode, feature)
        for mk in mks:
            for md in mds:
                for feat in ("v", "kv"):
                    f = nn_path(mk, md, feat)
                    if not os.path.exists(f) and not dry:
                        continue
                    run(["python3", "plot_nn.py", "--files", f, "--mode", "all",
                         "--out", f"{OUT}/fig_{mk}_{md}_{feat}.png",
                         "--title", f"{MODELS[mk][1]}  {md}  "
                                    f"{'V' if feat=='v' else 'K+V'}"], dry)

        # per model: the three modes overlaid, for each feature
        for mk in mks:
            for feat in ("v", "kv"):
                files = [nn_path(mk, md, feat) for md in mds
                         if dry or os.path.exists(nn_path(mk, md, feat))]
                if len(files) < 2:
                    continue
                run(["python3", "plot_nn.py", "--files", *files,
                     "--out", f"{OUT}/fig_modes_{mk}_{feat}.png",
                     "--title", f"{MODELS[mk][1]}  "
                                f"{'V' if feat=='v' else 'K+V'}  "
                                f"across quantization modes"], dry)

        # per mode: all models overlaid, for each feature
        for md in mds:
            for feat in ("v", "kv"):
                files = [nn_path(mk, md, feat) for mk in mks
                         if dry or os.path.exists(nn_path(mk, md, feat))]
                if len(files) < 2:
                    continue
                run(["python3", "plot_nn.py", "--files", *files,
                     "--out", f"{OUT}/fig_models_{md}_{feat}.png",
                     "--title", f"{'V' if feat=='v' else 'K+V'}  {md}  "
                                f"across models"], dry)

        # per (model, mode): V against K+V
        for mk in mks:
            for md in mds:
                fv, fkv = nn_path(mk, md, "v"), nn_path(mk, md, "kv")
                if not dry and not (os.path.exists(fv) and os.path.exists(fkv)):
                    continue
                run(["python3", "plot_nn.py", "--files", fv, fkv,
                     "--out", f"{OUT}/fig_vkv_{mk}_{md}.png",
                     "--title", f"{MODELS[mk][1]}  {md}  V vs K+V"], dry)

    print("\n=== summary ===")
    print(f"  completed {len(ok)}")
    if failed:
        print(f"  failed {len(failed)}:")
        for f in failed:
            print(f"    {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
