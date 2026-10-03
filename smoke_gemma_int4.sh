#!/bin/bash
# =================================================================
# Smoke test for the Gemma/int4 ambiguity fix
#
# Runs profile_kv_meta.py over only the first 200 tokens, twice:
#   int8  -> control; must stay ~100% clean (proves the slot-selection
#            change did NOT break a mode that already worked)
#   int4  -> the one that came back all-ambiguous before the fix
#
# Writes to a throwaway npz (NOT /mnt/f), so it touches nothing the real
# pipeline depends on.  Takes a few minutes: two engine starts + 2x200 tokens.
#
# WHAT TO LOOK AT in the output of each run:
#     clean     NNN/200      <- want this near 200 for BOTH modes
#     ambiguous NNN          <- want this near 0
#     first failures:        <- if int4 is still ambiguous, these lines carry
#       123: n_changed=2 (raw=2)    the raw= numbers Claude needs to diagnose
#
# VERDICT:
#   int8 clean ~200  AND  int4 clean ~200   -> fix works, run the full pipeline
#   int4 still mostly ambiguous             -> STOP, send Claude the
#                                              "first failures" lines (raw=...)
# =================================================================

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

cd "$(dirname "$0")" || exit 1

MODEL="google/gemma-2-9b"
SMOKE_OUT="${SMOKE_OUT:-/tmp/kvsmoke}"   # throwaway; override if /tmp is small
mkdir -p "$SMOKE_OUT"

echo "############################################################"
echo "# CONTROL: Gemma-2-9B / int8  (should be ~200/200 clean)"
echo "############################################################"
python3 profile_kv_meta.py \
    --model "$MODEL" \
    --kv-dtype int8_per_token_head \
    --gpu-util 0.90 --blocks 128 \
    --smoke 200 \
    --out "$SMOKE_OUT/smoke_gemma_int8.npz"

echo ""
echo "############################################################"
echo "# TEST: Gemma-2-9B / int4  (the one that was all-ambiguous)"
echo "############################################################"
python3 profile_kv_meta.py \
    --model "$MODEL" \
    --kv-dtype int4_per_token_head \
    --gpu-util 0.90 --blocks 128 \
    --smoke 200 \
    --out "$SMOKE_OUT/smoke_gemma_int4.npz"

echo ""
echo "############################################################"
echo "# Read the two 'clean / ambiguous' blocks above."
echo "#   both clean ~200  -> run:  bash run_8b.sh"
echo "#   int4 ambiguous   -> send the int4 'first failures' lines"
echo "############################################################"
