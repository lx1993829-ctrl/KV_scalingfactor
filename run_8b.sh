#!/bin/bash
# =================================================================
# 8B Model Testing Pipeline
# Run on an H200 (80GB HBM3e) or equivalent
#
# Four models:
#   Llama-3.1-8B     8 KV heads, head_dim=128, vocab=128256, 32 layers, RoPE
#   Qwen2.5-7B       4 KV heads, head_dim=128, vocab=152064, 28 layers, RoPE
#   Mistral-7B-v0.3  8 KV heads, head_dim=128, vocab=32768,  32 layers, RoPE
#   Gemma-2-9B      16 KV heads, head_dim=256, vocab=256000, 42 layers, RoPE
#
# Each model is tested under all three quantization modes:
#   int8_per_token_head  (symmetric, scale = absmax/127)
#   fp8_per_token_head   (symmetric, scale = absmax/448)
#   int4_per_token_head  (asymmetric, scale+zp, Hadamard rotation)
#
# Expected runtime on H200:
#   profile_kv_meta (128k vocab): ~45 min per (model, mode)
#   profile_kv_meta (152k vocab): ~55 min per (model, mode)
#   profile_kv_meta (32k vocab):  ~12 min per (model, mode)
#   profile_kv_meta (256k vocab): ~90 min per (model, mode)
#   Total profiling: ~12 × avg ~50 min ≈ 10 hours
#   Layer stacking profiling: ~same as single-layer per model
#   Analysis / NN / plots: minutes
#
# The scripts are all in the same directory as this file.
# Output goes to /mnt/f/ (change OUT below if needed).
#
# NOTE: Gemma-2-9B requires accepting Google's license on HuggingFace
#       before downloading: https://huggingface.co/google/gemma-2-9b
# =================================================================
set -e

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

OUT="/mnt/f"
cd "$(dirname "$0")"

echo "============================================================"
echo "Stage 1: Layer sweep (position/context invariance, quick)"
echo "============================================================"

# Layer sweep: 3 tokens × 6 positions × 4 contexts, layers 0,1,8
# Each takes ~5 min including engine startup

for MODEL_KEY in "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" "mistralai/Mistral-7B-v0.3" "google/gemma-2-9b"; do
    TAG=$(echo "$MODEL_KEY" | sed 's|.*/||; s|\.||g')
    CSV="$OUT/sweep_${TAG}_int8.csv"
    if [ -f "$CSV" ]; then
        echo "  skip $CSV (exists)"
        continue
    fi
    echo ""
    echo "--- layer sweep: $MODEL_KEY ---"
    python3 layer_sweep.py \
        --model "$MODEL_KEY" \
        --kv-dtype int8_per_token_head \
        --layers 0,1,8 \
        --gpu-util 0.90 --blocks 128 \
        --csv "$CSV"
done

echo ""
echo "============================================================"
echo "Stage 2: Full-vocabulary profiling (all 3 modes per model)"
echo "============================================================"

# This is the slow part. ~37 min average per (model, mode).
# Uses --resume so it is safe to Ctrl-C and re-run.

python3 run_grid.py --models llama8b,qwen7b,mistral7b,gemma9b --stage profile

echo ""
echo "============================================================"
echo "Stage 3: Analysis"
echo "============================================================"

for MK in llama8b qwen7b mistral7b gemma9b; do
    for MD in int8 fp8 int4; do
        NPZ="$OUT/kvmeta_${MK}_${MD}.npz"
        if [ ! -f "$NPZ" ]; then
            echo "  skip $MK/$MD: no profile"
            continue
        fi
        echo ""
        echo "--- analyze $MK / $MD ---"
        python3 analyze_kv_meta.py --path "$NPZ"
    done
done

echo ""
echo "============================================================"
echo "Stage 4: NN distances + plots"
echo "============================================================"

python3 run_grid.py --models llama8b,qwen7b,mistral7b,gemma9b --stage nn
python3 run_grid.py --models llama8b,qwen7b,mistral7b,gemma9b --stage plot

echo ""
echo "============================================================"
echo "Stage 5: Layer stacking (position 0, all layers)"
echo "============================================================"

# Qwen2.5-7B: 4 KV heads → might need stacking (like 1.5B with 2 heads)
# Llama/Mistral: 8 heads → likely 100% at layer 0, but verify
# Gemma-2-9B: 16 KV heads (MHA) → almost certainly 100% at layer 0

for MODEL_KEY in "Qwen/Qwen2.5-7B" "meta-llama/Llama-3.1-8B" "mistralai/Mistral-7B-v0.3" "google/gemma-2-9b"; do
    TAG=$(echo "$MODEL_KEY" | sed 's|.*/||; s|\.||g')
    NPZ="$OUT/kvlayers_${TAG}_int8.npz"
    if [ -f "$NPZ" ]; then
        echo "  skip layer profiling for $TAG (exists)"
    else
        echo ""
        echo "--- profile layers: $MODEL_KEY ---"
        python3 profile_layers.py \
            --model "$MODEL_KEY" \
            --kv-dtype int8_per_token_head \
            --gpu-util 0.90 --blocks 128 \
            --out "$NPZ"
    fi
    echo ""
    echo "--- analyze layers: $TAG ---"
    python3 analyze_layers.py --path "$NPZ" --per-layer
done

echo ""
echo "============================================================"
echo "Stage 6: Analytic table (transposed_eval, no vLLM needed)"
echo "============================================================"

# Validates that analytic layer-0 V matches vLLM profiling.
# Also evaluates transposed grouping defense.

for MODEL_KEY in "meta-llama/Llama-3.1-8B" "Qwen/Qwen2.5-7B" "mistralai/Mistral-7B-v0.3" "google/gemma-2-9b"; do
    TAG=$(echo "$MODEL_KEY" | sed 's|.*/||; s|\.||g')
    CACHE="$OUT/layer0_v_${TAG}.npz"
    echo ""
    echo "--- transposed eval: $MODEL_KEY ---"
    python3 transposed_eval.py \
        --model "$MODEL_KEY" \
        --save "$CACHE" \
        --groups 2,4,8,16,32
done

echo ""
echo "============================================================"
echo "Done. Check $OUT for outputs."
echo "============================================================"
