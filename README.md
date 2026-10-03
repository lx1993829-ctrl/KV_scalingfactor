export VLLM_USE_FLASHINFER_SAMPLER=0 

export VLLM_WSL2_ENABLE_PIN_MEMORY=1

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

bash smoke_gemma_int4.sh    # 先几分钟验证
bash run_8b.sh              # 冒烟过了再跑

The default OUT path is OUT="/mnt/f"
To change it, modify the OUT variable in run_8b.sh and run_grid.py
