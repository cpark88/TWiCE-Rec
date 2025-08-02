export VLLM_USE_V1="0"
# export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /home/jovyan/cp-gpu-4-datavol-one-model/one-model-v4/one_model_v4_work/open-r1_20250411/data/google_gemma-3-1b-it_sft_001_amazon_Amazon_Fashion_lora --tensor-parallel-size 1 --max-model-len 13000 --dtype bfloat16 --port 8001
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model google/gemma-3-27b-it --tensor-parallel-size 1 --max-model-len 13000 --dtype bfloat16 --port 8001
# CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /home/jovyan/cp-gpu-4-datavol-one-model/one-model-v4/one_model_v4_work/open-r1_20250411/data/curriculum_reco_low_lr_202506_merged --tensor-parallel-size 1 --max-model-len 13000 --dtype bfloat16 --port 8001