CUDA_VISIBLE_DEVICES=1 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2_grpo.yaml --num_processes 1 \
    src/open_r1/grpo.py --config recipes/CP_001/grpo/grpo_config_amazon.yaml
    
python src/open_r1/model_save.py --config recipes/CP_001/grpo/grpo_config_amazon.yaml