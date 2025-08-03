# Change batch size, number of epochs etc
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file recipes/accelerate_configs/zero2.yaml src/open_r1/sft.py \
    --config recipes/CP_001/sft/sft_config_amazon.yaml #zero2

#lora merge save    
python src/open_r1/model_save.py --config recipes/CP_001/sft/sft_config_amazon.yaml