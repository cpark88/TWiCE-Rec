# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import argparse
import yaml


def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    
    # 먼저 --config 받아오기
    temp_args, _ = parser.parse_known_args()

    # YAML 로드
    with open(temp_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    # YAML 값으로 전체 arg 세팅
    for k, v in config_dict.items():
        parser.add_argument(f'--{k}', default=v)

    args = parser.parse_args()    
    
    
#     parser = argparse.ArgumentParser(description="Run with a YAML config")
#     parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
#     args = parser.parse_args()
#     yaml_config = load_yaml_config(args.config)
    
#     # Parse the YAML config
#     args = parser.parse_dict(yaml_config)    

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name_or_path", default="negative", type=str)
    # parser.add_argument('--output_dir', default='output/', type=str)
    # args = parser.parse_args()
    
    ###
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    print(base_model.dtype)
    model_to_merge = PeftModel.from_pretrained(base_model, args.output_dir) # output_dir에는 이미 lora adapter 존재
    merged_model = model_to_merge.merge_and_unload()
    merged_model = merged_model.half() #fp16
    merged_model.save_pretrained(args.output_dir+'_lora', safe_serialization=True)
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, revision='main')
    tokenizer.save_pretrained(args.output_dir+'_lora')
    
    # gemma 3 27b case
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)  # 예시
        processor.save_pretrained(args.output_dir+'_lora')
    except:
        pass
    
    ###
    
if __name__ == "__main__":
    main()