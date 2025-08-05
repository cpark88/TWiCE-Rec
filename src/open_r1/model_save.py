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
    
    temp_args, _ = parser.parse_known_args()

    # YAML load
    with open(temp_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)

    for k, v in config_dict.items():
        parser.add_argument(f'--{k}', default=v)

    args = parser.parse_args()    
    
    ###
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16)
    print(base_model.dtype)
    model_to_merge = PeftModel.from_pretrained(base_model, args.output_dir)
    merged_model = model_to_merge.merge_and_unload()
    merged_model = merged_model.half() #fp16
    merged_model.save_pretrained(args.output_dir+'_lora', safe_serialization=True)
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code, revision='main')
    tokenizer.save_pretrained(args.output_dir+'_lora')
    
    try:
        processor = AutoProcessor.from_pretrained(args.model_name_or_path) 
        processor.save_pretrained(args.output_dir+'_lora')
    except:
        pass
    
    ###
    
if __name__ == "__main__":
    main()