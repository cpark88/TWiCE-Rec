# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys

import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from src.open_r1.configs import SFTConfig
from src.open_r1.utils import get_tokenizer, get_model
from src.open_r1.utils.callbacks import get_callbacks
from src.open_r1.utils.wandb_logging import init_wandb_training
from src.open_r1.utils.prompt import Prompt

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from trl import DataCollatorForCompletionOnlyLM

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoConfig
import random
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import login

logger = logging.getLogger(__name__)

os.environ['VLLM_USE_V1'] = '0'

login(token='xxx')


def main(script_args, training_args, model_args):
    
    
    os.environ["WANDB_MODE"] = "offline"
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)


        
    ################
    # Load datasets
    ################
    

        
    # amazon case
    
    domain = "Grocery_and_Gourmet_Food"
    dataset_train = []

    # task1: meta
    # with open(f"./src/open_r1/sasrec/amazon_dataset/meta_dataset/amazon_{domain}_llm_train_20250521.json", 'r', encoding='utf-8') as f:
    #     dataset_train_3 = json.load(f)
    #     random.shuffle(dataset_train_3)
    #     # dataset_train_3 = dataset_train_3[:200]
    #     print("train1:", len(dataset_train_3))
    #     dataset_train.extend(dataset_train_3)

    # # task2: sr
    # with open(f"./src/open_r1/sasrec/amazon_dataset/llm_dataset/amazon_{domain}_llm_train_case_2_20250521.json", 'r', encoding='utf-8') as f:
    #     dataset_train_2 = json.load(f)
    #     print("train2:", len(dataset_train_2))
    #     random.shuffle(dataset_train_2)
    #     # dataset_train_2 = dataset_train_2[:200]
    #     dataset_train.extend(dataset_train_2)

#task3: srr
    with open(f"./src/open_r1/sasrec/amazon_dataset/llm_dataset/amazon_{domain}_llm_train_case_1_20250521.json", 'r', encoding='utf-8') as f:
        dataset_train_1 = json.load(f)
        random.shuffle(dataset_train_1)
        # dataset_train_1 = dataset_train_1[:100]
        print("train3:", len(dataset_train_1))
        dataset_train.extend(dataset_train_1)
            
        
    random.shuffle(dataset_train)
        
        
        
    with open(f"./src/open_r1/sasrec/amazon_dataset/llm_dataset/amazon_{domain}_llm_test_20250521.json", 'r', encoding='utf-8') as f:
        dataset_test = json.load(f)
        random.shuffle(dataset_test)
        dataset_test = dataset_test[:500]  
        
        
    dataset_train = Dataset.from_list(dataset_train)
    dataset_test = Dataset.from_list(dataset_test)
    
    dataset = DatasetDict({
                "train": dataset_train,
                "test": dataset_test})

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)
    # tokenizer.pad_token = tokenizer.eos_token # 향후 뺴야함

    ###################
    # Load model
    ###################
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)
    
    
    ################
    # chat-template
    ################
    
    # amazon case
    def make_sft_conversation(example, prompt_column: str = 'input', completion_column: str = 'output', add_generation_prompt = False):
        prompt = []

        prompt.append({"role": "system", "content": example["system_prompt"]})        
        prompt.append({"role": "user", "content": example[prompt_column]})
        prompt.append({"role": "assistant", "content": example[completion_column] + tokenizer.eos_token})
        return {'text':tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=add_generation_prompt)}
    
    
    
    data_seed = 42
    # dataset['train'] = dataset['train'].shuffle(42).map(make_sft_conversation)
    dataset['train'] = dataset['train'].shuffle(42).map(make_sft_conversation) # curriculum no shuffle
    dataset['test'] = dataset['test'].shuffle(42).map(make_sft_conversation)
    print(dataset['train']['text'][0])
    
    ############################
    # Initialize the SFT Trainer
    ############################
    # Alphaca Style
    # response_template = "<|im_start|>assistant"
    response_template = "<start_of_turn>model"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)#, pad_to_multiple_of=8)  

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],# if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        data_collator=collator,
    )

    ###############
    # Training loop
    ###############
    
    
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        # trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)