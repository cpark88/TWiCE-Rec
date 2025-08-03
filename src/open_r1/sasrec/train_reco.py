# -*- coding:utf-8 -*-
# __author__ = Chung Park
# __date__ = 2024/3/2


from util import neg_sample, neg_sample_set, get_sample_scores
from customized_model import OneModelV3
# from customized_model_noise import OneModelV3
import json
from typing import Union, Dict, Optional, Sequence
import tqdm
import numpy as np
import os

import copy
import logging

import torch
import transformers
import util
from torch.utils.data import Dataset
from transformers import Trainer


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
from safetensors import safe_open

from outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback
from dataset import make_supervised_data_module_v3
from util import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int, CustomTrainer, CustomCallback


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 

def train():

    wandb.init(mode="offline") #offline disabled
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names=['answer_id', 'test_neg','input_ids_item', 'pos_ids_item', 'neg_ids_item'] # add inputs (multi-task) 
    
    with open(f'amazon_dataset/vocab/{data_args.data_name}_vocab_{data_args.domain}.json', 'r') as f:
        vocab_dict = json.load(f)

    data_args.len_vocab_dict_tokenized = len(vocab_dict)+1

    one_model = OneModelV3(model_args, data_args, training_args)


    data_args.model_type = 'train'
    data_args.pad_token_id = 2
    data_module = make_supervised_data_module_v3(data_args=data_args) # ->dict
    betas = (0.9, 0.999)
    optim = torch.optim.Adam(one_model.parameters(), lr=0.0001, betas=betas, weight_decay=0)
    trainer = Trainer(model=one_model, tokenizer=None, args=training_args, **data_module) #CustomTrainer Trainer


    trainer.train()
    print("Training End!")
    trainer.save_state()
    print("Save State!")


    if int(os.environ.get("LOCAL_RANK", 0)) == 0:

        one_model.eval()
        data_args.model_type = 'test'
        data_module = make_supervised_data_module_v3(data_args=data_args)
        batch_size=256
        test_dataloader= DataLoader(data_module['train_dataset'], batch_size=batch_size, collate_fn=data_module['data_collator'], shuffle=False)
        str_code='test'
        epoch=0
        rec_data_iter = tqdm.tqdm(enumerate(test_dataloader),
                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
                          total=len(test_dataloader),
                          bar_format="{l_bar}{r_bar}")

        pred_list_llm = None
        type_pos_list=[]

        with torch.no_grad():
            for i, batch in rec_data_iter:
                answer_id, test_neg, input_ids_item = batch['answer_id'], batch['test_neg'], batch['input_ids_item']
                if input_ids_item.shape[0]==batch_size:
                    test_logits_llm = one_model.evaluate(answer_id=answer_id.cuda(), test_neg=test_neg.cuda(), input_ids_item=input_ids_item.cuda())
                    test_logits_llm = test_logits_llm.cpu().detach().numpy().copy()

                    if i == 0:
                        pred_list_llm = test_logits_llm
                    else:
                        pred_list_llm = np.append(pred_list_llm, test_logits_llm, axis=0)


        print("Model Performance for LLM")
        print("================================================")
        print(get_sample_scores(epoch, pred_list_llm))
        print("================================================")

if __name__ == "__main__":
    train()