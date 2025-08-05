# -*- coding:utf-8 -*-

from util import neg_sample, neg_sample_set, get_sample_scores, neg_sample_unigram
from customized_model import OneModelV3
from typing import Union, Dict, Optional, Sequence
import numpy as np
import os

import copy
import logging
from dataclasses import dataclass

import torch
import transformers
import util
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np


import json
from util import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int
from outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback




def custom_replace_neg(tensor, next_index, ignore_index):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor!=next_index] = ignore_index    
    return res

def custom_replace_pos(tensor, next_index, ignore_index):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==next_index] = ignore_index    
    return res

def preprocess_v2(
    targets_id: Sequence[str],
) -> Dict:
    """Preprocess the data by tokenizing.
       Add the labels_id which indicates the real indices of items.
    """
    target_id_split_original = [example.split(',') for example in targets_id]
    target_id_split = [torch.tensor([int(j) for j in i][1:]) for i in target_id_split_original ] 
    input_target_id_split = [torch.tensor([int(j) for j in i][:-1]) for i in target_id_split_original ] 
    return dict(input_ids_item=input_target_id_split, pos_ids_item=target_id_split)


class SupervisedDatasetv3(Dataset):
    """Dataset for supervised fine-tuning.
       Add the "labels_id"
       Add the negative samples and the test negative samples.
    """

    def __init__(self, data_path: str, model_type: str, len_vocab_dict_tokenized:int, neg_sample_type:str):
        super(SupervisedDatasetv3, self).__init__()
        self.model_type = model_type
        self.len_vocab_dict_tokenized = len_vocab_dict_tokenized
        
        self.neg_sample_type = neg_sample_type
        

        logging.warning("Loading data...")
        print("data_path", data_path)
        list_data_dict = util.jload(data_path)

        logging.warning("Formatting inputs...")

        sources = [example for example in list_data_dict]

        self.answer_id=[]
        if self.model_type=='inference':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) + ','+example['positive_item_id'] }" for example in list_data_dict]
            self.answer_id = [ int(example['positive_item_id'])   for example in list_data_dict]

        elif self.model_type=='train':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) + ','+example['positive_item_id'] }" for example in list_data_dict]
            self.answer_id = [ int(example['positive_item_id'])   for example in list_data_dict] 

        elif self.model_type=='valid':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) + ','+example['positive_item_id'] }" for example in list_data_dict]
            self.answer_id = [ int(example['positive_item_id'])   for example in list_data_dict]

        elif self.model_type=='test':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) + ','+example['positive_item_id'] }" for example in list_data_dict]
            self.answer_id = [ int(example['positive_item_id'])   for example in list_data_dict]

        else:
            raise ValueError("model_type should be either 'inference' or 'train' or 'valid' or 'test', but got {}".format(self.model_type))

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_v2(targets_id)

        self.input_ids_item = data_dict['input_ids_item']
        self.pos_ids_item = data_dict['pos_ids_item']


    def __len__(self):
        return len(self.input_ids_item)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        test_neg = []
        if self.model_type=='test':
            seq_set = set(self.pos_ids_item[i])
            seq_set.update({0,1,2}) 
            for _ in range(29): #99
                test_neg.append(neg_sample(seq_set, self.len_vocab_dict_tokenized)) # vocab

        # neg item
        neg_ids_item = []
        seq_set_item = set(self.pos_ids_item[i])
        seq_set_item.update({0,1,2})
        for num in range(len(self.pos_ids_item[i])):
            if self.neg_sample_type == 'basic':
                neg_ids_item.append(neg_sample(seq_set_item, self.len_vocab_dict_tokenized))

        return dict(answer_id=self.answer_id[i], test_neg=test_neg, input_ids_item=self.input_ids_item[i], pos_ids_item=self.pos_ids_item[i], neg_ids_item=torch.tensor(neg_ids_item))


def customized_pad_sequence(
    sequences: Union[torch.Tensor, list[torch.Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
    pos: str = 'right',
) -> torch.Tensor:

    """
    This function returns a Tensor of size T x B x * or B x T x * where T is the length of the longest sequence. This function assumes trailing dimensions and type of all the Tensors in sequences are same.
    """
    if pos=='right':
        padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    elif pos=='left':
        sequences = tuple(map(lambda s: s.flip(0), sequences))
        padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
        _seq_dim = padded_sequence.dim()
        padded_sequence = padded_sequence.flip(-_seq_dim+batch_first)
    else:
        raise ValueError("pos should be either 'right' or 'left', but got {}".format(pos))
    return padded_sequence

@dataclass
class DataCollatorForSupervisedDatasetv3(object):
    """Collate examples for supervised fine-tuning."""

    # tokenizer: transformers.PreTrainedTokenizer
    pad_token_id: int
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        answer_id, test_neg, input_ids_item, pos_ids_item, neg_ids_item = tuple([instance[key] for instance in instances] for key in ("answer_id", "test_neg", "input_ids_item", "pos_ids_item", "neg_ids_item")) #full

        answer_id = torch.tensor(answer_id) # B
        test_neg = torch.tensor(test_neg) # B x 0

        input_ids_item = customized_pad_sequence(input_ids_item, batch_first=True, padding_value=self.pad_token_id, pos='left')
        pos_ids_item = customized_pad_sequence(pos_ids_item, batch_first=True, padding_value=self.pad_token_id, pos='left')
        neg_ids_item = customized_pad_sequence(neg_ids_item, batch_first=True, padding_value=self.pad_token_id, pos='left')

        return dict(
            answer_id=answer_id, 
            test_neg=test_neg, 
            input_ids_item=input_ids_item,
            pos_ids_item=pos_ids_item,
            neg_ids_item=neg_ids_item,
        )

def make_supervised_data_module_v3(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDatasetv3(data_path=data_args.data_path, model_type=data_args.model_type, len_vocab_dict_tokenized=data_args.len_vocab_dict_tokenized, neg_sample_type=data_args.neg_sample_type)
    data_collator = DataCollatorForSupervisedDatasetv3(pad_token_id=data_args.pad_token_id)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)