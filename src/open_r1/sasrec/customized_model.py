# -*- coding:utf-8 -*-
# __author__ = Chung Park & TAESAN
# __create_date__ = 2024/3/2
# __last_modified_date__ = 2024/4/5

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import json
import numpy as np
import copy
import random

# from sequential_reco import Encoder, EmbHead, LayerNorm, PositionEmbedding, EncoderItem, Intermediate, IntermediateItem, ItemEncoder
# from outputs import CustomModelOutput
import argparse
from types import SimpleNamespace
# from util import dict_str_key_to_int, sequential_loss, clm_loss, sequential_loss_item, kl_divergence_loss, AdaptiveLossWeighting, normalize_loss, init_weights


class OneModelV3(nn.Module):
    def __init__(self, model_args, data_args, training_args):
        super(OneModelV3, self).__init__()
        self.model_args, self.data_args, self.training_args = model_args, data_args, training_args
        print(f'Initializing language decoder ...')
        # Item seq
        self.training_args.hidden_size_item =  128
        self.training_args.num_attention_heads_item =  2
        self.training_args.attention_probs_dropout_prob_item =  0.5
        self.training_args.hidden_act_item = 'gelu'
        self.training_args.hidden_dropout_prob_item =  0.5
        self.training_args.num_hidden_layers_item =  2
        self.item_embedding = PositionEmbedding(self.training_args, self.data_args)
        self.seq_encoder_item = EncoderItem(self.training_args)
        self.prediction_layer_item = IntermediateItem(self.training_args)
        
        self.init_weights=init_weights
        self.apply(self.init_weights)
        print('SASRec decoder initialized.')

    def forward(self,  test_neg, answer_id, input_ids_item, pos_ids_item, neg_ids_item):
        """
        Note that all elements in DataCollector should be used as input in forward function.  
        Extracting the last hidden state and user embedding
        """
        if input_ids_item is not None:#only training (sometimes for inference)
            # id-based seqeunce reco modeling
            input_ids_item_emb = self.item_embedding(input_ids_item)
            pooled_output_item = self.seq_encoder_item(input_ids_item_emb, input_ids_item, self.data_args.pad_token_id, output_all_encoded_layers=True) #input_ids_item나 pos_ids_item나 상관없음 self.IGNORE_INDEX
            pooled_output_item = pooled_output_item[-1] # BxSxH
            pooled_output_item = self.prediction_layer_item(pooled_output_item)
            pooled_last_output_item = pooled_output_item[:,-1,:]#only for item temp 나중에 지워야함. 

        else:#only inference
            pass

        if test_neg is None:# before cross-attention, only inference
            return pooled_output_item, pooled_last_output_item#only for item temp 나중에 지워야함.
            
        else: #only training
            loss_item_seq = sequential_loss_item(self.item_embedding.item_embeddings, pooled_output_item, pos_ids_item, neg_ids_item, self.training_args, self.data_args)


            return CustomModelOutput(
                loss=loss_item_seq,
                sequential_loss=None,#loss_llm_seq,
                llm_loss=None,#loss_llm_clm,
                item_loss=loss_item_seq,
                kl_loss=None,#loss_kl,
                logits=pooled_output_item,#pooled_output,
                past_key_values=None, #outputs.past_key_values,
                hidden_states=None, #outputs.hidden_states,
                attentions=None #outputs.attentions,
            )

    def evaluate(self, answer_id, test_neg, input_ids_item):
        """
        Leave-one-out Evaluation
        answer_id : seq내 마지막 item index (정답) 
        test_neg : test 용 100개 item index
        input_ids_item : input item seq
        """
        _, seq_out = self.forward(test_neg=None, answer_id=None, input_ids_item=input_ids_item, pos_ids_item=None, neg_ids_item=None)
        test_items = torch.cat((answer_id.unsqueeze(-1), test_neg), -1) # B x 1 + B x 100  -> B x 101 
        test_item_emb = self.item_embedding.item_embeddings(test_items)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits
    
    def extract_logits(self, answer_id, test_neg, input_ids_item):
        """
        Leave-one-out Evaluation
        answer_id : seq내 마지막 item index (정답) 
        test_neg : test 용 100개 item index
        input_ids_item : input item seq
        """
        _, seq_out = self.forward(test_neg=None, answer_id=None, input_ids_item=input_ids_item, pos_ids_item=None, neg_ids_item=None)
        # test_items = torch.cat((answer_id.unsqueeze(-1), test_neg), -1) # B x 1 + B x 100  -> B x 101
        test_items = answer_id.unsqueeze(-1) # B x 1
        test_item_emb = self.item_embedding.item_embeddings(test_items)
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 1]
        test_logits = torch.sigmoid(test_logits)
        return test_logits

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding:utf-8 -*-
# __author__ = Chung Park, Kim TaeSan
# __date__ = 2024/3/2

import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ReLU, Tanh

def initialize_non_glu(module, input_dim, output_dim):
    gain_value = np.sqrt((input_dim + output_dim) / np.sqrt(4 * input_dim))
    torch.nn.init.xavier_normal_(module.weight, gain=gain_value)
    # torch.nn.init.zeros_(module.bias)
    return

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class EmbHead(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, training_args):
        super(EmbHead, self).__init__()
        layers = [
            torch.nn.Linear(training_args.hidden_size, training_args.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(training_args.hidden_size, training_args.hidden_size),
        ]
        self.mlp_head = torch.nn.Sequential(*layers)

    def forward(self, embeddings):
        embeddings = self.mlp_head(embeddings)
        
        return embeddings
    
class SelfAttention(nn.Module):
    def __init__(self, training_args):
        super(SelfAttention, self).__init__()
        if training_args.hidden_size % training_args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (training_args.hidden_size, training_args.num_attention_heads))
        self.num_attention_heads = training_args.num_attention_heads
        self.attention_head_size = int(training_args.hidden_size / training_args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.key = nn.Linear(training_args.hidden_size, self.all_head_size)
        self.value = nn.Linear(training_args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(training_args.attention_probs_dropout_prob)

        self.dense = nn.Linear(training_args.hidden_size, training_args.hidden_size)
        self.LayerNorm = LayerNorm(training_args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(training_args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self, training_args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(training_args.hidden_size, training_args.hidden_size * 4)
        if isinstance(training_args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[training_args.hidden_act]
        else:
            self.intermediate_act_fn = training_args.hidden_act

        self.dense_2 = nn.Linear(training_args.hidden_size * 4, training_args.hidden_size)
        self.LayerNorm = LayerNorm(training_args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(training_args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class Layer(nn.Module):
    """
    Self-Attention + Fully-Connected Layer
    """
    def __init__(self, training_args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(training_args)
        self.intermediate = Intermediate(training_args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    """
    Final Self-Attention Layer
    """
    def __init__(self, training_args):
        super(Encoder, self).__init__()
        layer = Layer(training_args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(training_args.num_hidden_layers)])
        
    def make_att_mask(self, input_ids, pad_token_id):
        seq_attention_mask = (input_ids > pad_token_id).long()
        extended_attention_mask = seq_attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = seq_attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # 0 or 1
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, hidden_states, input_ids, pad_token_id, output_all_encoded_layers=True):
        attention_mask = self.make_att_mask(input_ids, pad_token_id)
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


    
class PositionEmbedding(nn.Module):
    def __init__(self, training_args, data_args):
        super(PositionEmbedding, self).__init__()
        self.item_embeddings = nn.Embedding(data_args.len_vocab_dict_tokenized, training_args.hidden_size_item, padding_idx=data_args.pad_token_id)
        self.position_embeddings = nn.Embedding(training_args.model_max_length, training_args.hidden_size_item)
        self.LayerNorm = LayerNorm(training_args.hidden_size_item, eps=1e-12)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, input_ids_item): # BxS
        seq_length = input_ids_item.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids_item.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids_item)
        item_embeddings = self.item_embeddings(input_ids_item)
        position_embeddings = self.position_embeddings(position_ids)

        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb



class SelfAttentionItem(nn.Module):
    def __init__(self, training_args):
        super(SelfAttentionItem, self).__init__()
        if training_args.hidden_size_item % training_args.num_attention_heads_item != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (training_args.hidden_size_item, training_args.num_attention_heads_item))
        self.num_attention_heads = training_args.num_attention_heads_item
        self.attention_head_size = int(training_args.hidden_size_item / training_args.num_attention_heads_item)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(training_args.hidden_size_item, self.all_head_size)
        self.key = nn.Linear(training_args.hidden_size_item, self.all_head_size)
        self.value = nn.Linear(training_args.hidden_size_item, self.all_head_size)

        self.attn_dropout = nn.Dropout(training_args.attention_probs_dropout_prob_item)

        self.dense = nn.Linear(training_args.hidden_size_item, training_args.hidden_size_item)
        self.LayerNorm = LayerNorm(training_args.hidden_size_item, eps=1e-12)
        self.out_dropout = nn.Dropout(training_args.hidden_dropout_prob_item)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class IntermediateItem(nn.Module):
    def __init__(self, training_args):
        super(IntermediateItem, self).__init__()
        self.dense_1 = nn.Linear(training_args.hidden_size_item, training_args.hidden_size_item * 4)
        if isinstance(training_args.hidden_act_item, str):
            self.intermediate_act_fn = ACT2FN[training_args.hidden_act_item]
        else:
            self.intermediate_act_fn = training_args.hidden_act_item

        self.dense_2 = nn.Linear(training_args.hidden_size_item * 4, training_args.hidden_size_item)
        self.LayerNorm = LayerNorm(training_args.hidden_size_item, eps=1e-12)
        self.dropout = nn.Dropout(training_args.hidden_dropout_prob_item)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class LayerItem(nn.Module):
    """
    Self-Attention + Fully-Connected Layer
    """
    def __init__(self, training_args):
        super(LayerItem, self).__init__()
        self.attention = SelfAttentionItem(training_args)
        self.intermediate = IntermediateItem(training_args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class EncoderItem(nn.Module):
    """
    Final Self-Attention Layer
    """
    def __init__(self, training_args):
        super(EncoderItem, self).__init__()
        layer = LayerItem(training_args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(training_args.num_hidden_layers_item)])
        
    def make_att_mask(self, input_ids, pad_token_id):
        seq_attention_mask = (input_ids > pad_token_id).long()
        extended_attention_mask = seq_attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = seq_attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) # 0 or 1
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, hidden_states, input_ids, pad_token_id, output_all_encoded_layers=True):
        attention_mask = self.make_att_mask(input_ids, pad_token_id)
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class ItemEncoder: # sequence item에 해당하는 sub token emb 평균 추출 sample_id
    """
    Extracting trainable item embedding (pos/neg)
    """
    def __init__(self, llm_model, emb_head, ignore_index, data_args, training_args):
        super(ItemEncoder, self).__init__()
        self.model = llm_model
        self.emb_head = emb_head
        self.ignore_index = ignore_index
        self.data_args = data_args
        self.training_args = training_args
        
    def forward(self, padded_instances, item_enc_type):
        padded_att = torch.logical_not(torch.isin(padded_instances, torch.tensor([self.data_args.pad_token_id, ignore_index, self.data_args.unk_token_id]).cuda() )).long()

        if item_enc_type=='fc_layer':
            item_emb = self.model.get_input_embeddings()(padded_instances.cuda()) # B x S x #sub_token x H
            item_emb = self.emb_head(item_emb) # B x S x #sub_token x H
            padded_att = padded_att.unsqueeze(-1).expand(item_emb.size()) # B x S x #sub_token x H
            sum_embeddings = torch.sum(item_emb * padded_att, 2) # B x S x H
            sum_mask = torch.clamp(padded_att.sum(2), min=1e-9) # B x S x H
            item_emb = sum_embeddings/sum_mask   # B x S x H

        elif item_enc_type == 'shared_llm':
            # padded_instances --> B x S x #sub-token
            num_sub_tokens = padded_instances.shape[-1] # #sub-token
            seq_len = padded_instances.shape[1] # S

            padded_instances = padded_instances.view(-1, num_sub_tokens) # BS x #sub-token
            attention_mask = padded_instances.ne(data_args.pad_token_id)

            item_emb = self.model(input_ids=padded_instances, attention_mask=attention_mask, labels=None, output_hidden_states=True)
            item_emb = item_emb.hidden_states[-1] # BS x #sub-token x H
            item_emb = item_emb.detach()

            #option 1
            item_emb = item_emb[:,-1,:] # BS x H
            item_emb = item_emb.view(-1, seq_len, self.model.config.hidden_size) # B x S x H

        else:
            raise ValueError("item_enc_type should be either 'mean_pooling' or 'mean_transformer' or 'mean_fc' or 'llm_share', but got {}".format(item_enc_type))

        return item_emb
    
    
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
from transformers.utils import ModelOutput
import transformers
from transformers import TrainerCallback

@dataclass
class CustomModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    sequential_loss: Optional[torch.Tensor] = None
    llm_loss: Optional[torch.Tensor] = None
    item_loss: Optional[torch.Tensor] = None
    kl_loss: Optional[torch.Tensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    loss_type: str = field(default=None)#added
    lora_r: int = field(default=None)#added #16,#self.args['lora_r'],
    lora_alpha: int= field(default=None)  #16,#self.args['lora_alpha'],
    lora_dropout: float = field(default=None) #added
    pretrained_model_path: str = field(default=None) #added

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    model_type: str = field(default=None, metadata={"help": "full_train, train, valid, test."})
    len_vocab_dict_tokenized: int = field(default=None, metadata={"help": "num of original tokens."})
    data_name: str = field(default=None, metadata={"help": "skt or amazon."})
    default_next_token : str = field(default=None, metadata={"help": "<|n|>"})
    default_query_token : str = field(default=None, metadata={"help": "<q>"})
    neg_sample_type : str = field(default='basic', metadata={"help": "basic"})
    pad_token_id : int = field(default=None, metadata={"help": "2"})
    

    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    lora_yn: str = field(default=None)#added
    token: str = field(default=None)#added
    clm_loss: str = field(default=None)#added
    item_enc_type: str = field(default=None)#added
    peft_method: str = field(default=None)#added
    
    
    
class MyCallback(TrainerCallback):
    "A callback that prints a grad at every step"
       
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(logs)
            
            

            
            
            
            
            
import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union

# import openai
import tqdm
# from openai import openai_object
import copy
import random
# StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]
from typing import Dict, Optional, Sequence
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, BaseModelOutput, CausalLMOutput, CausalLMOutputWithPast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer

from transformers import Trainer, TrainingArguments, TrainerCallback
import logging
from torch.nn import Linear, BatchNorm1d, ReLU, Tanh

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")

    
def init_weights(module):
    """ Initialize the weights.
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
            
            
def smart_tokenizer_and_embedding_resize_v3(
    special_tokens_dict: Dict,
    added_tokens : str,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """
    Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    num_added_tokens = tokenizer.add_tokens(added_tokens)
    num_new_tokens = num_added_tokens+num_new_tokens

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def sequential_loss(pooled_output, labels_id, neg_sample_id, labels_token_id, loss_type, training_args, num_virtual_tokens, tokenizer, item_encoder, item_enc_type, backbone_model):
    if loss_type == 'bpr':
        if training_args.peft_method=='p_tuning':
            tensor = torch.full((labels_id.size(0), num_virtual_tokens, labels_id.size(2)), tokenizer.pad_token_id).to(labels_id.device) #추가#20240904
            # tensor = torch.full((labels_id.size(0), num_virtual_tokens), tokenizer.pad_token_id).to(labels_id.device) #추가
            labels_id = torch.cat([tensor, labels_id], dim = 1) #추가 Bx16xsubtoken |BxSxsubtoken 
            neg_sample_id = torch.cat([tensor, neg_sample_id], dim = 1) #추가
            
            tensor = torch.full((labels_token_id.size(0), num_virtual_tokens), tokenizer.pad_token_id).to(labels_token_id.device) #추가
            labels_token_id = torch.cat([tensor, labels_token_id], dim = -1)
            del tensor

        pos_emb = item_encoder(padded_instances=labels_id, item_enc_type=item_enc_type)
        neg_emb = item_encoder(padded_instances=neg_sample_id, item_enc_type=item_enc_type)

        # [B*S X H]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = pooled_output.view(-1, backbone_model.config.hidden_size) # [B*S X H]
        pos_logits = torch.sum(pos * seq_emb, -1) # [B*S]
        neg_logits = torch.sum(neg * seq_emb, -1) # [B*S]

        istarget = (labels_token_id > tokenizer.pad_token_id).view(labels_token_id.size(0) * labels_token_id.size(1)).float() # [B*S]#20240904

        loss_seq = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)

    elif loss_type == 'ce':
        loss_seq = nn.CrossEntropyLoss()
        pooled_logits = self.score(pooled_output) # B x S x Item_size(original vocab size)
        loss_seq = loss_seq(pooled_logits[labels_id!=self.tokenizer.pad_token_id], labels_id[labels_id!=self.tokenizer.pad_token_id].cuda())
    else:
        return ValueError("loss_type should be either 'bpr' or 'ce', but got {}".format(self.model_args.loss_type))
    return loss_seq

def clm_loss(outputs, clm_loss, training_args):
    if clm_loss=='y':
        loss_clm = outputs.loss
    elif clm_loss=='n':
        loss_clm = torch.tensor(0)
    else:
        return ValueError("clm_loss should be either 'y' or 'n', but got {}".format(training_args.clm_loss))
    return loss_clm

def sequential_loss_item(item_embeddings, seq_out, pos_id, neg_id, training_args, data_args):
        # ID seq loss
        pos_emb = item_embeddings(pos_id)
        neg_emb = item_embeddings(neg_id)

        # [B*S X H]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))

        seq_emb = seq_out.view(-1, seq_out.size(2))#training_args.hidden_size_item) # [B*S X H]
        pos_logits = torch.sum(pos * seq_emb, -1) # [B*S]
        neg_logits = torch.sum(neg * seq_emb, -1) # [B*S]
        istarget = (pos_id > data_args.pad_token_id).view(pos_id.size(0) * pos_id.size(1)).float() # [B*S]

        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / (torch.sum(istarget)+1e-24)
        
        return loss

def kl_divergence_loss(student_output, teacher_output, temperature):
    """
    student_output: 학생 모델의 attention layer 출력
    teacher_output: 교사 모델의 attention layer 출력
    temperature: 온도 매개변수
    """
    # 온도를 반영한 softmax 분포 계산
    teacher_probs = F.softmax(teacher_output / temperature, dim=2) # B x S x H
    student_log_probs = F.log_softmax(student_output / temperature, dim=2) # B x S x H

    # KL Divergence 계산 (교사와 학생의 분포 차이를 계산)
    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return kl_div
    

# Adaptive weight를 위한 클래스 정의
class AdaptiveLossWeighting(nn.Module):
    def __init__(self, num_losses):
        super(AdaptiveLossWeighting, self).__init__()
        # 각 손실에 대한 log variance를 학습할 파라미터로 설정
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, losses):
        weighted_losses = 0
        for i, loss in enumerate(losses):
            # log variance를 사용하여 가중치를 계산하고 손실에 반영
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses += weighted_loss
        return weighted_losses

# 손실을 표준화하기 위한 함수
def normalize_loss(loss, mean, std):
    return (loss - mean) / (std + 1e-6)  # 분모에 작은 값을 더해 0으로 나누는 것을 방지

def dict_str_key_to_int(target_dict):
    """
    String key --> Int key
    """
    return {int(k):v for k,v in target_dict.items()}
    
def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)


def get_sample_scores(epoch, pred_list):
    pred_list = (-pred_list).argsort().argsort()[:, 0]
    HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
    HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
    HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
    post_fix = {
        "Epoch": epoch,
        "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
        "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
        "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
        "MRR": '{:.4f}'.format(MRR),
    }
    print(post_fix)
    return ([HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix))


def neg_sample(item_set, item_size):  
    item = random.randint(2, item_size - 1)
    while item in item_set:
        item = random.randint(2, item_size - 1)
    return item

def neg_sample_set(item_set, item_total):  
    item = int(random.choice(item_total))
    while (item in item_set) | (item == 0) | (item == 1):
        item = int(random.choice(item_total))
    return item
    
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def weighted_sample(totals, sample_size):
    rnd = random.random() * totals[-1]
    idx = np.searchsorted(totals,rnd,'right')
    sample = idx
    return sample

def neg_sample_unigram(item_set, item_size, weight):  
    item = weighted_sample(weight,1)
    while item in item_set:
        item = weighted_sample(weight,1)
    return item





class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Forward pass
        outputs = model(**inputs)
        # logits = outputs.logits
        
        # Compute your custom losses (as an example)
        sequential_loss = outputs.sequential_loss  # Custom loss 1 computation
        llm_loss = outputs.llm_loss  # Custom loss 2 computation
        item_loss = outputs.item_loss
        kl_loss = outputs.kl_loss
        # Combine losses if necessary (optional)
        total_loss = outputs.loss

        # Log individual losses
        self.log({'total_loss':round(total_loss.item(), 2),'sequential_loss': round(sequential_loss.item(), 2), 'llm_loss': round(llm_loss.item(), 2), 'item_loss': round(item_loss.item(), 2), 'kl_loss': round(kl_loss.item(), 2)})

        return (total_loss, outputs) if return_outputs else total_loss

# Custom callback to log additional losses and round epoch
class CustomCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        # # Compute the current epoch as a fraction
        # current_epoch = state.global_step / state.max_steps * args.num_train_epochs
        # if current_epoch - math.floor(current_epoch) < 0.5:
        #     # Log if the current epoch is near the 0.01 mark
        #     control.should_log = True

        # Ensure logging happens only on the main process (local_rank == 0)
        if args.local_rank != 0:
            return  # Skip logging for non-main processes
        
        # Compute the current epoch as a fraction
        current_epoch = state.global_step / state.max_steps * args.num_train_epochs
        
        # Check if the current epoch is close to an integer (1.0, 2.0, etc.)
        if args.local_rank == 0 and abs(current_epoch % 10.0) < (1 / state.max_steps):  # Small tolerance to capture 1.0 intervals
            # Force logging if the current epoch is near the 1.0 mark
            control.should_log = True
        else:
            # Suppress logging for other steps
            control.should_log = False


    def on_log(self, args, state, control, logs=None, **kwargs):
        # if logs is not None and 'epoch' in logs:
        #     # Round the epoch to 0.01
        #     logs['epoch'] = round(logs['epoch'], 2)
        if args.local_rank != 0:
            return  # Skip logging for non-main processes        

        if logs is not None and 'epoch' in logs:
            # Round the epoch to 0.5
            # logs['epoch'] = round(logs['epoch'] * 2) / 2  # Round to nearest 0.5
            logs['epoch'] = round(logs['epoch'])# Round to nearest 0.5

        # Add other custom log values if needed
        if logs is not None and 'total_loss' in logs:
            logs['total_loss'] = logs['total_loss']
        if logs is not None and 'sequential_loss' in logs:
            logs['sequential_loss'] = logs['sequential_loss']
        if logs is not None and 'llm_loss' in logs:
            logs['llm_loss'] = logs['llm_loss']
        if logs is not None and 'item_loss' in logs:
            logs['item_loss'] = logs['item_loss']
        if logs is not None and 'kl_loss' in logs:
            logs['kl_loss'] = logs['kl_loss']

        # Call the parent class's on_log to handle default logs like grad_norm, learning_rate, etc.
        # super().on_log(args, state, control, logs)
        
# # GradNorm 클래스 정의
# class GradNorm:
#     def __init__(self, num_losses, alpha=0.5):
#         self.weights = nn.Parameter(torch.ones(num_losses))  # 초기 가중치는 1로 설정
#         self.alpha = alpha  # 하이퍼파라미터: 기울기 균형 조정을 위한 지수값

#     def compute_grad_norm(self, model, losses):
#         # 각 손실의 기울기 norm 계산
#         norms = []
#         for loss in losses:
#             model.zero_grad()
#             loss.backward(retain_graph=True)
#             grad_norm = 0
#             for param in model.parameters():
#                 if param.grad is not None:
#                     grad_norm += torch.norm(param.grad, p=2)  # L2 norm
#             norms.append(grad_norm.item())
#         return torch.tensor(norms)

#     def update_weights(self, losses, initial_grad_norm, current_grad_norm):
#         # 각 손실의 상대적인 기울기 norm 계산
#         relative_norms = current_grad_norm / initial_grad_norm
#         mean_relative_norm = relative_norms.mean()

#         # 가중치 업데이트 규칙 적용
#         target_norms = mean_relative_norm * (relative_norms ** self.alpha)
#         self.weights.data = self.weights.data * (target_norms / current_grad_norm)

#     def get_weighted_loss(self, losses):
#         weighted_losses = 0
#         for i, loss in enumerate(losses):
#             weighted_losses += self.weights[i] * loss
#         return weighted_losses