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