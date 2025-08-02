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