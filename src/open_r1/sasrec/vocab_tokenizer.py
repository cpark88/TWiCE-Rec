# -*- coding:utf-8 -*-
# __author__ = Chung Park
# __date__ = 2024/3/2


# from skt.gcp import get_bigquery_client, bq_insert_overwrite, get_max_part, bq_to_df, bq_to_pandas, pandas_to_bq_table, load_query_result_to_table, PROJECT_ID
from aladin import bigquery
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import tqdm
import datetime
import os
import pandas as pd

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    RobertaModel,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    pipeline,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedTokenizer,
    HfArgumentParser
)


"""
vocab-tokenizer mapping
"""


def extract_unigram_weight_rev(item_cnt, vocab_tokenizer_data):
    
    # 아예등장하지 않은 경우는?
    # vocab에는 있으나, one_model_instruct_dataset_adot에는 없는 item_id는 0에 가까운 값을 할당 및 특수 토큰은 0.0으로 할당해서 샘플링할 때 안나오도록 함.
    temp_item = pd.DataFrame([list(range(len(vocab_tokenizer_data))), [0.0001]*len(vocab_tokenizer_data)],index=['item_index','add_cnt']).T
    temp_item['item_index'] = temp_item['item_index'].astype(int)

    item_cnt = item_cnt.merge(temp_item, on="item_index", how="outer")
    item_cnt = item_cnt.fillna(0.0)
    item_cnt['sum_cnt'] = item_cnt['item_cnt'] + item_cnt['add_cnt']
    item_cnt.loc[:5,'sum_cnt'] = 0.0
    
    # unigram distribution의 확률값 계산하는 부분
    ### 3/4 하는 이유는 skewness에 따라 인기있는 아이템이 계속 샘플링되는 정도는 완화시켜주기 위함.
    ### 이 값은 휴리스틱하게 정의되어 통상적으로 사용되는 값이며, 우리 데이터에 따라 다른 값을 줘도 됨.
    item_cnt['cnt_pow'] = pow(item_cnt['sum_cnt'], 3/4)
    item_cnt['cnt_prop'] = item_cnt['cnt_pow']/np.sum(item_cnt['cnt_pow'])
    item_cnt = item_cnt.sort_values('item_index',ascending=True).reset_index(drop=True)

    neg_totals = np.cumsum(item_cnt['cnt_prop'].tolist())
    
    
    return neg_totals


def value_padding(value:list, max_length:int, padding_index:int):
    if max_length >= len(value):
        padding_len = max_length - len(value)
        result = [padding_index for _ in range(padding_len)]+value
    else:
        result = value[:max_length]
    return result


def vocab_tokenizer_mapping():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, help='EleutherAI/polyglot-ko-3.8b')   
    parser.add_argument('--subtoken_max_length', default=64, type=int)
    parser.add_argument('--padding', default='y', type=str, help='y or n')
    parser.add_argument('--data_name', default='skt', type=str, help='skt or amazon')
    parser.add_argument('--default_next_token', default="<|n|>", type=str)
    parser.add_argument('--default_query_token', default="<q>", type=str)
    args = parser.parse_args()

    proxies = {
    "http": "http://10.40.84.223:10203",
    "https": "http://10.40.84.223:10203",
    }
    token='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'

    # IGNORE_INDEX = 0 #-100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"

    DEFAULT_NEXT_TOKEN = args.default_next_token
    DEFAULT_QUERY_TOKEN = args.default_query_token 

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=1e15,
        padding_side="right",
        use_fast=False,
        # proxies=proxies,
        # token=token
        )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    num_added_tokens = tokenizer.add_tokens([DEFAULT_NEXT_TOKEN])
    num_added_tokens = tokenizer.add_tokens([DEFAULT_QUERY_TOKEN])
    
    tokenizer.pad_token_id = tokenizer.eos_token_id #added 20240517


    conn = bigquery.connect()
    # vocab=bigquery.table_to_dataframe(conn,'x1112020','svc_amazon_log_20231201_llm_voca')
    
    
    
    if args.data_name=='skt':
        fixed_date=datetime.date(2024, 5, 9)
        vocab=bigquery.table_to_dataframe(conn,'adot_reco','onemodelV3_recgpt_final_vocabbook_index_prd')
        vocab=vocab[vocab['dt']==fixed_date].reset_index(drop=True)
    else:#amazon
        vocab=bigquery.table_to_dataframe(conn,'x1112020','svc_amazon_log_20231201_llm_voca_v3_2')
        vocab.columns=['item','item_index','type','item_cnt']
        vocab=vocab[~vocab['item'].isnull()]
        vocab=vocab[~(vocab['item']=='')]
        vocab=vocab[~vocab[['item','item_index']].duplicated()].reset_index(drop=True)
        vocab_tmp0 = pd.DataFrame({'item' : ['null0'], 'item_index' : [0], 'type' : ['etc'],'item_cnt':[0]})
        vocab_tmp1 = pd.DataFrame({'item' : ['null1'], 'item_index' : [1], 'type' : ['etc'],'item_cnt':[0]})
        vocab_tmp2 = pd.DataFrame({'item' : ['null2'], 'item_index' : [2], 'type' : ['etc'],'item_cnt':[0]})
        vocab=pd.concat([vocab,vocab_tmp0,vocab_tmp1]).reset_index(drop=True)

    

    # original vocab dict
    vocab_dict={str(index):nm for index, nm in zip(vocab['item_index'],vocab['item'])} # if index>3}
    # tokenzier vocab
    # vocab_dict_tokenized={key:tokenizer(value,return_attention_mask=False ,add_special_tokens=False)['input_ids'][:args.subtoken_max_length] for key, value in tqdm.tqdm(vocab_dict.items())}
    vocab_dict_tokenized={key:tokenizer(value,return_attention_mask=False ,add_special_tokens=False)['input_ids'] for key, value in tqdm.tqdm(vocab_dict.items())}
    # fill [0] with null index
    for i in range(max([int(i) for i in vocab_dict_tokenized.keys()])):
        try:
            vocab_dict_tokenized[str(i)]=vocab_dict_tokenized[str(i)]
        except KeyError:
            vocab_dict_tokenized[str(i)]=[0]

    # padding (set fixted length for sub-tokens)
    if args.padding == 'y':
        len_voca_tokens=[len(v) for k, v in vocab_dict_tokenized.items()]
        #option1
        # token_max_length = np.max(len_voca_tokens)
        #option2
        token_max_length = args.subtoken_max_length
        vocab_dict_tokenized = {k:value_padding(v, token_max_length, tokenizer.pad_token_id) for k, v in vocab_dict_tokenized.items()}
    elif args.padding == 'n':
        pass
    else:
        raise ValueError("padding should be either 'y' or 'n', but got {}".format(args.padding))
    
    vocab_type = vocab.groupby('type')['item_cnt'].sum().reset_index().sort_values(by='item_cnt',ascending=False).reset_index(drop=True).reset_index()
    vocab_type.columns=['type_index','type','cnt']
    vocab_type['type_index']=vocab_type['type_index']+5 # special token 4개 고려 
    print(vocab_type)
    vocab=pd.merge(vocab,vocab_type[['type','type_index']],on='type')
    
    vocab_item_type_dict={str(index):str(typ) for index, typ in zip(vocab['item_index'],vocab['type_index'])}
    
    vocab_type_set_dict={}
    for typ in list(set(vocab['type_index'])):
        typ_item_set = list(vocab[vocab['type_index']==typ]['item_index'].drop_duplicates())
        item_set = []
        for item in typ_item_set:
            item_set.append(str(item))
        vocab_type_set_dict[str(typ)] = item_set

        
    vocab_cnt = vocab[['item_index','item_cnt']]
    neg_totals = extract_unigram_weight_rev(vocab_cnt, vocab_dict_tokenized)

    if not os.path.exists('./token_mapping'):
        os.makedirs('./token_mapping')
    # json save
    model_path = '_'.join(args.model_name.split('/'))
    
    np.save(f'token_mapping/{model_path}_vocab_neg_wei_{args.data_name}_v3', neg_totals)    
    

        
    with open(f'token_mapping/{model_path}_vocab_mapping_{args.data_name}_v3.json', 'w') as f : 
        json.dump(vocab_dict, f, indent=4)

    with open(f'token_mapping/{model_path}_vocab_tokenizer_mapping_{args.data_name}_v3.json', 'w') as f : 
        json.dump(vocab_dict_tokenized, f, indent=4)
        
    with open(f'token_mapping/{model_path}_vocab_id_type_mapping_{args.data_name}_v3.json', 'w') as f : 
        json.dump(vocab_item_type_dict, f, indent=4)
        
    with open(f'token_mapping/{model_path}_vocab_type_set_mapping_{args.data_name}_v3.json', 'w') as f : 
        json.dump(vocab_type_set_dict, f, indent=4)

    print("Complete saving vocab-tokenized dictionary!")
    
if __name__ == "__main__":
    vocab_tokenizer_mapping()

