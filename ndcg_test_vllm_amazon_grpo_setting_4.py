# run_vllm.py
import multiprocessing as mp
import numpy as np
# from blocker_numpy import blocker
# Data training arguments
from datasets import load_from_disk

import torch
# from blocker_torch import blocker
# from src.open_r1.utils.prompt import Prompt
import tqdm
import json
import re
import pandas as pd
import sys
import subprocess
import os
import math
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList, BitsAndBytesConfig
# from transformers import AutoProcessor, Gemma3ForConditionalGeneration
os.environ['VLLM_USE_V1'] = '0'

from huggingface_hub import login

login(token='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')

import re

def filter_valid_asins(string_list):
    """
    10자리 알파벳 대문자/숫자로만 된 문자열만 필터링합니다.

    Args:
        string_list (list of str): 입력 문자열 리스트

    Returns:
        list of str: 유효한 10자리 문자열 리스트
    """
    pattern = re.compile(r'^[A-Z0-9]{10}$')
    return [s for s in string_list if pattern.match(s)]

def extract_items(text):
    try:
        items = re.findall(r"<item_id>(.*?)</item_id>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'

def extract_items_name(text):
    try:
        items = re.findall(r"<item_nm>(.*?)</item_nm>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'

def extract_reasoning(text):
    try:
        items = re.findall(r"<think>(.*?)</think>", text, re.DOTALL)[0].strip()
    except:
        items = []
    return items if items else '//no'


def calculate_hr_ndcg(outputs: list, ground_truth: str, top_n: int = 5):
    """
    Args:
        outputs (list): LLM이 생성한 추천 결과 리스트 (우선순위 순)
        ground_truth (str): 실제 정답 아이템
        top_n (int): 평가할 top-N 범위

    Returns:
        hr (float): Hit Rate@N
        ndcg (float): NDCG@N
    """
    # 상위 top_n까지 자르기
    top_outputs = outputs[:top_n]

    # 정답 아이템이 top-N 내에 있는지
    if ground_truth in top_outputs:
        hr = 1.0
        rank = top_outputs.index(ground_truth) + 1  # 1-based index
        ndcg = 1 / math.log2(rank + 1)
    else:
        hr = 0.0
        ndcg = 0.0

    return hr, ndcg

def evaluate_all(predictions: list, ground_truths: list, top_n: int = 10):
    """
    predictions: list of list, 각 유저에 대한 생성 결과
    ground_truths: list of str, 각 유저의 정답 아이템
    """
    hr_total, ndcg_total = 0.0, 0.0
    num_samples = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        hr, ndcg = calculate_hr_ndcg(pred, gt, top_n)
        hr_total += hr
        ndcg_total += ndcg

    return hr_total / num_samples, ndcg_total / num_samples


def dedup_preserve_order(seq):
    return list(dict.fromkeys(seq))


def inference():
    task = 'case1'#case2 case1 basic cross_domain
    type_ = '001'
    stage = 'stdpo'    
    domain = "Amazon_Fashion"
    print(domain)
    model_name = f"/home/jovyan/cp-gpu-4-datavol-one-model/one-model-v4/one_model_v4_work/open-r1_20250411/data/google_gemma-3-1b-it_{stage}_{type_}_amazon_{domain}_lora"
    # model_name = f"/home/jovyan/cp-gpu-4-datavol-one-model/one-model-v4/one_model_v4_work/open-r1_20250411/data/google_gemma-3-1b-it_{stage}_{type_}_amazon_Amazon_Fashion_lora"
    # model_name = f"google/gemma-3-1b-it"
    llm = LLM(model=model_name, max_model_len=13000, tensor_parallel_size=1, dtype=torch.bfloat16, trust_remote_code=True)#torch.bfloat16)

    tokenizer = llm.get_tokenizer()
    
    with open(f'./src/open_r1/sasrec/amazon_dataset/llm_dataset/amazon_{domain}_llm_test_20250521.json', 'r', encoding='utf-8') as f:
        df_final_amazon = json.load(f)
    # with open(f'./src/open_r1/sasrec/amazon_dataset/llm_dataset/amazon_{domain}_llm_test_case2_20250521.json', 'r', encoding='utf-8') as f:
        # df_final_amazon = json.load(f)
        
    ground_truth = [ extract_items(df_final_amazon[index]['output']) for index in range(len(df_final_amazon)) ]
        
        
    def make_sft_conversation(example, prompt_column: str = 'input'): # prompt / completion
        prompt = []

        prompt.append({"role": "system", "content": example["system_prompt"]})        
        prompt.append({"role": "user", "content": example[prompt_column]})

        # return prompt #vllm serve
        return {'text':tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)}   

    # deterministic
    # Exploratory (High-Creativity)
    # sampling_params_with_processor = SamplingParams(
    #     temperature=0.7,#0.2,#0.7,#0.1,  # 창의적 생성 조정
    #     top_k=50,#50,  # 확률이 높은 k개만 선택
    #     top_p=0.9,#0.95,#0.1, # 누적 확률이 p 이상이면 중단
    #     # repetition_penalty=1.1,  # 중복 단어 방지
    #     max_tokens=300,
    #     n=8,
    #     stop = ["</item_id>"], # 어차피 뒤에서 자름
    # )
    
    # sampling_params_with_processor = SamplingParams(
    #     temperature=1.2,#0.2,#0.7,#0.1,  # 창의적 생성 조정
    #     top_k=100,#50,  # 확률이 높은 k개만 선택
    #     top_p=0.95,#0.95,#0.1, # 누적 확률이 p 이상이면 중단
    #     # repetition_penalty=1.1,  # 중복 단어 방지
    #     max_tokens=300,
    #     n=8,
    #     stop = ["</item_id>"], # 어차피 뒤에서 자름
    # )
    
#     sampling_params_with_processor = SamplingParams(
#         temperature=0.2,#0.2,#0.7,#0.1,  # 창의적 생성 조정
#         top_k=20,#50,  # 확률이 높은 k개만 선택
#         top_p=0.9,#0.95,#0.1, # 누적 확률이 p 이상이면 중단
#         # repetition_penalty=1.1,  # 중복 단어 방지
#         max_tokens=300,
#         n=8,
#         stop = ["</item_id>"], # 어차피 뒤에서 자름
#     )
    num_gen = 10
    sampling_params_with_processor = SamplingParams(
        temperature=1.5, # 창의적 생성 조정 임시로 1로 조정
        top_k=120,#50,  # 확률이 높은 k개만 선택
        top_p=0.95,#0.95,#0.1, # 누적 확률이 p 이상이면 중단
        # repetition_penalty=1.1,  # 중복 단어 방지
        max_tokens=300,
        n=num_gen,
        stop = ["</item_id>"], # 어차피 뒤에서 자름
    )

    # rationals_name_ = f"""./{dataset_name}_dataset/rationals/rationals_list_{saved_model_name}_gudok.jsonl"""
    rationals_name_ = f"""./amazon_dataset/inference_list_{domain}_test_{stage}_ndcg_{type_}_{task}_4.jsonl"""  
    
    prompts = [make_sft_conversation(x)['text'] for x in df_final_amazon]#[:17]
    # 배치 크기 설정
    batch_size = 128
    
    
    with open(rationals_name_, 'w', encoding='utf-8') as f:
        rationals = []
        rationals_ = []
        
        # 배치 추론 루프
        for index in tqdm.tqdm(range(0, len(prompts), batch_size)):
            # print(index)
            batch_prompts = prompts[index:index + batch_size]       
            output_text = llm.generate(batch_prompts, sampling_params_with_processor)
            # print(output_text)
            print(output_text[0].outputs[3].text)
            # output_text = [x.outputs[0].text+"</item_id>" for x in output_text]
            output_text = [ filter_valid_asins (  dedup_preserve_order(list(set( [ extract_items(x.outputs[k].text+"</item_id>") for k in range(num_gen)] ))) ) for x in output_text]
            # print("="*70)
            # print(output_text)
            
            for j, output in enumerate(output_text):
                record = {str(index): output}
                rationals.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            rationals_.extend(output_text)
            
            # if index > 2:
            #     break        
        

        
    # 파일에 저장 (각 요소를 개별 줄에 저장)
    rationals_name = f"""./amazon_dataset/inference_list_{domain}_test_{stage}_ndcg_{type_}_{task}_4.json"""
    with open(rationals_name, "w") as f:
        json.dump(rationals, f)  # JSON 형식으로 저장

        
    print("="*100)
    print(domain, 'with ',stage)
    k_=5    
    hr_at_k, ndcg_at_k = evaluate_all(rationals_, ground_truth, top_n=k_)
    print(f"HR@{k_}: {hr_at_k:.4f}, NDCG@{k_}: {ndcg_at_k:.4f}")  
    
    k_=1   
    hr_at_k, ndcg_at_k = evaluate_all(rationals_, ground_truth, top_n=k_)
    print(f"HR@{k_}: {hr_at_k:.4f}, NDCG@{k_}: {ndcg_at_k:.4f}")  
        
        
        

    
    
if __name__ == "__main__":
    inference()