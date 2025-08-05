# coding=utf-8
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

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

import re
from typing import Optional


###
from sasrec.util import neg_sample, neg_sample_set, get_sample_scores, neg_sample_unigram
from sasrec.customized_model import OneModelV3
from typing import Union, Dict, Optional, Sequence
import numpy as np
import os

import copy
import logging
from dataclasses import dataclass

import torch
import transformers
import sasrec.util
from torch.utils.data import Dataset

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import tqdm
import copy

from sasrec.util import dict_str_key_to_int, sequential_loss, clm_loss, sequential_loss_item, kl_divergence_loss, AdaptiveLossWeighting, normalize_loss, init_weights
from sasrec.sequential_reco import *
from sasrec.customized_model import *
from sasrec.outputs import ModelArguments, DataArguments, TrainingArguments, MyCallback
from sasrec.util import smart_tokenizer_and_embedding_resize_v3, dict_str_key_to_int, CustomTrainer, CustomCallback
from safetensors.torch import load_file
from sasrec.customized_model import OneModelV3
import json
from typing import Union, Dict, Optional, Sequence
import tqdm
import numpy as np
import os

import copy
import logging

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import wandb
import numpy as np
from safetensors import safe_open


model_args = ModelArguments()
data_args = DataArguments()
training_args = TrainingArguments()
training_args.label_names=['answer_id', 'test_neg','input_ids_item', 'pos_ids_item', 'neg_ids_item']

domain = "Grocery_and_Gourmet_Food"

with open(f'./src/open_r1/sasrec/amazon_dataset/vocab/amazon_vocab_{domain}.json', 'r') as f:
    vocab_dict = json.load(f)
vocab_dict_inverse = {v:k for k, v in vocab_dict.items()}

data_args.len_vocab_dict_tokenized = len(vocab_dict)+1
model_args.pretrained_model_path = f"./src/open_r1/sasrec/output_dir/{domain}/checkpoint-60/model.safetensors"
data_args.pad_token_id = 2
training_args.model_max_length = 100
training_args.item_enc_type = 'fc_layer'
training_args.neg_sample_type = 'basic'
# training_args.hidden_size_item = 64
one_model = OneModelV3(model_args, data_args, training_args) # this is the pretrained CF-Rec model.

state_dict = load_file(model_args.pretrained_model_path)
one_model.load_state_dict(state_dict)
print("Model Load Complete!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
one_model.to(device)
one_model.eval()
print(one_model)


# irm_final_rewards    
def collaborative_guided_reward(completions: list[list[dict[str, str]]], output: list[str], output_id: list[str], lambda_: float = 3.0, cf_threshold: float = 0.3, **kwargs) -> list[Optional[float]]:
    """
    collaborative_guided_reward
    
    - item_rewards: 0 or 1
    - cf_rewards: float (0~1)
    - lambda_: weight for CF reward if item is matched
    - delta: penalty if CF is high but item is wrong
    - cf_threshold: threshold to define 'high' CF score
    """
    cf_rewards = collaborative_filtering_reward(completions, output_id)
    item_match_rewards = item_id_match_reward(completions, output)
    assert len(item_match_rewards) == len(cf_rewards), "different lengths between cf and item!"

    final_rewards = []
    for item_r, cf_r in zip(item_match_rewards, cf_rewards):
        if item_r == 1 and cf_r >= cf_threshold:
            reward = 1.0 + lambda_ * cf_r
        elif item_r == 1 and cf_r < cf_threshold:
            reward = 1.0
        elif item_r == 0 and cf_r >= cf_threshold:
            reward = cf_r#-delta
        else:  # item_r == 0 and cf_r < threshold
            reward = 0.0
        final_rewards.append(reward)

    return final_rewards    
    
    
    

def collaborative_filtering_reward(completions: list[list[dict[str, str]]], output_id: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vllm의 output 형태 
    rewards = []
    for content, id_ in zip(contents, output_id):#solution
        input_ids = torch.tensor([int(i) for i in id_.split(",")]).unsqueeze(0)
        input_ids_item = customized_pad_sequence(input_ids, batch_first=True, padding_value=2, pos='left') 
        match = re.search(r"<item_nm>(.*?)</item_nm>", content, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            try:
                llm_infer_answer_id = torch.tensor(int(vocab_dict_inverse[item_text])).unsqueeze(0)
                # sasrec의 logit
                with torch.no_grad():
                    print("input_id_item", input_ids_item)
                    print("answer_id", llm_infer_answer_id)
                    reward = one_model.extract_logits(answer_id = llm_infer_answer_id.cuda(), test_neg=None, input_ids_item=input_ids_item.cuda() ).cpu().detach().numpy().copy()[0][0]
                    print("reward", reward)
            except:
                reward = 0.0
            
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards   



def item_match_reward(completions: list[list[dict[str, str]]], output: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vllm의 output 형태 
    rewards = []
    for content, sol in zip(contents, output):#solution
        match = re.search(r"<item_nm>(.*?)</item_nm>", content, re.DOTALL) #amazon
        sol_match = re.search(r"<item_nm>(.*?)</item_nm>", sol, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            sol_text = sol_match.group(1).strip()
            # matching
            reward = 1.0 if item_text == sol_text else 0.0
        else:
            reward = 0.0
        rewards.append(reward)
    return rewards

def item_id_match_reward(completions: list[list[dict[str, str]]], output: list[str], **kwargs) -> list[Optional[float]]: #solution
    """Reward function that checks if the text inside <item>...</item> exactly matches the solution."""
    contents = [completion[0]["content"] for completion in completions] # vllm의 output 형태 
    rewards = []
    for content, sol in zip(contents, output):#solution
        match = re.search(r"<item_id>(.*?)</item_id>", content, re.DOTALL) #amazon
        sol_match = re.search(r"<item_id>(.*?)</item_id>", sol, re.DOTALL) #amazon
        if match:
            item_text = match.group(1).strip()
            sol_text = sol_match.group(1).strip()
            reward = 1.0 if item_text == sol_text else 0.0
        else:
            reward = 0.0
        rewards.append(reward)

    return rewards   
    
def only_expected_tags_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    """
    Reward is 1.0 only if the completion contains ONLY the allowed tags:
    <think>, <item>, <sales> (with matching closing tags),
    and NO other custom tags like <xxx>...</xxx>.
    """
    allowed_tags = {"think", "item_nm","item_id", "sales"}
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content in contents:
        tags = re.findall(r"</?([a-zA-Z0-9_]+)>", content)
        unique_tags = set(tags)
        if unique_tags.issubset(allowed_tags):
            reward = 1.0
        else:
            reward = 0.0

        rewards.append(reward)

    return rewards    
    
def tag_presence_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[Optional[float]]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    required_tags = ["item_id", "item_nm", "think"]

    for content in contents:
        all_present = True
        for tag in required_tags:
            pattern = fr"<{tag}>.*?</{tag}>"
            if not re.search(pattern, content, re.DOTALL):
                all_present = False
                break
        reward = 1.0 if all_present else 0.0
        rewards.append(reward)

    return rewards    

    
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """    
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<item_nm>") == 1:
            count += 0.25
        if text.count("</item_nm>") == 1:
            count += 0.25
        if text.count("<item_id>") == 1:
            count += 0.25
        if text.count("</item_id>") == 1:
            count += 0.25
        if text.count("<think>") == 1:
            count += 0.25
        if text.count("</think>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "tag_count": tag_count_reward,
        "item_accuracy" : item_match_reward,
        "item_id_accuracy" : item_id_match_reward,
        "tag_presence" : tag_presence_reward,
        "only_expected_tags" : only_expected_tags_reward,
        "cf_model_reward" : collaborative_filtering_reward,
        "collaborative_guided_reward" : collaborative_guided_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs





###sasrec part
###### sasrec import 
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
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict] 

        elif self.model_type=='train':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:-2]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]

        elif self.model_type=='valid':
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]
            # self.answer_id = [0]

        elif self.model_type=='test':
            targets_id = [f"{ ','.join(example['output_id'].split(',')[:]) }" for example in list_data_dict]
            self.answer_id = [ int(example['output_id'].split(',')[-1])   for example in list_data_dict]

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
            for _ in range(100):
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
