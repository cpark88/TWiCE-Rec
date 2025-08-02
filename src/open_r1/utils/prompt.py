import os
import random


random.seed(1958)
class Prompt:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.replace("\\n", "\n").strip() for p in raw_prompts]
        # self.templates = [p.strip() for p in raw_prompts]
            
        self.historyList = []
        self.itemList = []
        self.reasoning_process = ""
        self.positive_item = ""
        self.trueSelection = ""
        self.mno_profile_feature = ""

    def __str__(self) -> str:
        prompt = self.templates[random.randint(0, len(self.templates)-1)] # template를 랜덤하게 선택
        history = ", ".join(self.historyList)
        cans = "\n\n".join(self.itemList)
        prompt = prompt.replace("[HistoryHere]", history)
        prompt = prompt.replace("[CansHere]", cans)
        prompt = prompt.replace("[ReasoningHere]", self.reasoning_process)
        prompt = prompt.replace("[PosItemHere]", self.positive_item)
        prompt = prompt.replace("[ProfileHere]", self.mno_profile_feature) # only for skt
        prompt += " "
        return prompt


# origianl    
# class Prompt:
#     def __init__(self, prompt_path) -> None:
#         assert os.path.isfile(prompt_path), "Please specify a prompt template"
#         with open(prompt_path, 'r') as f:
#             raw_prompts = f.read().splitlines()
#         self.templates = [p.strip() for p in raw_prompts]
            
#         self.historyList = []
#         self.itemList = []
#         self.trueSelection = ""

#     def __str__(self) -> str:
#         prompt = self.templates[random.randint(0, len(self.templates)-1)]
#         history = ", ".join(self.historyList)
#         cans = ", ".join(self.itemList)
#         prompt = prompt.replace("[HistoryHere]", history)
#         prompt = prompt.replace("[CansHere]", cans)
#         prompt += " "

            
#         return prompt
        
        
        
        