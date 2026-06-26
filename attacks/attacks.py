import torch
import numpy as np
import json
import gc
from AutoDAN.utils.string_utils import autodan_SuffixManager
from AutoDAN.utils.eval_utils import generate
#import lib.perturbations as perturbations

## Not used yet, will certainly be used to wrap returns later
class Prompt:
    def __init__(self, full_prompt, perturbable_prompt, max_new_tokens):
        self.full_prompt = full_prompt
        self.perturbable_prompt = perturbable_prompt

class DANPrompt(Prompt):
    def __init__(self, goal, target, final_suffix, user_text_prompt, suffix_manager):
        self.goal = goal
        self.target = target
        self.user_text_prompt = user_text_prompt

class Attack:
    def __init__(self, logfile, target_model):
        self.logfile = logfile
        self.target_model = target_model

class AutoDAN(Attack):
    """
    AutoDAN attack
    
    """

    def __init__(self, logfile=None, target_model=None, tokenizer = None, conv_template = None):

        super(AutoDAN, self).__init__(logfile, target_model)

        self.tokenizer = tokenizer
        self.conv_template = conv_template

        with open(self.logfile, 'r') as f:
            log = json.load(f)

        # Enables obj[i]
        self.prompts = [
            self.create_prompt(goal=artifact["goal"], target=artifact["target"], final_suffix=artifact["final_suffix"])
            for artifact in log.values()
        ]

    def create_prompt(self, goal, target, final_suffix):

        suffix_manager = autodan_SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=goal,
            target=target,
            adv_string=final_suffix,
        )
        text_prompt = suffix_manager.get_prompt(adv_string=final_suffix)

        end_of_user_text = text_prompt.find("[/INST]") - 1
        assert text_prompt.startswith("[INST]")
        start_of_user_text = text_prompt.find("[INST]") + len("[INST]") + 1
        user_text_prompt = text_prompt[start_of_user_text:end_of_user_text]

        return DANPrompt(goal, target, final_suffix, user_text_prompt, suffix_manager)

### This would have been a nice implementation,
### but the smooth llm code words it differently
"""
    def __getitem__(self, index, tokenizer, conv_template, goal, target, adv_string):
        suffix_manager = autodan_SuffixManager(
            tokenizer=self.tokenizer,
            conv_template=self.conv_template,
            instruction=goal,
            target=target,
            adv_string=adv_suffix,
        )

        artifact = self.data[index]
        adv_string = artifact['adv_string']
        prompt = suffix_manager.get_prompt(adv_string=adv_string)

        return self.data[index]

    # Enables len(obj)
    def __len__(self):
        return len(self.data)
"""
