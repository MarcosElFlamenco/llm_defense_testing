import os
import torch
import pandas as pd
import json
import random
import time
from tqdm.auto import tqdm
import argparse
import time

#import lib.perturbations as perturbations
#import lib.attacks as attacks
#import lib.language_models as language_models
#import lib.model_configs as model_configs
from defense_factory import get_defense

import numpy as np
import torch.nn as nn

from utils.opt_utils import (
    autodan_sample_control,
    autodan_sample_control_hga,
    get_score_autodan,
    load_model_and_tokenizer,
)
from utils.string_utils import autodan_SuffixManager, load_conversation_template
from utils.eval_utils import check_for_attack_success


##CODE START

SEED=20

MODEL_PATH_DICTS = {
    "llama2": "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590/",
    "llama3": "~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
    "vicuna": "./models/vicuna/vicuna-7b-v1.3",
    "guanaco": "./models/guanaco/guanaco-7B-HF",
    "WizardLM": "./models/WizardLM/WizardLM-7B-V1.0",
    "mpt-chat": "./models/mpt/mpt-7b-chat",
    "mpt-instruct": "./models/mpt/mpt-7b-instruct",
    "falcon": "./models/falcon/falcon-7b-instruct",
}


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    start_time = time.time()
    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)


    set_seed()
    device = f"cuda:{args.device}"
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.target_model])
    print(f"model path is {model_path}")

    template_name = args.target_model

    target_model , tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )

    conv_template = load_conversation_template(template_name)


    defense = get_defense(
        defense_type=args.defense_type,
        target_model=target_model,
        tokenizer=tokenizer,
        conv_template=conv_template,
        args=args
    )

    
    ##artifact load
    path = "results/autodan_ga/llama2_0_normal.json" if attack_mode == "ga" else "results/autodan_hga/llama2_0_normal.json"
    with open(path, "r") as f:
        artifact_dataset = json.load(f)
    example = artifact_dataset["0"]
    goal = example["goal"]
    target = example["target"]
    adv_suffix = example["final_suffix"]

    start_time = time.time()
    output = defense(goal, target, adv_suffix, batch_size=64, max_new_len=64)
    inference_time = time.time() - start_time
    print(f"Output: {output}")
    print(f"Inference time: {inference_time} seconds")


 
if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results'
    )
    parser.add_argument(
        '--trial',
        type=int,
        default=0
    )

    # Targeted LLM
    parser.add_argument(
        '--target_model',
        type=str,
        default='llama2',
        choices=['vicuna', 'llama2']
    )

    # Attacking LLM
    parser.add_argument(
        '--attack',
        type=str,
        default='GCG',
        choices=['GCG', 'PAIR']
    )
    parser.add_argument(
        '--attack_logfile',
        type=str,
        default='data/GCG/vicuna_behaviors.json'
    )

    # SmoothLLM
    parser.add_argument(
        '--smoothllm_num_copies',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--smoothllm_pert_pct',
        type=int,
        default=10
    )
    parser.add_argument(
        '--verbose',
        action="store_true"
    )
    parser.add_argument(
        '--quantize',
        action="store_true"
    )


    parser.add_argument(
        '--smoothllm_pert_type',
        type=str,
        default='RandomSwapPerturbation',
        choices=[
            'RandomSwapPerturbation',
            'RandomPatchPerturbation',
            'RandomInsertPerturbation'
        ]
    )

    parser.add_argument(
        '--defense_type',
        type=str,
        default='NoDefense',
        choices=[
            'NoDefense',
            'SmoothLLM'
        ]
    )

    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()
    main(args)
