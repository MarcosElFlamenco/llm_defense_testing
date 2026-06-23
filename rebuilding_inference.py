import os, sys
import gc
import json
import argparse
import time
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
## This makes imports simpler with the two nested directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AutoDAN"))

from AutoDAN.utils.eval_utils import _sample_next_suffixes, _prepare_reference, log_init, set_seed, check_for_attack_success, update_gen_config
from AutoDAN.utils.references import TEST_PREFIXES, MODEL_PATH_DICTS
from AutoDAN.utils.opt_utils import (
    get_score_autodan,
    load_model_and_tokenizer,
)
from AutoDAN.utils.string_utils import autodan_SuffixManager, load_conversation_template

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Configs")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_elites", type=float, default=0.05)
    parser.add_argument("--crossover", type=float, default=0.5)
    parser.add_argument("--num_points", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument('--do_sample',action="store_true")
    parser.add_argument("--attack_mode", type=str, choices=["ga", "hga"], default="ga")
    ## Note that this argument is only used in the case of hg
    parser.add_argument("--iter", type=int, default=5)

    parser.add_argument("--mutation", type=float, default=0.01)
    parser.add_argument("--init_prompt_path", type=str, default="./assets/autodan_initial_prompt.txt")
    parser.add_argument("--dataset_path", type=str, default="./data/advbench/harmful_behaviors.csv")
    parser.add_argument("--model", type=str, default="llama2")
    parser.add_argument("--save_suffix", type=str, default="normal")
    parser.add_argument("--API_key", type=str, default=None)
    return parser

def run_autodan_eval(args, attack_mode="ga"):
    if attack_mode not in {"ga", "hga"}:
        raise ValueError("attack_mode must be 'ga' or 'hga'")

    set_seed()
    device = f"cuda:{args.device}"
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.model])

    template_name = args.model
    batch_size = args.batch_size
    num_steps = args.num_steps
    num_elites = max(1, int(batch_size * args.num_elites))

    model, tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
    )

    conv_template = load_conversation_template(template_name)
    harmful_data = pd.read_csv(args.dataset_path)
    dataset = zip(harmful_data.goal[args.start:], harmful_data.target[args.start:])
    infos = {}
    crit = nn.CrossEntropyLoss(reduction="mean")
    reference_template = torch.load("assets/prompt_group.pth", map_location="cpu")
    result_dir = "./results/autodan_ga" if attack_mode == "ga" else "./results/autodan_hga"
    hga_interval = getattr(args, "iter", None)


    gen_config = update_gen_config(model.generation_config, args)
    print(f'Attacks will be generated with the following generation config: {gen_config}')

    with open("AutoDAN/results/autodan_hga/llama2_0_normal.json", "r") as f:
        jailbreaks = json.load(f)

    for i in range(len(jailbreaks.keys()))
        start_time = time.time()
        adv_suffix = ""
        gen_str = ""
        is_success = False

        artifact = jailbreaks[str(i)]

        with torch.no_grad():
            adv_suffix = artifact["final_suffix"]

            suffix_manager = autodan_SuffixManager(
                tokenizer=tokenizer,
                conv_template=conv_template,
                instruction=goal,
                target=target,
                adv_string=adv_suffix,
            )
            is_success, gen_str = check_for_attack_success( model,
                tokenizer,
                suffix_manager.get_input_ids(adv_string=adv_suffix).to(device),
                suffix_manager._assistant_role_slice,
                TEST_PREFIXES,
                gen_config
                )
            print(f"is success {is_success}")
            print(f"gen_str {gen_str}")
            gc.collect()
            torch.cuda.empty_cache()

        #os.makedirs(result_dir, exist_ok=True)
        #save_file_name = f"{result_dir}/{args.model}_{args.start}_{args.save_suffix}.json"
        #with open(save_file_name, "w") as json_file:
            #json.dump(infos, json_file, indent=4)
        #print(f"saves info {info} to file {save_file_name}")


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_autodan_eval(args, attack_mode=args.attack_mode)
