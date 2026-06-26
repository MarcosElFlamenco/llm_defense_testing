import json
import os
from typing import Dict, Iterable, Set


def build_results_path(results_dir: str, attack: str, defense_type: str, target_model: str, save_suffix: str) -> str:
    full_results_dir = os.path.join(results_dir, attack, defense_type)
    file_name = f"{target_model}_{save_suffix}.json"
    return os.path.join(full_results_dir, file_name)


def load_results_file(results_path: str) -> Dict[str, dict]:
    if not os.path.exists(results_path):
        return {}

    try:
        with open(results_path, "r") as json_file:
            loaded_results = json.load(json_file)
    except json.JSONDecodeError:
        return {}

    if not isinstance(loaded_results, dict):
        raise ValueError(f"Expected results file to contain a JSON object: {results_path}")

    return loaded_results


def get_processed_prompts(results: Dict[str, dict]) -> Set[str]:
    processed_prompts: Set[str] = set()
    for result in results.values():
        if isinstance(result, dict):
            prompt_text = result.get("user_text_prompt")
            if isinstance(prompt_text, str):
                processed_prompts.add(prompt_text)
    return processed_prompts


def save_results_file(results_path: str, results: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    temp_path = f"{results_path}.tmp"
    with open(temp_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    os.replace(temp_path, results_path)


def build_pending_prompts(prompts: Iterable[object], processed_prompts: Set[str]):
    pending_prompts = []
    seen_prompts = set(processed_prompts)
    for index, prompt in enumerate(prompts):
        if prompt.user_text_prompt in seen_prompts:
            continue
        pending_prompts.append((index, prompt))
        seen_prompts.add(prompt.user_text_prompt)
    return pending_prompts