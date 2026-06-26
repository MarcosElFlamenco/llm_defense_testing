import json
import os
from typing import Dict, Iterable, List, Set, Tuple


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


def get_tested_goals(results: Dict[str, dict]) -> Set[str]:
    tested_goals: Set[str] = set()
    for result in results.values():
        if not isinstance(result, dict):
            continue
        goal = result.get("goal")
        if isinstance(goal, str):
            tested_goals.add(goal)
    return tested_goals


def get_tested_indices(results: Dict[str, dict]) -> Set[int]:
    tested_indices: Set[int] = set()
    for key in results.keys():
        try:
            tested_indices.add(int(key))
        except (TypeError, ValueError):
            continue
    return tested_indices


def build_pending_prompts(
    prompts: Iterable[Tuple[int, str, str]],
    results: Dict[str, dict],
) -> List[Tuple[int, str, str]]:
    tested_goals = get_tested_goals(results)
    tested_indices = get_tested_indices(results)

    pending_prompts: List[Tuple[int, str, str]] = []
    for index, goal, target in prompts:
        if index in tested_indices or goal in tested_goals:
            continue
        pending_prompts.append((index, goal, target))
    return pending_prompts


def save_results_file(results_path: str, results: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    temp_path = f"{results_path}.tmp"
    with open(temp_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    os.replace(temp_path, results_path)