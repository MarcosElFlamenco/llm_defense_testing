## Confirm determinism
import json

def first_difference_index(a, b):
    max_idx = min(len(a), len(b))
    for i in range(max_idx):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return max_idx
    return None

def highlight_difference_window(text, diff_idx, prefix_chars=6, window_chars=120):
    start = max(0, diff_idx - prefix_chars)
    end = min(len(text), diff_idx + window_chars)
    snippet = text[start:end].replace("\n", "\\n")
    local_idx = diff_idx - start

    if local_idx < len(snippet):
        snippet = (
            snippet[:local_idx]
            + "\033[31m" + snippet[local_idx] + "\033[0m"
            + snippet[local_idx + 1:]
        )
    else:
        snippet = snippet + "\033[31m∅\033[0m"

    return start, snippet

def confirm_determinism(attack_logfile, defense_testing_results):
    count = 0
    with open(attack_logfile, "r") as f:
        attacks = json.load(f)
    with open(defense_testing_results, "r") as f:
        defense_results = json.load(f)

    for key in attacks.keys():
        jailbreak_output = attacks[key].get("final_respond", attacks[key].get("output", "")) or ""
        inference_output = defense_results[key].get("final_respond", defense_results[key].get("output", "")) or ""

        if jailbreak_output.startswith(inference_output) or inference_output.startswith(jailbreak_output):
            count += 1
            continue

        diff_idx = first_difference_index(jailbreak_output, inference_output)
        a_start, a_snippet = highlight_difference_window(jailbreak_output, diff_idx)
        b_start, b_snippet = highlight_difference_window(inference_output, diff_idx)

        print(f"Mismatch for key {key} at char {diff_idx}:")
        print(f"  Jailbreak  [{a_start}:...]: {a_snippet}")
        print(f"  Inference  [{b_start}:...]: {b_snippet}")
        print()

    num_queries = len(attacks.keys())
    if count == num_queries:
        print(f"Determinism confirmed:\n {attack_logfile}\n {defense_testing_results}")
    else:
        print(f"Nondeterministic on {num_queries - count} out of {num_queries} queries")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Confirm determinism between attack and defense outputs.")
    parser.add_argument("--reference_outputs", type=str, help="Path to the attack logfile (JSON).")
    parser.add_argument("--new_outputs", type=str, help="Path to the defense testing results (JSON).")

    args = parser.parse_args()

    confirm_determinism(args.reference_outputs, args.new_outputs)
