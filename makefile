
MODEL_NAME = lmsys/vicuna-7b-v1.5

download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2
LOG_FILE = data/AutoDAN/llama-2-7b-chat-hf_behaviors.json
ATTACK = AUTODAN
SAVE_SUFFIX = non 

evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/llama2_0_regular.json" \
		--max_new_tokens 512 \
		--save_suffix $(SAVE_SUFFIX) \
		--inference_batch_size 6

confirm_determinism:
	python dictionary_utils.py \
		--reference_outputs defense_testing_results/AUTODAN/NoDefense/llama2_cleanerbatch.json \
		--new_outputs defense_testing_results/AUTODAN/NoDefense/llama2_$(SAVE_SUFFIX).json \

smooth_llm_evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--defense SmoothLLM \
		--attack_logfile "AutoDAN/results/autodan_hga/llama2_0_normal_debug.json" \
		--max_new_tokens 512 \
		--save_suffix smoothllm \
		--inference_batch_size 4 \
		--defense SmoothLLM \
		--smoothllm_pert_type RandomSwapPerturbation \
		--smoothllm_pert_pct 10 \
		--smoothllm_num_copies 3		

autodan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--dataset_path data/advbench/smaller_behaviors.csv \
		--max_new_tokens 128 \
		--save_suffix normal \
		--model gemma-7b \

nightrun:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/llama2_0_complete.json" \
		--max_new_tokens 512 \
		--save_suffix complete \
		--inference_batch_size 8	
megadan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--max_new_tokens 128 \
		--save_suffix megadan \
		--continue_after_jailbroken \
		--dataset_path ./data/advbench/smaller_behaviors.csv

generate_behavior_files:
	python generate_behavior_files.py
