
MODEL_NAME = meta-llama/Llama-2-7b-chat-hf


download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2
LOGFILE = data/GCG/llama2_behaviors_mini.json


vanilla_inference:
	python main.py \
		--defense_type Empty \
		--results_dir ./results \
		--target_model $(TARGET_MODEL) \
		--attack GCG \
		--attack_logfile $(LOGFILE) \
		--verbose

smooth_llm:
	python main.py \
		--defense_type SmoothLLM \
		--results_dir ./results \
		--target_model $(TARGET_MODEL) \
		--attack GCG \
		--attack_logfile $(LOGFILE) \
		--smoothllm_pert_type RandomSwapPerturbation \
		--smoothllm_pert_pct 10 \
		--smoothllm_num_copies 3 \
		--verbose
