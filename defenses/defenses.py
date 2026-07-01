import torch
import numpy as np
import gc
from AutoDAN.utils.string_utils import autodan_SuffixManager
from AutoDAN.utils.eval_utils import generate, generate_from_user_text, check_for_attack_success

class Defense:

    def __init__(self, target_model, tokenizer, conv_template):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.conv_template = conv_template

    def forward_autodan(self, user_text_prompt, gen_config):

        input_text_prompt = "[INST] " + user_text_prompt + " [/INST]"
        input_toks = self.tokenizer(input_text_prompt).input_ids
        input_ids_user = torch.tensor(input_toks)

        gen_str = self.tokenizer.decode(
            generate_from_user_text(
                self.target_model,
                input_ids_user,
                gen_config=gen_config
            )
        ).strip()

        gc.collect()
        torch.cuda.empty_cache()
        return gen_str

    def forward_autodan_batch(self, user_text_prompts, gen_config, batch_size=8):
        """Batched inference for a list of user_text_prompts.

        Returns a list of decoded string outputs in the same order as inputs.
        """
        if not isinstance(user_text_prompts, (list, tuple)):
            raise ValueError(f"user_text_prompts must be a list or tuple for batched inference \n Currently {user_text_prompts}")

        # Format prompts
        input_texts = ["[INST] " + p + " [/INST]" for p in user_text_prompts]
        all_outputs = []

        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]

            # Tokenize batch with padding
            batch_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                padding_side='left',
                truncation=False,
                return_tensors='pt'
            )
            batch_input_ids = batch_inputs['input_ids'].to(self.target_model.device)
            batch_attention_mask = batch_inputs['attention_mask'].to(self.target_model.device)

            try:
                outputs = self.target_model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    generation_config=gen_config,
                )
            except RuntimeError:
                # Fall back to single-sample loop on OOM
                print(f"Warning: OOM encountered for batch {i}..{i+len(batch_texts)-1}. Falling back to single-sample inference.")
                for t in batch_texts:
                    all_outputs.append(self.forward_autodan(t, gen_config))
                torch.cuda.empty_cache()
                continue

            # Decode and strip the prompt prefix from generated text
            # Use token boundaries for the prompt/generation split so batching
            # does not introduce character-level slicing drift.
            gen_start_idx = batch_input_ids.size(1)

            for j in range(outputs.size(0)):
                generated_tokens = outputs[j, gen_start_idx:]
                gen_str = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                all_outputs.append(gen_str)

            gc.collect()
            torch.cuda.empty_cache()

        return all_outputs

class NoDefense(Defense):
    """
    No defense
    
    """

    def __init__(self, 
        target_model,
        tokenizer,
        conv_template
    ):
        super(NoDefense, self).__init__(target_model,tokenizer, conv_template)

    @torch.no_grad()
    def __call__(self, inputs, gen_config, batch_size=64):
        ## Single string inference [this should never happen actually]
        if isinstance(inputs, str):
            self.forward_autodan(inputs, gen_config)
        else:
            ### Batched inference
            outputs = self.forward_autodan_batch(input, gen_config, batch_size=batch_size)
            return outputs

