# defense_factory.py
from defenses.defenses import NoDefense 
from defenses.smoothllm import SmoothLLM

def get_defense(defense_type: str, target_model, tokenizer, conv_template, args=None):
    cls = defense_type.lower()
    if cls == "nodefense":
        return NoDefense(
            target_model=target_model,
            tokenizer=tokenizer,
            conv_template=conv_template
        )
    if cls == "smoothllm":
        return SmoothLLM(
            target_model=target_model,
            tokenizer=tokenizer,
            conv_template=conv_template,
            pert_type=args.smoothllm_pert_type,
            pert_pct=args.smoothllm_pert_pct,
            num_copies=args.smoothllm_num_copies,
            smoothllm_batch_size=args.smoothllm_batch_size
        )
    print(f'Unknown defense type {defense_type}')
 
        
