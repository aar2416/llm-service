"""Unsloth model loading shared by SFT and RL.

IMPORTANT: `unsloth` must be imported before trl/transformers/peft so its
patches apply, so every entry script imports this module first.
"""
from unsloth import FastLanguageModel  # noqa: I001  (must come first)

BASE_MODEL = "unsloth/Qwen3-14B"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_qwen_lora(model_name: str = BASE_MODEL, max_seq_length: int = 4096, lora_r: int = 16):
    """Load Qwen3-14B in 4-bit and attach LoRA adapters via Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,  # auto-detect bf16/fp16
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0,
        target_modules=LORA_TARGET_MODULES,
        use_gradient_checkpointing="unsloth",  # long-context memory savings
        random_state=3407,
    )
    return model, tokenizer


def save_outputs(model, tokenizer, output_dir: str, merge: bool = True) -> None:
    """Save LoRA adapters, and optionally a merged 16-bit model SGLang can serve."""
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if merge:
        model.save_pretrained_merged(f"{output_dir}/merged", tokenizer, save_method="merged_16bit")
