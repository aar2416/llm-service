"""Reinforcement learning (GRPO) on Qwen3-14B with Unsloth + TRL.

The model samples several output sheets per prompt; rewards (format, cell accuracy
against the gold sheet) are computed by finetune/rewards.py and GRPO
pushes the policy toward higher-reward samples.

    python -m finetune.rl --dataset finetune/data/sample_rl.jsonl
    python -m finetune.rl --init-adapter outputs/qwen3-14b-sft   # start from the SFT result

Dataset: JSONL with {"email", "input_sheet", "output_sheet"} per line (CSV text; output_sheet = gold).
"""
import argparse

from .common import BASE_MODEL, load_qwen_lora, save_outputs  # imports unsloth first

from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from .prompts import build_prompt_messages, load_jsonl  # noqa: E402
from .rewards import REWARD_FUNCS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--init-adapter", default=None,
                   help="Path to a merged/SFT model to start from (overrides --model)")
    p.add_argument("--dataset", default="finetune/data/sample_rl.jsonl")
    p.add_argument("--output-dir", default="outputs/qwen3-14b-grpo")
    p.add_argument("--max-prompt-length", type=int, default=1024)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--no-merge", action="store_true")
    args = p.parse_args()

    max_seq = args.max_prompt_length + args.max_completion_length
    model, tokenizer = load_qwen_lora(args.init_adapter or args.model, max_seq, args.lora_r)

    rows = load_jsonl(args.dataset)
    dataset = Dataset.from_dict({
        "prompt": [build_prompt_messages(r) for r in rows],  # chat format; TRL applies the template
        "answer": [r["output_sheet"] for r in rows],          # gold CSV, forwarded to reward functions
    })

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCS,
        train_dataset=dataset,
        args=GRPOConfig(
            output_dir=args.output_dir,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            optim="adamw_8bit",
            weight_decay=0.1,
            warmup_ratio=0.1,
            per_device_train_batch_size=args.num_generations,  # must be divisible by num_generations
            gradient_accumulation_steps=1,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_completion_length=args.max_completion_length,
            max_steps=args.max_steps,
            logging_steps=1,
            save_steps=50,
            report_to="none",
            seed=3407,
        ),
    )
    trainer.train()
    save_outputs(model, tokenizer, args.output_dir, merge=not args.no_merge)


if __name__ == "__main__":
    main()
