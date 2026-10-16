"""Supervised fine-tuning of Qwen3-14B with Unsloth (email + input sheet -> output sheet).

    python -m finetune.sft --dataset finetune/data/sample_sft.jsonl

Dataset: JSONL with {"email", "input_sheet", "output_sheet"} per line (sheets as CSV text).
"""
import argparse

from .common import BASE_MODEL, load_qwen_lora, save_outputs  # imports unsloth first

from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

from .prompts import build_full_messages, load_jsonl  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--dataset", default="finetune/data/sample_sft.jsonl")
    p.add_argument("--output-dir", default="outputs/qwen3-14b-sft")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--no-merge", action="store_true", help="Skip saving the merged 16-bit model")
    args = p.parse_args()

    model, tokenizer = load_qwen_lora(args.model, args.max_seq_length, args.lora_r)

    rows = load_jsonl(args.dataset)
    texts = [
        tokenizer.apply_chat_template(build_full_messages(r), tokenize=False, add_generation_prompt=False)
        for r in rows
    ]
    dataset = Dataset.from_dict({"text": texts})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=args.output_dir,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.05,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
            logging_steps=1,
            report_to="none",
            seed=3407,
        ),
    )
    # Compute loss only on the assistant reply, not the system/user prompt.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    trainer.train()
    save_outputs(model, tokenizer, args.output_dir, merge=not args.no_merge)


if __name__ == "__main__":
    main()
