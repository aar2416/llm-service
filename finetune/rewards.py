"""Reward functions for GRPO (no heavy imports, unit-testable).

Task: email + input sheet -> output sheet in the single OUTPUT_COLUMNS format.
The main reward is the percentage of output cells that are correct.

TRL calls each as reward(prompts=..., completions=..., **dataset_columns) and
expects one float per completion. Completions are chat-format:
[{"role": "assistant", "content": "..."}]. `answer` is the gold output CSV.
"""
from .prompts import extract_code
from .sheets import cell_accuracy, header_matches, parse_csv

FORMAT_REWARD = 0.5
ACCURACY_REWARD_SCALE = 2.0  # 100% of cells correct -> 2.0


def _text(completion) -> str:
    return completion[0]["content"] if isinstance(completion, list) else completion


def format_reward(completions, **kwargs) -> list[float]:
    """+0.5 if the reply has one fenced block whose header is exactly OUTPUT_COLUMNS."""
    out = []
    for c in completions:
        text = _text(c)
        code = extract_code(text) if text.count("```") == 2 else None
        out.append(FORMAT_REWARD if code and header_matches(parse_csv(code)) else 0.0)
    return out


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """2.0 * (fraction of gold cells reproduced correctly)."""
    out = []
    for c, gold in zip(completions, answer):
        code = extract_code(_text(c))
        out.append(ACCURACY_REWARD_SCALE * cell_accuracy(code, gold) if code else 0.0)
    return out


REWARD_FUNCS = [format_reward, accuracy_reward]
