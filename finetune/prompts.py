"""Instruction prompts and chat formatting shared by SFT and RL (no heavy imports).

Example fields: {"email": str, "input_sheet": CSV str, "output_sheet": CSV str}.
"""
import json
import re

from .sheets import OUTPUT_COLUMNS

_COLUMNS = ", ".join(OUTPUT_COLUMNS)

SYSTEM_PROMPT = (
    "You are a data-entry assistant. You will be given an email and an input spreadsheet "
    "(as CSV). Read the email, update/extend the input spreadsheet with the information "
    "it contains, and return the final spreadsheet.\n"
    "Rules:\n"
    f"- Output a single CSV with exactly these columns, in this order: {_COLUMNS}.\n"
    "- The first line is the header row. One record per row.\n"
    "- Dates as YYYY-MM-DD; numbers without currency symbols or thousands separators.\n"
    "- Keep every row of the input sheet unless the email says to change or remove it.\n"
    "- Never invent values; leave a cell empty if the email and sheet do not provide it.\n"
    "- Reply with only the CSV inside one ```csv fenced block, nothing else."
)

USER_TEMPLATE = "### Email\n{email}\n\n### Input sheet (CSV)\n```csv\n{sheet}\n```"

CODE_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)


def build_user_message(email: str, input_sheet: str) -> str:
    return USER_TEMPLATE.format(email=email.strip(), sheet=input_sheet.strip("\n"))


def format_assistant_message(output_sheet: str) -> str:
    return f"```csv\n{output_sheet.strip(chr(10))}\n```"


def build_prompt_messages(example: dict) -> list[dict]:
    """System + user turns for one example."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(example["email"], example["input_sheet"])},
    ]


def build_full_messages(example: dict) -> list[dict]:
    """Prompt turns + the gold assistant answer."""
    return build_prompt_messages(example) + [
        {"role": "assistant", "content": format_assistant_message(example["output_sheet"])}
    ]


def extract_code(text: str) -> str | None:
    """Return the first fenced block in a model reply, or None."""
    m = CODE_BLOCK_RE.search(text)
    return m.group(1).strip("\n") if m else None


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
