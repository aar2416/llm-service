import pytest

from finetune.prompts import (
    build_full_messages, build_prompt_messages, extract_code, load_jsonl,
)
from finetune.rewards import accuracy_reward, format_reward
from finetune.sheets import OUTPUT_COLUMNS, cell_accuracy, cells_equal

H = ",".join(OUTPUT_COLUMNS)
GOLD = H + "\n2025-03-02,Acme Corp,Laptop stand,10,25.00,250.00\n2025-03-05,Globex,Monitor,4,180.00,720.00"


def reply(csv_text):
    return [{"role": "assistant", "content": f"```csv\n{csv_text}\n```"}]


def test_sample_data_builds_messages():
    for path in ("finetune/data/sample_sft.jsonl", "finetune/data/sample_rl.jsonl"):
        for row in load_jsonl(path):
            msgs = build_full_messages(row)
            assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
            assert extract_code(msgs[-1]["content"]) == row["output_sheet"].strip("\n")
            assert len(build_prompt_messages(row)) == 2
            assert cell_accuracy(row["output_sheet"], row["output_sheet"]) == 1.0


def test_cells_equal_normalizes_numbers_and_text():
    assert cells_equal("$1,250.00", "1250")
    assert cells_equal("  acme   CORP", "Acme Corp")
    assert not cells_equal("10", "12")


def test_cell_accuracy_is_percentage_of_cells():
    assert cell_accuracy(GOLD, GOLD) == 1.0
    # one wrong cell out of 12
    one_wrong = GOLD.replace("Globex", "Initech")
    assert cell_accuracy(one_wrong, GOLD) == pytest.approx(11 / 12)
    # missing second row: 6 of 12 cells
    assert cell_accuracy(GOLD.split("\n")[0] + "\n" + GOLD.split("\n")[1], GOLD) == pytest.approx(0.5)
    # extra row also penalised: 12 correct of 18
    extra = GOLD + "\n2025-03-09,X,Y,1,1,1"
    assert cell_accuracy(extra, GOLD) == pytest.approx(12 / 18)


def test_column_order_does_not_matter_for_accuracy():
    reordered = "Vendor,Date,Description,Quantity,Unit Price,Total\nAcme Corp,2025-03-02,Laptop stand,10,25.00,250.00"
    gold = H + "\n2025-03-02,Acme Corp,Laptop stand,10,25.00,250.00"
    assert cell_accuracy(reordered, gold) == 1.0


def test_rewards():
    assert accuracy_reward([reply(GOLD)], [GOLD]) == [2.0]
    assert accuracy_reward([reply(GOLD.replace("Globex", "Initech"))], [GOLD])[0] == pytest.approx(2 * 11 / 12)
    assert accuracy_reward([[{"role": "assistant", "content": "no sheet"}]], [GOLD]) == [0.0]
    assert format_reward([reply(GOLD)]) == [0.5]
    assert format_reward([reply("A,B\n1,2")]) == [0.0]
    assert format_reward([[{"role": "assistant", "content": "none"}]]) == [0.0]
