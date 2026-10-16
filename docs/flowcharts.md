# Flowcharts and tuning guide

Diagrams are Mermaid and render on GitHub / VS Code (Markdown preview with Mermaid support).

## 1. LLM service flow

```mermaid
flowchart TD
    A[HTTP client<br/>POST /generate] --> V
    Q[(RabbitMQ<br/>llm.requests)] -->|fetch only when budget has free tokens| C[RabbitConsumer loop]
    C -->|invalid JSON| E1[error payload]
    C --> V[plan: validate request]
    V --> M{model in<br/>LLM_MODEL_ENDPOINTS?}
    M -- no --> E2[UnknownModel<br/>HTTP 404]
    M -- yes --> T[count_tokens<br/>tokenizer or ~4 chars/token]
    T --> B{input tokens ><br/>2/3 of 30k = 20k?}
    B -- yes --> E3[TokenLimitExceeded<br/>HTTP 422]
    B -- no --> X[cost = input + output reservation<br/>output = max_tokens or 1000, input+output <= 30k]
    X --> W{free budget >= cost?<br/>shared 30k pool}
    W -- no, HTTP --> H[wait up to 60s<br/>then 503]
    W -- no, RabbitMQ --> R0[wait; other messages<br/>stay in the queue]
    H --> W
    R0 --> W
    W -- yes --> RES[reserve cost]
    RES --> G[SGLang server<br/>/v1/chat/completions]
    G -->|HTTP error| E4[UpstreamError<br/>HTTP 502]
    G --> REL[release cost<br/>wakes waiting requests]
    E4 --> REL
    REL --> R[GenerateResponse<br/>content + usage]
    R --> A
    R --> P[(RabbitMQ<br/>reply_to or llm.results)]
    E1 & E2 & E3 & E4 -. status=error .-> P
```

Example: with a 30k pool, a request with 3000 input tokens reserves 4000 (3000 + 1000 output), so seven run in parallel; the eighth waits (in RabbitMQ for queued requests) until one finishes.

## 2. Fine-tuning flow

```mermaid
flowchart TD
    D1[Real emails + input .xlsx + gold output .xlsx] --> D2[xlsx_to_csv<br/>finetune/sheets.py]
    D2 --> D3[JSONL: email, input_sheet, output_sheet]
    D3 --> P[prompts.py<br/>system prompt + chat messages]
    P --> SFT
    subgraph SFT [Stage 1: SFT - finetune/sft.py]
        S1[Unsloth load Qwen3-14B 4-bit + LoRA] --> S2[SFTTrainer<br/>loss on assistant reply only]
    end
    SFT --> SV[(outputs/qwen3-14b-sft<br/>LoRA + merged)]
    SV -->|--init-adapter| RL
    P --> RL
    subgraph RL [Stage 2: RL - finetune/rl.py]
        R1[Unsloth load model + LoRA] --> R2[GRPO: sample N sheets per prompt]
        R2 --> R3[rewards.py<br/>format 0.5 + 2.0 x cell accuracy]
        R3 --> R4[policy update]
        R4 --> R2
    end
    RL --> RV[(outputs/qwen3-14b-grpo/merged)]
    RV --> SG[Serve with SGLang<br/>add to LLM_MODEL_ENDPOINTS]
    SG --> SVC[LLM service]
```

## 3. Parameters to improve fine-tuning

Flags marked **CLI** can be set on the command line; **code** means edit the script/config.

### Data (usually the biggest lever)
| What | Guidance |
|---|---|
| Number of examples | SFT: hundreds to a few thousand real email/sheet pairs. The sample files are only for format checking. |
| Coverage | Include add-row, edit-row, delete-row, multi-item emails, messy formatting, missing values. |
| Gold quality | Wrong gold sheets directly teach wrong answers and distort the cell-accuracy reward. |
| Held-out set | Keep ~10% unseen; score it with `cell_accuracy` before/after training. |
| `OUTPUT_COLUMNS` (code, `sheets.py`) | Must match the real format exactly. |

### SFT (`finetune/sft.py`)
| Param | Default | Tune |
|---|---|---|
| `--epochs` CLI | 2 | 1-3. Raise if underfitting; lower if train loss drops but validation accuracy stalls (overfit). |
| `--lr` CLI | 2e-4 | 1e-4 to 2e-4 for LoRA. Lower if loss is unstable. |
| `--lora-r` CLI | 16 | 16 to 64. Higher = more capacity and more memory. Keep alpha = r. |
| `--batch-size` x `--grad-accum` CLI | 2 x 4 | Effective batch 8-32. Raise grad-accum to stabilise without extra memory. |
| `--max-seq-length` CLI | 4096 | Must fit email + input sheet + output sheet. Large sheets need more (memory cost). |
| warmup_ratio, lr scheduler (code) | 0.05, linear | Cosine can help on longer runs. |
| LoRA dropout, target modules (code, `common.py`) | 0, all linear layers | Add small dropout (0.05) if overfitting. |

### RL / GRPO (`finetune/rl.py`)
| Param | Default | Tune |
|---|---|---|
| `--num-generations` CLI | 4 | 8-16 gives a better reward baseline per prompt; costs memory and time. |
| `--lr` CLI | 5e-6 | RL needs much lower LR than SFT. 1e-6 to 1e-5. |
| `--max-steps` CLI | 250 | Watch mean reward; stop when it plateaus. |
| `--max-completion-length` CLI | 1024 | Must fit the whole output sheet, or long sheets get truncated and score 0. |
| `--max-prompt-length` CLI | 1024 | Must fit email + input sheet. |
| `--init-adapter` CLI | none | Always start RL from the SFT model; RL from the base model is slow and unstable. |
| Reward weights (code, `rewards.py`) | format 0.5, accuracy 2.0 | Drop the format weight once the model always gets the header right. |
| Sampling temperature, KL `beta` (code, `GRPOConfig`) | TRL defaults | Higher temperature = more exploration; higher beta = stays closer to the SFT model. |

### Evaluating improvement
1. Run the held-out set through the model, compute `cell_accuracy` per example.
2. Track mean accuracy, % of exact-match sheets, and % with a valid header.
3. Compare base vs SFT vs SFT+GRPO. Only keep GRPO if it beats SFT on the held-out set.
