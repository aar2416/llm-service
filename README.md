# llm-service

FastAPI + SGLang (Qwen) service with RabbitMQ triggers, a shared token budget, and Unsloth fine-tuning.

```
src/llm_service_main/   service code
finetune/               Unsloth SFT + GRPO (Qwen3-14B)
```

## Quick start
```bash
pip install -e ".[dev]"
python -m sglang.launch_server --model-path Qwen/Qwen3-14B --port 30000
cp .env.example .env
uvicorn llm_service_main.main:app --app-dir src --port 8000
```
`POST /generate` `{"model": "Qwen/Qwen3-14B", "messages": [...], "stream": false}`

## Request flow
```mermaid
flowchart TD
    A[HTTP /generate] --> V[validate model + count tokens]
    Q[(RabbitMQ llm.requests)] --> V
    V -->|unknown model| E1[404]
    V -->|input > 2/3 of 30k| E2[422]
    V --> W{free budget >= cost?}
    W -- no --> Wait[HTTP: wait 60s then 503<br/>RabbitMQ: stays in queue]
    Wait --> W
    W -- yes --> G[SGLang]
    G --> R[release budget]
    R --> Out[JSON / SSE stream / RabbitMQ reply]
```

## Token budget
```mermaid
flowchart LR
    P[(30k token pool)] --> R1[req A: 3000 in + 1000 out = 4000]
    P --> R2[req B: 4000]
    P --> R3[... up to 7 x 4000]
    P -. full .-> R4[req 8 waits]
    R1 -- done, releases 4000 --> P
```

## Fine-tuning
```mermaid
flowchart LR
    D[email + input xlsx + gold xlsx] --> C[CSV JSONL]
    C --> S[sft.py<br/>Unsloth SFT]
    S --> RL[rl.py<br/>GRPO]
    RL --> M[merged model]
    M --> SG[serve with SGLang]
    RL -. reward .- RW[0.5 format + 2.0 x % correct cells]
```
```bash
pip install -e ".[finetune]"   # needs NVIDIA GPU
python -m finetune.sft
python -m finetune.rl --init-adapter outputs/qwen3-14b-sft/merged
```
Output columns are set in `finetune/sheets.py` (`OUTPUT_COLUMNS`). Tuning parameters: [docs/flowcharts.md](docs/flowcharts.md).
