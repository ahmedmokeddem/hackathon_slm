import json

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_NEW_TOKENS = 256

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)
model.eval()


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "Task: generate exactly one valid Polars Python expression assigned to result.\n"

                "Rules:\n"
                "- Output only Python code\n"
                "- No imports, comments, or explanations\n"
                "- Assume pl is already available\n"
                "- Never use df\n"
                "- Use only provided table names and exact column names\n"
                "- Result must start with: result =\n"

                "Polars rules:\n"
                "- Use group_by, not groupby\n"
                "- After group_by, always use .agg(...)\n"
                "- Never use with_columns after group_by\n"
                "- Use with_columns only before aggregation\n"
                "- Use pl.col(...).sum(), not pl.sum(...)\n"
                "- Use head(k), not limit(k)\n"
                "- Use is_null() / is_not_null() for null checks\n"
                "- For 'never' queries, prefer join(..., how='anti')\n"
                "- For ranking within groups, use rank(...).over(...)\n"

                "Useful patterns:\n"
                "- Revenue = unit_price * quantity * (1 - discount)\n"
                "- Compute revenue before group_by, then aggregate with sum().round(2)\n"
                "- For top-k queries: sort(..., descending=True).head(k)\n"
                "- For boolean columns, use pl.col(\"flag\") directly, not == 1 or == 0\n"
                "- For year/month from string dates, use str.slice(...)\n"
                "- If duplicate column names appear after joins, use Polars suffixes like _right\n"

                f"TABLES = {json.dumps(payload.tables, ensure_ascii=False)}"
            )
        },
        {
            "role": "user",
            "content": payload.message,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.2,
        top_p=1.0,
        top_k=20,
        num_beams=1,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))