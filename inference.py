import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import strip_code_fence

MODEL_NAME      = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_NEW_TOKENS  = 256

tokenizer: AutoTokenizer | None = None
model: AutoModelForCausalLM | None = None


def load_model() -> None:
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )


@torch.inference_mode()
def generate(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return strip_code_fence(raw)
