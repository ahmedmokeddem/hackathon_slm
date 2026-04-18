import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import strip_code_fence

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER    = "Sokheng/qwen2.5-coder-7b-polars"
MAX_NEW_TOKENS = 256

tokenizer = None
model     = None

def load_model() -> None:
    global tokenizer, model
    tokenizer   = AutoTokenizer.from_pretrained(BASE_MODEL)
    base        = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype       = torch.bfloat16,
        device_map  = "auto",
    )
    model = PeftModel.from_pretrained(base, ADAPTER)
    model = model.merge_and_unload()
    model.eval()

@torch.inference_mode()
def generate(messages: list[dict]) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize    = False,
        add_generation_prompt=True,
    )
    inputs  = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens = MAX_NEW_TOKENS,
        do_sample   = False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache   =True,
    )
    return strip_code_fence(tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ))