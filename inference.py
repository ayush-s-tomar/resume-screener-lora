"""
Standalone inference script for the LoRA fine-tuned resume screener.
Loads Qwen2.5-0.5B-Instruct base model + LoRA adapter, returns structured JSON verdict.

Usage:
    python inference.py
"""

import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def screen_resume(model, tokenizer, resume_text: str, role: str) -> dict:
    prompt = f"Screen this resume for a {role} position and return a structured verdict.\n\nResume: {resume_text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(inputs, max_new_tokens=150, do_sample=False)
    raw = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}


if __name__ == "__main__":
    model, tokenizer = load_model()

    example_resume = (
        "Bachelor's degree in Computer Science. 4 years of experience building "
        "distributed backend systems. Skilled in Python, PostgreSQL, Docker, Kubernetes."
    )

    verdict = screen_resume(model, tokenizer, example_resume, role="Backend Engineer")
    print(json.dumps(verdict, indent=2))