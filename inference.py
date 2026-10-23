"""
Standalone inference script for the LoRA fine-tuned resume screener.
Loads the Qwen2.5-0.5B-Instruct base model plus the LoRA adapter and returns a
structured JSON verdict.

Usage:
    python inference.py                                   # built-in example
    python inference.py --role "Backend Engineer" --resume path/to/resume.txt
    python inference.py --role "Backend Engineer" --resume "raw resume text ..."
"""

import argparse
import json
import os
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "resume-screener-lora-adapter"
)

EXAMPLE_RESUME = (
    "Bachelor's degree in Computer Science. 4 years of experience building "
    "distributed backend systems. Skilled in Python, PostgreSQL, Docker, Kubernetes."
)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if torch.cuda.is_available():
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype="auto", device_map="auto"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32, device_map="cpu"
        )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def parse_verdict(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"raw_output": raw, "parse_error": True}


def screen_resume(model, tokenizer, resume_text: str, role: str) -> dict:
    prompt = (
        f"Screen this resume for a {role} position and return a structured verdict."
        f"\n\nResume: {resume_text}"
    )
    messages = [{"role": "user", "content": prompt}]
    # return_dict=True gives input_ids + attention_mask on every transformers version
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)

    raw = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return parse_verdict(raw)


def read_resume(value: str) -> str:
    """--resume may be a path to a text file or the raw resume text itself."""
    if os.path.isfile(value):
        with open(value, encoding="utf-8") as f:
            return f.read().strip()
    return value.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Screen a resume with the LoRA fine-tuned model."
    )
    parser.add_argument("--role", default="Backend Engineer", help="Target role")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a resume text file, or the raw resume text. Omit to use a built-in example.",
    )
    args = parser.parse_args()

    resume = read_resume(args.resume) if args.resume else EXAMPLE_RESUME
    model, tokenizer = load_model()
    verdict = screen_resume(model, tokenizer, resume, args.role)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()