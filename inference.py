"""
Standalone inference script for the LoRA fine-tuned resume screener.
Loads Qwen2.5-0.5B-Instruct base model + LoRA adapter, returns structured JSON verdict.

Usage:
    python inference.py
    python inference.py --role "Backend Engineer" --resume path/to/resume.txt
    python inference.py --role "Backend Engineer" --resume "Bachelor's in CS, 4 years..."
"""

import argparse
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"

EXAMPLE_RESUME = (
    "Bachelor's degree in Computer Science. 4 years of experience building "
    "distributed backend systems. Skilled in Python, PostgreSQL, Docker, Kubernetes."
)
EXAMPLE_ROLE = "Backend Engineer"


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


def resolve_resume_text(resume_arg: str) -> str:
    """If resume_arg is a path to an existing file, read it. Otherwise treat it as raw text."""
    if resume_arg and os.path.isfile(resume_arg):
        with open(resume_arg, "r", encoding="utf-8") as f:
            return f.read()
    return resume_arg


def main():
    parser = argparse.ArgumentParser(description="Screen a resume with the fine-tuned LoRA adapter.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a resume text file, or raw resume text as a string. Defaults to a built-in example if omitted.",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="Target role to screen the resume against. Defaults to a built-in example if omitted.",
    )
    args = parser.parse_args()

    resume_text = resolve_resume_text(args.resume) if args.resume else EXAMPLE_RESUME
    role = args.role if args.role else EXAMPLE_ROLE

    if args.resume is None and args.role is None:
        print("[No --resume/--role provided -- running built-in example]\n")

    model, tokenizer = load_model()
    verdict = screen_resume(model, tokenizer, resume_text, role=role)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()