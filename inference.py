"""
Standalone inference script for the LoRA fine-tuned resume screener.
Loads Qwen2.5-0.5B-Instruct base model + LoRA adapter, returns a structured JSON verdict.

When a borderline margin is enabled (see borderline.py), the verdict also says whether its score sits close to a verdict
boundary, so that an application can send those resumes to a human instead of deciding them automatically.

Usage:
    python inference.py                                   # built-in example resume
    python inference.py --role "Data Engineer" --resume-file resume.txt
    python inference.py --margin 3                        # flag scores within 3 points of a boundary
    python inference.py --margin 0                        # never flag
"""

import argparse
import json
from pathlib import Path

try:
    import borderline
except ImportError:  # inference.py was copied on its own, without borderline.py
    borderline = None

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"

EXAMPLE_RESUME = (
    "Bachelor's degree in Computer Science. 4 years of experience building "
    "distributed backend systems. Skilled in Python, PostgreSQL, Docker, Kubernetes."
)


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    try:
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype="auto", device_map="auto")
    except (TypeError, ValueError):  # older transformers versions only know torch_dtype
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def screen_resume(model, tokenizer, resume_text: str, role: str, margin=None) -> dict:
    """Score one resume.

    margin is the borderline margin in score points: None uses the default from borderline.py (off if the analysis
    recommended none), 0 turns the flag off.
    """
    prompt = f"Screen this resume for a {role} position and return a structured verdict.\n\nResume: {resume_text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = (inputs["input_ids"] if hasattr(inputs, "keys") else inputs).to(model.device)

    output = model.generate(input_ids, max_new_tokens=150, do_sample=False)
    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    try:
        verdict = json.loads(raw)
        if not isinstance(verdict, dict):
            raise ValueError("the output is not a JSON object")
    except ValueError:
        verdict = {"raw_output": raw, "parse_error": True}
    return borderline.annotate(verdict, margin) if borderline else verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="Screen a resume with the LoRA adapter")
    ap.add_argument("--role", default="Backend Engineer", help="target role (default: Backend Engineer)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--resume", help="resume text")
    src.add_argument("--resume-file", help="path to a UTF-8 text file containing the resume")
    ap.add_argument("--margin", type=float, default=None,
                    help="borderline margin in score points (default: from results/selective_v1.json; 0 turns the flag off)")
    args = ap.parse_args(argv)

    if args.resume_file:
        resume = Path(args.resume_file).read_text(encoding="utf-8-sig")
    else:
        resume = args.resume or EXAMPLE_RESUME

    model, tokenizer = load_model()
    verdict = screen_resume(model, tokenizer, resume, role=args.role, margin=args.margin)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
