"""
compare_baseline_vs_finetuned.py
----------------------------------
Runs the eval set through two configurations of the SAME base model:
  1. Zero-shot: base model, prompted to return JSON, no fine-tuning
  2. Fine-tuned: base model + LoRA adapter trained on our synthetic dataset

Compares both against ground truth on:
  - Valid JSON rate (does it even parse?)
  - Exact verdict match rate (strong/moderate/weak_match correct?)
  - ATS score mean absolute error

Run this AFTER downloading resume-screener-lora-final.zip from Colab and
unzipping it into this project folder.

Usage:
    python compare_baseline_vs_finetuned.py
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-final"
EVAL_PATH = "./data/eval.jsonl"

ZERO_SHOT_INSTRUCTION = (
    "You are a resume screening assistant. Given a resume, respond with ONLY "
    "a JSON object with these exact keys: role (string), ats_score (0-100 "
    "integer), verdict (one of: strong_match, moderate_match, weak_match), "
    "matched_skills (list of strings), missing_skills (list of strings), "
    "years_experience (integer). Do not include any text outside the JSON."
)


def load_eval_set(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def extract_json(text):
    """Pull the first {...} block out of a model response, tolerating extra text."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate(model, tokenizer, messages, max_new_tokens=200):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)


def score_predictions(predictions, ground_truths):
    valid_json_count = 0
    verdict_correct = 0
    score_errors = []

    for pred_text, truth in zip(predictions, ground_truths):
        parsed = extract_json(pred_text)
        if parsed is None:
            continue
        valid_json_count += 1

        if parsed.get("verdict") == truth["verdict"]:
            verdict_correct += 1

        if isinstance(parsed.get("ats_score"), (int, float)):
            score_errors.append(abs(parsed["ats_score"] - truth["ats_score"]))

    n = len(ground_truths)
    return {
        "valid_json_rate": round(valid_json_count / n, 3),
        "verdict_accuracy": round(verdict_correct / n, 3),
        "mean_absolute_score_error": round(sum(score_errors) / len(score_errors), 2) if score_errors else None,
    }


def main():
    eval_examples = load_eval_set(EVAL_PATH)
    ground_truths = [json.loads(ex["completion"]) for ex in eval_examples]

    print(f"Loaded {len(eval_examples)} eval examples\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, device_map="cpu"
    )

    # --- Zero-shot baseline ---
    print("Running zero-shot baseline...")
    zero_shot_preds = []
    for ex in eval_examples:
        messages = [
            {"role": "system", "content": ZERO_SHOT_INSTRUCTION},
            {"role": "user", "content": ex["prompt"]},
        ]
        zero_shot_preds.append(generate(base_model, tokenizer, messages))

    zero_shot_results = score_predictions(zero_shot_preds, ground_truths)

    # --- Fine-tuned model ---
    print("Loading fine-tuned adapter and running eval...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    finetuned_preds = []
    for ex in eval_examples:
        messages = [{"role": "user", "content": ex["prompt"]}]
        finetuned_preds.append(generate(finetuned_model, tokenizer, messages))

    finetuned_results = score_predictions(finetuned_preds, ground_truths)

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"{'Metric':<28} {'Zero-shot':<12} {'Fine-tuned':<12}")
    print("-" * 50)
    print(f"{'Valid JSON rate':<28} {zero_shot_results['valid_json_rate']:<12} {finetuned_results['valid_json_rate']:<12}")
    print(f"{'Verdict accuracy':<28} {zero_shot_results['verdict_accuracy']:<12} {finetuned_results['verdict_accuracy']:<12}")
    print(f"{'Mean abs score error':<28} {zero_shot_results['mean_absolute_score_error']:<12} {finetuned_results['mean_absolute_score_error']:<12}")

    with open("comparison_results.json", "w") as f:
        json.dump({"zero_shot": zero_shot_results, "fine_tuned": finetuned_results}, f, indent=2)
    print("\nSaved full results to comparison_results.json")


if __name__ == "__main__":
    main()
