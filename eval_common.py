"""Shared helpers for the evaluation scripts: load an adapter, generate one JSON verdict per prompt, score a run."""
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ORDER = {"weak_match": 0, "moderate_match": 1, "strong_match": 2}
PROMPT = "Screen this resume for a {role} position and return a structured verdict.\n\nResume: {resume}"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_adapter(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_json(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return extract_json(text)


def prompt_of(example):
    if "prompt" in example:
        return example["prompt"]
    return PROMPT.format(role=example["role"], resume=example["resume"])


def truth_of(example):
    if "completion" in example:
        truth = json.loads(example["completion"])
        return truth["verdict"], truth.get("ats_score")
    return example["label"], None


def num(parsed):
    value = parsed.get("ats_score") if isinstance(parsed, dict) else None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def verdict_of(parsed):
    value = parsed.get("verdict") if isinstance(parsed, dict) else None
    return value if value in ORDER else "invalid"


def score_run(preds, truths):
    n = len(truths)
    labels = list(ORDER) + ["invalid"]
    confusion = {t: {p: 0 for p in labels} for t in ORDER}
    correct = severe = 0
    errors = []
    for parsed, (truth, true_score) in zip(preds, truths):
        pred = verdict_of(parsed)
        confusion[truth][pred] += 1
        correct += int(pred == truth)
        severe += int(pred != "invalid" and abs(ORDER[pred] - ORDER[truth]) == 2)
        if true_score is not None and num(parsed) is not None:
            errors.append(abs(num(parsed) - true_score))
    concordant = pairs = 0.0
    for i, (ti, _) in enumerate(truths):
        for j, (tj, _) in enumerate(truths):
            si, sj = num(preds[i]), num(preds[j])
            if ORDER[ti] < ORDER[tj] and si is not None and sj is not None:
                pairs += 1
                concordant += 1.0 if si < sj else (0.5 if si == sj else 0.0)
    return {
        "n_examples": n,
        "valid_json_rate": round(sum(p is not None for p in preds) / n, 3),
        "verdict_accuracy": round(correct / n, 3),
        "verdict_correct": correct,
        "severe_errors": severe,
        "mean_absolute_score_error": round(sum(errors) / len(errors), 2) if errors else None,
        "pairwise_ordering": round(concordant / pairs, 3) if pairs else None,
        "confusion": confusion,
    }