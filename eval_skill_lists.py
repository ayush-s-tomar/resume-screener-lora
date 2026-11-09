"""
Skill-list evaluation for the LoRA adapter on the held-out set.

Compares the adapter's matched_skills and missing_skills with the generator's lists (micro precision, recall, F1 and
exact-set match), and runs two label-independent consistency checks against the resume text:
  - how many skills the adapter lists as missing actually appear in the resume (contradictions)
  - how many skills it lists as matched do not appear in the resume (unsupported)
The text check is an approximate string match. It also recomputes verdict accuracy and score error from the same outputs.

Usage: python eval_skill_lists.py --adapter resume-screener-lora-adapter --out results/skill_lists_v1.json
"""
import argparse
import json
import os
import re
import tempfile
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DATA = "data/eval.jsonl"
KEYS = ("matched_skills", "missing_skills")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_adapter(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    except TypeError:  # older transformers only know torch_dtype
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tokenizer


def generate_text(model, tokenizer, prompt):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def norm(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def skills_of(obj, key):
    items = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(items, list):
        return None
    return {norm(x) for x in items if str(x).strip()}


def appears(skill, text):
    return any(v and v in text for v in {skill, skill.rstrip("s")})


def main():
    ap = argparse.ArgumentParser(description="Skill-list evaluation")
    ap.add_argument("--adapter", default="resume-screener-lora-adapter")
    ap.add_argument("--out", default="results/skill_lists_v1.json")
    args = ap.parse_args()

    examples = load_jsonl(DATA)
    cache = os.path.join(tempfile.gettempdir(), "skill_lists_raw_%s.jsonl" % os.path.basename(args.adapter))
    raw = {}
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    raw[d["i"]] = d["text"]
        print("Resuming: %d outputs already cached" % len(raw), flush=True)

    model = tok = None
    for i, ex in enumerate(examples):
        if i not in raw:
            if model is None:
                model, tok = load_adapter(args.adapter)
            raw[i] = generate_text(model, tok, ex["prompt"])
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps({"i": i, "text": raw[i]}) + "\n")
        print("  %d/%d" % (i + 1, len(examples)), flush=True)

    stats = {k: {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "n": 0, "pred_total": 0, "pred_bad": 0, "gt_total": 0, "gt_bad": 0} for k in KEYS}
    invalid = correct = 0
    errors = []
    for i, ex in enumerate(examples):
        truth = json.loads(ex["completion"])
        parsed = extract_json(raw[i])
        if parsed is None:
            invalid += 1
        else:
            correct += int(parsed.get("verdict") == truth.get("verdict"))
            score = parsed.get("ats_score")
            if isinstance(score, (int, float)) and not isinstance(score, bool) and truth.get("ats_score") is not None:
                errors.append(abs(float(score) - float(truth["ats_score"])))
        resume = norm(ex["prompt"].partition("Resume: ")[2])
        for key in KEYS:
            gt = skills_of(truth, key) or set()
            pred = skills_of(parsed, key) or set()
            s = stats[key]
            s["tp"] += len(pred & gt)
            s["fp"] += len(pred - gt)
            s["fn"] += len(gt - pred)
            s["exact"] += int(pred == gt)
            s["n"] += 1
            # matched skills should appear in the resume; missing skills should not
            bad = (lambda sk: not appears(sk, resume)) if key == "matched_skills" else (lambda sk: appears(sk, resume))
            s["pred_total"] += len(pred)
            s["pred_bad"] += sum(1 for sk in pred if bad(sk))
            s["gt_total"] += len(gt)
            s["gt_bad"] += sum(1 for sk in gt if bad(sk))

    def summary(s):
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {
            "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(f1, 3),
            "exact_set_match": round(s["exact"] / s["n"], 3),
            "inconsistent_with_resume_rate": round(s["pred_bad"] / s["pred_total"], 3) if s["pred_total"] else 0.0,
            "generator_inconsistent_with_resume_rate": round(s["gt_bad"] / s["gt_total"], 3) if s["gt_total"] else 0.0,
            "predicted_skills": s["pred_total"],
        }

    n = len(examples)
    result = {
        "adapter": args.adapter,
        "date": time.strftime("%Y-%m-%d"),
        "n_examples": n,
        "invalid_outputs": invalid,
        "matched": summary(stats["matched_skills"]),
        "missing": summary(stats["missing_skills"]),
        "verdict_accuracy": round(correct / n, 3),
        "mean_absolute_score_error": round(sum(errors) / len(errors), 2) if errors else None,
        "note": "inconsistent_with_resume_rate: for matched skills, the share NOT found in the resume text; for missing skills, the share that IS found in it (approximate string match).",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    for key, label in (("matched", "matched_skills"), ("missing", "missing_skills")):
        r = result[key]
        print("%-15s F1 %.3f (P %.3f, R %.3f) exact-set %.3f | inconsistent with the resume: %.3f (generator's own lists: %.3f)" % (
            label, r["f1"], r["precision"], r["recall"], r["exact_set_match"],
            r["inconsistent_with_resume_rate"], r["generator_inconsistent_with_resume_rate"]))
    print("verdict accuracy %.3f, mean score error %s, invalid outputs %d" % (result["verdict_accuracy"], result["mean_absolute_score_error"], invalid))
    print("Saved to " + args.out)


if __name__ == "__main__":
    main()