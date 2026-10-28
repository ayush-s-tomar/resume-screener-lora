"""
eval_groq_baseline.py
---------------------
Scores the held-out eval set with a larger, prompted (not fine-tuned) model:
openai/gpt-oss-20b on Groq. It uses the same system prompt (JSON schema plus the
verdict -> ats_score ranges measured on the training set) and the same user
prompts as the "zero-shot + rubric" baseline in compare_baseline_vs_finetuned.py,
so the only thing that changes is the model.

Usage:
    export GROQ_API_KEY=...            # Windows: $env:GROQ_API_KEY="..."
    python eval_groq_baseline.py

Options:
    --limit N   only the first N eval examples (smoke test)
    --out PATH  results file (default: groq_baseline_results.json)
    --pace S    seconds to wait between requests (default: 4)

Progress is cached in the system temp folder, so an interrupted or rate-limited
run resumes where it stopped.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"
TRAIN_PATH = "./data/train.jsonl"
EVAL_PATH = "./data/eval.jsonl"

# Kept identical to compare_baseline_vs_finetuned.py so that only the model differs.
ZERO_SHOT_INSTRUCTION = (
    "You are a resume screening assistant. Given a resume, respond with ONLY "
    "a JSON object with these exact keys: role (string), ats_score (0-100 "
    "integer), verdict (one of: strong_match, moderate_match, weak_match), "
    "matched_skills (list of strings), missing_skills (list of strings), "
    "years_experience (integer). Do not include any text outside the JSON."
)


class QuotaExhausted(Exception):
    pass


class FatalApiError(Exception):
    pass


class GaveUp(Exception):
    pass


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_rubric(train_path):
    """Min/max ats_score per verdict, measured on the training set only."""
    ranges = {}
    for ex in load_jsonl(train_path):
        label = json.loads(ex["completion"])
        score = label["ats_score"]
        lo, hi = ranges.get(label["verdict"], (score, score))
        ranges[label["verdict"]] = (min(lo, score), max(hi, score))
    return dict(sorted(ranges.items(), key=lambda kv: -kv[1][0]))


def rubric_instruction(ranges):
    spec = ", ".join(
        f"{verdict}: ats_score {lo}-{hi}" for verdict, (lo, hi) in ranges.items()
    )
    return (
        ZERO_SHOT_INSTRUCTION
        + " The verdict must agree with ats_score using these ranges: "
        + spec
        + "."
    )


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


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


def call_groq(api_key, system, user, use_effort):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 2000,
    }
    if use_effort:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            # a custom User-Agent avoids Cloudflare rejecting the default urllib one
            "User-Agent": "resume-screener-eval/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content") or ""


def score_one(api_key, system, user, max_tries=8):
    use_effort = True
    for attempt in range(1, max_tries + 1):
        try:
            return call_groq(api_key, system, user, use_effort)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            if exc.code == 429:
                try:
                    wait = float(exc.headers.get("retry-after") or 0)
                except ValueError:
                    wait = 0.0
                wait = wait or min(60.0, 10.0 * attempt)
                if wait > 180:
                    raise QuotaExhausted("Groq asks to wait %.0fs: %s" % (wait, detail))
                print("    rate limited, waiting %.0fs" % wait, flush=True)
                time.sleep(wait + 1)
            elif exc.code == 400 and use_effort and "reason" in detail.lower():
                use_effort = False
            elif exc.code in (500, 502, 503, 504):
                time.sleep(5 * attempt)
            else:
                raise FatalApiError("HTTP %s: %s" % (exc.code, detail))
        except (urllib.error.URLError, OSError):
            time.sleep(5 * attempt)
    raise GaveUp("no valid response after %d tries" % max_tries)


def main():
    ap = argparse.ArgumentParser(description="Score the eval set with gpt-oss-20b on Groq.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="groq_baseline_results.json")
    ap.add_argument("--pace", type=float, default=4.0)
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY is not set.")

    eval_examples = load_jsonl(EVAL_PATH)
    if args.limit:
        eval_examples = eval_examples[: args.limit]
    truths = [json.loads(ex["completion"]) for ex in eval_examples]

    ranges = build_rubric(TRAIN_PATH)
    system = rubric_instruction(ranges)
    print("Model: %s | examples: %d | rubric (train set): %s" % (MODEL, len(eval_examples), ranges), flush=True)

    key = hashlib.sha1((MODEL + system).encode("utf-8")).hexdigest()[:8]
    cache_path = os.path.join(tempfile.gettempdir(), "groq_baseline_cache_%s.jsonl" % key)
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["i"]] = row["text"]
        print("Resuming: %d answers already cached" % len(cache), flush=True)

    preds = []
    api_errors = 0
    with open(cache_path, "a", encoding="utf-8") as cf:
        for i, ex in enumerate(eval_examples):
            if i in cache:
                text = cache[i]
            else:
                try:
                    text = score_one(api_key, system, ex["prompt"])
                except QuotaExhausted as exc:
                    print("Rate limit or daily quota reached (%s)." % exc)
                    print("Progress is saved. Re-run later to resume.")
                    sys.exit(3)
                except FatalApiError as exc:
                    print("Groq API error: %s" % exc)
                    sys.exit(2)
                except GaveUp as exc:
                    print("  example %d: %s" % (i + 1, exc))
                    text = None
                if text is not None:
                    cf.write(json.dumps({"i": i, "text": text}) + "\n")
                    cf.flush()
                time.sleep(args.pace)
            if text is None:
                api_errors += 1
                text = ""
            preds.append(text)
            print("  groq %d/%d" % (i + 1, len(eval_examples)), flush=True)

    results = score_predictions(preds, truths)
    results["n_examples"] = len(eval_examples)
    results["api_errors"] = api_errors
    results["model"] = MODEL
    results["prompt"] = "same system prompt, score ranges and user prompts as the zero-shot + rubric baseline"
    results["reasoning_effort"] = "low"
    results["temperature"] = 0
    results["rubric_ranges"] = {k: list(v) for k, v in ranges.items()}
    results["date"] = time.strftime("%Y-%m-%d")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nRESULTS (%s)" % MODEL)
    print("  valid JSON rate:      %s" % results["valid_json_rate"])
    print("  verdict accuracy:     %s" % results["verdict_accuracy"])
    print("  mean abs score error: %s" % results["mean_absolute_score_error"])
    print("  API errors:           %d" % api_errors)
    print("Saved to %s" % args.out)


if __name__ == "__main__":
    main()