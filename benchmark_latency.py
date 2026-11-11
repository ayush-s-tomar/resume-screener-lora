"""
Latency and footprint benchmark: the local LoRA adapter on CPU versus the hosted demo (gpt-oss-20b on Groq).

Local: per-resume generation time on CPU (after one warm-up run), tokens per second, resident memory.
Hosted: end-to-end latency of /score requests to the live demo (after waking it), including the network.
Sizes: adapter file, base model in the local Hugging Face cache, and the published GGUF files.

Usage: python benchmark_latency.py --adapter resume-screener-lora-adapter --out results/latency_v1.json
"""
import argparse
import json
import os
import platform
import re
import statistics
import time
import urllib.request

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEMO_URL = "https://resume-screener-finetune.onrender.com"
GGUF_REPO = "Kus-hal/resume-screener-gguf"
DATA = "data/eval.jsonl"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pctl(values, q):
    s = sorted(values)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "median": round(statistics.median(values), 2),
        "p90": round(pctl(values, 0.9), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def bench_local(adapter, examples, n):
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    except TypeError:  # older transformers versions only know torch_dtype
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    load_seconds = time.perf_counter() - t0

    def run(prompt):
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        seconds = time.perf_counter() - start
        prompt_tokens = inputs["input_ids"].shape[1]
        return seconds, out.shape[1] - prompt_tokens, prompt_tokens

    run(examples[0]["prompt"])  # warm-up, not counted
    latencies, generated, prompts = [], [], []
    for i, ex in enumerate(examples[:n], 1):
        seconds, new_tokens, prompt_tokens = run(ex["prompt"])
        latencies.append(seconds)
        generated.append(new_tokens)
        prompts.append(prompt_tokens)
        print("  local %d/%d: %.1f s, %d new tokens" % (i, n, seconds, new_tokens), flush=True)
    rss_gb = None
    try:
        import psutil

        rss_gb = round(psutil.Process().memory_info().rss / 2**30, 2)
    except Exception:
        pass
    return {
        "load_seconds": round(load_seconds, 1),
        "latency_seconds": summarize(latencies),
        "tokens_per_second_median": round(statistics.median(g / s for g, s in zip(generated, latencies)), 1),
        "generated_tokens_median": int(statistics.median(generated)),
        "prompt_tokens_median": int(statistics.median(prompts)),
        "resident_memory_gb": rss_gb,
    }


def post_json(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def bench_hosted(url, examples, n):
    awake = False
    for _ in range(14):
        try:
            with urllib.request.urlopen(url + "/health", timeout=30) as resp:
                if resp.status == 200:
                    awake = True
                    break
        except Exception:
            time.sleep(5)
    if not awake:
        print("  hosted demo did not wake; skipping", flush=True)
        return None

    def payload(ex):
        head, _, body = ex["prompt"].partition("Resume: ")
        m = re.search(r"for an? (.+?) position", head)
        return {"target_role": m.group(1) if m else "Software Engineer", "resume_text": body}

    try:
        post_json(url + "/score", payload(examples[n % len(examples)]))  # warm-up, not counted
    except Exception as exc:
        print("  hosted warm-up failed: %s" % exc, flush=True)
    latencies, errors = [], 0
    for i, ex in enumerate(examples[:n], 1):
        start = time.perf_counter()
        try:
            post_json(url + "/score", payload(ex))
            latencies.append(time.perf_counter() - start)
            print("  hosted %d/%d: %.1f s" % (i, n, latencies[-1]), flush=True)
        except Exception as exc:
            errors += 1
            print("  hosted %d/%d failed: %s" % (i, n, exc), flush=True)
    result = summarize(latencies)
    if result is not None:
        result["errors"] = errors
    return result


def sizes(adapter):
    out = {}
    path = os.path.join(adapter, "adapter_model.safetensors")
    if os.path.exists(path):
        out["adapter_mb"] = round(os.path.getsize(path) / 2**20, 1)
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == BASE_MODEL:
                out["base_mb"] = round(repo.size_on_disk / 2**20)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(GGUF_REPO, files_metadata=True)
        files = {s.rfilename: round(s.size / 2**20) for s in info.siblings if s.rfilename.endswith(".gguf") and s.size}
        if files:
            out["gguf_files"] = files
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description="Latency and footprint benchmark")
    ap.add_argument("--adapter", default="resume-screener-lora-adapter")
    ap.add_argument("--out", default="results/latency_v1.json")
    ap.add_argument("--n", type=int, default=12, help="local examples")
    ap.add_argument("--hosted-n", type=int, default=5, help="hosted requests")
    ap.add_argument("--demo-url", default=DEMO_URL)
    ap.add_argument("--no-demo", action="store_true")
    args = ap.parse_args()

    examples = load_jsonl(DATA)
    print("Local benchmark (CPU)...", flush=True)
    local = bench_local(args.adapter, examples, args.n)
    hosted = None
    if not args.no_demo:
        print("Hosted benchmark (live demo)...", flush=True)
        hosted = bench_hosted(args.demo_url, examples, args.hosted_n)
    result = {
        "date": time.strftime("%Y-%m-%d"),
        "hardware": {"os": platform.system(), "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads()},
        "local": local,
        "hosted": hosted,
        "sizes": sizes(args.adapter),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    lat = local["latency_seconds"]
    print("\nLocal adapter: median %.1f s (p90 %.1f s), %.1f tokens/s, load %.0f s, memory %s GB" % (
        lat["median"], lat["p90"], local["tokens_per_second_median"], local["load_seconds"], local["resident_memory_gb"]))
    if hosted:
        print("Hosted demo:   median %.1f s (p90 %.1f s), %d errors" % (hosted["median"], hosted["p90"], hosted["errors"]))
    print("Sizes: %s" % json.dumps(result["sizes"]))
    print("Saved to " + args.out)


if __name__ == "__main__":
    main()