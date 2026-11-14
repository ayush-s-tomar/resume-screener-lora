"""
Score an adapter on a labelled JSONL set (data/eval.jsonl or data/handwritten_eval.jsonl).

python eval_adapter.py --adapter resume-screener-lora-adapter --data data/handwritten_eval.jsonl --out results/handwritten_v1.json
"""
import argparse
import json
import os
import time

from eval_common import generate_json, load_adapter, load_jsonl, num, prompt_of, score_run, truth_of, verdict_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="resume-screener-lora-adapter")
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    examples = load_jsonl(args.data)
    model, tok = load_adapter(args.adapter)
    preds = []
    for i, ex in enumerate(examples, 1):
        preds.append(generate_json(model, tok, prompt_of(ex)))
        print("  %s %d/%d" % (os.path.basename(args.adapter), i, len(examples)), flush=True)

    truths = [truth_of(ex) for ex in examples]
    result = score_run(preds, truths)
    result.update(adapter=args.adapter, data=args.data, date=time.strftime("%Y-%m-%d"))
    result["per_example"] = [
        {"id": ex.get("id", i), "role": ex.get("role"), "label": t[0], "pred": verdict_of(p), "pred_score": num(p)}
        for i, (ex, t, p) in enumerate(zip(examples, truths, preds))
    ]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("\nverdict accuracy %.3f (%d/%d), severe errors %d, pairwise ordering %s, mean score error %s" % (
        result["verdict_accuracy"], result["verdict_correct"], result["n_examples"], result["severe_errors"],
        result["pairwise_ordering"], result["mean_absolute_score_error"]))
    print("Saved to " + args.out)


if __name__ == "__main__":
    main()