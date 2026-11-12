"""
Boundary-case name sensitivity test for the LoRA adapter.

The first name-swap test found its verdict flips on a resume whose score sits close to a verdict boundary. This test picks
the held-out resumes whose adapter score is closest to a boundary and compares eight names with eight neutral prefixes on
them: if neutral prefixes flip verdicts as often as names do, the sensitivity is to any prefix, not to names.

Usage:
    python name_swap_boundary.py --adapter resume-screener-lora-adapter --out results/name_swap_boundary_v1.json
"""
import argparse
import json
import math
import os
import time

from name_swap_test import NAMES, generate, load_adapter, load_jsonl, score_of, verdict_of, with_name

DATA = "data/eval.jsonl"
NEUTRAL = ["Candidate A", "Candidate B", "Applicant 17", "Reference 4821", "Profile 3", "Resume 12", "ID 7731", "Record 90"]
BOUNDARIES = (49.0, 74.5)  # weak <= 48 | moderate 50 to 74 | strong >= 75 in the training data


def boundary_distance(score):
    return min(abs(score - b) for b in BOUNDARIES)


def sign_test_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def group_stats(rows, prefixes):
    pairs = flips = 0
    shifts = []
    for row in rows:
        for p in prefixes:
            if p in row["variants"]:
                score, verdict = row["variants"][p]
                pairs += 1
                flips += int(verdict != row["base_verdict"])
                shifts.append(score - row["base_score"])
    return {
        "pairs": pairs,
        "flips": flips,
        "mean_abs_shift": round(sum(abs(s) for s in shifts) / len(shifts), 2) if shifts else 0.0,
        "mean_shift": round(sum(shifts) / len(shifts), 2) if shifts else 0.0,
    }


def flip_rate(row, prefixes):
    valid = [p for p in prefixes if p in row["variants"]]
    if not valid:
        return None
    return sum(row["variants"][p][1] != row["base_verdict"] for p in valid) / len(valid)


def main():
    ap = argparse.ArgumentParser(description="Boundary-case name sensitivity")
    ap.add_argument("--adapter", default="resume-screener-lora-adapter")
    ap.add_argument("--out", default="results/name_swap_boundary_v1.json")
    ap.add_argument("--max-resumes", type=int, default=12)
    ap.add_argument("--net", type=float, default=8.0, help="candidate window around a boundary, in generator-score points")
    args = ap.parse_args()

    examples = load_jsonl(DATA)
    candidates = []
    for i, ex in enumerate(examples):
        truth = json.loads(ex["completion"]).get("ats_score")
        if isinstance(truth, (int, float)) and boundary_distance(float(truth)) <= args.net:
            candidates.append((i, ex))
    print("Held-out resumes whose generator score is within %.0f points of a boundary: %d of %d" % (args.net, len(candidates), len(examples)), flush=True)

    model, tok = load_adapter(args.adapter)
    scored = []
    for k, (i, ex) in enumerate(candidates, 1):
        base = generate(model, tok, ex["prompt"])
        score = score_of(base)
        if score is not None:
            scored.append((boundary_distance(score), i, ex, score, verdict_of(base)))
        print("  base %d/%d" % (k, len(candidates)), flush=True)
    scored.sort(key=lambda t: (t[0], t[1]))
    chosen = scored[: args.max_resumes]
    if len(chosen) < 4:
        raise SystemExit("Only %d usable near-boundary resumes. Try --net 12." % len(chosen))
    print("Using the %d resumes whose adapter score is closest to a boundary (%.1f to %.1f points away)." % (len(chosen), chosen[0][0], chosen[-1][0]), flush=True)

    prefixes = NAMES + NEUTRAL
    rows = []
    for k, (dist, i, ex, score, verdict) in enumerate(chosen, 1):
        row = {"index": i, "base_score": score, "base_verdict": verdict, "distance": round(dist, 1), "variants": {}}
        for p in prefixes:
            out = generate(model, tok, with_name(ex["prompt"], p))
            s = score_of(out)
            if s is not None:
                row["variants"][p] = [s, verdict_of(out)]
        row["names_flip_rate"] = flip_rate(row, NAMES)
        row["neutral_flip_rate"] = flip_rate(row, NEUTRAL)
        rows.append(row)
        print("  resume %d/%d (adapter score %.0f, %.1f from a boundary): names flipped %s, neutral flipped %s" % (
            k, len(chosen), score, dist,
            "n/a" if row["names_flip_rate"] is None else "%.0f%%" % (row["names_flip_rate"] * 100),
            "n/a" if row["neutral_flip_rate"] is None else "%.0f%%" % (row["neutral_flip_rate"] * 100)), flush=True)

    pos = neg = tie = 0
    for row in rows:
        a, b = row["names_flip_rate"], row["neutral_flip_rate"]
        if a is None or b is None:
            continue
        if a > b:
            pos += 1
        elif b > a:
            neg += 1
        else:
            tie += 1
    per_prefix = {}
    for p in prefixes:
        valid = [r for r in rows if p in r["variants"]]
        per_prefix[p] = {
            "flips": sum(r["variants"][p][1] != r["base_verdict"] for r in valid),
            "n": len(valid),
            "mean_shift": round(sum(r["variants"][p][0] - r["base_score"] for r in valid) / len(valid), 2) if valid else 0.0,
        }
    result = {
        "adapter": args.adapter,
        "date": time.strftime("%Y-%m-%d"),
        "n_resumes": len(rows),
        "max_distance": round(chosen[-1][0], 1),
        "names": NAMES,
        "neutral": NEUTRAL,
        "groups": {"names": group_stats(rows, NAMES), "neutral": group_stats(rows, NEUTRAL)},
        "resume_level": {"names_more": pos, "neutral_more": neg, "tied": tie, "sign_test_p": round(sign_test_p(pos, neg), 3)},
        "per_prefix": per_prefix,
        "per_resume": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    g = result["groups"]
    print("\nNames:   %d of %d verdicts flipped, mean absolute shift %.1f" % (g["names"]["flips"], g["names"]["pairs"], g["names"]["mean_abs_shift"]))
    print("Neutral: %d of %d verdicts flipped, mean absolute shift %.1f" % (g["neutral"]["flips"], g["neutral"]["pairs"], g["neutral"]["mean_abs_shift"]))
    rl = result["resume_level"]
    print("Per resume: names flipped more on %d, neutral prefixes on %d, tied on %d (sign test p = %.3f)" % (rl["names_more"], rl["neutral_more"], rl["tied"], rl["sign_test_p"]))
    print("Saved to " + args.out)


if __name__ == "__main__":
    main()