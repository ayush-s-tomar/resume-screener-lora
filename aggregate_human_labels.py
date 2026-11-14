"""
Aggregate human labels for the hand-authored set and compare them with the adapter and with my rubric labels.

Each annotator labels every resume as strong_match, moderate_match or weak_match with labeling/label_resumes.html and
sends back a labels_<id>.json file. Put those files in a folder (default: labels/) and run:

    python aggregate_human_labels.py --labels labels --who "volunteer annotators"

Add --selftest to check the pipeline on simulated annotators; it writes nothing to the repo.
"""
import argparse
import glob
import itertools
import json
import math
import os
import re
import statistics
import tempfile
import time
from collections import Counter

ORD = {"weak_match": 0, "moderate_match": 1, "strong_match": 2}
NAMES = ["weak_match", "moderate_match", "strong_match"]


def load_jsonl(path):
    with open(path, encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


def kappa(a, b, weighted):
    n, k = len(a), 3
    obs = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[ORD[x]][ORD[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i, j):
        return abs(i - j) / (k - 1) if weighted else float(i != j)

    num = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    den = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return 1 - num / den if den else float("nan")


def pair_stats(a, b):
    return {"acc": accuracy(a, b), "wkappa": kappa(a, b, True), "kappa": kappa(a, b, False)}


def mean(rows, key):
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return sum(vals) / len(vals) if vals else float("nan")


def pc(x):
    return "n/a" if math.isnan(x) else "%.0f%%" % (x * 100)


def kf(x):
    return "n/a" if math.isnan(x) else "%.2f" % x


def clean(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def rel(x, ref):
    if math.isnan(x) or math.isnan(ref):
        return "(not computable)"
    gap = ref - x
    if gap <= 0.10:
        return "about as well as the annotators agree with each other"
    if gap <= 0.25:
        return "somewhat less well than the annotators agree with each other"
    return "clearly less well than the annotators agree with each other"


def patch_readme(path, block):
    with open(path, encoding="utf-8") as f:
        text = f.read().replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- hl-start -->") + r".*?" + re.escape("<!-- hl-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
    elif text.count("<!-- ns-start -->") == 1:
        text = text.replace("<!-- ns-start -->", block + "\n\n<!-- ns-start -->", 1)
    else:
        print("README NOT changed ('<!-- ns-start -->' was found %d times). Paste this section where you want it:\n\n%s" % (text.count("<!-- ns-start -->"), block))
        return
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("README updated.")


def main():
    ap = argparse.ArgumentParser(description="Aggregate human labels")
    ap.add_argument("--labels", default="labels")
    ap.add_argument("--data", default="data/handwritten_eval.jsonl")
    ap.add_argument("--adapter-results", default="results/handwritten_v1.json")
    ap.add_argument("--out", default="results/human_labels_v1.json")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--who", default="volunteer annotators")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    rubric_rows = load_jsonl(args.data)
    rubric = {r["id"]: r["label"] for r in rubric_rows}
    if args.selftest:
        import random

        rng = random.Random(7)
        tmp = tempfile.mkdtemp()
        for k in range(3):
            labels = {}
            for r in rubric_rows:
                lab = r["label"]
                if rng.random() < 0.15:
                    lab = rng.choice([x for x in NAMES if x != lab])
                labels[r["id"]] = lab
            with open(os.path.join(tmp, "labels_demo%d.json" % k), "w", encoding="utf-8") as f:
                json.dump({"annotator": "demo%d" % k, "labels": labels}, f)
        args.labels, args.out, args.readme = tmp, os.path.join(tmp, "out.json"), None
        print("SELF-TEST: three simulated annotators (15% random disagreement with the rubric). Nothing is written to the repo.\n")

    with open(args.adapter_results, encoding="utf-8-sig") as f:
        adapter = {e["id"]: e["pred"] for e in json.load(f)["per_example"]}
    raw = []
    for fpath in sorted(glob.glob(os.path.join(args.labels, "*.json"))):
        with open(fpath, encoding="utf-8-sig") as f:
            d = json.load(f)
        raw.append({k: v for k, v in d.get("labels", {}).items() if k in rubric and v in ORD})
    if len(raw) < 2:
        raise SystemExit("Need label files from at least 2 annotators in %s (found %d)." % (args.labels, len(raw)))
    items = [i for i in rubric if all(i in labs for labs in raw)]
    if len(items) < 20:
        raise SystemExit("Only %d resumes are labelled by every annotator; need at least 20. Ask the annotators to finish." % len(items))
    names = ["A%d" % (k + 1) for k in range(len(raw))]
    annots = {names[k]: {i: raw[k][i] for i in items} for k in range(len(raw))}
    n_ann, n_items = len(names), len(items)

    hh = [pair_stats([annots[p][i] for i in items], [annots[q][i] for i in items]) for p, q in itertools.combinations(names, 2)]
    adapter_items = [i for i in items if adapter.get(i) in ORD]
    ah = [pair_stats([annots[p][i] for i in adapter_items], [adapter[i] for i in adapter_items]) for p in names]
    rh = [pair_stats([annots[p][i] for i in items], [rubric[i] for i in items]) for p in names]
    ar = pair_stats([rubric[i] for i in adapter_items], [adapter[i] for i in adapter_items])
    n_pairs = len(hh)
    h_acc, h_wk = mean(hh, "acc"), mean(hh, "wkappa")
    a_acc, a_wk = mean(ah, "acc"), mean(ah, "wkappa")
    r_acc, r_wk = mean(rh, "acc"), mean(rh, "wkappa")

    def consensus(item):
        votes = [annots[p][item] for p in names]
        if n_ann == 2:
            return votes[0] if votes[0] == votes[1] else None
        counts = Counter(votes)
        top = max(counts.values())
        leaders = [v for v, c in counts.items() if c == top]
        if len(leaders) == 1:
            return leaders[0]
        return NAMES[statistics.median_low([ORD[v] for v in votes])]

    cons = {i: consensus(i) for i in items}
    unanimous = sum(len({annots[p][i] for p in names}) == 1 for i in items)
    cons_items = [i for i in adapter_items if cons[i] is not None]
    conf_text = ""
    if len(cons_items) >= 10:
        conf = {t: {p: 0 for p in NAMES} for t in NAMES}
        for i in cons_items:
            conf[cons[i]][adapter[i]] += 1
        rows_md = "\n".join("| %s | %d | %d | %d |" % (t.split("_")[0], conf[t]["weak_match"], conf[t]["moderate_match"], conf[t]["strong_match"]) for t in NAMES)
        conf_text = ("\n\nAgainst the human consensus (%d resumes with a clear majority; rows are the humans, columns the adapter):\n\n"
                     "| | weak | moderate | strong |\n|---|---|---|---|\n%s" % (len(cons_items), rows_md))

    interp = ("The adapter agrees with the annotators %s (weighted kappa %s against %s between the annotators), and my rubric labels agree with them %s (%s)."
              % (rel(a_wk, h_wk), kf(a_wk), kf(h_wk), rel(r_wk, h_wk), kf(r_wk)))
    pair_word = "pair" if n_pairs == 1 else "pairs"
    block = f"""<!-- hl-start -->
### Human labels on a hand-authored test set

The biggest weakness of this project is that its labels come from my own generator. To get a look at human judgment, I wrote a separate set of {n_items} resumes (6 roles and three layouts, drafted with an AI assistant; see [`data/HANDWRITTEN_RUBRIC.md`](data/HANDWRITTEN_RUBRIC.md)) and had {n_ann} {args.who} label each one as a strong, moderate or weak match for its target role, without seeing my rubric or the model's output. `labeling/label_resumes.html` is the form they used.

| Comparison | Accuracy | Weighted kappa |
|---|---|---|
| Annotator vs annotator (mean of {n_pairs} {pair_word}) | {pc(h_acc)} | {kf(h_wk)} |
| Adapter v1 vs annotator (mean) | {pc(a_acc)} | {kf(a_wk)} |
| My rubric vs annotator (mean) | {pc(r_acc)} | {kf(r_wk)} |
| Adapter v1 vs my rubric | {pc(ar['acc'])} | {kf(ar['wkappa'])} |

{interp} The annotators were unanimous on {unanimous} of {n_items} resumes.{conf_text}

This is {n_ann} annotators and {n_items} resumes, the resumes are AI-drafted and the annotators are not necessarily professional recruiters, so the numbers are a sanity check, not a benchmark: one resume moves an accuracy figure by about {100 / n_items:.0f} points. Raw label files are not published to keep annotators anonymous; anonymized per-resume labels are in `results/human_labels_v1.json`, and `python aggregate_human_labels.py` recomputes everything.
<!-- hl-end -->"""

    result = {
        "date": time.strftime("%Y-%m-%d"),
        "n_annotators": n_ann,
        "n_items": n_items,
        "human_vs_human": {"accuracy": clean(h_acc), "weighted_kappa": clean(h_wk), "pairs": n_pairs},
        "adapter_vs_human": {"accuracy": clean(a_acc), "weighted_kappa": clean(a_wk)},
        "rubric_vs_human": {"accuracy": clean(r_acc), "weighted_kappa": clean(r_wk)},
        "adapter_vs_rubric": {"accuracy": clean(ar["acc"]), "weighted_kappa": clean(ar["wkappa"])},
        "unanimous_items": unanimous,
        "consensus_items": len(cons_items),
        "labels_anonymized": annots,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Annotators: %d | resumes labelled by all: %d | unanimous: %d" % (n_ann, n_items, unanimous))
    print("Annotator vs annotator: accuracy %s, weighted kappa %s (%d %s)" % (pc(h_acc), kf(h_wk), n_pairs, pair_word))
    print("Adapter vs annotator:   accuracy %s, weighted kappa %s" % (pc(a_acc), kf(a_wk)))
    print("Rubric vs annotator:    accuracy %s, weighted kappa %s" % (pc(r_acc), kf(r_wk)))
    print("Adapter vs rubric:      accuracy %s, weighted kappa %s" % (pc(ar["acc"]), kf(ar["wkappa"])))
    if args.readme:
        patch_readme(args.readme, block)
        print("Saved " + args.out)
    else:
        print("\nPreview of the README section (SELF-TEST numbers, not real):\n\n" + block)
        print("\nSelf-test finished; nothing was written to your repo.")


if __name__ == "__main__":
    main()