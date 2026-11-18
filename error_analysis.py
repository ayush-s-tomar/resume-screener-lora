"""
Error analysis of the adapter, computed only from the saved result files (no model run).

Reads results/handwritten_v1.json, data/handwritten_eval.jsonl, results/heldout_v1.json and data/eval.jsonl, and writes
docs/ERROR_ANALYSIS.md, results/error_analysis_v1.json and a short summary in the README.

Usage:
    python error_analysis.py
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

ORD = {"weak_match": 0, "moderate_match": 1, "strong_match": 2}
NAMES = ["weak_match", "moderate_match", "strong_match"]
SHORT = {"weak_match": "weak", "moderate_match": "moderate", "strong_match": "strong"}
BOUNDS = (49.0, 74.5)  # weak up to 48, moderate 50 to 74, strong 75 and above in the training data
NEAR = 6.0
LABEL_KEYS = ("label", "truth", "gold", "expected", "true_verdict", "expected_verdict")
PRED_KEYS = ("pred", "prediction", "pred_verdict", "predicted", "verdict")
SCORE_KEYS = ("pred_score", "predicted_score", "score", "pred_ats_score")
TRUE_KEYS = ("true_score", "gold_score", "expected_score", "label_score")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def pick(d, keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def num(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def norm_label(x):
    if x is None:
        return None
    s = str(x).strip().lower().replace(" ", "_").replace("-", "_")
    return s + "_match" if s in ("weak", "moderate", "strong") else s


def verdict_from_score(s):
    return "weak_match" if s < BOUNDS[0] else ("moderate_match" if s < BOUNDS[1] else "strong_match")


def per_example(res, name):
    for k in ("per_example", "examples", "results", "details"):
        if isinstance(res.get(k), list):
            return res[k]
    raise SystemExit("%s: could not find the per-example list. Top-level keys: %s" % (name, ", ".join(res.keys())))


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (x * 100)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dist_to_boundary(s):
    return min(abs(s - b) for b in BOUNDS)


def get_handwritten(root):
    res = load_json(root / "results" / "handwritten_v1.json")
    data = {r["id"]: r for r in load_jsonl(root / "data" / "handwritten_eval.jsonl")}
    rows = []
    for e in per_example(res, "handwritten_v1.json"):
        d = data.get(e.get("id"), {})
        rows.append({
            "id": e.get("id"),
            "role": e.get("role") or d.get("role") or "unknown",
            "label": norm_label(pick(e, LABEL_KEYS) or d.get("label")),
            "pred": norm_label(pick(e, PRED_KEYS)),
            "score": num(pick(e, SCORE_KEYS)),
            "resume": d.get("resume", ""),
            "meta": d,
        })
    return rows, data


def get_heldout(root):
    res = load_json(root / "results" / "heldout_v1.json")
    pe = per_example(res, "heldout_v1.json")
    data = load_jsonl(root / "data" / "eval.jsonl")
    rows = []
    for i, e in enumerate(pe):
        comp = {}
        if len(data) == len(pe):
            try:
                comp = json.loads(data[i]["completion"])
            except (KeyError, ValueError, TypeError):
                comp = {}
        true_score = num(pick(e, TRUE_KEYS))
        if true_score is None:
            true_score = num(comp.get("ats_score"))
        label = norm_label(pick(e, LABEL_KEYS) or comp.get("verdict"))
        if label not in ORD and true_score is not None:
            label = verdict_from_score(true_score)
        rows.append({"i": i, "label": label, "pred": norm_label(pick(e, PRED_KEYS)), "score": num(pick(e, SCORE_KEYS)), "true_score": true_score})
    return rows


def acc_of(rows):
    return sum(r["label"] == r["pred"] for r in rows) / len(rows) if rows else None


def confusion(rows):
    c = {t: {p: 0 for p in NAMES + ["invalid"]} for t in NAMES}
    for r in rows:
        if r["label"] in c:
            c[r["label"]][r["pred"] if r["pred"] in ORD else "invalid"] += 1
    return c


def band(s, t1, t2):
    return 0 if s < t1 else (1 if s < t2 else 2)


def fit_thresholds(pairs):
    scores = sorted({s for s, _ in pairs})
    pts = [-1e9] + [(a + b) / 2.0 for a, b in zip(scores, scores[1:])] + [1e9]
    best_key, best = None, None
    for i, t1 in enumerate(pts):
        for t2 in pts[i:]:
            hits = sum(band(s, t1, t2) == y for s, y in pairs)
            dist = min(abs(t1 - BOUNDS[0]), 1000) + min(abs(t2 - BOUNDS[1]), 1000)
            key = (-hits, dist)
            if best_key is None or key < best_key:
                best_key, best = key, (t1, t2, hits)
    return best


def loocv(pairs):
    hits = 0
    for i in range(len(pairs)):
        t1, t2, _ = fit_thresholds(pairs[:i] + pairs[i + 1:])
        s, y = pairs[i]
        hits += int(band(s, t1, t2) == y)
    return hits / len(pairs)


def fmt_t(t):
    return "none" if abs(t) > 1e6 else "%.1f" % t


def main():
    ap = argparse.ArgumentParser(description="Error analysis from saved results")
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    ha, ha_data = get_handwritten(root)
    ho = get_heldout(root)
    print("Schema check: hand-authored rows %d, held-out rows %d" % (len(ha), len(ho)))
    print("  hand-authored data fields: " + ", ".join(sorted(next(iter(ha_data.values())).keys())))
    for name, rows in (("hand-authored", ha), ("held-out", ho)):
        bad = [r for r in rows if r["label"] not in ORD]
        if bad:
            raise SystemExit("%s: %d rows have an unrecognised label (first: %r). Paste me this message." % (name, len(bad), bad[0]["label"]))

    out = {"date": time.strftime("%Y-%m-%d")}
    md = ["# Error analysis", "",
          "Computed by `error_analysis.py` on %s from the saved result files (`results/heldout_v1.json`, `results/handwritten_v1.json`) and the data files, with no model run. "
          "Every number can be recomputed with `python error_analysis.py`. The hand-authored labels were drafted with an AI assistant and have not been checked by humans, "
          "and the held-out labels come from the generator, so both describe agreement with a rubric, not with human judgment." % out["date"], ""]

    # ---------- held-out ----------
    n_ho = len(ho)
    ho_err = [r for r in ho if r["label"] != r["pred"]]
    out["heldout"] = {"n": n_ho, "errors": len(ho_err), "accuracy": acc_of(ho)}
    md += ["## 1. Held-out set (%d synthetic examples)" % n_ho, "",
           "The adapter got %d right (%s) and %d wrong." % (n_ho - len(ho_err), pct(acc_of(ho)), len(ho_err)), ""]
    have_true = all(r["true_score"] is not None for r in ho)
    if ho_err:
        rows_md = []
        for r in sorted(ho_err, key=lambda r: r["i"]):
            d = dist_to_boundary(r["true_score"]) if r["true_score"] is not None else None
            rows_md.append([r["i"], "n/a" if r["true_score"] is None else "%.0f" % r["true_score"], "n/a" if r["score"] is None else "%.0f" % r["score"],
                            SHORT.get(r["label"], r["label"]), SHORT.get(r["pred"], r["pred"]), "n/a" if d is None else "%.1f" % d])
        md += [md_table(["Example", "Rubric score", "Adapter score", "Rubric verdict", "Adapter verdict", "Rubric score's distance to nearest boundary"], rows_md), ""]
    ho_near_share = ho_err_near = None
    if have_true:
        near_all = sum(dist_to_boundary(r["true_score"]) <= NEAR for r in ho)
        ho_err_near = sum(dist_to_boundary(r["true_score"]) <= NEAR for r in ho_err)
        ho_near_share = near_all / n_ho
        two_band = sum(1 for r in ho_err if r["pred"] in ORD and abs(ORD[r["pred"]] - ORD[r["label"]]) == 2)
        out["heldout"].update({"near_boundary_all": near_all, "near_boundary_errors": ho_err_near, "two_band_errors": two_band})
        md += ["Within %d score points of a verdict boundary: **%d of the %d errors** (%s), against %d of all %d examples (%s). Errors that skip a whole band (weak against strong): %d." % (
            NEAR, ho_err_near, len(ho_err), pct(ho_err_near / len(ho_err)) if ho_err else "n/a", near_all, n_ho, pct(near_all / n_ho), two_band), ""]
    else:
        md += ["The rubric scores are not available for every held-out example, so the distance to a boundary was not computed.", ""]
    scored = [r for r in ho if r["score"] is not None and r["true_score"] is not None]
    if scored:
        ok = [abs(r["score"] - r["true_score"]) for r in scored if r["label"] == r["pred"]]
        bad = [abs(r["score"] - r["true_score"]) for r in scored if r["label"] != r["pred"]]
        if ok and bad:
            md += ["Mean absolute score error: %.1f on the %d correct verdicts and %.1f on the %d wrong ones." % (sum(ok) / len(ok), len(ok), sum(bad) / len(bad), len(bad)), ""]

    # ---------- hand-authored ----------
    n_ha = len(ha)
    valid = [r for r in ha if r["pred"] in ORD]
    invalid = n_ha - len(valid)
    conf = confusion(ha)
    md += ["## 2. Hand-authored set (%d resumes)" % n_ha, "",
           "Verdict accuracy %s (%d of %d); %d output%s were not valid JSON and count as wrong." % (
               pct(acc_of(ha)), sum(r["label"] == r["pred"] for r in ha), n_ha, invalid, "" if invalid == 1 else "s"), "",
           "### Confusion matrix (rows: rubric label, columns: adapter verdict)", "",
           md_table(["", "weak", "moderate", "strong", "invalid"], [[SHORT[t]] + [conf[t][p] for p in NAMES + ["invalid"]] for t in NAMES]), "",
           "### Per class", ""]
    cls_rows = []
    for t in NAMES:
        tot, tp = sum(conf[t].values()), conf[t][t]
        predicted = sum(conf[x][t] for x in NAMES)
        cls_rows.append([SHORT[t], tot, "%d/%d (%s)" % (tp, tot, pct(tp / tot if tot else None)), "%d/%d (%s)" % (tp, predicted, pct(tp / predicted if predicted else None))])
    md += [md_table(["Rubric class", "Resumes", "Recall", "Precision"], cls_rows), ""]

    direction = Counter()
    for r in valid:
        direction[ORD[r["pred"]] - ORD[r["label"]]] += 1
    under, over = direction[-1] + direction[-2], direction[1] + direction[2]
    errs_valid = under + over
    out["handwritten"] = {"n": n_ha, "accuracy": acc_of(ha), "invalid": invalid, "under_predictions": under, "over_predictions": over,
                          "off_by_two": direction[-2] + direction[2], "confusion": conf}
    md += ["### Direction of the errors", "",
           md_table(["Adapter vs rubric", "Resumes"], [["two bands lower", direction[-2]], ["one band lower", direction[-1]], ["same", direction[0]], ["one band higher", direction[1]], ["two bands higher", direction[2]]]), "",
           "Of the %d wrong verdicts with a valid output, **%d are under-predictions** and %d over-predictions." % (errs_valid, under, over), ""]

    roles = {}
    for r in ha:
        roles.setdefault(r["role"], []).append(r)
    role_rows = [[role, len(rs), "%d (%s)" % (sum(x["label"] == x["pred"] for x in rs), pct(acc_of(rs)))] for role, rs in sorted(roles.items())]
    out["handwritten"]["by_role"] = {role: acc_of(rs) for role, rs in roles.items()}
    md += ["### By role", "", md_table(["Role", "Resumes", "Correct"], role_rows), ""]

    skip = {"id", "role", "label", "resume", "prompt", "completion"}
    meta_keys = sorted({k for r in ha for k in r["meta"] if k not in skip})
    meta_md = []
    for k in meta_keys:
        vals = [r["meta"].get(k) for r in ha]
        if any(v is None or isinstance(v, (dict, list, float)) for v in vals) or any(isinstance(v, str) and len(v) > 40 for v in vals):
            continue
        uniq = sorted({str(v) for v in vals})
        if not 2 <= len(uniq) <= 6:
            continue
        meta_md += [md_table([k, "Resumes", "Correct"], [[u, sum(str(r["meta"].get(k)) == u for r in ha), "%d (%s)" % (
            sum(str(r["meta"].get(k)) == u and r["label"] == r["pred"] for r in ha), pct(acc_of([r for r in ha if str(r["meta"].get(k)) == u])))] for u in uniq]), ""]
    if meta_md:
        md += ["### By other fields in the data file (exploratory)", ""] + meta_md
    if all(r["resume"] for r in ha):
        srt = sorted(ha, key=lambda r: len(r["resume"].split()))
        third = len(srt) // 3
        parts = [srt[:third], srt[third:2 * third], srt[2 * third:]]
        len_rows = [[nm, "%d to %d words" % (len(g[0]["resume"].split()), len(g[-1]["resume"].split())), "%d (%s)" % (sum(x["label"] == x["pred"] for x in g), pct(acc_of(g)))]
                    for nm, g in zip(["shortest third", "middle third", "longest third"], parts)]
        md += ["### By resume length (exploratory)", "", md_table(["Group", "Length", "Correct"], len_rows), ""]

    sc_rows = []
    for t in NAMES:
        s = [r["score"] for r in ha if r["label"] == t and r["score"] is not None]
        if s:
            sc_rows.append([SHORT[t], len(s), "%.1f" % (sum(s) / len(s)), "%.0f to %.0f" % (min(s), max(s))])
    md += ["### Adapter score by rubric label", "", md_table(["Rubric label", "Scored resumes", "Mean adapter score", "Range"], sc_rows), ""]

    # ---------- thresholds ----------
    pairs = [(r["score"], ORD[r["label"]]) for r in ha if r["score"] is not None and r["label"] in ORD]
    thr = None
    if len(pairs) >= 20:
        default_acc = sum(band(s, BOUNDS[0], BOUNDS[1]) == y for s, y in pairs) / len(pairs)
        t1, t2, hits = fit_thresholds(pairs)
        thr = {"n": len(pairs), "default": default_acc, "in_sample": hits / len(pairs), "loocv": loocv(pairs), "t1": t1, "t2": t2}
        out["thresholds"] = thr
        md += ["## 3. Would moving the score thresholds fix it?", "",
               "The adapter's score is the number behind the verdict, so one cheap question is whether the verdict gap is mostly a shifted threshold. "
               "On the %d resumes with a score I searched for the two thresholds (weak/moderate and moderate/strong) that give the highest accuracy, and also checked them with leave-one-out validation, "
               "which refits the thresholds without each resume in turn and tests on that resume." % len(pairs), "",
               md_table(["Thresholds", "Accuracy on the %d scored resumes" % len(pairs)], [
                   ["Training bands (%.0f and %.1f)" % (BOUNDS[0], BOUNDS[1]), pct(default_acc)],
                   ["Best on these resumes (weak below %s, strong from %s), in-sample" % (fmt_t(t1), fmt_t(t2)), pct(hits / len(pairs))],
                   ["Same search under leave-one-out validation", pct(thr["loocv"])]]), "",
               "With %d resumes the in-sample figure is optimistic by construction; the leave-one-out figure is the fairer estimate, and it is itself noisy." % len(pairs), ""]

    # ---------- reading ----------
    md += ["## 4. What this suggests", ""]
    if ho_near_share is not None and ho_err:
        md += ["- **Held-out errors:** %d of %d sit within %d points of a verdict boundary, against %s of all examples, which fits the boundary fragility seen in the name-swap boundary test." % (
            ho_err_near, len(ho_err), NEAR, pct(ho_near_share))]
    if errs_valid:
        md += ["- **Hand-authored errors:** %d of %d are under-predictions, so on differently written resumes the adapter mostly rates the candidate lower than the rubric does, not randomly." % (under, errs_valid)]
    if thr:
        gain = thr["loocv"] - thr["default"]
        md += ["- **Thresholds:** leave-one-out accuracy moves from %s to %s. %s" % (pct(thr["default"]), pct(thr["loocv"]), (
            "That is a real gain even under validation, which points more to a shifted threshold than to a ranking problem, but it needs fresh data to confirm."
            if gain >= 0.10 else "That is not a trustworthy gain, so recalibrating the thresholds alone would not fix the verdict gap."))]
    md += ["", "## 5. What this cannot tell", "",
           "- It cannot say which part of a resume caused an error; that needs interpretability work or controlled edits of the resumes.",
           "- The hand-authored groups have 6 to 12 resumes, so one resume moves a group's accuracy by 8 to 17 points. Treat the breakdowns as leads, not findings.",
           "- Both label sets are rubric labels. Independent human labels (see `labeling/`) are needed to tell an adapter problem from a rubric problem.", ""]

    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "ERROR_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    (root / "results" / "error_analysis_v1.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8", newline="\n")

    # ---------- README summary ----------
    sents = []
    if ho_near_share is not None and ho_err:
        sents.append("Of the %d held-out errors, %d sit within %d score points of a verdict boundary, although only %s of all held-out examples do." % (len(ho_err), ho_err_near, NEAR, pct(ho_near_share)))
    if errs_valid:
        sents.append("On the hand-authored set %d of the %d wrong verdicts with a valid output are under-predictions (the adapter rated the resume lower than the rubric) and %s." % (under, errs_valid, ("1 is an over-prediction" if over == 1 else "%d are over-predictions" % over)))
    if thr:
        if thr["loocv"] - thr["default"] >= 0.10:
            sents.append("Re-fitting the two score thresholds on those resumes would raise accuracy from %s to %s even under leave-one-out validation, which points to a shifted threshold more than a ranking problem, but it is %d resumes and needs fresh data." % (
                pct(thr["default"]), pct(thr["loocv"]), thr["n"]))
        else:
            sents.append("Re-fitting the two score thresholds on the %d scored resumes raises accuracy from %s to %s in-sample but to only %s under leave-one-out validation, so recalibration alone would not be a trustworthy fix." % (
                thr["n"], pct(thr["default"]), pct(thr["in_sample"]), pct(thr["loocv"])))
    block = "<!-- ea-start -->\n**Error analysis.** " + " ".join(sents) + " Details and caveats: [`docs/ERROR_ANALYSIS.md`](docs/ERROR_ANALYSIS.md); reproduce with `python error_analysis.py`.\n<!-- ea-end -->"
    print("Wrote docs/ERROR_ANALYSIS.md and results/error_analysis_v1.json")
    for s in sents:
        print("  - " + s)
    if args.no_readme:
        return
    rp = root / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- ea-start -->") + r".*?" + re.escape("<!-- ea-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
    elif text.count("<!-- hw-end -->") == 1:
        text = text.replace("<!-- hw-end -->", "<!-- hw-end -->\n\n" + block, 1)
    else:
        print("README NOT changed ('<!-- hw-end -->' found %d times). Paste this where you want it:\n\n%s" % (text.count("<!-- hw-end -->"), block))
        return
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("README updated.")


if __name__ == "__main__":
    main()