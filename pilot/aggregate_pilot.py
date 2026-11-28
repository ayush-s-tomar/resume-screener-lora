"""
Aggregate the pilot survey (questions in pilot/questions.json, answers in pilot/responses/*.json) into aggregate numbers only.
Free-text comments are never published. Results are written only with at least 10 valid responses.

Usage (from the repository root):
    python pilot/aggregate_pilot.py --check path/to/pilot_xxxx.json    # is a downloaded answers file valid?
    python pilot/aggregate_pilot.py                                     # print the summary of pilot/responses/
    python pilot/aggregate_pilot.py --write                             # also write results/pilot_v1.json, docs/PILOT_RESULTS.md and a README block
    python pilot/aggregate_pilot.py --selftest                          # try the pipeline on simulated answers; writes nothing
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

MIN_PUBLISH = 10
ORD = {"weak": 0, "moderate": 1, "strong": 2}
BAD_FEELINGS = ("a_bit_worried", "discouraged")
ANCHORS = ("<!-- us-end -->", "<!-- fb-end -->", "<!-- ds-end -->", "<!-- dm-end -->", "<!-- pk-end -->", "<!-- ru-end -->", "<!-- sp-end -->", "<!-- hw-end -->")


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pc(x):
    return "n/a" if x is None else "%.0f%%" % (x * 100)


def load_questions(root):
    data = json.loads((Path(root) / "pilot" / "questions.json").read_text(encoding="utf-8-sig"))
    return data["questions"]


def problems(r, qs):
    if not isinstance(r, dict):
        return ["not a JSON object"]
    out = []
    if r.get("consent") is not True:
        out.append("consent is not true")
    if not r.get("id"):
        out.append("no id")
    a = r.get("answers")
    if not isinstance(a, dict):
        return out + ["no answers"]
    for q in qs:
        v = a.get(q["id"])
        if v is None or v == "":
            if q.get("required"):
                out.append("missing answer: " + q["id"])
            continue
        t = q["type"]
        if t == "choice" and v not in [o[0] for o in q["options"]]:
            out.append("unknown option for %s: %r" % (q["id"], v))
        elif t == "scale" and (isinstance(v, bool) or not isinstance(v, int) or not q["min"] <= v <= q["max"]):
            out.append("bad rating for %s: %r" % (q["id"], v))
        elif t == "number" and (isinstance(v, bool) or not isinstance(v, (int, float)) or not q["min"] <= v <= q["max"]):
            out.append("bad number for %s: %r" % (q["id"], v))
        elif t == "text" and (not isinstance(v, str) or len(v) > q.get("max", 300)):
            out.append("bad text for " + q["id"])
    return out


def load_responses(folder, qs):
    good, bad, seen = [], [], set()
    for f in sorted(Path(folder).glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError:
            bad.append((f.name, ["not valid JSON"]))
            continue
        p = problems(r, qs)
        if p:
            bad.append((f.name, p))
        elif r["id"] in seen:
            bad.append((f.name, ["duplicate id"]))
        else:
            seen.add(r["id"])
            good.append(r)
    return good, bad


def summarize(rs, qs):
    s = {"n": len(rs), "questions": {}}
    for q in qs:
        vals = [r["answers"][q["id"]] for r in rs if r["answers"].get(q["id"]) not in (None, "")]
        info = {"type": q["type"], "n": len(vals)}
        if q["type"] == "choice":
            info["counts"] = dict(Counter(vals))
        elif q["type"] == "scale":
            agree = sum(v >= 4 for v in vals)
            lo, hi = wilson(agree, len(vals))
            info.update({
                "counts": {str(k): c for k, c in sorted(Counter(vals).items())},
                "mean": (sum(vals) / len(vals)) if vals else None, "median": statistics.median(vals) if vals else None,
                "agree": agree, "agree_share": (agree / len(vals)) if vals else None, "agree_lo": lo, "agree_hi": hi,
            })
        elif q["type"] == "number":
            info.update({"mean": (sum(vals) / len(vals)) if vals else None, "median": statistics.median(vals) if vals else None})
        s["questions"][q["id"]] = info
    pairs = [(r["answers"]["self_fit"], r["answers"]["verdict"]) for r in rs if r["answers"].get("self_fit") in ORD and r["answers"].get("verdict") in ORD]
    s["self_vs_model"] = {
        "pairs": len(pairs), "same": sum(a == b for a, b in pairs),
        "model_lower": sum(ORD[b] < ORD[a] for a, b in pairs), "model_higher": sum(ORD[b] > ORD[a] for a, b in pairs),
    }

    def agree_share(qid):
        info = s["questions"].get(qid) or {}
        return info.get("agree_share")

    comfort = s["questions"].get("comfort") or {}
    n_c = comfort.get("n", 0)
    bad_n = sum((comfort.get("counts") or {}).get(k, 0) for k in BAD_FEELINGS)
    s["uncomfortable"] = {"n": bad_n, "of": n_c}
    crit = [
        {"id": "l_again", "text": "At least 60% would use a tool like this again (rating 4 or 5)", "share": agree_share("l_again"), "threshold": 0.60, "direction": "at least"},
        {"id": "l_missing", "text": "At least 60% found the missing-skills list useful (rating 4 or 5)", "share": agree_share("l_missing"), "threshold": 0.60, "direction": "at least"},
        {"id": "comfort", "text": "At most 20% felt worried or discouraged", "share": (bad_n / n_c) if n_c else None, "threshold": 0.20, "direction": "at most"},
    ]
    for c in crit:
        if c["share"] is None:
            c["met"] = None
        else:
            c["met"] = (c["share"] >= c["threshold"]) if c["direction"] == "at least" else (c["share"] <= c["threshold"])
    s["criteria"] = crit
    s["useful"] = all(c["met"] for c in crit) if all(c["met"] is not None for c in crit) else None
    return s


def render_block(s):
    q = s["questions"]
    again, missing, sm, unc = q["l_again"], q["l_missing"], s["self_vs_model"], s["uncomfortable"]
    parts = ["In a small voluntary pilot (self-selected students, answers collected anonymously) they tried the demo."]
    parts.append("%d of %d (%s; 95%% interval %s to %s) said they would use a tool like this again before applying, and %d of %d rated the missing-skills list useful (4 or 5)." % (
        again["agree"], again["n"], pc(again["agree_share"]), pc(again["agree_lo"]), pc(again["agree_hi"]), missing["agree"], missing["n"]))
    if sm["pairs"]:
        parts.append("The model's verdict matched the student's own judgement, given before they saw the result, in %d of %d cases; it was lower in %d and higher in %d." % (
            sm["same"], sm["pairs"], sm["model_lower"], sm["model_higher"]))
    parts.append("%d of %d said the result left them worried or discouraged." % (unc["n"], unc["of"]))
    verdict = {True: "all met", False: "not all met", None: "not computable"}[s["useful"]]
    parts.append("The criteria fixed before the pilot (see the protocol) were %s." % verdict)
    parts.append("The sample is small and self-selected, and the comparison is the student's own expectation, not an expert's judgement.")
    return ("<!-- pl-start -->\n**Pilot with %d students.** " % s["n"] + " ".join(parts)
            + " Details: [`docs/PILOT_RESULTS.md`](docs/PILOT_RESULTS.md); protocol: [`pilot/PROTOCOL.md`](pilot/PROTOCOL.md).\n<!-- pl-end -->")


def render_docs(s, qs, date):
    byid = {q["id"]: q for q in qs}
    L = ["# Pilot results", "",
         "Computed by `pilot/aggregate_pilot.py` on %s from %d anonymous response files. Only aggregate numbers are published; free-text comments are not. "
         "The protocol, including the criteria below, was written before any data was collected: [PROTOCOL.md](../pilot/PROTOCOL.md)." % (date, s["n"]), ""]

    def options_line(qid):
        q, info = byid[qid], s["questions"][qid]
        return ", ".join("%s %d" % (o[1], info["counts"].get(o[0], 0)) for o in q["options"] if info["counts"].get(o[0], 0)) or "no answers"

    L += ["## Who took part", ""]
    for qid in ("year", "role_family", "resume_used", "input_method"):
        if qid in byid:
            L.append("- **%s** %s" % (byid[qid]["label"], options_line(qid)))
    L += ["", "## Ratings (1 = strongly disagree, 5 = strongly agree)", "", "| Statement | Answers | Mean | Median | Rated 4 or 5 (95% interval) |", "|---|---|---|---|---|"]
    for q in qs:
        if q["type"] == "scale":
            i = s["questions"][q["id"]]
            L.append("| %s | %d | %s | %s | %d (%s; %s to %s) |" % (q["label"], i["n"], "n/a" if i["mean"] is None else "%.1f" % i["mean"],
                                                                 "n/a" if i["median"] is None else "%g" % i["median"], i["agree"], pc(i["agree_share"]), pc(i["agree_lo"]), pc(i["agree_hi"])))
    sm = s["self_vs_model"]
    L += ["", "## The model's verdict against the student's own expectation", ""]
    if sm["pairs"]:
        L.append("Students said how well they thought the resume fits the role before they saw the result. In the %d cases where they gave a view, the model's verdict was the same in %d, lower in %d and higher in %d. "
                 "This is the student's own expectation, not an expert's judgement." % (sm["pairs"], sm["same"], sm["model_lower"], sm["model_higher"]))
    else:
        L.append("No student gave a view before seeing the result, so there is no comparison.")
    L += ["", "## How the result felt, and trust", ""]
    for qid in ("comfort", "trust_decisions"):
        if qid in byid:
            L.append("- **%s** %s" % (byid[qid]["label"], options_line(qid)))
    L += ["", "## Criteria fixed before the pilot", "", "| Criterion | Observed | Threshold | Met |", "|---|---|---|---|"]
    for c in s["criteria"]:
        L.append("| %s | %s | %s | %s |" % (c["text"], pc(c["share"]), pc(c["threshold"]), {True: "yes", False: "no", None: "n/a"}[c["met"]]))
    L += ["", "Overall: %s." % {True: "all criteria met", False: "not all criteria met", None: "not computable"}[s["useful"]], "",
          "## Limits", "",
          "- %d students, self-selected from my own contacts, so they are not a representative sample and may be kinder than strangers." % s["n"],
          "- The comparison with the model is the student's own expectation. It is not an expert's judgement and not a measure of accuracy.",
          "- The ratings are about the hosted demo, which scores with a larger model, not with the LoRA adapter.",
          "- Free-text comments (%d) are not published; I read them for problems the ratings miss." % s["questions"].get("comment", {}).get("n", 0), ""]
    return "\n".join(L)


def fake_responses(qs, n=14, seed=3):
    rng = random.Random(seed)
    out = []
    for k in range(n):
        a = {}
        for q in qs:
            if q["type"] == "choice":
                a[q["id"]] = rng.choice([o[0] for o in q["options"]])
            elif q["type"] in ("scale", "number"):
                a[q["id"]] = rng.randint(q["min"], q["max"])
        out.append({"version": 1, "id": "selftest%02d" % k, "created": "2026-01-01", "consent": True, "answers": a})
    return out


def patch_readme(root, block):
    rp = Path(root) / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- pl-start -->") + r".*?" + re.escape("<!-- pl-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
    else:
        anchor = next((a for a in ANCHORS if text.count(a) == 1), None)
        if not anchor:
            print("README NOT changed (no anchor found). Paste this where you want it:\n\n" + block)
            return
        text = text.replace(anchor, anchor + "\n\n" + block, 1)
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("README updated.")


def main():
    ap = argparse.ArgumentParser(description="Aggregate the pilot survey")
    ap.add_argument("--root", default=".")
    ap.add_argument("--responses")
    ap.add_argument("--check")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    qs = load_questions(root)

    if args.check:
        r = json.loads(Path(args.check).read_text(encoding="utf-8-sig"))
        p = problems(r, qs)
        if p:
            print("NOT VALID:\n  " + "\n  ".join(p))
            sys.exit(1)
        print("OK: a valid answers file (%d answers)" % len(r["answers"]))
        return

    bad = []
    if args.selftest:
        rs = fake_responses(qs)
        print("SELF-TEST: %d simulated responses. Nothing is written.\n" % len(rs))
    else:
        folder = Path(args.responses) if args.responses else root / "pilot" / "responses"
        rs, bad = load_responses(folder, qs) if folder.exists() else ([], [])
        print("Valid responses: %d in %s" % (len(rs), folder))
        for name, p in bad:
            print("  skipped %s: %s" % (name, "; ".join(p)))
    if not rs:
        print("No responses yet.")
        return
    s = summarize(rs, qs)
    print(json.dumps({k: v for k, v in s.items() if k != "questions"}, indent=2))
    print("\nREADME block:\n" + render_block(s))
    if args.selftest or not args.write:
        return
    if s["n"] < MIN_PUBLISH:
        raise SystemExit("\nOnly %d valid responses; results are written from %d. Keep collecting." % (s["n"], MIN_PUBLISH))
    (root / "results").mkdir(exist_ok=True)
    (root / "docs").mkdir(exist_ok=True)
    (root / "results" / "pilot_v1.json").write_text(json.dumps(dict(s, date=time.strftime("%Y-%m-%d")), indent=2), encoding="utf-8", newline="\n")
    (root / "docs" / "PILOT_RESULTS.md").write_text(render_docs(s, qs, time.strftime("%Y-%m-%d")), encoding="utf-8", newline="\n")
    print("\nWrote results/pilot_v1.json and docs/PILOT_RESULTS.md")
    patch_readme(root, render_block(s))


if __name__ == "__main__":
    main()
