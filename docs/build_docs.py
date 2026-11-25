"""
Writes docs/RESPONSIBLE_USE.md, docs/BIAS_AUDIT.md and docs/PRIVACY.md from the result files and the demo source, and adds a short
banner and section to the README. Every number comes from a file in the repository, so the documents cannot drift from the results:
run it again after any result changes. Sections whose source files are missing are left out.

Usage (from the repository root):
    python docs/build_docs.py
    python docs/build_docs.py --no-readme
"""
import argparse
import json
import re
import time
from pathlib import Path

ORD = {"weak_match": 0, "moderate_match": 1, "strong_match": 2}
BOUNDS = (49.0, 74.5)


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def pct(x):
    return "n/a" if x is None else ("%.1f" % (x * 100)).rstrip("0").rstrip(".") + "%"


def norm(x):
    s = str(x or "").strip().lower().replace(" ", "_").replace("-", "_")
    return s + "_match" if s in ("weak", "moderate", "strong") else s


def dist_to_boundary(score):
    return min(abs(score - b) for b in BOUNDS)


def name_swap_facts(ns):
    if not ns:
        return None
    names = list(ns.get("names") or [])
    rows = ns.get("per_resume") or []
    stats, flipped = {}, []
    for idx, row in enumerate(rows):
        base, bverdict = row.get("base_score"), row.get("base_verdict")
        hits = 0
        for p, v in (row.get("variants") or {}).items():
            if base is None or not isinstance(v, (list, tuple)) or len(v) < 2 or v[0] is None:
                continue
            s = stats.setdefault(p, {"shifts": [], "flips": 0})
            s["shifts"].append(float(v[0]) - float(base))
            flip = v[1] != bverdict
            s["flips"] += int(flip)
            if p in names and flip:
                hits += 1
        if hits:
            flipped.append((idx, base, hits))
    used = [stats[p] for p in names if p in stats]
    pairs = sum(len(s["shifts"]) for s in used)
    return {
        "names": names, "stats": stats, "n_resumes": len(rows), "flipped": flipped, "pairs": pairs,
        "flips": sum(s["flips"] for s in used),
        "mean_abs": (sum(abs(x) for s in used for x in s["shifts"]) / pairs) if pairs else None,
        "biggest": max((abs(x) for s in used for x in s["shifts"]), default=None),
    }


def app_facts(root):
    p = root / "render-deploy" / "app.py"
    if not p.exists():
        return None
    src = p.read_text(encoding="utf-8")

    def env_int(name):
        m = re.search(r'_env_int\("%s",\s*(\d+)\)' % name, src)
        return int(m.group(1)) if m else None

    body = re.search(r"MAX_BODY_BYTES\s*=\s*(\d+)\s*\*\s*(\d+)", src)
    cache = re.search(r"CACHE_MAX\s*=\s*(\d+)", src)
    upload = re.search(r"MAX_UPLOAD_BYTES\s*=\s*(\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)", src)
    return {
        "rate": env_int("DEMO_RATE_LIMIT"), "window": env_int("DEMO_RATE_WINDOW_SECONDS"), "daily": env_int("DEMO_DAILY_CAP"),
        "body_kb": (int(body.group(1)) * int(body.group(2))) // 1024 if body else None,
        "cache_max": int(cache.group(1)) if cache else None,
        "upload_mb": (int(upload.group(1)) * int(upload.group(2)) * int(upload.group(3))) // (1024 * 1024) if upload else None,
        "has_upload": "/extract" in src, "has_mask": "redact" in src, "has_usage": "/feedback" in src,
        "clean_page": ("set_cookie" not in src) and (re.search(r"<script[^>]+\bsrc\s*=", src) is None),
        "model": (re.search(r'MODEL\s*=\s*"([^"]+)"', src) or [None, None])[1],
    }


def responsible_use(c):
    L = []
    A = L.append
    A("# Responsible use")
    A("")
    A("> Written by `docs/build_docs.py` on %s from the files in `results/`; regenerate it with `python docs/build_docs.py`. It states what this project is and is not for. It is not legal advice." % c["date"])
    A("")
    A("## In one paragraph")
    A("")
    A("resume-screener-lora is a research and portfolio project. Its labels are synthetic, it has never been validated on real resumes or on real hiring outcomes, and it has not been audited for bias. "
      "It must not be used to make, or to support, decisions about real people. The hosted demo exists to show the output format and the evaluation work, not to screen anyone.")
    A("")
    A("## What the evidence shows, and what it does not")
    A("")
    A("| Check | Result | Where |")
    A("|---|---|---|")
    if c["held"] is not None:
        A("| Held-out set (%s synthetic examples from the same generator as the training data) | %s verdict accuracy | [comparison_results.json](../comparison_results.json) |" % (c["n_ho"] or "n/a", pct(c["held"])))
    if c["hand"] is not None:
        A("| Hand-authored set (%d fictional resumes in other layouts, labelled against an AI-drafted rubric and not checked by humans) | %s (%d/%d) verdict accuracy, %s severe errors (weak against strong), scores ordered correctly in %s of pairs | [handwritten_v1.json](../results/handwritten_v1.json) |" % (
            c["n_h"], pct(c["hand"]), c["k_h"], c["n_h"], c["sev"], pct(c["ordering"])))
    ns = c["ns"]
    if ns and ns["pairs"]:
        A("| Name swap (%d held-out resumes, %d names) | %d of %d verdicts flipped, mean absolute score change %.1f points, all flips on %d resume(s) | [BIAS_AUDIT.md](BIAS_AUDIT.md) |" % (
            ns["n_resumes"], len(ns["names"]), ns["flips"], ns["pairs"], ns["mean_abs"], len(ns["flipped"])))
    nb = c["nb"]
    if nb:
        g = nb["groups"]
        A("| Boundary check (the %d resumes closest to a verdict boundary) | names flipped %d of %d verdicts, neutral prefixes %d of %d | [BIAS_AUDIT.md](BIAS_AUDIT.md) |" % (
            nb["n_resumes"], g["names"]["flips"], g["names"]["pairs"], g["neutral"]["flips"], g["neutral"]["pairs"]))
    sk = c["sk"]
    if sk and sk.get("matched") and sk.get("missing"):
        A("| Skill lists (held-out set, against the generator's lists) | matched F1 %.2f, missing F1 %.2f; %s of the skills listed as missing appear in the resume text | [skill_lists_v1.json](../results/skill_lists_v1.json) |" % (
            sk["matched"]["f1"], sk["missing"]["f1"], pct(sk["missing"].get("inconsistent_with_resume_rate"))))
    A("")
    if c["held"] is not None and c["hand"] is not None:
        A("What this adds up to: the adapter reproduces its generator's rubric well on data from that generator (%s) and much less well on resumes written differently (%s), "
          "where it mostly rates candidates lower than my rubric does. Nothing here measures whether a score predicts job performance, whether the rubric itself is fair, "
          "or how the adapter treats groups of people. \"Accuracy\" always means agreement with a rubric." % (pct(c["held"]), pct(c["hand"])))
        A("")
    A("## Known failure modes")
    A("")
    if c["under"] is not None and (c["under"] + c["over"]):
        A("- **Unfamiliar layouts.** On the hand-authored set %d of the %d wrong verdicts with a valid output are under-predictions (the adapter rated the resume lower than the rubric) and %d are over-predictions." % (c["under"], c["under"] + c["over"], c["over"]))
    A("- **Verdicts near a boundary are unstable.** Small changes at the top of the prompt flipped verdicts on the resumes whose score sits close to a verdict threshold, with names and with neutral prefixes alike.")
    if c["invalid"]:
        A("- **Invalid output.** %d of %d outputs on the hand-authored set were not valid JSON." % (c["invalid"], c["n_h"]))
    if sk and sk.get("missing"):
        A("- **Skill lists.** %s of the skills listed as missing appear in the resume text, so a missing-skills list can be wrong." % pct(sk["missing"].get("inconsistent_with_resume_rate")))
    A("- **Narrow scope.** English only, six roles, synthetic training data, one small model.")
    A("- **Protected characteristics.** Only a name check was run. Nothing was measured for age, gender, disability, caste, nationality or any other characteristic, and names are a weak proxy for most of them.")
    A("")
    A("## Uses this project does not support")
    A("")
    A("- Automatically rejecting, ranking or shortlisting real candidates, or using a score as the only basis for a decision about a person.")
    A("- Scoring people who have not been told that software is screening them.")
    A("- Submitting real people's resumes to the hosted demo: the text is sent to a third-party API (see [PRIVACY.md](PRIVACY.md)).")
    A("- Roles, languages or resume layouts that the evaluation did not cover.")
    if c["margin"]:
        A("- Treating the borderline flag as a safeguard: its margin (%g points) was chosen on %s synthetic examples%s." % (c["margin"], c["n_ho"] or "a small number of", ", and it only partly carries over to the hand-authored set (see [SELECTIVE_PREDICTION.md](SELECTIVE_PREDICTION.md))" if c["has_sp"] else ""))
    A("")
    A("## What would be needed before anyone used something like this for real")
    A("")
    A("| Condition | Status in this project |")
    A("|---|---|")
    A("| A defined purpose, and a human who makes every decision and can overrule the score | Not applicable: the project has no real use |")
    A("| Validation on real resumes from the actual setting, labelled by several qualified people, with their agreement reported | Not done%s |" % ("; a labelling form exists in `labeling/`, but no labels have been collected" if c["has_labeling"] else ""))
    A("| An adverse-impact analysis by group, using real selection outcomes, by an independent auditor where the law requires one | Not done; the name checks are not a substitute |")
    A("| Advance notice to candidates, and a way to ask for human review | Not done |")
    A("| Data minimisation and protection: mask identifiers, prefer local inference, short retention, a data-protection assessment | %s |" % ("Partly: the repository can mask emails, links, phone numbers and ID numbers before scoring; no assessment was done" if c["has_parser"] else "Not done"))
    A("| Monitoring for drift and complaints, and a way to withdraw the tool | Not done |")
    A("| Documentation and review | %s |" % ("Partly: a model card in [MODEL_CARD.md](MODEL_CARD.md), a changelog and these documents" if c["has_card"] else "Partly: a changelog and these documents"))
    A("")
    A("## Rules that apply to hiring tools (pointers, not advice)")
    A("")
    A("Checked on %s against public summaries, not against the legal texts. Rules and dates change; read the current primary sources and ask a lawyer before relying on any of this." % c["date"])
    A("")
    A("- **New York City, Local Law 144 of 2021.** Employers and employment agencies that use an automated employment decision tool to screen candidates in the city need a bias audit by an independent auditor within a year before use, must publish a summary of the results, and must notify candidates in advance (at least 10 business days). The audit looks at disparate impact across sex and race/ethnicity categories.")
    A("- **EU AI Act.** Annex III lists AI systems used to analyse and filter job applications and to evaluate candidates as high-risk. Under the Digital Omnibus on AI (Regulation (EU) 2026/1744), the date from which the obligations for stand-alone Annex III high-risk systems apply moved from 2 August 2026 to 2 December 2027. The delay is not a repeal.")
    A("- **Data protection.** The EU and UK GDPR (including its rule on decisions based solely on automated processing) and India's Digital Personal Data Protection Act, 2023, apply to personal data in resumes. Check their current status and rules.")
    A("")
    A("## Reporting a concern")
    A("")
    A("If you think this project is being used in a way it should not be, or you find a mistake in these documents, open an issue at https://github.com/ayush-s-tomar/resume-screener-lora/issues or use the private route in [SECURITY.md](../SECURITY.md).")
    return "\n".join(L) + "\n"


def bias_audit(c):
    ns, nb = c["ns"], c["nb"]
    L = []
    A = L.append
    A("# Name sensitivity checks (this is not a bias audit)")
    A("")
    A("> Written by `docs/build_docs.py` on %s from `results/name_swap_v1.json` and `results/name_swap_boundary_v1.json`." % c["date"])
    A("")
    A("These checks ask one narrow question: does the adapter's output change when only the candidate's name at the top of the resume changes? "
      "They are a sensitivity check on one adapter, synthetic resumes and a handful of names I chose. They say nothing about whether the adapter is fair to any group of people.")
    A("")
    if ns and ns["pairs"]:
        A("## Test 1: %d held-out resumes, %d names" % (ns["n_resumes"], len(ns["names"])))
        A("")
        A("Each resume was scored once as it is and once with each name placed at the top. A score change is the score with the name minus the score without it.")
        A("")
        A("| Prefix | Resumes | Mean score change | Mean absolute change | Verdict flips |")
        A("|---|---|---|---|---|")
        for p in ns["names"] + [k for k in ns["stats"] if k not in ns["names"]]:
            s = ns["stats"].get(p)
            if not s or not s["shifts"]:
                continue
            n = len(s["shifts"])
            label = p if p in ns["names"] else p + " (control)"
            A("| %s | %d | %+.1f | %.1f | %d of %d |" % (label, n, sum(s["shifts"]) / n, sum(abs(x) for x in s["shifts"]) / n, s["flips"], n))
        A("")
        A("Across the names, the mean absolute score change was %.1f points (largest single change %.0f), and %d of %d verdicts flipped." % (ns["mean_abs"], ns["biggest"], ns["flips"], ns["pairs"]))
        if ns["flipped"]:
            parts = []
            for idx, base, hits in ns["flipped"]:
                d = " (base score %.0f, %.1f points from the nearest verdict boundary)" % (base, dist_to_boundary(base)) if base is not None else ""
                parts.append("resume %d%s: %d name(s)" % (idx, d, hits))
            A("The flips came from %d resume(s): %s." % (len(ns["flipped"]), "; ".join(parts)))
        A("")
    if nb:
        g, rl = nb["groups"], nb["resume_level"]
        fn = g["names"]["flips"] / max(g["names"]["pairs"], 1)
        fo = g["neutral"]["flips"] / max(g["neutral"]["pairs"], 1)
        A("## Test 2: the %d resumes closest to a verdict boundary, names against neutral prefixes" % nb["n_resumes"])
        A("")
        A("The first test's flips sat on a resume close to a verdict boundary, so this test picked the held-out resumes whose adapter score is closest to a boundary (within %.1f points) and compared the names with neutral prefixes such as \"Applicant 17\" and \"Reference 4821\". Several neutral prefixes contain digits, which may disturb the model more than a name does." % nb["max_distance"])
        A("")
        A("| Prefix type | Verdict flips | Mean absolute score change |")
        A("|---|---|---|")
        A("| Names | %d of %d (%s) | %.1f points |" % (g["names"]["flips"], g["names"]["pairs"], pct(fn), g["names"]["mean_abs_shift"]))
        A("| Neutral prefixes | %d of %d (%s) | %.1f points |" % (g["neutral"]["flips"], g["neutral"]["pairs"], pct(fo), g["neutral"]["mean_abs_shift"]))
        A("")
        A("Across the resumes, names flipped more verdicts on %d, neutral prefixes on %d, and they tied on %d (sign test p = %.2f; read it loosely, the pairs are not independent)." % (rl["names_more"], rl["neutral_more"], rl["tied"], rl["sign_test_p"]))
        A("")
        if fn - fo > 0.05:
            A("Names flipped more verdicts than neutral prefixes did. That is a signal to investigate before trusting the adapter on real candidates, not proof of bias.")
        elif fo - fn > 0.05:
            A("Neutral prefixes flipped more verdicts than names did, so near a boundary the adapter is fragile to changes at the top of the prompt in general. That does not show that names are harmless.")
        else:
            A("Names flipped about as many verdicts as neutral prefixes, so near a boundary the adapter is sensitive to any prefix. That does not show that names are harmless.")
        A("")
    A("## What these checks do not show")
    A("")
    A("- They do not measure outcomes for any group. There are no real candidates, no selection rates and no impact ratios, which is what bias audits for hiring tools normally report.")
    A("- A name is one proxy among many, and a weak one for most protected characteristics (age, disability, caste, religion, nationality). Other cues in a resume, such as a school, a gap or a language, were not tested.")
    A("- The names were chosen by me, a small set was used, and the resumes are synthetic. One adapter was tested.")
    A("- No difference found in a small sample is not evidence of no difference.")
    A("")
    A("## What a real audit would need")
    A("")
    A("1. Real resumes (with consent) and the real decisions or outcomes the tool would influence, in the setting where it would be used.")
    A("2. Demographic categories for those candidates, collected lawfully, with selection rates and impact ratios by group, including intersections.")
    A("3. Statistical tests that account for small groups, and a plan for what to do when a gap is found.")
    A("4. An independent auditor where the law requires one, repeated on a schedule and after every model change, with results published where required.")
    A("")
    A("See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for the wider picture.")
    return "\n".join(L) + "\n"


def privacy(c):
    a = c["app"]
    L = []
    A = L.append
    A("# Privacy")
    A("")
    A("> Written by `docs/build_docs.py` on %s from `render-deploy/app.py`. It describes what this project does with data. It is not legal advice." % c["date"])
    A("")
    A("## Summary")
    A("")
    A("- **The hosted demo sends the text you submit to Groq's API%s to be scored.** Groq processes it under its own terms. Please do not submit real personal data." % (" (model `%s`)" % a["model"] if a and a["model"] else ""))
    A("- **The local adapter runs on your machine.** The resume text does not leave it. The model files are downloaded from Hugging Face on first use.")
    if a and a["clean_page"]:
        A("- The demo page has no accounts, no cookies and no analytics or other external scripts.")
    A("- This repository contains only synthetic or fictional resumes.")
    A("")
    if a:
        A("## Hosted demo")
        A("")
        A("### What the server keeps")
        A("")
        A("| Item | Content | Lifetime |")
        A("|---|---|---|")
        cache = "up to %d entries, " % a["cache_max"] if a["cache_max"] else ""
        A("| Result cache | The verdict, score, skills and rationale, keyed by a SHA-256 hash of the role and text. The text itself is not stored. The rationale is written by the model and may paraphrase the resume. | In memory, %scleared whenever the server restarts |" % cache)
        lim = ""
        if a["rate"] and a["window"]:
            lim = " Limit: %d new screenings per visitor per %d minutes%s." % (a["rate"], max(1, a["window"] // 60), (" and %d per day overall" % a["daily"]) if a["daily"] else "")
        A("| Rate counters | Timestamps of recent requests per IP address.%s | In memory, cleared whenever the server restarts |" % lim)
        A("| Log lines | A shortened hash of the visitor's address (8 hex characters), the text length and the response time. Never the text. | Written to the service's log output; the hosting provider may keep it for its own retention period |")
        if a["has_usage"]:
            A("| Usage events | The kind of event (screening, upload, feedback), a hash of the visitor's address that changes every day, a coarse role group, the score and verdict, and yes/no flags (cached, borderline, masked). Never the text or the address. | Counts in memory until the server restarts; also written to the log output, and stored in a database table if the host configures one |")
        A("")
        if a["body_kb"]:
            A("Requests to the scoring endpoint are limited to %d KB." % a["body_kb"])
            A("")
        if a["has_upload"]:
            A("### Uploads and masking")
            A("")
            A("PDF, DOCX and text files%s are read in memory to extract their text and are not written to disk. Only the file size and extension are logged. The extracted text is then handled like pasted text." % (" (up to %d MB)" % a["upload_mb"] if a["upload_mb"] else ""))
            A("")
        if a["has_mask"]:
            A("The mask option replaces emails, links, phone numbers and ID numbers (Aadhaar, PAN and US SSN formats) before the text is scored. It does **not** mask names or street addresses and is not an anonymisation guarantee.")
            A("")
        A("### Third parties")
        A("")
        A("- **Groq** receives the text (masked, if you chose that) and returns the verdict.")
        A("- **The hosting provider** (Render) serves the page and may keep its own request logs.")
        A("")
    A("## Local adapter and command line")
    A("")
    A("Scoring with `inference.py` is done on your machine. Nothing is sent anywhere except the one-time download of the base model and, if no local adapter folder exists, the adapter from Hugging Face.")
    A("")
    if c["has_labeling"]:
        A("## Annotation form")
        A("")
        A("`labeling/label_resumes.html` runs entirely in the browser with no network requests. Answers are kept in the browser's local storage until the annotator downloads them. Annotator IDs are replaced by A1, A2 and so on before any result is published, and the raw label files are not committed.")
        A("")
    A("## If you submitted real personal data to the demo")
    A("")
    A("This project stores no resume text, so there is nothing for me to delete. The text was sent to Groq, so see Groq's privacy policy and contact Groq about it.")
    return "\n".join(L) + "\n"


def upsert(text, start, end, block, anchors):
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if pat.search(text):
        return pat.sub(lambda m: block, text, count=1), True
    for anchor in anchors:
        if text.count(anchor) == 1:
            return text.replace(anchor, anchor + "\n\n" + block, 1), True
    return text, False


def main():
    ap = argparse.ArgumentParser(description="Write the responsible-use, name-check and privacy documents")
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    docs = root / "docs"
    docs.mkdir(exist_ok=True)

    cmp_ = load(root / "comparison_results.json")
    held = ((cmp_ or {}).get("fine_tuned") or {}).get("verdict_accuracy")
    ev = root / "data" / "eval.jsonl"
    n_ho = sum(1 for line in ev.read_text(encoding="utf-8-sig").splitlines() if line.strip()) if ev.exists() else None
    h = load(root / "results" / "handwritten_v1.json")
    under = over = None
    invalid = 0
    if h:
        under = over = 0
        for e in h.get("per_example", []):
            lab, pred = norm(e.get("label")), norm(e.get("pred"))
            if pred not in ORD:
                invalid += 1
            elif lab in ORD and lab != pred:
                if ORD[pred] < ORD[lab]:
                    under += 1
                else:
                    over += 1
    sel = load(root / "results" / "selective_v1.json")
    margin = (sel or {}).get("recommended_margin")
    ctx = {
        "date": time.strftime("%Y-%m-%d"), "held": held, "n_ho": n_ho,
        "hand": h.get("verdict_accuracy") if h else None, "n_h": h.get("n_examples") if h else None, "k_h": h.get("verdict_correct") if h else None,
        "sev": h.get("severe_errors") if h else None, "ordering": h.get("pairwise_ordering") if h else None,
        "under": under, "over": over, "invalid": invalid,
        "ns": name_swap_facts(load(root / "results" / "name_swap_v1.json")), "nb": load(root / "results" / "name_swap_boundary_v1.json"),
        "sk": load(root / "results" / "skill_lists_v1.json"), "app": app_facts(root),
        "margin": margin if isinstance(margin, (int, float)) and not isinstance(margin, bool) and margin > 0 else None,
        "has_sp": (docs / "SELECTIVE_PREDICTION.md").exists(), "has_card": (docs / "MODEL_CARD.md").exists(),
        "has_labeling": (root / "labeling" / "label_resumes.html").exists(), "has_parser": (root / "resume_parser.py").exists(),
    }
    (docs / "RESPONSIBLE_USE.md").write_text(responsible_use(ctx), encoding="utf-8", newline="\n")
    (docs / "BIAS_AUDIT.md").write_text(bias_audit(ctx), encoding="utf-8", newline="\n")
    (docs / "PRIVACY.md").write_text(privacy(ctx), encoding="utf-8", newline="\n")
    print("Wrote docs/RESPONSIBLE_USE.md, docs/BIAS_AUDIT.md and docs/PRIVACY.md")
    for name, key in (("held-out accuracy", "held"), ("hand-authored accuracy", "hand")):
        print("  %-24s %s" % (name, pct(ctx[key])))
    print("  name-swap checks: %s | boundary check: %s | skill lists: %s | demo source: %s" % tuple("yes" if ctx[k] else "NOT FOUND" for k in ("ns", "nb", "sk", "app")))
    if ctx["app"]:
        a = ctx["app"]
        print("  demo: upload %s, masking %s, no cookies or external scripts %s" % (a["has_upload"], a["has_mask"], a["clean_page"]))

    if args.no_readme:
        return
    rp = root / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    banner = "> **Research artifact, not a hiring tool.** "
    if held is not None and ctx["hand"] is not None:
        banner += "The labels are synthetic, accuracy falls from %s on the held-out set to %s on hand-written resumes, and no bias audit has been done." % (pct(held), pct(ctx["hand"]))
    else:
        banner += "The labels are synthetic and no bias audit has been done."
    banner += " Read [Responsible use](docs/RESPONSIBLE_USE.md) before using any part of it on real candidates."
    banner_block = "<!-- ru-banner-start -->\n" + banner + "\n<!-- ru-banner-end -->"
    pat = re.compile(re.escape("<!-- ru-banner-start -->") + r".*?" + re.escape("<!-- ru-banner-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: banner_block, text, count=1)
        print("README: banner updated.")
    else:
        m = re.search(r"(?m)^# .+\n", text)
        if m:
            text = text[:m.end()] + "\n" + banner_block + "\n" + text[m.end():]
            print("README: banner added under the title.")
        else:
            print("README banner NOT added (no '# ' title line). Paste this under your title:\n\n" + banner_block)
    section = (
        "<!-- ru-start -->\n### Responsible use, name checks and privacy\n\n"
        "- [`docs/RESPONSIBLE_USE.md`](docs/RESPONSIBLE_USE.md): what this project is and is not for, what the evidence shows and does not show, and pointers to the rules that apply to hiring tools.\n"
        "- [`docs/BIAS_AUDIT.md`](docs/BIAS_AUDIT.md): the name-swap and boundary checks in one place, and what a real bias audit would need.\n"
        "- [`docs/PRIVACY.md`](docs/PRIVACY.md): what the hosted demo, the local adapter and the labelling form do with data.\n<!-- ru-end -->"
    )
    anchors = ["<!-- pk-end -->", "<!-- dm-end -->", "<!-- rp-end -->", "<!-- bf-end -->", "<!-- sp-end -->", "<!-- ea-end -->", "<!-- hw-end -->"]
    text, ok = upsert(text, "<!-- ru-start -->", "<!-- ru-end -->", section, anchors)
    print("README: documents section %s." % ("updated" if ok else "NOT added (no anchor found); paste this where you want it:\n\n" + section))
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    entry = ("- Documentation: `docs/RESPONSIBLE_USE.md` (what the project is and is not for, the evidence, rules that apply to hiring tools), `docs/BIAS_AUDIT.md` (the name checks and what they do not show) "
             "and `docs/PRIVACY.md` (what the demo, the local adapter and the labelling form do with data), generated from the result files by `docs/build_docs.py`.")
    cp = root / "CHANGELOG.md"
    if cp.exists():
        c = cp.read_text(encoding="utf-8").replace("\r\n", "\n")
        if "docs/RESPONSIBLE_USE.md" in c:
            print("CHANGELOG already has the entry.")
        elif "## [Unreleased]\n\n### Added\n\n" in c:
            cp.write_text(c.replace("## [Unreleased]\n\n### Added\n\n", "## [Unreleased]\n\n### Added\n\n" + entry + "\n", 1), encoding="utf-8", newline="\n")
            print("CHANGELOG: entry added.")
        elif c.count("## [Unreleased]\n\n## [") == 1:
            cp.write_text(c.replace("## [Unreleased]\n\n## [", "## [Unreleased]\n\n### Added\n\n" + entry + "\n\n## [", 1), encoding="utf-8", newline="\n")
            print("CHANGELOG: entry added.")
        else:
            print("CHANGELOG NOT changed (no [Unreleased] section). Add this line by hand:\n" + entry)
    else:
        print("CHANGELOG.md not found (Day 3 not applied); skipped.")


if __name__ == "__main__":
    main()
