"""
Writes docs/DATASET_CARD.md for the synthetic and hand-authored data, and can stage the files for the Hugging Face Hub.
Counts and label distributions are read from the data files, so the card cannot drift from them.

Usage (from the repository root):
    python docs/build_dataset_card.py                  # write docs/DATASET_CARD.md and print a personal-data scan
    python docs/build_dataset_card.py --stage DIR      # also copy the data files and the card (with front matter) into DIR
    python docs/build_dataset_card.py --finalize       # after publishing: link the dataset from the README, CHANGELOG and model card
"""
import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

REPO_ID = "Kus-hal/resume-screener-synthetic"
HF_URL = "https://huggingface.co/datasets/" + REPO_ID
GITHUB = "https://github.com/ayush-s-tomar/resume-screener-lora"
ORDER = ("weak_match", "moderate_match", "strong_match")
FENCE = "`" * 3
ANCHORS = ("<!-- fb-end -->", "<!-- dm-end -->", "<!-- pk-end -->", "<!-- ru-end -->", "<!-- rp-end -->", "<!-- bf-end -->", "<!-- sp-end -->", "<!-- ea-end -->", "<!-- hw-end -->")
OLD_SENTENCE = "The data is not published yet."
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
URL = re.compile(r"(?i)\bhttps?://\S+|\bwww\.\S+|\b(?:linkedin|github)\.com/\S+")
PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t().\-]{8,}\d(?![\w])")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def completion(row):
    c = row.get("completion")
    if isinstance(c, str):
        try:
            c = json.loads(c)
        except ValueError:
            return {}
    return c if isinstance(c, dict) else {}


def role_of(row):
    m = re.search(r"for an? (.+?) position", str(row.get("prompt", "")))
    return m.group(1).strip() if m else None


def summarize(rows):
    comps = [completion(r) for r in rows]
    verdicts = Counter(c["verdict"] for c in comps if c.get("verdict"))
    ranges = {}
    for c in comps:
        v, s = c.get("verdict"), c.get("ats_score")
        if v and isinstance(s, (int, float)) and not isinstance(s, bool):
            lo, hi = ranges.get(v, (s, s))
            ranges[v] = (min(lo, s), max(hi, s))
    keys = sorted(set().union(*[set(c) for c in comps])) if comps else []
    roles = Counter(r for r in (role_of(row) for row in rows) if r)
    return {"n": len(rows), "verdicts": verdicts, "ranges": ranges, "keys": keys, "roles": roles}


def verdict_table(s):
    lines = ["| Verdict | Rows | Score range in the data |", "|---|---|---|"]
    for v in ORDER:
        if s["verdicts"].get(v):
            lo, hi = s["ranges"].get(v, (None, None))
            lines.append("| `%s` | %d | %s |" % (v, s["verdicts"][v], ("%g to %g" % (lo, hi)) if lo is not None else "n/a"))
    return lines


def front_matter(total):
    size = "n<1K" if total < 1000 else "1K<n<10K"
    return "\n".join([
        "---", "license: mit", "language:", "- en", "pretty_name: resume-screener-synthetic", "size_categories:", "- " + size,
        "task_categories:", "- text-generation", "tags:", "- synthetic", "- resume-screening", "- evaluation",
        "configs:", "- config_name: default", "  data_files:", "  - split: train", "    path: train.jsonl", "  - split: test", "    path: eval.jsonl",
        "- config_name: handwritten", "  data_files:", "  - split: test", "    path: handwritten_eval.jsonl", "---", "", ""])


def build(root):
    train = summarize(read_jsonl(root / "data" / "train.jsonl"))
    held = summarize(read_jsonl(root / "data" / "eval.jsonl"))
    hw = read_jsonl(root / "data" / "handwritten_eval.jsonl")
    hw_keys = sorted(set().union(*[set(r) for r in hw])) if hw else []
    hw_labels = Counter(r.get("label") for r in hw)
    hw_roles = Counter(r.get("role") for r in hw if r.get("role"))
    human = (root / "results" / "human_labels_v1.json").exists()
    L = []
    A = L.append
    A("# resume-screener-synthetic")
    A("")
    A("Synthetic training and test data for [resume-screener-lora](%s), a LoRA adapter that reads a resume and a target role and returns a structured JSON verdict "
      "(adapter: https://huggingface.co/Kus-hal/resume-screener-lora). Every resume here is synthetic or fictional; no real person's resume or personal data was used." % GITHUB)
    A("")
    A("## Contents")
    A("")
    A("| File | Split | Rows | What it is |")
    A("|---|---|---|---|")
    A("| `train.jsonl` | train | %d | Training examples written by a template-based generator with a written rubric. |" % train["n"])
    A("| `eval.jsonl` | test | %d | Held-out examples from the same generator (the held-out set in the project's evaluation). |" % held["n"])
    A("| `handwritten_eval.jsonl` | test, config `handwritten` | %d | Fictional resumes in layouts the generator never produces, labelled against a written rubric. |" % len(hw))
    A("")
    A("## Fields")
    A("")
    A("`train.jsonl` and `eval.jsonl` have two fields: `prompt` (the instruction \"Screen this resume for a {role} position and return a structured verdict.\" followed by the resume text) "
      "and `completion` (a JSON string with the keys %s)." % (", ".join("`%s`" % k for k in train["keys"]) or "n/a"))
    A("")
    A("`handwritten_eval.jsonl` has the fields %s. `label` is `strong_match`, `moderate_match` or `weak_match`." % (", ".join("`%s`" % k for k in hw_keys) or "n/a"))
    A("")
    A("## Verdict distribution")
    A("")
    A("**train.jsonl** (%d rows)" % train["n"])
    A("")
    L.extend(verdict_table(train))
    A("")
    A("**eval.jsonl** (%d rows)" % held["n"])
    A("")
    L.extend(verdict_table(held))
    A("")
    A("**handwritten_eval.jsonl** (%d rows): %s." % (len(hw), ", ".join("%d `%s`" % (hw_labels[v], v) for v in ORDER if hw_labels.get(v)) or "n/a"))
    A("")
    if train["roles"]:
        A("Roles in `train.jsonl`: %s." % ", ".join("%s (%d)" % (k, v) for k, v in sorted(train["roles"].items())[:12]))
        A("")
    if hw_roles:
        A("Roles in `handwritten_eval.jsonl`: %s." % ", ".join("%s (%d)" % (k, v) for k, v in sorted(hw_roles.items())))
        A("")
    A("## How it was made")
    A("")
    A("`train.jsonl` and `eval.jsonl` were produced by a template-based generator with a written rubric; the verdict follows from the score. The generator and the rubric are in the project repository. "
      "`handwritten_eval.jsonl` was drafted with an AI assistant against the rubric in [HANDWRITTEN_RUBRIC.md](%s/blob/main/data/HANDWRITTEN_RUBRIC.md) and has not been checked by humans." % GITHUB)
    A("")
    A("## Intended use")
    A("")
    A("- Reproducing the evaluations in the project, and studying how a small model behaves on a task whose labels come from a rubric.")
    A("- Teaching and demonstration.")
    A("")
    A("Not for training or evaluating tools that screen real candidates, and not a measure of what makes a good candidate.")
    A("")
    A("## Limitations")
    A("")
    A("- The data is templated and synthetic, and the labels are the generator's rubric, not human judgments. A model can learn the rubric and look accurate: the adapter reaches 88.5% on `eval.jsonl` but 44.4% on `handwritten_eval.jsonl` (see the project README for the current numbers).")
    A("- The labels of `handwritten_eval.jsonl` are rubric labels drafted with an AI assistant. %s" % ("Independent human labels were collected on this set; see the project README for how they compare." if human else "Independent human labels have not been collected yet."))
    A("- English only, a handful of roles and under a thousand rows, so any number computed on it carries wide uncertainty.")
    A("- Verdicts follow score bands, so examples close to a band edge are ambiguous.")
    A("")
    A("## Links and citation")
    A("")
    A("- Code, evaluation and results: %s" % GITHUB)
    A("- Adapter and model card: https://huggingface.co/Kus-hal/resume-screener-lora")
    A("")
    A(FENCE + "bibtex")
    A("@dataset{tomar2026resumescreenersynthetic,")
    A("  author = {Tomar, Ayush Singh},")
    A("  title  = {resume-screener-synthetic},")
    A("  year   = {2026},")
    A("  url    = {%s}" % HF_URL)
    A("}")
    A(FENCE)
    A("")
    A("Generated from the data files by `docs/build_dataset_card.py` on %s." % time.strftime("%Y-%m-%d"))
    return front_matter(train["n"] + held["n"] + len(hw)), "\n".join(L) + "\n"


def write_card(root, body, published):
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    note = ("> Published at %s. This copy is kept in the repository so it is versioned with the code and the data.\n\n" % HF_URL) if published else ""
    (docs / "DATASET_CARD.md").write_text(note + body, encoding="utf-8", newline="\n")


def scan(root):
    found = {"emails": [], "phones": [], "links": []}
    for rel, field in (("data/train.jsonl", "prompt"), ("data/eval.jsonl", "prompt"), ("data/handwritten_eval.jsonl", "resume")):
        for i, row in enumerate(read_jsonl(root / rel), 1):
            text = str(row.get(field, ""))
            for m in EMAIL.findall(text):
                found["emails"].append("%s row %d: %s" % (rel, i, m))
            for m in URL.findall(text):
                found["links"].append("%s row %d: %s" % (rel, i, m[:60]))
            for m in PHONE.findall(text):
                digits = re.sub(r"\D", "", m)
                if 10 <= len(digits) <= 15 and not re.fullmatch(r"(?:(?:19|20)\d{2}\D*)+", m.strip()):
                    found["phones"].append("%s row %d: %s" % (rel, i, m.strip()))
    return found


def print_scan(found):
    print("\nPersonal-data scan of the data files (the card says no real personal data is included):")
    for kind in ("emails", "phones", "links"):
        print("  %s: %d" % (kind, len(found[kind])))
        for item in found[kind][:3]:
            print("     " + item)
    if not any(found.values()):
        print("  nothing found.")


def stage(root, out, front, body):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for src, dst in (("train.jsonl", "train.jsonl"), ("eval.jsonl", "eval.jsonl"), ("handwritten_eval.jsonl", "handwritten_eval.jsonl")):
        text = (root / "data" / src).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        (out / dst).write_text(text, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(front + body, encoding="utf-8", newline="\n")


def replace_once(path, old, new):
    p = Path(path)
    if not p.exists():
        return "not found"
    t = p.read_text(encoding="utf-8").replace("\r\n", "\n")
    if new in t:
        return "already updated"
    if t.count(old) != 1:
        return "sentence not found exactly once (%d); edit it by hand" % t.count(old)
    p.write_text(t.replace(old, new, 1), encoding="utf-8", newline="\n")
    return "updated"


def finalize(root, hf_card):
    front, body = build(root)
    write_card(root, body, True)
    n_tr = len(read_jsonl(root / "data" / "train.jsonl"))
    n_ev = len(read_jsonl(root / "data" / "eval.jsonl"))
    n_hw = len(read_jsonl(root / "data" / "handwritten_eval.jsonl"))
    block = ("<!-- ds-start -->\n**Data.** The synthetic training set (%d rows), the held-out set (%d) and the %d hand-authored resumes are published at %s with a dataset card "
             "([`docs/DATASET_CARD.md`](docs/DATASET_CARD.md)). All resumes are synthetic or fictional, and the labels are a generator's rubric and an AI-drafted rubric, not human judgments.\n<!-- ds-end -->" % (n_tr, n_ev, n_hw, HF_URL))
    rp = root / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- ds-start -->") + r".*?" + re.escape("<!-- ds-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
        print("README: data note updated.")
    else:
        anchor = next((a for a in ANCHORS if text.count(a) == 1), None)
        if anchor:
            text = text.replace(anchor, anchor + "\n\n" + block, 1)
            print("README: data note added after " + anchor)
        else:
            print("README NOT changed (no anchor found). Paste this where you want it:\n\n" + block)
            text = None
    if text is not None:
        with open(rp, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    entry = "- Data: the synthetic training and held-out sets and the hand-authored set are published on Hugging Face as `%s`, with a dataset card (`docs/DATASET_CARD.md`, generated by `docs/build_dataset_card.py`)." % REPO_ID
    cp = root / "CHANGELOG.md"
    if cp.exists():
        c = cp.read_text(encoding="utf-8").replace("\r\n", "\n")
        if REPO_ID in c:
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
        print("CHANGELOG.md not found; skipped.")
    new_sentence = "The data is published as [%s](%s), together with the %d hand-authored resumes." % (REPO_ID, HF_URL, n_hw)
    print("docs/MODEL_CARD.md: " + replace_once(root / "docs" / "MODEL_CARD.md", OLD_SENTENCE, new_sentence))
    if hf_card:
        print("Hugging Face model card text: " + replace_once(hf_card, OLD_SENTENCE, new_sentence))


def main():
    ap = argparse.ArgumentParser(description="Dataset card for the synthetic and hand-authored data")
    ap.add_argument("--root", default=".")
    ap.add_argument("--stage")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--hf-card")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    front, body = build(root)
    if args.finalize:
        finalize(root, args.hf_card)
        return
    write_card(root, body, False)
    print("Wrote docs/DATASET_CARD.md")
    print_scan(scan(root))
    if args.stage:
        stage(root, args.stage, front, body)
        print("\nStaged the data files and the card in " + args.stage)


if __name__ == "__main__":
    main()
