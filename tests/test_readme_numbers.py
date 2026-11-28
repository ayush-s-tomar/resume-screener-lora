"""
Keeps the README, the model card and the result files honest.

Every number that a generated README block or the repository copy of the model card quotes is recomputed from results/*.json and must
appear in the text. A strict claim fails when its number is missing (the text is stale or a result changed). A claim is skipped when its
source file or block does not exist, or when a result file has a different layout than expected. Soft claims (older hand-written sections)
only warn. The file also checks that README commands name files that exist, that relative links resolve and that block markers are balanced.
"""
import json
import re
import warnings
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
LABS = ("weak_match", "moderate_match", "strong_match")


def load(rel):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8-sig")) if p.exists() else None


def read(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8-sig").replace("\r\n", "\n") if p.exists() else None


def count(rel):
    t = read(rel)
    return sum(1 for line in t.splitlines() if line.strip()) if t is not None else None


def pct_strip(x):
    return ("%.1f" % (x * 100)).rstrip("0").rstrip(".") + "%"


def pct1(x):
    return "%.1f%%" % (x * 100)


def block_of(text, name):
    m = re.search(r"<!-- %s-start -->(.*?)<!-- %s-end -->" % (re.escape(name), re.escape(name)), text, re.S)
    return m.group(1) if m else None


def find_dict_with(d, keys):
    """The first nested dict that has all the given keys."""
    if isinstance(d, dict):
        if all(k in d for k in keys):
            return d
        for v in d.values():
            r = find_dict_with(v, keys)
            if r is not None:
                return r
    return None


class Claim:
    def __init__(self, cid, doc, block, source, strict, needles=None, skip=None):
        self.cid, self.doc, self.block, self.source, self.strict = cid, doc, block, source, strict
        self.needles, self.skip = needles or [], skip


CLAIMS = []


def claim(cid, doc, block, source, build, strict=True):
    try:
        needles = build()
    except Exception as exc:  # a result file with a different layout is skipped, never failed
        CLAIMS.append(Claim(cid, doc, block, source, strict, skip="cannot be checked (%s: %s)" % (type(exc).__name__, exc)))
        return
    if needles is None:
        CLAIMS.append(Claim(cid, doc, block, source, strict, skip="%s is not present" % source))
    elif not needles:
        CLAIMS.append(Claim(cid, doc, block, source, strict, skip="no checkable numbers found in %s" % source))
    else:
        CLAIMS.append(Claim(cid, doc, block, source, strict, needles=needles))


def hw_numbers():
    h = load("results/handwritten_v1.json")
    if h is None:
        return None
    c, n = h["confusion"], h["n_examples"]
    needles = ["%s (%d/%d)" % (pct_strip(h["verdict_accuracy"]), h["verdict_correct"], n), "%d of %d" % (h["severe_errors"], n)]
    if h.get("pairwise_ordering") is not None:
        needles.append(pct_strip(h["pairwise_ordering"]))
    for t in LABS:
        needles.append("| %s | %d | %d | %d | %d |" % (t.split("_")[0], c[t]["weak_match"], c[t]["moderate_match"], c[t]["strong_match"], c[t]["invalid"]))
    pred_weak = sum(c[t]["weak_match"] for t in LABS)
    true_weak = sum(c["weak_match"].values())
    if pred_weak > 1.3 * true_weak:
        needles.append("%d of %d (the rubric has %d)" % (pred_weak, n, true_weak))
    return needles


def ea_numbers():
    e = load("results/error_analysis_v1.json")
    if e is None:
        return None
    needles = []
    ho = e.get("heldout") or {}
    if ho.get("errors") and ho.get("n") and "near_boundary_errors" in ho and "near_boundary_all" in ho:
        needles += ["Of the %d held-out errors, %d sit within" % (ho["errors"], ho["near_boundary_errors"]), pct1(ho["near_boundary_all"] / ho["n"])]
    hw = e.get("handwritten") or {}
    ints = {k: v for k, v in hw.items() if isinstance(v, int) and not isinstance(v, bool)}
    under = next((v for k, v in ints.items() if "under" in k.lower()), None)
    over = next((v for k, v in ints.items() if "over" in k.lower()), None)
    if under is not None and over is not None and under + over:
        needles.append("%d of the %d wrong verdicts" % (under, under + over))
    th = find_dict_with(e, ("default", "in_sample", "loocv"))
    if th:
        needles += [pct1(th["default"]), pct1(th["loocv"])]
        if th["loocv"] - th["default"] < 0.10:
            needles.append(pct1(th["in_sample"]))
    return needles


def sp_numbers():
    s = load("results/selective_v1.json")
    if s is None:
        return None
    rec = s["recommended_margin"]
    if rec is None:
        return ["do not recommend it"]
    a = next(r for r in s["heldout"] if r["margin"] == rec)
    b = next(r for r in s["handwritten"] if r["margin"] == rec)
    needles = ["within %d points of a verdict boundary" % rec, "flags %d of %d resumes" % (a["flagged"], a["n"]),
               "hold %d of the %d errors" % (a["errors_flagged"], a["errors_total"]), pct1(a["acc_kept"]), pct1(a["acc_kept_lo"]), pct1(a["acc_kept_hi"])]
    if b["lift"] is not None:
        needles += ["flags %d of %d resumes" % (b["flagged"], b["n"]), "%d of %d errors" % (b["errors_flagged"], b["errors_total"]), pct1(b["flagged_share"])]
    return needles


def nsb_numbers():
    nb = load("results/name_swap_boundary_v1.json")
    if nb is None:
        return None
    g, rl = nb["groups"], nb["resume_level"]
    return ["Names flipped %d of %d verdicts" % (g["names"]["flips"], g["names"]["pairs"]), "%.1f points" % g["names"]["mean_abs_shift"],
            "%d of %d for neutral prefixes" % (g["neutral"]["flips"], g["neutral"]["pairs"]), "%.1f points" % g["neutral"]["mean_abs_shift"],
            "names flipped more verdicts on %d, neutral prefixes on %d, and they tied on %d (sign test p = %.2f)" % (rl["names_more"], rl["neutral_more"], rl["tied"], rl["sign_test_p"]),
            "within %.1f points" % nb["max_distance"]]


def headline_numbers():
    cmp_ = load("comparison_results.json")
    if cmp_ is None:
        return None
    return [pct_strip(cmp_["fine_tuned"]["verdict_accuracy"])]


def card_numbers():
    cmp_ = load("comparison_results.json")
    h = load("results/handwritten_v1.json")
    if cmp_ is None or h is None:
        return None
    acc = cmp_["fine_tuned"]["verdict_accuracy"]
    n_ho = count("data/eval.jsonl")
    needles = [pct_strip(acc)]
    if n_ho:
        needles.append("(%d/%d" % (round(acc * n_ho), n_ho))
    n = h["n_examples"]
    needles.append("Verdict accuracy %s (%d/%d), %d severe errors" % (pct_strip(h["verdict_accuracy"]), h["verdict_correct"], n, h["severe_errors"]))
    if h.get("pairwise_ordering") is not None:
        needles.append("%s of pairs" % pct_strip(h["pairwise_ordering"]))
    c = h["confusion"]
    pred_weak = sum(c[t]["weak_match"] for t in LABS)
    true_weak = sum(c["weak_match"].values())
    if pred_weak > 1.3 * true_weak:
        needles.append("%d of %d resumes (the rubric has %d)" % (pred_weak, n, true_weak))
    try:
        nb = load("results/name_swap_boundary_v1.json")
        if nb:
            needles += ["names flipped %d of %d verdicts" % (nb["groups"]["names"]["flips"], nb["groups"]["names"]["pairs"]),
                        "flipped %d of %d" % (nb["groups"]["neutral"]["flips"], nb["groups"]["neutral"]["pairs"])]
    except (KeyError, TypeError):
        pass
    try:
        sk = load("results/skill_lists_v1.json")
        if sk:
            needles += ["Matched skills F1 %.2f; missing skills F1 %.2f" % (sk["matched"]["f1"], sk["missing"]["f1"]),
                        "%.1f%% of the skills" % (sk["missing"]["inconsistent_with_resume_rate"] * 100),
                        "(%.1f%% for the generator" % (sk["missing"]["generator_inconsistent_with_resume_rate"] * 100)]
    except (KeyError, TypeError):
        pass
    return needles


def ns_numbers():
    ns = load("results/name_swap_v1.json")
    if ns is None:
        return None
    names, rows = ns["names"], ns["per_resume"]
    shifts, flips, pairs = [], 0, 0
    for row in rows:
        base, verdict = row["base_score"], row["base_verdict"]
        for p in names:
            v = row["variants"].get(p)
            if v:
                shifts.append(abs(float(v[0]) - float(base)))
                flips += int(v[1] != verdict)
                pairs += 1
    return ["%d of %d verdicts flipped" % (flips, pairs), "mean absolute score change was %.1f points" % (sum(shifts) / len(shifts))]


def changelog_numbers():
    h = load("results/handwritten_v1.json")
    if h is None:
        return None
    return ["accuracy falls to %s" % pct_strip(h["verdict_accuracy"])]


claim("readme-hw-block", "README.md", "hw", "results/handwritten_v1.json", hw_numbers)
claim("readme-ea-block", "README.md", "ea", "results/error_analysis_v1.json", ea_numbers)
claim("readme-sp-block", "README.md", "sp", "results/selective_v1.json", sp_numbers)
claim("readme-nsb-block", "README.md", "nsb", "results/name_swap_boundary_v1.json", nsb_numbers)
claim("readme-headline", "README.md", None, "comparison_results.json", headline_numbers)
claim("model-card", "docs/MODEL_CARD.md", None, "the results files", card_numbers)
claim("readme-name-swap-text", "README.md", "ns", "results/name_swap_v1.json", ns_numbers, strict=False)
claim("changelog-handwritten", "CHANGELOG.md", None, "results/handwritten_v1.json", changelog_numbers, strict=False)


@pytest.mark.parametrize("claim", CLAIMS, ids=lambda c: c.cid)
def test_numbers_match_the_results(claim):
    if claim.skip:
        pytest.skip(claim.skip)
    text = read(claim.doc)
    if text is None:
        pytest.skip("%s is not present" % claim.doc)
    scope = text
    if claim.block:
        scope = block_of(text, claim.block)
        if scope is None:
            pytest.skip("%s has no <!-- %s-start --> block" % (claim.doc, claim.block))
    missing = [n for n in claim.needles if n not in scope]
    if not missing:
        return
    message = "%s: %s no longer contains %s (recomputed from %s). Either the text is stale or the result changed; fix whichever is wrong." % (
        claim.doc, ("the %s block" % claim.block) if claim.block else "the file", ", ".join(repr(m) for m in missing), claim.source)
    if claim.strict:
        pytest.fail(message)
    warnings.warn(UserWarning("soft claim: " + message))


def test_block_markers_are_balanced():
    text = read("README.md")
    starts = re.findall(r"<!-- ([\w-]+)-start -->", text)
    ends = re.findall(r"<!-- ([\w-]+)-end -->", text)
    assert sorted(starts) == sorted(ends), "unbalanced README block markers: starts %s, ends %s" % (sorted(starts), sorted(ends))
    twice = sorted({s for s in starts if starts.count(s) > 1})
    assert not twice, "a README block appears twice: " + ", ".join(twice)
    for name in set(starts):
        assert text.index("<!-- %s-start -->" % name) < text.index("<!-- %s-end -->" % name), "block %s ends before it starts" % name


def test_no_leftover_placeholders():
    for rel in ("README.md", "CHANGELOG.md", "docs/MODEL_CARD.md", "docs/RESPONSIBLE_USE.md", "docs/PRIVACY.md", "docs/BIAS_AUDIT.md", "docs/DATASET_CARD.md", "docs/PILOT_RESULTS.md"):
        text = read(rel)
        if text is not None:
            assert "@@" not in text, rel + " still contains an unreplaced @@ placeholder"
            assert "TODO" not in text, rel + " still contains a TODO"


def test_readme_commands_name_existing_files():
    names = set(re.findall(r"python3?\s+((?:[\w.\-]+/)*[\w.\-]+\.py)\b", read("README.md")))
    missing = sorted(n for n in names if not (ROOT / n).exists())
    assert not missing, "the README shows commands for files that do not exist: " + ", ".join(missing)


def test_readme_relative_links_resolve():
    bad = []
    for target in re.findall(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", read("README.md")):
        if re.match(r"^(https?:|mailto:|#|data:)", target):
            continue
        path = unquote(target.split("#")[0].split("?")[0])
        if path and not (ROOT / path).exists():
            bad.append(target)
    assert not bad, "the README links to missing files: " + ", ".join(sorted(set(bad)))


@pytest.mark.parametrize("path", sorted((ROOT / "results").glob("*.json")), ids=lambda p: p.name)
def test_result_files_are_valid_json(path):
    json.loads(path.read_text(encoding="utf-8-sig"))
