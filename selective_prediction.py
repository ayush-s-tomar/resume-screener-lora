"""
Selective prediction: what is a "borderline" flag worth?

A verdict is flagged as borderline when the adapter's own score lies within a margin of a verdict boundary (or when the output
is not valid JSON, which an application can detect). Flagged resumes would be sent to a human instead of being decided
automatically. This script uses only saved results: it chooses the margin on the held-out set with a rule fixed in advance,
then checks the same margin, without refitting, on the hand-authored set.

Usage:
    python selective_prediction.py
"""
import argparse
import json
import math
import re
import time
from pathlib import Path

from error_analysis import BOUNDS, ORD, dist_to_boundary, get_handwritten, get_heldout, md_table, pct

MARGINS = [0, 1, 2, 3, 4, 5, 6, 8, 10]
CANDIDATES = [1, 2, 3, 4, 5, 6, 8]
MAX_FLAGGED = 0.30
MIN_FLAGGED = 4


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def acc_all(rows):
    return sum(r["label"] == r["pred"] for r in rows) / len(rows)


def curve(rows):
    n = len(rows)
    wrong_all = sum(r["label"] != r["pred"] for r in rows)
    out = []
    for m in MARGINS:
        flagged, kept = [], []
        for r in rows:
            unusable = r["pred"] not in ORD or r["score"] is None
            if unusable or (m > 0 and dist_to_boundary(r["score"]) < m):
                flagged.append(r)
            else:
                kept.append(r)
        ok = sum(r["label"] == r["pred"] for r in kept)
        wf = sum(r["label"] != r["pred"] for r in flagged)
        lo, hi = wilson(ok, len(kept))
        out.append({
            "margin": m, "n": n, "flagged": len(flagged), "flagged_share": len(flagged) / n, "kept": len(kept), "coverage": len(kept) / n,
            "acc_kept": ok / len(kept) if kept else None, "acc_kept_lo": lo, "acc_kept_hi": hi,
            "errors_total": wrong_all, "errors_flagged": wf, "caught": wf / wrong_all if wrong_all else None,
            "lift": (wf / wrong_all - len(flagged) / n) if wrong_all else None,
        })
    return out


def pick_margin(c):
    best = None
    for r in c:
        if r["margin"] in CANDIDATES and r["flagged"] >= MIN_FLAGGED and r["flagged_share"] <= MAX_FLAGGED and r["lift"] is not None and r["lift"] > 0:
            key = (r["lift"], -r["margin"])
            if best is None or key > best[0]:
                best = (key, r)
    return best[1]["margin"] if best else None


def fmt_ci(r):
    if r["acc_kept"] is None:
        return "n/a"
    return "%s (%s to %s)" % (pct(r["acc_kept"]), pct(r["acc_kept_lo"]), pct(r["acc_kept_hi"]))


def table(c):
    rows = []
    for r in c:
        gain = "n/a" if r["lift"] is None else "%+.0f pts" % (r["lift"] * 100)
        rows.append([r["margin"] or "none", "%d (%s)" % (r["flagged"], pct(r["flagged_share"])), pct(r["coverage"]), fmt_ci(r),
                     "%d of %d" % (r["errors_flagged"], r["errors_total"]), gain])
    return md_table(["Margin (points)", "Flagged", "Coverage", "Accuracy on unflagged (95% interval)", "Errors among flagged", "Gain over random flagging"], rows)


def svg_chart(path, series, rec):
    W, H, L, R, T, B = 600, 360, 64, 24, 36, 56
    xs = [p[0] for s in series for p in s["pts"]]
    ys = [p[1] for s in series for p in s["pts"]]
    x0 = min(math.floor(min(xs) * 10) / 10, 0.9)
    y0 = min(math.floor(min(ys) * 10) / 10, 0.9)

    def sx(v):
        return L + (v - x0) / (1.0 - x0) * (W - L - R)

    def sy(v):
        return T + (1 - (v - y0) / (1.0 - y0)) * (H - T - B)

    p = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="Arial, Helvetica, sans-serif" font-size="12">' % (W, H, W, H),
         '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
         '<text x="%d" y="22" font-size="14" font-weight="bold" fill="#222222">Accuracy on unflagged resumes against coverage</text>' % L]
    for v in range(int(round(x0 * 10)), 11):
        x = sx(v / 10.0)
        p.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e5e5"/>' % (x, T, x, H - B))
        p.append('<text x="%.1f" y="%d" text-anchor="middle" fill="#555555">%d%%</text>' % (x, H - B + 16, v * 10))
    for v in range(int(round(y0 * 10)), 11):
        y = sy(v / 10.0)
        p.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e5e5"/>' % (L, y, W - R, y))
        p.append('<text x="%d" y="%.1f" text-anchor="end" fill="#555555">%d%%</text>' % (L - 6, y + 4, v * 10))
    p.append('<text x="%d" y="%d" text-anchor="middle" fill="#333333">Coverage (share of resumes not flagged)</text>' % ((L + W - R) // 2, H - 14))
    p.append('<text transform="rotate(-90 16 %d)" x="16" y="%d" text-anchor="middle" fill="#333333">Accuracy on unflagged</text>' % ((T + H - B) // 2, (T + H - B) // 2))
    for i, s in enumerate(series):
        pts = " ".join("%.1f,%.1f" % (sx(a), sy(b)) for a, b, _ in s["pts"])
        p.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>' % (pts, s["color"]))
        for a, b, m in s["pts"]:
            if rec is not None and m == rec:
                p.append('<circle cx="%.1f" cy="%.1f" r="7" fill="none" stroke="#222222" stroke-width="2"/>' % (sx(a), sy(b)))
            p.append('<circle cx="%.1f" cy="%.1f" r="3.5" fill="%s"/>' % (sx(a), sy(b), s["color"]))
        ly = T + 14 + i * 18
        p.append('<rect x="%d" y="%d" width="12" height="12" fill="%s"/>' % (L + 12, ly - 10, s["color"]))
        p.append('<text x="%d" y="%d" fill="#222222">%s</text>' % (L + 30, ly, s["label"]))
    if rec is not None:
        p.append('<text x="%d" y="%d" fill="#222222">Circle: chosen margin (%d points)</text>' % (L + 12, T + 14 + len(series) * 18, rec))
    p.append("</svg>")
    Path(path).write_text("\n".join(p), encoding="utf-8", newline="\n")


def main():
    ap = argparse.ArgumentParser(description="Selective prediction analysis")
    ap.add_argument("--root", default=".")
    ap.add_argument("--no-readme", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    ha, _ = get_handwritten(root)
    ho = get_heldout(root)
    for name, rows in (("held-out", ho), ("hand-authored", ha)):
        have = sum(r["score"] is not None for r in rows)
        if have < 0.9 * len(rows):
            raise SystemExit("%s: only %d of %d rows have an adapter score. Paste me the schema check from Day 4." % (name, have, len(rows)))
    c_ho, c_ha = curve(ho), curve(ha)
    rec = pick_margin(c_ho)
    n_ho, n_ha = len(ho), len(ha)
    today = time.strftime("%Y-%m-%d")

    md = ["# Flagging borderline verdicts", "",
          "Computed by `selective_prediction.py` on %s from the saved result files, with no model run." % today, "",
          "## Idea", "",
          "The name-swap boundary test showed that verdicts close to a threshold are unstable. An application can act on that: mark a verdict **borderline** when the adapter's own score "
          "lies within a margin of a verdict boundary (%.0f and %.1f), and send those resumes to a human instead of deciding them automatically. An output that is not valid JSON is always flagged, "
          "because the application can detect that. The question is what the flag buys: how much accuracy rises on the resumes that are not flagged, at what cost in coverage, "
          "and whether the flag catches more errors than flagging the same number of resumes at random would." % BOUNDS, "",
          "## Rule for choosing the margin (fixed before looking at the numbers)", "",
          "On the **held-out set only**, among margins of %s points, take the one with the largest gain over random flagging (share of errors caught minus share of resumes flagged), "
          "provided it flags at most %d%% of resumes and at least %d. If no margin has a positive gain, recommend no flag. The hand-authored set is not used to choose; "
          "it is the out-of-sample check." % (", ".join(str(m) for m in CANDIDATES), int(MAX_FLAGGED * 100), MIN_FLAGGED), "",
          "## Held-out set (%d examples, %d errors)" % (n_ho, c_ho[0]["errors_total"]), "", table(c_ho), "",
          "## Hand-authored set (%d resumes, %d errors), same margins, nothing refitted" % (n_ha, c_ha[0]["errors_total"]), "", table(c_ha), ""]
    base_ho, base_ha = acc_all(ho), acc_all(ha)
    md += ["Overall accuracy without any flag: %s on the held-out set and %s on the hand-authored set (an invalid output counts as wrong)." % (pct(base_ho), pct(base_ha)), "",
           "## Result", ""]
    if rec is None:
        md += ["No margin met the rule on the held-out set, so I do not recommend a borderline flag from this evidence.", ""]
    else:
        a, b = next(r for r in c_ho if r["margin"] == rec), next(r for r in c_ha if r["margin"] == rec)
        md += ["**Recommended margin: %d points.** On the held-out set it flags %d of %d resumes (%s) and those hold %d of the %d errors; accuracy on the unflagged resumes is %s against %s overall." % (
            rec, a["flagged"], a["n"], pct(a["flagged_share"]), a["errors_flagged"], a["errors_total"], fmt_ci(a), pct(base_ho)), ""]
        if b["lift"] is None:
            tail = "There were no errors to catch."
        elif b["lift"] > 0.10:
            tail = "That is more than the %s of errors that flagging the same share at random would catch, so the flag helps on differently written resumes too." % pct(b["flagged_share"])
        elif b["lift"] >= -0.05:
            tail = "That is about what flagging the same share at random would catch (%s), so the flag does not transfer to differently written resumes." % pct(b["flagged_share"])
        else:
            tail = "That is fewer than flagging the same share at random would catch (%s), so the flag does not help on differently written resumes." % pct(b["flagged_share"])
        md += ["**Out-of-sample check.** On the hand-authored set the same margin flags %d of %d resumes (%s) and those hold %d of the %d errors; accuracy on the unflagged resumes is %s against %s overall. %s" % (
            b["flagged"], b["n"], pct(b["flagged_share"]), b["errors_flagged"], b["errors_total"], fmt_ci(b), pct(base_ha), tail), ""]
    md += ["## Limits", "",
           "- The held-out set has %d errors, so the margin is chosen from a very small number of events and could easily be different on another sample." % c_ho[0]["errors_total"],
           "- Flagged resumes are not scored as right or wrong by a human; they are removed from the automatic decisions. The cost is the coverage given up.",
           "- The boundaries (%.0f and %.1f) are the ones in the training data. The error analysis shows the adapter's errors on the hand-authored set are mostly under-predictions that a margin around a boundary cannot catch unless they sit near it." % BOUNDS,
           "- Both label sets are rubric labels, not human judgments.", ""]
    docs = root / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "SELECTIVE_PREDICTION.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    out = {"date": today, "rule": {"candidates": CANDIDATES, "max_flagged_share": MAX_FLAGGED, "min_flagged": MIN_FLAGGED, "chosen_on": "held-out"},
           "recommended_margin": rec, "boundaries": list(BOUNDS), "heldout": c_ho, "handwritten": c_ha}
    (root / "results" / "selective_v1.json").write_text(json.dumps(out, indent=2), encoding="utf-8", newline="\n")

    def pts(c):
        return [(r["coverage"], r["acc_kept"], r["margin"]) for r in c if r["acc_kept"] is not None]

    svg_chart(root / "assets" / "selective_prediction.svg",
              [{"label": "Held-out (%d synthetic examples)" % n_ho, "color": "#1f77b4", "pts": pts(c_ho)},
               {"label": "Hand-authored (%d resumes)" % n_ha, "color": "#d62728", "pts": pts(c_ha)}], rec)

    if rec is None:
        sentence = "I also checked whether flagging verdicts near a boundary as borderline would help; no margin gained more than random flagging would on the held-out set, so I do not recommend it."
    else:
        a, b = next(r for r in c_ho if r["margin"] == rec), next(r for r in c_ha if r["margin"] == rec)
        if b["lift"] is None:
            ha_part = "the hand-authored set had no errors to catch"
        elif b["lift"] > 0.10:
            ha_part = "on the hand-authored set, which was not used to choose the margin, the same margin flags %d of %d resumes and those hold %d of %d errors, more than random flagging of the same share (%s) would catch" % (
                b["flagged"], b["n"], b["errors_flagged"], b["errors_total"], pct(b["flagged_share"]))
        else:
            ha_part = "on the hand-authored set, which was not used to choose the margin, the same margin flags %d of %d resumes and those hold only %d of %d errors, no better than random flagging of the same share (%s), so the flag does not transfer to those resumes" % (
                b["flagged"], b["n"], b["errors_flagged"], b["errors_total"], pct(b["flagged_share"]))
        sentence = ("Verdicts close to a threshold are unstable, so a verdict can be marked borderline when the adapter's own score is within %d points of a verdict boundary. "
                    "Chosen on the held-out set by a rule fixed in advance, that margin flags %d of %d resumes and the flagged resumes hold %d of the %d errors; accuracy on the unflagged resumes is %s against %s overall. "
                    "On the other side, %s. The margin rests on %d held-out errors, so treat it as a starting point." % (
                        rec, a["flagged"], a["n"], a["errors_flagged"], a["errors_total"], fmt_ci(a), pct(base_ho), ha_part, a["errors_total"]))
    block = ("<!-- sp-start -->\n**Borderline flag.** " + sentence + " Details: [`docs/SELECTIVE_PREDICTION.md`](docs/SELECTIVE_PREDICTION.md); reproduce with `python selective_prediction.py`.\n\n"
             "![Accuracy on unflagged resumes against coverage](assets/selective_prediction.svg)\n<!-- sp-end -->")
    print("Wrote docs/SELECTIVE_PREDICTION.md, results/selective_v1.json and assets/selective_prediction.svg")
    print("Recommended margin: %s" % ("none" if rec is None else "%d points" % rec))
    print(sentence)
    if args.no_readme:
        return
    rp = root / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- sp-start -->") + r".*?" + re.escape("<!-- sp-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
    else:
        anchor = "<!-- ea-end -->" if text.count("<!-- ea-end -->") == 1 else "<!-- hw-end -->"
        if text.count(anchor) != 1:
            print("README NOT changed (anchor %s found %d times). Paste this where you want it:\n\n%s" % (anchor, text.count(anchor), block))
            return
        text = text.replace(anchor, anchor + "\n\n" + block, 1)
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("README updated.")


if __name__ == "__main__":
    main()