"""
Summarise the demo's usage events (aggregates only) from the database table written by render-deploy/app.py.

The events hold no resume text and no addresses (see docs/PRIVACY.md). This script needs the project URL and the service-role key.
Keep that key on your machine only: never put it on the host or in the repository.

Usage (set SUPABASE_URL and SUPABASE_SERVICE_KEY first):
    python usage_report.py            # print the summary
    python usage_report.py --write    # also write results/usage_v1.json and a README block
"""
import argparse
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path


def fetch_rows(url, key, table="demo_events", page=1000):
    rows, start = [], 0
    while True:
        req = urllib.request.Request(
            "%s/rest/v1/%s?select=*&order=id.asc" % (url.rstrip("/"), table),
            headers={"apikey": key, "Authorization": "Bearer " + key, "Range-Unit": "items", "Range": "%d-%d" % (start, start + page - 1)},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            chunk = json.loads(resp.read().decode("utf-8"))
        rows += chunk
        if len(chunk) < page:
            return rows
        start += page


def aggregate(rows):
    scores = [r for r in rows if r.get("kind") == "score"]
    uploads = [r for r in rows if r.get("kind") == "upload"]
    rated = [r for r in rows if r.get("kind") == "feedback" and r.get("rating") in ("up", "down")]
    days = sorted({str(r["created_at"])[:10] for r in rows if r.get("created_at")})
    visitor_days = {(str(r.get("created_at", ""))[:10], r["visitor"]) for r in rows if r.get("visitor")}
    up = sum(r["rating"] == "up" for r in rated)
    flagged = [r for r in scores if r.get("borderline") is not None]
    return {
        "events": len(rows),
        "screenings": len(scores),
        "fresh_screenings": sum(not r.get("cached") for r in scores),
        "uploads": len(uploads),
        "visitor_days": len(visitor_days),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "active_days": len(days),
        "feedback_up": up,
        "feedback_down": len(rated) - up,
        "helpful_share": (up / len(rated)) if rated else None,
        "down_reasons": dict(Counter(r["reason"] for r in rated if r["rating"] == "down" and r.get("reason"))),
        "roles": dict(Counter(r["role_bucket"] for r in scores if r.get("role_bucket"))),
        "masked_share": (sum(bool(r.get("masked")) for r in scores) / len(scores)) if scores else None,
        "borderline_share": (sum(bool(r["borderline"]) for r in flagged) / len(flagged)) if flagged else None,
    }


def plural(n, word):
    return "%d %s%s" % (n, word, "" if n == 1 else "s")


def render_block(a):
    if not a["screenings"]:
        return None
    text = "From %s to %s, the hosted demo logged %s (%d not served from the cache) and %s across %s (a visitor is counted once per day, because the visitor hash changes daily)." % (
        a["first_day"], a["last_day"], plural(a["screenings"], "screening"), a["fresh_screenings"], plural(a["uploads"], "file upload"), plural(a["visitor_days"], "visitor-day"))
    rated = a["feedback_up"] + a["feedback_down"]
    if rated:
        reasons = ""
        if a["down_reasons"]:
            reasons = " (" + ", ".join("%s %d" % (k.replace("_", " "), v) for k, v in sorted(a["down_reasons"].items())) + ")"
        text += " Visitors rated %s: %d helpful and %d not helpful%s." % (plural(rated, "result"), a["feedback_up"], a["feedback_down"], reasons)
    text += " The counts include my own testing; only aggregate counts are published, and the event log never holds resume text."
    return "<!-- us-start -->\n**Usage so far.** " + text + "\n<!-- us-end -->"


def main():
    ap = argparse.ArgumentParser(description="Summarise the demo's usage events")
    ap.add_argument("--write", action="store_true", help="write results/usage_v1.json and the README block")
    ap.add_argument("--table", default="demo_events")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    url, key = os.environ.get("SUPABASE_URL", ""), os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY first.")
    a = aggregate(fetch_rows(url, key, args.table))
    print(json.dumps(a, indent=2))
    block = render_block(a)
    print("\nREADME block:\n" + (block or "(not enough data yet: no screenings recorded)"))
    if not args.write:
        return
    if not block:
        raise SystemExit("Nothing to write yet.")
    root = Path(args.root).resolve()
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "usage_v1.json").write_text(json.dumps(dict(a, date=time.strftime("%Y-%m-%d")), indent=2), encoding="utf-8", newline="\n")
    rp = root / "README.md"
    text = rp.read_text(encoding="utf-8").replace("\r\n", "\n")
    pat = re.compile(re.escape("<!-- us-start -->") + r".*?" + re.escape("<!-- us-end -->"), re.S)
    if pat.search(text):
        text = pat.sub(lambda m: block, text, count=1)
    else:
        anchor = next((x for x in ("<!-- fb-end -->", "<!-- dm-end -->", "<!-- pk-end -->", "<!-- ru-end -->", "<!-- sp-end -->", "<!-- hw-end -->") if text.count(x) == 1), None)
        if not anchor:
            print("README NOT changed (no anchor found). Paste the block above where you want it.")
            return
        text = text.replace(anchor, anchor + "\n\n" + block, 1)
    with open(rp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("Wrote results/usage_v1.json and updated the README.")


if __name__ == "__main__":
    main()
