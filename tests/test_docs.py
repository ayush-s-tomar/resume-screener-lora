"""Checks that the documents in docs/ agree with the result files and that their relative links resolve."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def pct(x):
    return ("%.1f" % (x * 100)).rstrip("0").rstrip(".") + "%"


@pytest.mark.parametrize("path", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_relative_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        target = target.split("#")[0]
        if target:
            assert (path.parent / target).resolve().exists(), "%s links to %s, which does not exist" % (path.name, target)


def test_responsible_use_numbers_match_the_results():
    doc = DOCS / "RESPONSIBLE_USE.md"
    hand = ROOT / "results" / "handwritten_v1.json"
    cmp_ = ROOT / "comparison_results.json"
    if not (doc.exists() and hand.exists() and cmp_.exists()):
        pytest.skip("the document or a result file is not present")
    text = doc.read_text(encoding="utf-8")
    h = json.loads(hand.read_text(encoding="utf-8-sig"))
    held = json.loads(cmp_.read_text(encoding="utf-8-sig"))["fine_tuned"]["verdict_accuracy"]
    assert pct(h["verdict_accuracy"]) in text
    assert pct(held) in text


def test_privacy_document_matches_the_demo_limits():
    doc = DOCS / "PRIVACY.md"
    app = ROOT / "render-deploy" / "app.py"
    if not (doc.exists() and app.exists()):
        pytest.skip("the document or the demo source is not present")
    m = re.search(r'_env_int\("DEMO_RATE_LIMIT",\s*(\d+)\)', app.read_text(encoding="utf-8"))
    if m:
        assert "%s new screenings" % m.group(1) in doc.read_text(encoding="utf-8")
