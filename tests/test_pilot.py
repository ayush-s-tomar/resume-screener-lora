"""Tests for the pilot kit: the question file, the generated survey page and the aggregation."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pilot"))
import aggregate_pilot as ap  # noqa: E402

DATA = json.loads((ROOT / "pilot" / "questions.json").read_text(encoding="utf-8"))
QS = DATA["questions"]


def base(**over):
    a = {"role_family": "backend", "resume_used": "sample", "input_method": "sample_button", "self_fit": "moderate", "score": 56, "verdict": "moderate",
         "l_verdict": 4, "l_matched": 4, "l_missing": 4, "l_new": 3, "l_again": 5, "comfort": "no_change", "trust_decisions": "no"}
    a.update(over)
    return a


def resp(i, **over):
    return {"version": 1, "id": "r%d" % i, "created": "2026-01-01", "consent": True, "answers": base(**over)}


def test_question_file_is_consistent():
    ids = [q["id"] for q in QS]
    assert len(ids) == len(set(ids))
    for q in QS:
        assert q["type"] in ("choice", "scale", "number", "text")
        if q["type"] == "choice":
            values = [o[0] for o in q["options"]]
            assert values and len(values) == len(set(values))
        if q["type"] in ("scale", "number"):
            assert q["min"] < q["max"]
    for needed in ("self_fit", "verdict", "l_again", "l_missing", "comfort"):
        assert needed in ids


def test_the_base_answers_are_valid():
    assert ap.problems(resp(1), QS) == []


def test_survey_page_embeds_the_current_questions():
    page = ROOT / "pilot" / "survey.html"
    if not page.exists():
        pytest.skip("pilot/survey.html has not been built")
    m = re.search(r"var DATA = (.*?);\nvar answers = \{\};", page.read_text(encoding="utf-8"), re.S)
    assert m, "the page does not contain the question data"
    assert json.loads(m.group(1).replace("<\\/", "</")) == DATA, "rebuild the page with: python pilot/build_survey.py"


def test_simulated_responses_are_valid():
    assert all(ap.problems(r, QS) == [] for r in ap.fake_responses(QS))


def test_problems_are_detected():
    assert "consent is not true" in ap.problems(dict(resp(1), consent=False), QS)
    assert any("unknown option" in p for p in ap.problems(resp(1, verdict="great"), QS))
    assert any("bad rating" in p for p in ap.problems(resp(1, l_again=6), QS))
    assert any("bad number" in p for p in ap.problems(resp(1, score=101), QS))
    assert any("bad text" in p for p in ap.problems(resp(1, comment="x" * 301), QS))
    broken = resp(1)
    del broken["answers"]["self_fit"]
    assert "missing answer: self_fit" in ap.problems(broken, QS)
    assert ap.problems([], QS) == ["not a JSON object"]


def test_summary_numbers():
    rs = [resp(1), resp(2, self_fit="strong", verdict="weak", l_again=2, comfort="discouraged"), resp(3, self_fit="weak", verdict="strong"), resp(4, self_fit="unsure")]
    s = ap.summarize(rs, QS)
    assert s["n"] == 4
    assert s["self_vs_model"] == {"pairs": 3, "same": 1, "model_lower": 1, "model_higher": 1}
    again = s["questions"]["l_again"]
    assert again["agree"] == 3 and again["agree_share"] == pytest.approx(0.75)
    assert s["uncomfortable"] == {"n": 1, "of": 4}
    by = {c["id"]: c for c in s["criteria"]}
    assert by["l_again"]["met"] is True and by["l_missing"]["met"] is True and by["comfort"]["met"] is False
    assert s["useful"] is False


def test_block_mentions_the_counts():
    s = ap.summarize([resp(1), resp(2), resp(3)], QS)
    block = ap.render_block(s)
    assert block.startswith("<!-- pl-start -->") and block.endswith("<!-- pl-end -->")
    assert "3 students" in block and "3 of 3" in block


def test_wilson_interval():
    assert ap.wilson(0, 0) == (None, None)
    lo, hi = ap.wilson(5, 10)
    assert lo < 0.5 < hi


def test_load_responses_skips_bad_files(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(resp(1)), encoding="utf-8")
    (tmp_path / "b.json").write_text("not json", encoding="utf-8")
    (tmp_path / "c.json").write_text(json.dumps(resp(1)), encoding="utf-8")
    (tmp_path / "d.json").write_text(json.dumps(dict(resp(4), consent=False)), encoding="utf-8")
    good, bad = ap.load_responses(tmp_path, QS)
    assert len(good) == 1 and len(bad) == 3


def test_docs_render_without_free_text():
    rs = [resp(i, comment="private remark %d" % i) for i in range(10)]
    doc = ap.render_docs(ap.summarize(rs, QS), QS, "2026-01-01")
    assert "private remark" not in doc and "Free-text comments (10)" in doc
