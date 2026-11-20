"""Tests for the borderline flag and for inference.screen_resume (no model needed: the model and tokenizer are fakes)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import borderline  # noqa: E402
import inference  # noqa: E402


class FakeInputs:
    shape = (1, 4)

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self, text):
        self.text = text
        self.prompt = None

    def apply_chat_template(self, messages, **kwargs):
        self.prompt = messages[0]["content"]
        return FakeInputs()

    def decode(self, ids, skip_special_tokens=True):
        return self.text


class FakeModel:
    device = "cpu"

    def generate(self, inputs, **kwargs):
        return [[1, 2, 3, 4, 5, 6]]


def run(text, **kwargs):
    tok = FakeTokenizer(text)
    return inference.screen_resume(FakeModel(), tok, "some resume", "Backend Engineer", **kwargs), tok


@pytest.mark.parametrize("score,expected", [(49, 0), (74.5, 0), (60, 11), (48, 1), (76, 1.5), (0, 49), (100, 25.5)])
def test_distance_to_boundary(score, expected):
    assert borderline.distance_to_boundary(score) == pytest.approx(expected)


def test_near_boundary_is_flagged():
    out = borderline.annotate({"ats_score": 76, "verdict": "strong_match"}, margin=3)
    assert out["borderline"] is True
    assert out["borderline_reason"] == "near_boundary"
    assert out["distance_to_boundary"] == pytest.approx(1.5)
    assert out["borderline_margin"] == 3.0
    assert out["ats_score"] == 76


def test_far_from_a_boundary_is_not_flagged():
    out = borderline.annotate({"ats_score": 60}, margin=3)
    assert out["borderline"] is False
    assert out["borderline_reason"] is None
    assert out["distance_to_boundary"] == pytest.approx(11)


def test_the_margin_is_strict():
    # Day 5 flagged a verdict only when its distance to a boundary was strictly smaller than the margin.
    assert borderline.annotate({"ats_score": 76}, margin=1.5)["borderline"] is False
    assert borderline.annotate({"ats_score": 76}, margin=1.51)["borderline"] is True


def test_invalid_output_is_flagged():
    out = borderline.annotate({"raw_output": "oops", "parse_error": True}, margin=3)
    assert out["borderline"] is True
    assert out["borderline_reason"] == "invalid_output"
    assert out["distance_to_boundary"] is None


@pytest.mark.parametrize("verdict", [{"verdict": "weak_match"}, {"ats_score": "60"}, {"ats_score": None}, {"ats_score": True}, {"ats_score": float("nan")}])
def test_missing_or_unusable_score_is_flagged(verdict):
    out = borderline.annotate(verdict, margin=3)
    assert out["borderline"] is True
    assert out["borderline_reason"] == "missing_score"


@pytest.mark.parametrize("margin", [0, -1])
def test_flag_off_returns_the_verdict_unchanged(margin):
    verdict = {"ats_score": 76, "verdict": "strong_match"}
    assert borderline.annotate(verdict, margin=margin) == verdict


def test_annotate_does_not_modify_its_input():
    verdict = {"ats_score": 76}
    borderline.annotate(verdict, margin=3)
    assert verdict == {"ats_score": 76}


def test_default_margin_is_used_when_none_is_given(monkeypatch):
    monkeypatch.setattr(borderline, "DEFAULT_MARGIN", 3.0)
    assert borderline.annotate({"ats_score": 76})["borderline"] is True
    monkeypatch.setattr(borderline, "DEFAULT_MARGIN", None)
    assert borderline.annotate({"ats_score": 76}) == {"ats_score": 76}


@pytest.mark.parametrize("content,expected", [
    ('{"recommended_margin": 3}', 3.0),
    ('{"recommended_margin": 2.5}', 2.5),
    ('{"recommended_margin": null}', None),
    ('{"recommended_margin": 0}', None),
    ('{"recommended_margin": -1}', None),
    ('{"recommended_margin": true}', None),
    ("[1, 2]", None),
    ("not json", None),
    ("{}", None),
])
def test_recommended_margin_parsing(tmp_path, content, expected):
    path = tmp_path / "selective.json"
    path.write_text(content, encoding="utf-8")
    assert borderline.recommended_margin(path) == expected


def test_recommended_margin_missing_file(tmp_path):
    assert borderline.recommended_margin(tmp_path / "nope.json") is None


def test_boundaries_and_default_match_the_saved_analysis():
    path = ROOT / "results" / "selective_v1.json"
    if not path.exists():
        pytest.skip("results/selective_v1.json is not present")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    assert tuple(data["boundaries"]) == borderline.BOUNDARIES
    assert borderline.DEFAULT_MARGIN == borderline.recommended_margin(path)


def test_prompt_format_is_unchanged():
    _, tok = run("{}", margin=0)
    assert tok.prompt == "Screen this resume for a Backend Engineer position and return a structured verdict.\n\nResume: some resume"


def test_valid_json_passes_through_with_the_flag_off():
    verdict = {"ats_score": 60, "verdict": "moderate_match", "matched_skills": ["Python"], "missing_skills": []}
    out, _ = run(json.dumps(verdict), margin=0)
    assert out == verdict


def test_valid_json_gets_flag_fields_when_enabled():
    out, _ = run(json.dumps({"ats_score": 76}), margin=3)
    assert out["ats_score"] == 76
    assert out["borderline"] is True


def test_invalid_json_is_a_flagged_parse_error():
    out, _ = run("not json", margin=3)
    assert out["parse_error"] is True
    assert out["raw_output"] == "not json"
    assert out["borderline"] is True
    assert out["borderline_reason"] == "invalid_output"


def test_json_that_is_not_an_object_is_a_parse_error():
    out, _ = run("[1, 2, 3]", margin=0)
    assert out == {"raw_output": "[1, 2, 3]", "parse_error": True}
