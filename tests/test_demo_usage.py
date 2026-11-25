"""Tests for the demo's usage counters and feedback (render-deploy/app.py) and for usage_report.py. The model call is replaced by a fake."""
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("groq")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "render-deploy"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import usage_report  # noqa: E402

SECRET = "SECRET-RESUME-TEXT-123"
HEADERS = {"Content-Type": "application/octet-stream"}


@pytest.fixture(scope="module")
def mod():
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    spec = importlib.util.spec_from_file_location("demo_usage_app", DEMO / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo(mod, monkeypatch):
    def fake_score(role, resume):
        return {"verdict_json": {"role": role, "ats_score": 76, "verdict": "strong_match", "matched_skills": ["Python"],
                                 "missing_skills": [], "years_experience": 4}, "rationale": "ok"}

    monkeypatch.setattr(mod, "client", object())
    monkeypatch.setattr(mod, "_score_uncached", fake_score)
    monkeypatch.setattr(mod, "SUPABASE_URL", "")
    monkeypatch.setattr(mod, "SUPABASE_KEY", "")
    mod.CACHE.clear()
    mod.RATE.clear()
    mod.FEEDBACK_RATE.clear()
    mod.RECENT_EVENTS.clear()
    mod.STATS["days"].clear()
    mod.DAILY["day"], mod.DAILY["count"] = "", 0
    return SimpleNamespace(mod=mod, client=TestClient(mod.app))


def score(demo, text="Python developer, 4 years", role="Backend Engineer"):
    return demo.client.post("/score", json={"resume_text": text, "target_role": role})


def totals(demo):
    return demo.client.get("/stats").json()["totals"]


def test_scoring_is_unchanged_and_counted(demo):
    first, second = score(demo), score(demo)
    assert first.status_code == 200 and first.json()["cached"] is False
    assert first.headers["x-cache"] == "miss" and second.headers["x-cache"] == "hit"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert second.json()["cached"] is True
    t = totals(demo)
    assert t["scores"] == 2 and t["cache_hits"] == 1 and t["visitor_days"] == 1


@pytest.mark.parametrize("role,bucket", [
    ("Backend Engineer", "backend"), ("Front-end developer", "frontend"), ("DevOps Engineer", "devops"), ("Data Engineer", "data"),
    ("AI Engineer", "ai"), ("ML Engineer", "ai"), ("Software Developer", "software"), ("Graphic Designer", "other"), ("", "other"), ("retail email", "other"),
])
def test_role_buckets(mod, role, bucket):
    assert mod.role_bucket(role) == bucket


def test_events_never_contain_the_resume_text_or_the_address(demo, caplog):
    caplog.set_level(logging.INFO, logger="resume_demo")
    score(demo, text="Backend developer. " + SECRET)
    demo.client.post("/feedback", json={"rating": "down", "reason": "too_high", "score": 76, "verdict": "strong_match", "role": "Backend Engineer"})
    blob = json.dumps(list(demo.mod.RECENT_EVENTS)) + json.dumps(demo.client.get("/stats").json()) + caplog.text
    assert SECRET not in blob and "testclient" not in blob and "Backend developer" not in blob
    assert len(demo.mod.RECENT_EVENTS) == 2


def test_feedback_is_validated_and_counted(demo):
    assert demo.client.post("/feedback", json={"rating": "maybe"}).status_code == 422
    assert demo.client.post("/feedback", json={"rating": "down", "reason": "bad"}).status_code == 422
    assert demo.client.post("/feedback", json={"rating": "up"}).json() == {"ok": True}
    assert demo.client.post("/feedback", json={"rating": "down", "reason": "wrong_skills"}).status_code == 200
    t = totals(demo)
    assert t["up"] == 1 and t["down"] == 1


def test_feedback_has_a_rate_limit(demo, monkeypatch):
    monkeypatch.setattr(demo.mod, "FEEDBACK_LIMIT", 2)
    codes = [demo.client.post("/feedback", json={"rating": "up"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_stats_shape(demo):
    score(demo)
    s = demo.client.get("/stats").json()
    assert set(s) == {"since", "durable_storage", "today", "totals", "note"}
    assert s["durable_storage"] is False
    assert set(s["totals"]) == {"scores", "cache_hits", "uploads", "masked", "borderline", "up", "down", "visitor_days"}


def test_events_are_stored_when_a_database_is_configured(demo, monkeypatch):
    sent = []
    monkeypatch.setattr(demo.mod, "SUPABASE_URL", "https://example.invalid")
    monkeypatch.setattr(demo.mod, "SUPABASE_KEY", "anon-key")
    monkeypatch.setattr(demo.mod, "_post_event", lambda event: sent.append(event))
    score(demo)
    for _ in range(100):
        if sent:
            break
        time.sleep(0.02)
    assert sent and sent[0]["kind"] == "score"
    assert demo.client.get("/stats").json()["durable_storage"] is True


def test_stored_rows_only_contain_allowed_columns(mod, monkeypatch):
    captured = {}

    class FakeResponse:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["data"] = json.loads(req.data)
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return FakeResponse()

    monkeypatch.setattr(mod, "SUPABASE_URL", "https://example.invalid")
    monkeypatch.setattr(mod, "SUPABASE_KEY", "anon-key")
    monkeypatch.setattr(mod._urlrequest, "urlopen", fake_urlopen)
    mod._post_event({"kind": "score", "visitor": "abc", "score": 50, "extra": "must not be sent"})
    assert captured["url"] == "https://example.invalid/rest/v1/demo_events"
    assert captured["data"] == {"kind": "score", "visitor": "abc", "score": 50}
    assert captured["headers"]["apikey"] == "anon-key"


def test_page_has_the_feedback_controls(demo):
    html = demo.client.get("/").text
    for marker in ('id="fb"', "sendFeedback", 'id="usage"', "resetFeedback"):
        assert marker in html


def test_rejected_requests_record_nothing(demo):
    r = demo.client.post("/score", json={"resume_text": "a" * 70000, "target_role": "x"})
    assert r.status_code == 413
    assert len(demo.mod.RECENT_EVENTS) == 0


def test_uploads_are_counted(demo):
    if not any(getattr(route, "path", "") == "/extract" for route in demo.mod.app.routes):
        pytest.skip("file upload (Day 8) is not applied")
    try:
        r = demo.client.post("/extract?filename=a.txt", content=b"hello", headers=HEADERS)
    except TypeError:
        r = demo.client.post("/extract?filename=a.txt", data=b"hello", headers=HEADERS)
    assert r.status_code == 200
    assert totals(demo)["uploads"] == 1


ROWS = [
    {"created_at": "2026-11-20T10:00:00Z", "kind": "score", "visitor": "a", "role_bucket": "backend", "score": 76, "cached": False, "borderline": True, "masked": True},
    {"created_at": "2026-11-20T10:05:00Z", "kind": "score", "visitor": "a", "role_bucket": "backend", "score": 76, "cached": True, "borderline": True, "masked": False},
    {"created_at": "2026-11-21T09:00:00Z", "kind": "score", "visitor": "b", "role_bucket": "data", "score": 40, "cached": False, "borderline": False, "masked": False},
    {"created_at": "2026-11-21T09:01:00Z", "kind": "upload", "visitor": "b"},
    {"created_at": "2026-11-21T09:02:00Z", "kind": "feedback", "visitor": "b", "rating": "up"},
    {"created_at": "2026-11-21T09:03:00Z", "kind": "feedback", "visitor": "b", "rating": "down", "reason": "too_high"},
]


def test_report_aggregates():
    a = usage_report.aggregate(ROWS)
    assert a["screenings"] == 3 and a["fresh_screenings"] == 2 and a["uploads"] == 1
    assert a["visitor_days"] == 2
    assert (a["first_day"], a["last_day"], a["active_days"]) == ("2026-11-20", "2026-11-21", 2)
    assert (a["feedback_up"], a["feedback_down"]) == (1, 1) and a["helpful_share"] == 0.5
    assert a["down_reasons"] == {"too_high": 1}
    assert a["roles"] == {"backend": 2, "data": 1}
    assert a["masked_share"] == pytest.approx(1 / 3)
    assert a["borderline_share"] == pytest.approx(2 / 3)


def test_report_block_needs_data_and_reads_well():
    assert usage_report.render_block(usage_report.aggregate([])) is None
    block = usage_report.render_block(usage_report.aggregate(ROWS))
    assert "3 screenings" in block and "2 visitor-days" in block and "1 file upload " in block
    assert block.startswith("<!-- us-start -->") and block.endswith("<!-- us-end -->")
