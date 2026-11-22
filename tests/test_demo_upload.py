"""Tests for the demo's file upload, masking option and borderline badge (render-deploy/app.py). The model call is replaced by a fake."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("groq")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "render-deploy"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_resume_parser import make_docx, make_pdf  # noqa: E402

RESUME = "Asha Verma\nasha@example.com +91 98765 43210\nBackend developer, 4 years of Python."


@pytest.fixture(scope="module")
def mod():
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    spec = importlib.util.spec_from_file_location("demo_upload_app", DEMO / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def demo(mod, monkeypatch):
    calls, state = [], {"score": 76}

    def fake_score(role, resume):
        calls.append((role, resume))
        return {"verdict_json": {"role": role, "ats_score": state["score"], "verdict": "strong_match", "matched_skills": ["Python"],
                                 "missing_skills": [], "years_experience": 4}, "rationale": "ok"}

    monkeypatch.setattr(mod, "client", object())
    monkeypatch.setattr(mod, "_score_uncached", fake_score)
    monkeypatch.setattr(mod, "BORDERLINE_MARGIN", 3.0)
    mod.CACHE.clear()
    mod.RATE.clear()
    mod.EXTRACT_RATE.clear()
    mod.DAILY["day"], mod.DAILY["count"] = "", 0
    return SimpleNamespace(mod=mod, client=TestClient(mod.app), calls=calls, state=state)


def score(demo, text=RESUME, **extra):
    return demo.client.post("/score", json={"resume_text": text, "target_role": "Backend Engineer", **extra})


def upload(demo, name, data):
    url = "/extract?filename=" + name
    headers = {"Content-Type": "application/octet-stream"}
    try:
        return demo.client.post(url, content=data, headers=headers)
    except TypeError:
        return demo.client.post(url, data=data, headers=headers)


# ---------- borderline ----------

def test_flag_off_leaves_the_response_unchanged(demo, monkeypatch):
    monkeypatch.setattr(demo.mod, "BORDERLINE_MARGIN", 0.0)
    r = score(demo)
    assert r.status_code == 200
    assert set(r.json()) == {"verdict_json", "rationale", "cached"}


def test_borderline_near_a_boundary_and_on_a_cache_hit(demo):
    first, second = score(demo).json(), score(demo).json()
    assert first["borderline"] == {"is_borderline": True, "distance": 1.5, "margin": 3.0}
    assert second["cached"] is True and second["borderline"] == first["borderline"]


def test_not_borderline_far_from_a_boundary(demo):
    demo.state["score"] = 60
    assert score(demo).json()["borderline"] == {"is_borderline": False, "distance": 11.0, "margin": 3.0}


# ---------- masking ----------

def test_no_masking_by_default(demo):
    body = score(demo).json()
    assert "@" in demo.calls[0][1]
    assert "masked" not in body


def test_masking_happens_before_scoring(demo):
    body = score(demo, redact=True).json()
    sent = demo.calls[0][1]
    assert "[EMAIL]" in sent and "[PHONE]" in sent and "@" not in sent and "98765" not in sent
    assert body["masked"] == {"email": 1, "phone": 1}


def test_cache_key_uses_the_masked_text(demo):
    score(demo, "Contact a@b.com for Python work", redact=True)
    second = score(demo, "Contact c@d.org for Python work", redact=True).json()
    assert len(demo.calls) == 1
    assert second["cached"] is True


# ---------- extraction ----------

def test_extract_text_file(demo):
    r = upload(demo, "resume.txt", "Backend developer, Python".encode("utf-8"))
    assert r.status_code == 200
    assert r.json() == {"text": "Backend developer, Python", "chars": 25, "truncated": False}


def test_extract_docx(demo):
    r = upload(demo, "resume.docx", make_docx(["Asha Verma", "Python developer"], table_rows=[["Skills", "FastAPI"]]))
    assert r.status_code == 200
    assert r.json()["text"] == "Asha Verma\nPython developer\nSkills | FastAPI"


def test_extract_pdf(demo):
    pytest.importorskip("pypdf")
    r = upload(demo, "resume.pdf", make_pdf("Python developer with 4 years of experience"))
    assert r.status_code == 200
    assert "Python developer with 4 years of experience" in r.json()["text"]


def test_extract_rejects_unsupported_and_empty_files(demo):
    r = upload(demo, "resume.exe", b"x")
    assert r.status_code == 422 and "Unsupported" in r.json()["detail"]
    r = upload(demo, "resume.txt", b"")
    assert r.status_code == 422 and "empty" in r.json()["detail"]


def test_extract_too_large(demo):
    r = upload(demo, "resume.txt", b"a" * (demo.mod.MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413


def test_extract_truncates_long_text(demo):
    r = upload(demo, "resume.txt", b"a" * 13000).json()
    assert r["truncated"] is True and r["chars"] == 12000 and len(r["text"]) == 12000


def test_extract_has_its_own_rate_limit(demo, monkeypatch):
    monkeypatch.setattr(demo.mod, "EXTRACT_LIMIT", 2)
    assert upload(demo, "a.txt", b"hello").status_code == 200
    assert upload(demo, "a.txt", b"hello").status_code == 200
    r = upload(demo, "a.txt", b"hello")
    assert r.status_code == 429 and "retry-after" in r.headers


def test_scoring_body_limit_is_still_64_kb(demo):
    r = demo.client.post("/score", json={"resume_text": "a" * 70000, "target_role": "Backend Engineer"})
    assert r.status_code == 413


# ---------- deployment hygiene ----------

def test_vendored_parser_matches_the_root_copy():
    a = (ROOT / "resume_parser.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    b = (DEMO / "resume_parser.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    assert a == b, "render-deploy/resume_parser.py is out of date: copy resume_parser.py into render-deploy/"


def test_page_has_the_new_controls(demo):
    html = demo.client.get("/").text
    for marker in ('id="file"', 'id="mask"', 'id="borderline"', 'id="bnote"', 'id="privacy"', "/extract?filename="):
        assert marker in html
