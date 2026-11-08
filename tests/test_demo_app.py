import importlib.util
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def demo():
    pytest.importorskip("groq")
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "render-deploy" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_client(content):
    message = types.SimpleNamespace(content=content)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    completions = types.SimpleNamespace(create=lambda **kwargs: response)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def api(demo):
    from fastapi.testclient import TestClient

    return TestClient(demo.app)


def post_score(demo, resume="Python, Docker, 4 years.", role="Backend Engineer"):
    return api(demo).post("/score", json={"resume_text": resume, "target_role": role})


def test_health_endpoint(demo):
    assert api(demo).get("/health").json() == {"status": "ok"}


def test_home_page_has_the_six_example_buttons(demo):
    html = api(demo).get("/").text
    for label in ("Backend Engineer", "Frontend Engineer", "AI Engineer", "Software Developer", "DevOps Engineer", "Career mismatch"):
        assert label in html


def test_score_without_an_api_key_returns_503(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", None)
    assert post_score(demo).status_code == 503


def test_score_rejects_an_empty_resume(demo):
    assert post_score(demo, resume="").status_code == 422


def test_score_normalizes_messy_model_output(demo, monkeypatch):
    content = 'Here you go: {"ats_score": 140, "verdict": "Strong Match", "matched_skills": ["Python", ""], "missing_skills": "none", "years_experience": 5.4, "rationale": "Good fit."}'
    monkeypatch.setattr(demo, "client", fake_client(content))
    response = post_score(demo)
    assert response.status_code == 200
    body = response.json()
    verdict = body["verdict_json"]
    assert verdict["ats_score"] == 100
    assert verdict["verdict"] == "strong_match"
    assert verdict["matched_skills"] == ["Python"]
    assert verdict["missing_skills"] == []
    assert verdict["years_experience"] == 5
    assert body["rationale"] == "Good fit."


def test_score_returns_502_when_the_model_returns_no_json(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client("no json here"))
    assert post_score(demo).status_code == 502