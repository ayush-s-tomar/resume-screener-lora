import importlib.util
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOOD = '{"ats_score": 82, "verdict": "strong_match", "matched_skills": ["Python"], "missing_skills": [], "years_experience": 4, "rationale": "Good fit."}'


@pytest.fixture(scope="module")
def demo():
    pytest.importorskip("groq")
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location("demo_app", ROOT / "render-deploy" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_state(demo):
    demo.CACHE.clear()
    demo.RATE.clear()
    demo.DAILY.update(day="", count=0)


def fake_client(content, calls=None):
    message = types.SimpleNamespace(content=content)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return response

    completions = types.SimpleNamespace(create=create)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


def api(demo):
    from fastapi.testclient import TestClient

    return TestClient(demo.app)


def post_score(demo, resume="Python, Docker, 4 years.", role="Backend Engineer"):
    return api(demo).post("/score", json={"resume_text": resume, "target_role": role})


def test_health_endpoint_and_security_headers(demo):
    response = api(demo).get("/health")
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_home_page_has_the_six_example_buttons(demo):
    html = api(demo).get("/").text
    for label in ("Backend Engineer", "Frontend Engineer", "AI Engineer", "Software Developer", "DevOps Engineer", "Career mismatch"):
        assert label in html


def test_score_without_an_api_key_returns_503(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", None)
    assert post_score(demo).status_code == 503


def test_score_rejects_empty_and_blank_resumes(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client(GOOD))
    assert post_score(demo, resume="").status_code == 422
    assert post_score(demo, resume="   ").status_code == 422


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
    assert body["cached"] is False


def test_score_returns_502_when_the_model_returns_no_json(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client("no json here"))
    assert post_score(demo).status_code == 502


def test_identical_requests_are_served_from_the_cache(demo, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "client", fake_client(GOOD, calls))
    first, second = post_score(demo), post_score(demo)
    assert first.headers["x-cache"] == "miss" and first.json()["cached"] is False
    assert second.headers["x-cache"] == "hit" and second.json()["cached"] is True
    assert second.json()["verdict_json"] == first.json()["verdict_json"]
    assert len(calls) == 1


def test_a_different_role_is_not_a_cache_hit(demo, monkeypatch):
    calls = []
    monkeypatch.setattr(demo, "client", fake_client(GOOD, calls))
    post_score(demo, role="Backend Engineer")
    post_score(demo, role="Frontend Engineer")
    assert len(calls) == 2


def test_rate_limit_returns_429_with_retry_after(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client(GOOD))
    monkeypatch.setattr(demo, "RATE_LIMIT", 2)
    assert post_score(demo, resume="resume one").status_code == 200
    assert post_score(demo, resume="resume two").status_code == 200
    blocked = post_score(demo, resume="resume three")
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_cached_requests_do_not_count_against_the_rate_limit(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client(GOOD))
    monkeypatch.setattr(demo, "RATE_LIMIT", 1)
    assert all(post_score(demo).status_code == 200 for _ in range(4))


def test_daily_cap_returns_429(demo, monkeypatch):
    monkeypatch.setattr(demo, "client", fake_client(GOOD))
    monkeypatch.setattr(demo, "DAILY_CAP", 1)
    assert post_score(demo, resume="first resume").status_code == 200
    blocked = post_score(demo, resume="second resume")
    assert blocked.status_code == 429
    assert "quota" in blocked.json()["detail"]


def test_oversized_bodies_are_rejected_with_413(demo):
    response = api(demo).post("/score", json={"resume_text": "a" * 70000, "target_role": "x"})
    assert response.status_code == 413


def test_client_ip_prefers_the_cdn_header(demo):
    request = types.SimpleNamespace(headers={"cf-connecting-ip": "1.2.3.4", "x-forwarded-for": "9.9.9.9"}, client=None)
    assert demo.client_ip(request) == "1.2.3.4"
    request = types.SimpleNamespace(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1"}, client=None)
    assert demo.client_ip(request) == "9.9.9.9"