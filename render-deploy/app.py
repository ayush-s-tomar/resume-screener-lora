import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections import OrderedDict, deque

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq
from pydantic import BaseModel, Field

try:
    import resume_parser
except ImportError:  # the module was not deployed next to app.py: the demo still works, upload and masking are unavailable
    resume_parser = None

app = FastAPI()

API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if API_KEY else None

MODEL = "openai/gpt-oss-20b"

# Verdict labels and score ranges were read from the training data (data/*.jsonl)
# when this file was generated, so the demo uses the same label set as the adapter.
VERDICTS = ["moderate_match", "weak_match", "strong_match"]
RUBRIC = """- strong_match: ats_score 75-100
- weak_match: ats_score 8-48
- moderate_match: ats_score 50-74"""


class ResumeRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=12000)
    target_role: str = Field(default="Software Engineer", max_length=100)
    redact: bool = False


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    return text


def build_prompt(role: str, resume: str) -> str:
    rubric = ""
    if RUBRIC.strip():
        rubric = "\nScore ranges seen per verdict in the training data:\n" + RUBRIC + "\n"
    return (
        f"Screen this resume for a {role} position and return a structured verdict.\n\n"
        "Respond with ONLY a JSON object with these exact keys: "
        "role (string), ats_score (integer 0-100), "
        f"verdict (one of: {', '.join(VERDICTS)}), "
        "matched_skills (list of skills the resume has that the role needs), "
        "missing_skills (list of skills the role needs that the resume lacks), "
        "years_experience (integer), "
        "rationale (one or two sentences explaining the verdict). "
        "The verdict must agree with ats_score."
        f"{rubric}\n"
        "Do not include any text outside the JSON.\n\n"
        f"Resume: {resume}"
    )


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def as_int(value, lo=None, hi=None):
    try:
        n = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if lo is not None:
        n = max(lo, n)
    if hi is not None:
        n = min(hi, n)
    return n


def as_str_list(value):
    if not isinstance(value, list):
        return []
    items = [clean_text(str(v)).strip() for v in value]
    return [v for v in items if v][:12]


def normalize(data: dict, role: str):
    score = as_int(data.get("ats_score"), 0, 100)
    verdict = str(data.get("verdict", "")).strip().lower().replace(" ", "_")
    if score is None or verdict not in VERDICTS:
        return None
    return {
        "verdict_json": {
            "role": role,
            "ats_score": score,
            "verdict": verdict,
            "matched_skills": as_str_list(data.get("matched_skills")),
            "missing_skills": as_str_list(data.get("missing_skills")),
            "years_experience": as_int(data.get("years_experience"), 0, 60),
        },
        "rationale": clean_text(str(data.get("rationale", ""))).strip()[:600],
    }


def _score_uncached(role, resume):
    """Call the hosted model (up to two attempts) and return the normalized verdict."""
    if client is None:
        raise HTTPException(status_code=503, detail="Scoring backend is not configured (GROQ_API_KEY missing).")
    prompt = build_prompt(role, resume)
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0 if attempt == 0 else 0.4,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Upstream model error: " + type(exc).__name__) from exc
        content = response.choices[0].message.content or ""
        parsed = extract_json(content)
        result = normalize(parsed, role) if isinstance(parsed, dict) else None
        if result:
            return result
    raise HTTPException(status_code=502, detail="The model did not return a valid verdict. Please try again.")


# ---------------------------------------------------------------------------
# Public-demo guards: response cache, per-visitor rate limit and a daily cap, so one curious
# visitor (or a bot) cannot use up the shared Groq quota. State is held in memory, so it resets
# whenever the free Render instance sleeps and wakes. Tune with the DEMO_* environment variables.
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("resume_demo")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


RATE_LIMIT = _env_int("DEMO_RATE_LIMIT", 8)
RATE_WINDOW = _env_int("DEMO_RATE_WINDOW_SECONDS", 600)
DAILY_CAP = _env_int("DEMO_DAILY_CAP", 150)
CACHE_MAX = 256
MAX_BODY_BYTES = 64 * 1024
MAX_UPLOAD_BYTES = 3 * 1024 * 1024
MAX_RESUME_CHARS = 12000
EXTRACT_LIMIT = _env_int("DEMO_EXTRACT_LIMIT", 20)
# Verdict boundaries as in the training data (weak up to 48, moderate 50 to 74, strong 75 and above).
BOUNDARIES = (49.0, 74.5)
# Borderline margin in score points. The default comes from the selective-prediction analysis (docs/SELECTIVE_PREDICTION.md),
# which measured the local adapter, not this hosted model, so the page says so. 0 switches the badge off.
BORDERLINE_MARGIN = _env_float("DEMO_BORDERLINE_MARGIN", 6.0)

CACHE = OrderedDict()
RATE = {}
EXTRACT_RATE = {}
DAILY = {"day": "", "count": 0}
LOCK = threading.Lock()


@app.middleware("http")
async def hardening(request: Request, call_next):
    length = request.headers.get("content-length", "")
    limit = MAX_UPLOAD_BYTES if request.url.path == "/extract" else MAX_BODY_BYTES
    if length.isdigit() and int(length) > limit:
        return JSONResponse({"detail": "Request too large."}, status_code=413)
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    return response


def client_ip(request):
    ip = request.headers.get("cf-connecting-ip")
    if not ip:
        forwarded = request.headers.get("x-forwarded-for", "")
        ip = forwarded.split(",")[0].strip() if forwarded else ""
    return ip or (request.client.host if request.client else "unknown")


def cache_key(role, resume):
    return hashlib.sha256((role.lower() + "\n" + resume).encode("utf-8")).hexdigest()


def check_limits(ip, now):
    """Raise HTTP 429 when this visitor or the whole demo is over budget; otherwise record the call."""
    today = time.strftime("%Y-%m-%d", time.gmtime(now))
    with LOCK:
        if DAILY["day"] != today:
            DAILY["day"], DAILY["count"] = today, 0
        if DAILY["count"] >= DAILY_CAP:
            raise HTTPException(
                status_code=429,
                detail="The shared demo quota for today is used up and resets at midnight UTC. "
                "Repeating an earlier screening still works, or run the adapter locally (see the README).",
            )
        hits = RATE.setdefault(ip, deque())
        while hits and now - hits[0] > RATE_WINDOW:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            retry = int(RATE_WINDOW - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="You have reached the demo limit of %d new screenings per %d minutes. "
                "Please try again in %d seconds. Repeating an earlier screening is instant and does not count."
                % (RATE_LIMIT, max(1, RATE_WINDOW // 60), retry),
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
        DAILY["count"] += 1
        if len(RATE) > 5000:
            for stale in [k for k, v in RATE.items() if not v or now - v[-1] > RATE_WINDOW]:
                RATE.pop(stale, None)


def check_extract_limit(ip, now):
    """Raise HTTP 429 when this visitor has uploaded too many files recently (separate from the scoring limit)."""
    with LOCK:
        hits = EXTRACT_RATE.setdefault(ip, deque())
        while hits and now - hits[0] > RATE_WINDOW:
            hits.popleft()
        if len(hits) >= EXTRACT_LIMIT:
            retry = int(RATE_WINDOW - (now - hits[0])) + 1
            raise HTTPException(
                status_code=429,
                detail="You have reached the demo limit of %d file uploads per %d minutes. "
                "Please try again in %d seconds, or paste the text instead." % (EXTRACT_LIMIT, max(1, RATE_WINDOW // 60), retry),
                headers={"Retry-After": str(retry)},
            )
        hits.append(now)
        if len(EXTRACT_RATE) > 5000:
            for stale in [k for k, v in EXTRACT_RATE.items() if not v or now - v[-1] > RATE_WINDOW]:
                EXTRACT_RATE.pop(stale, None)


def borderline_info(result):
    """Is this score close to a verdict boundary? None when the flag is switched off (DEMO_BORDERLINE_MARGIN=0)."""
    if BORDERLINE_MARGIN <= 0:
        return None
    score = result["verdict_json"]["ats_score"]
    distance = min(abs(score - b) for b in BOUNDARIES)
    return {"is_borderline": distance < BORDERLINE_MARGIN, "distance": round(distance, 1), "margin": BORDERLINE_MARGIN}


def build_response(result, cached, masked):
    body = {**result, "cached": cached}
    info = borderline_info(result)
    if info is not None:
        body["borderline"] = info
    if masked is not None:
        body["masked"] = masked
    return body


async def read_limited(request, limit):
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail="Request too large.")
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/extract")
async def extract_resume_text(request: Request, filename: str = ""):
    """Turn an uploaded PDF, DOCX or text resume into plain text. Files are read in memory and never stored."""
    started = time.time()
    if resume_parser is None:
        raise HTTPException(status_code=503, detail="File upload is not available on this deployment.")
    ip = client_ip(request)
    visitor = hashlib.sha256(ip.encode("utf-8")).hexdigest()[:8]
    check_extract_limit(ip, started)
    data = await read_limited(request, MAX_UPLOAD_BYTES)
    name = (filename or "")[:200]
    try:
        text = await run_in_threadpool(resume_parser.extract_text, data, name, MAX_UPLOAD_BYTES)
    except resume_parser.ResumeParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.info("extract visitor=%s failed=%s", visitor, type(exc).__name__)
        raise HTTPException(status_code=422, detail="Could not read the file.") from exc
    truncated = len(text) > MAX_RESUME_CHARS
    if truncated:
        text = text[:MAX_RESUME_CHARS]
    logger.info("extract visitor=%s ext=%s bytes=%d chars=%d ms=%d", visitor, (os.path.splitext(name)[1].lower() or "none")[:8], len(data), len(text), int((time.time() - started) * 1000))
    return {"text": text, "chars": len(text), "truncated": truncated}


@app.post("/score")
def score_resume(req: ResumeRequest, request: Request, response: Response):
    started = time.time()
    role = clean_text(req.target_role).strip() or "Software Engineer"
    resume = clean_text(req.resume_text).strip()
    if not resume:
        raise HTTPException(status_code=422, detail="Please paste some resume text.")
    masked = None
    if req.redact:
        if resume_parser is None:
            raise HTTPException(status_code=503, detail="Masking is not available on this deployment.")
        resume, masked = resume_parser.redact_pii(resume)
    key = cache_key(role, resume)
    ip = client_ip(request)
    visitor = hashlib.sha256(ip.encode("utf-8")).hexdigest()[:8]

    with LOCK:
        cached = CACHE.get(key)
        if cached is not None:
            CACHE.move_to_end(key)
    if cached is not None:
        response.headers["X-Cache"] = "hit"
        logger.info("score visitor=%s cache=hit resume_chars=%d ms=%d", visitor, len(resume), int((time.time() - started) * 1000))
        return build_response(cached, True, masked)

    if client is None:
        raise HTTPException(status_code=503, detail="Scoring backend is not configured (GROQ_API_KEY missing).")
    check_limits(ip, started)
    result = _score_uncached(role, resume)
    with LOCK:
        CACHE[key] = result
        CACHE.move_to_end(key)
        while len(CACHE) > CACHE_MAX:
            CACHE.popitem(last=False)
    response.headers["X-Cache"] = "miss"
    logger.info("score visitor=%s cache=miss resume_chars=%d ms=%d", visitor, len(resume), int((time.time() - started) * 1000))
    return build_response(result, False, masked)

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Screener</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%236366f1'/%3E%3Cstop offset='100%25' stop-color='%23a78bfa'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='24' height='24' rx='6' fill='url(%23g)'/%3E%3Cpath d='M6 3h9l4 4v13H6V3z' fill='none' stroke='white' stroke-width='1.6'/%3E%3Cpath d='M9 12h6M9 15h6M9 18h3' stroke='white' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background:
            radial-gradient(circle at 20% 0%, rgba(99,102,241,0.10), transparent 40%),
            radial-gradient(circle at 80% 100%, rgba(139,92,246,0.08), transparent 40%),
            #0a0a0f;
        min-height: 100vh;
        padding: 56px 24px;
    }
    .wrap { max-width: 820px; width: 100%; margin: 0 auto; }
    .panel {
        background: #14141f;
        border: 1px solid #26263a;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 18px;
        transition: border-color 0.2s;
    }
    .panel:hover { border-color: #35355a; }
    .header-row { display: flex; align-items: center; gap: 16px; margin-bottom: 18px; }
    .icon-box {
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        flex-shrink: 0; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    }
    .icon-box svg { width: 24px; height: 24px; }
    h1 { font-size: 24px; color: #f1f5f9; font-weight: 700; }
    .desc { color: #9ca3af; font-size: 14.5px; line-height: 1.7; margin-bottom: 20px; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; }
    .badge {
        background: #1f1f30; border: 1px solid #35354d; color: #a5b4fc;
        font-size: 12px; font-weight: 600; padding: 6px 14px; border-radius: 999px;
    }
    details.panel summary {
        cursor: pointer; color: #cbd5e1; font-size: 15px; font-weight: 600;
        list-style: none; display: flex; align-items: center; gap: 8px;
    }
    details.panel summary::-webkit-details-marker { display: none; }
    details.panel summary::before {
        content: '\203A'; display: inline-block; transition: transform 0.2s;
        color: #818cf8; font-size: 17px;
    }
    details.panel[open] summary::before { transform: rotate(90deg); }
    details.panel .how-body { margin-top: 16px; color: #9ca3af; font-size: 14.5px; line-height: 1.8; }
    a { color: #a5b4fc; }
    label {
        display: flex; align-items: center; gap: 8px; color: #cbd5e1;
        font-size: 14.5px; font-weight: 600; margin-bottom: 10px;
    }
    input[type=text], textarea {
        width: 100%; background: #0e0e17; border: 1px solid #2c2c42; border-radius: 10px;
        padding: 13px 16px; color: #e5e7eb; font-size: 14.5px; font-family: inherit;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    input[type=text]:focus, textarea:focus {
        outline: none; border-color: #818cf8; box-shadow: 0 0 0 3px rgba(129,140,248,0.15);
    }
    textarea { min-height: 160px; resize: vertical; }
    .field { margin-bottom: 24px; }
    .examples { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-bottom: 22px; }
    .examples .lbl { color: #9ca3af; font-size: 13px; margin-right: 4px; }
    button {
        width: 100%; padding: 15px; font-size: 15.5px; font-weight: 700; color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6); border: none; border-radius: 10px;
        cursor: pointer; transition: transform 0.15s, box-shadow 0.15s; font-family: inherit;
    }
    button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 10px 24px rgba(99,102,241,0.35); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    button.ex, button.ghost {
        width: auto; padding: 7px 14px; font-size: 12.5px; font-weight: 600; color: #a5b4fc;
        background: #1f1f30; border: 1px solid #35354d; border-radius: 999px;
    }
    button.ex:hover:not(:disabled), button.ghost:hover:not(:disabled) {
        transform: none; box-shadow: none; border-color: #818cf8;
    }
    .hint { color: #6b7280; font-size: 13px; margin-top: 16px; text-align: center; }
    #msg { display: none; margin-top: 18px; padding: 14px 18px; border-radius: 10px; font-size: 14px; }
    #msg.show { display: flex; align-items: center; gap: 10px; }
    #msg.loading { color: #a5b4fc; background: #0e0e17; border: 1px solid #2c2c42; font-weight: 600; }
    #msg.error { color: #fca5a5; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.35); }
    #out { margin-top: 24px; animation: fadeIn 0.4s ease; }
    .score-row { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; }
    .score { font-size: 44px; font-weight: 800; color: #a5b4fc; line-height: 1; }
    .score .of { font-size: 14px; font-weight: 500; color: #9ca3af; margin-left: 6px; }
    .verdict {
        font-size: 12px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase;
        padding: 6px 12px; border-radius: 999px; border: 1px solid;
    }
    .v-good { color: #6ee7b7; background: rgba(16,185,129,0.12); border-color: rgba(16,185,129,0.4); }
    .v-mid { color: #fcd34d; background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.4); }
    .v-bad { color: #fca5a5; background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.4); }
    .meta { color: #9ca3af; font-size: 13px; }
    .bar { height: 8px; background: #1f1f30; border-radius: 999px; margin: 16px 0 22px; overflow: hidden; }
    .bar-fill { height: 100%; width: 0; border-radius: 999px; transition: width 0.6s ease; background: #6366f1; }
    .bar-fill.v-good { background: #10b981; }
    .bar-fill.v-mid { background: #f59e0b; }
    .bar-fill.v-bad { background: #ef4444; }
    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-bottom: 20px; }
    h3.sec { font-size: 13px; color: #cbd5e1; font-weight: 600; margin-bottom: 10px; }
    .chips { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip { font-size: 12.5px; padding: 5px 12px; border-radius: 999px; border: 1px solid; }
    .chip.ok { color: #6ee7b7; background: rgba(16,185,129,0.10); border-color: rgba(16,185,129,0.35); }
    .chip.gap { color: #fcd34d; background: rgba(245,158,11,0.10); border-color: rgba(245,158,11,0.35); }
    .chip.none { color: #6b7280; border-color: #2c2c42; }
    .tag {
        font-size: 11px; color: #818cf8; border: 1px solid #35354d; border-radius: 999px;
        padding: 2px 8px; margin-left: 6px; font-weight: 600;
    }
    #rationale { color: #d1d5db; font-size: 14px; line-height: 1.7; margin-bottom: 20px; }
    details.rawbox summary { cursor: pointer; color: #9ca3af; font-size: 13px; margin-bottom: 10px; }
    pre#raw {
        background: #0e0e17; border: 1px solid #2c2c42; border-radius: 10px; padding: 14px 16px;
        font-size: 12.5px; line-height: 1.6; color: #d1d5db; overflow-x: auto; margin-bottom: 10px;
    }
    .spinner {
        width: 15px; height: 15px; border: 2px solid #2c2c42; border-top-color: #a5b4fc;
        border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    footer { text-align: center; font-size: 12px; color: #4b5563; margin-top: 8px; }
    label.check { font-size: 13px; font-weight: 400; color: #9ca3af; margin: 12px 0 6px; }
    label.check input { width: auto; accent-color: #6366f1; }
    .upload-row { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 12px; }
    .bnote { color: #fcd34d; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.3); border-radius: 10px; padding: 10px 14px; font-size: 13px; line-height: 1.6; margin: -6px 0 18px; }
    .fineprint { color: #6b7280; font-size: 12.5px; line-height: 1.6; margin-top: 6px; }
    @media (max-width: 600px) {
        body { padding: 20px 12px; }
        .panel { padding: 20px; }
        .cols { grid-template-columns: 1fr; }
        .score { font-size: 38px; }
    }
</style>
</head>
<body>
<div class="wrap">
    <div class="panel">
        <div class="header-row">
            <div class="icon-box">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M6 2h9l5 5v15H6V2z" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
                    <path d="M15 2v5h5" stroke="white" stroke-width="1.6" stroke-linejoin="round"/>
                    <path d="M9 12h6M9 15h6M9 18h3" stroke="white" stroke-width="1.6" stroke-linecap="round"/>
                </svg>
            </div>
            <h1>Resume Screener</h1>
        </div>
        <p class="desc">Scores a resume against a target role and returns a structured JSON verdict, the same schema the LoRA fine-tuned model was trained to emit. This demo is served via Groq (gpt-oss-20b).</p>
        <div class="badges">
            <span class="badge">Adapter runs locally</span>
            <span class="badge">Groq API</span>
            <span class="badge">gpt-oss-20b</span>
            <span class="badge">FastAPI Backend</span>
        </div>
    </div>

    <details class="panel">
        <summary>How this model works</summary>
        <div class="how-body">
            The resume and target role are sent to Groq&rsquo;s hosted gpt-oss-20b, prompted to return the JSON schema that a LoRA fine-tuned Qwen2.5-0.5B was trained on: role, ats_score (0-100), verdict, matched_skills, missing_skills and years_experience. The demo adds a short rationale. The fine-tuned adapter itself runs locally with no API key; on 96 held-out examples it reached 88.5% verdict accuracy versus 17.7% for the zero-shot base model (18.8% when that baseline is given the verdict score ranges). This hosted demo uses Groq because Render&rsquo;s free tier (512 MB RAM) cannot run the model. Details are in the <a href="https://github.com/ayush-s-tomar/resume-screener-lora" target="_blank" rel="noopener">GitHub repo</a>.
        </div>
    </details>

    <details class="panel" id="privacy">
        <summary>Privacy</summary>
        <div class="how-body">
            Your text is sent to Groq&rsquo;s API to be scored, so Groq processes it under its own terms; please do not submit real personal data you would not share with a third party. This demo does not store resumes or uploaded files: files are read in memory and discarded. The server keeps only a temporary in-memory cache of scoring results (keyed by a hash of the role and text, without the text), rate counters keyed by IP address, and short log lines (a shortened hash of the visitor&rsquo;s address, the text length and timings). The cache and counters reset whenever the server restarts, and the hosting provider may keep its own request logs. The mask option replaces emails, links, phone numbers and ID numbers (Aadhaar, PAN and US SSN formats) before scoring; it does not mask names or addresses and is not an anonymisation guarantee. This is a research demo and must not be used for real hiring decisions.
        </div>
    </details>

    <div class="panel">
        <div class="examples">
            <span class="lbl">Try an example:</span>
            <button class="ex" type="button" onclick="loadExample('backend')">Backend Engineer</button>
            <button class="ex" type="button" onclick="loadExample('frontend')">Frontend Engineer</button>
            <button class="ex" type="button" onclick="loadExample('ai')">AI Engineer</button>
            <button class="ex" type="button" onclick="loadExample('software')">Software Developer</button>
            <button class="ex" type="button" onclick="loadExample('devops')">DevOps Engineer</button>
            <button class="ex" type="button" onclick="loadExample('mismatch')">Career mismatch</button>
        </div>
        <div class="field">
            <label for="role">&#127919; Target role</label>
            <input type="text" id="role" placeholder="Enter your role" maxlength="100">
        </div>
        <div class="field">
            <label for="resume">&#128196; Resume text</label>
            <div class="upload-row">
                <input type="file" id="file" accept=".pdf,.docx,.txt,.md" hidden>
                <button class="ghost" id="uploadBtn" type="button" onclick="pickFile()">Upload a PDF, DOCX or text file</button>
                <span class="meta" id="fileinfo">or paste the text below (up to 12,000 characters)</span>
            </div>
            <textarea id="resume" placeholder="Paste resume text here..." maxlength="12000"></textarea>
            <label class="check" for="mask"><input type="checkbox" id="mask" checked> Mask emails, links, phone numbers and ID numbers before scoring</label>
            <p class="fineprint">Names and addresses are not masked. The text is sent to Groq&rsquo;s API to be scored; please use fictional or non-sensitive text (see Privacy above).</p>
        </div>
        <button id="btn" type="button" onclick="scoreResume()">Screen resume &rarr;</button>
        <p class="hint" id="hint">Pick a sample resume above (all fictional) or paste your own, then press Screen resume (Ctrl+Enter also works).</p>
        <div id="msg"></div>

        <div id="out" hidden>
            <div class="score-row">
                <div class="score"><span id="scoreNum">0</span><span class="of">/ 100</span></div>
                <span id="verdict" class="verdict"></span>
                <span id="borderline" class="verdict v-mid" hidden>Borderline</span>
                <span id="years" class="meta"></span>
            </div>
            <div class="bar"><div id="barFill" class="bar-fill"></div></div>
            <p id="bnote" class="bnote" hidden></p>
            <div class="cols">
                <div><h3 class="sec">Matched skills</h3><div id="matched" class="chips"></div></div>
                <div><h3 class="sec">Missing skills</h3><div id="missing" class="chips"></div></div>
            </div>
            <h3 class="sec">Rationale <span class="tag">demo only</span></h3>
            <p id="rationale"></p>
            <p id="maskinfo" class="meta" hidden></p>
            <details class="rawbox">
                <summary>Raw JSON (the model&rsquo;s output schema)</summary>
                <pre id="raw"></pre>
                <button class="ghost" id="copyBtn" type="button" onclick="copyRaw()">Copy JSON</button>
            </details>
        </div>
    </div>

    <footer>Demo scored via Groq (gpt-oss-20b). The trained LoRA adapter runs locally, see the <a href="https://github.com/ayush-s-tomar/resume-screener-lora" target="_blank" rel="noopener">GitHub repo</a></footer>
</div>
<script>
const EXAMPLES = {
    backend: {
        role: "Backend Engineer",
        text: [
            "Backend Engineer, 5 years of experience.",
            "Skills: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, REST APIs, AWS.",
            "Experience: Built and scaled payment microservices handling 2M requests/day. Designed REST APIs with FastAPI and PostgreSQL, cut p95 latency by 40% with Redis caching, and ran services on Kubernetes with CI/CD.",
            "Education: B.Tech in Computer Science."
        ].join('\n')
    },
    frontend: {
        role: "Frontend Engineer",
        text: [
            "Frontend Developer, 2 years of experience.",
            "Skills: JavaScript, React, HTML, CSS, Git, REST API integration.",
            "Experience: Built responsive marketing pages and an internal admin dashboard in React for a 10-person startup. Fixed cross-browser layout bugs and integrated REST APIs. Raised the Lighthouse performance score of the main landing page from 62 to 85.",
            "Education: B.Tech in Information Technology."
        ].join('\n')
    },
    ai: {
        role: "AI Engineer",
        text: [
            "AI Engineer, 3 years of experience.",
            "Skills: Python, PyTorch, LangChain, LangGraph, RAG, Qdrant, pgvector, FastAPI, Docker, LLM evaluation, prompt engineering.",
            "Experience: Built a retrieval-augmented support assistant over 50,000 documents using hybrid search and reranking, raising answer accuracy from 71% to 89% on an internal eval set. Fine-tuned a 0.5B model with LoRA to return structured JSON and served it behind a FastAPI service. Built multi-agent workflows with LangGraph and added tracing and automated evals to catch regressions.",
            "Education: B.Tech in Computer Science."
        ].join('\n')
    },
    software: {
        role: "Software Developer",
        text: [
            "Recent graduate applying for Software Developer roles.",
            "Skills: Python, Java, SQL, Git, data structures and algorithms, basic HTML and CSS.",
            "Experience: 3-month software internship fixing bugs and writing unit tests in a Java Spring service. Built a library management web app in Flask with SQLite and a command-line expense tracker. Solved 250+ practice problems on competitive programming sites.",
            "Education: B.Tech in Computer Science."
        ].join('\n')
    },
    devops: {
        role: "DevOps Engineer",
        text: [
            "Systems Administrator moving into DevOps, 4 years of experience.",
            "Skills: Linux, Bash, Docker, Jenkins, AWS (EC2, S3), Git, Nagios.",
            "Experience: Administered 40 Linux servers and automated patching and backups with Bash scripts. Containerized three internal tools with Docker. Built Jenkins pipelines for nightly builds and cut manual deploy time from 2 hours to 20 minutes.",
            "Education: B.Sc in Information Technology."
        ].join('\n')
    },
    mismatch: {
        role: "Backend Engineer",
        text: [
            "Graphic Designer, 6 years of experience.",
            "Skills: Photoshop, Illustrator, Figma, Brand Identity, Typography.",
            "Experience: Designed branding and print campaigns for retail clients. Led a team of 3 designers on packaging projects.",
            "Education: Bachelor of Fine Arts."
        ].join('\n')
    }
};

const $ = (id) => document.getElementById(id);

function loadExample(key) {
    const ex = EXAMPLES[key];
    $('role').value = ex.role;
    $('resume').value = ex.text;
    $('resume').focus();
}

function verdictClass(v) {
    v = String(v).toLowerCase();
    if (/strong|high|good/.test(v)) return 'v-good';
    if (/weak|poor|low|no_|not|reject/.test(v)) return 'v-bad';
    return 'v-mid';
}

function fillChips(id, items, cls) {
    const box = $(id);
    box.textContent = '';
    const list = (items && items.length) ? items : [];
    if (!list.length) {
        const none = document.createElement('span');
        none.className = 'chip none';
        none.textContent = 'None listed';
        box.appendChild(none);
        return;
    }
    list.forEach((s) => {
        const c = document.createElement('span');
        c.className = 'chip ' + cls;
        c.textContent = s;
        box.appendChild(c);
    });
}

function setMsg(kind, text) {
    const m = $('msg');
    m.textContent = '';
    if (!kind) { m.className = ''; return; }
    m.className = 'show ' + kind;
    if (kind === 'loading') {
        const sp = document.createElement('span');
        sp.className = 'spinner';
        m.appendChild(sp);
    }
    m.appendChild(document.createTextNode(text));
}

function render(data) {
    const v = data.verdict_json;
    const cls = verdictClass(v.verdict);
    $('out').hidden = false;
    $('scoreNum').textContent = v.ats_score;
    const verdictEl = $('verdict');
    verdictEl.textContent = v.verdict.replace(/_/g, ' ');
    verdictEl.className = 'verdict ' + cls;
    $('years').textContent = (v.years_experience === null || v.years_experience === undefined) ? '' : v.years_experience + ' yrs experience';
    const fill = $('barFill');
    fill.className = 'bar-fill ' + cls;
    fill.style.width = '0%';
    requestAnimationFrame(() => { fill.style.width = v.ats_score + '%'; });
    fillChips('matched', v.matched_skills, 'ok');
    fillChips('missing', v.missing_skills, 'gap');
    $('rationale').textContent = data.rationale || 'No rationale returned.';
    $('raw').textContent = JSON.stringify(v, null, 2);
    const b = data.borderline;
    const isB = !!(b && b.is_borderline);
    $('borderline').hidden = !isB;
    const note = $('bnote');
    note.hidden = !isB;
    if (isB) {
        note.textContent = 'Borderline: this score is ' + b.distance.toFixed(1) + ' points from the nearest verdict boundary (margin ' + b.margin + '), so a small change in the text or the model could flip the verdict. Review it manually. The margin was measured on the local adapter; this hosted model has not been validated against it.';
    }
    const mk = data.masked;
    const parts = mk ? Object.keys(mk).map((k) => mk[k] + ' ' + k.replace('_', ' ')) : [];
    const mi = $('maskinfo');
    mi.hidden = !mk;
    mi.textContent = mk ? ('Masked before scoring: ' + (parts.join(', ') || 'nothing found') + '.') : '';
}

async function scoreResume() {
    const text = $('resume').value.trim();
    const role = $('role').value.trim() || 'Software Engineer';
    const btn = $('btn');
    $('out').hidden = true;

    if (!text) {
        setMsg('error', 'Please paste some resume text first.');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Screening...';
    setMsg('loading', ' Screening resume (may take up to 30s if the server was asleep)...');

    try {
        const res = await fetch('/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_text: text, target_role: role, redact: $('mask').checked })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed (' + res.status + '). Resume text can be up to 12,000 characters.');
        }
        render(data);
        setMsg('', '');
        $('hint').hidden = true;
    } catch (e) {
        setMsg('error', e.message || 'Something went wrong. Please try again.');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Screen resume \u2192';
    }
}

const MAX_UPLOAD = 3 * 1024 * 1024;

function pickFile() { $('file').click(); }

async function handleFile(file) {
    if (!file) return;
    if (file.size > MAX_UPLOAD) {
        setMsg('error', 'That file is larger than 3 MB.');
        $('file').value = '';
        return;
    }
    $('uploadBtn').disabled = true;
    setMsg('loading', ' Reading ' + file.name + '...');
    try {
        const res = await fetch('/extract?filename=' + encodeURIComponent(file.name), {
            method: 'POST',
            headers: { 'Content-Type': 'application/octet-stream' },
            body: file
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
            throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not read the file (' + res.status + ').');
        }
        $('resume').value = data.text;
        $('fileinfo').textContent = file.name + ': ' + data.chars + ' characters' + (data.truncated ? ' (shortened to the demo limit)' : '') + '. Check the text before scoring.';
        setMsg('', '');
    } catch (e) {
        setMsg('error', e.message || 'Could not read the file.');
    } finally {
        $('uploadBtn').disabled = false;
        $('file').value = '';
    }
}

$('file').addEventListener('change', (e) => handleFile(e.target.files[0]));

async function copyRaw() {
    const b = $('copyBtn');
    try {
        await navigator.clipboard.writeText($('raw').textContent);
        b.textContent = 'Copied';
    } catch (e) {
        b.textContent = 'Copy failed';
    }
    setTimeout(() => { b.textContent = 'Copy JSON'; }, 1500);
}

$('resume').addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') scoreResume();
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok"}