import json
import os
import re
import unicodedata

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from groq import Groq
from pydantic import BaseModel, Field

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


@app.post("/score")
def score_resume(req: ResumeRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Scoring backend is not configured (GROQ_API_KEY missing).")
    role = clean_text(req.target_role).strip() or "Software Engineer"
    prompt = build_prompt(role, clean_text(req.resume_text))
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
            <textarea id="resume" placeholder="Paste resume text here..." maxlength="12000"></textarea>
        </div>
        <button id="btn" type="button" onclick="scoreResume()">Screen resume &rarr;</button>
        <p class="hint" id="hint">Pick a sample resume above (all fictional) or paste your own, then press Screen resume (Ctrl+Enter also works).</p>
        <div id="msg"></div>

        <div id="out" hidden>
            <div class="score-row">
                <div class="score"><span id="scoreNum">0</span><span class="of">/ 100</span></div>
                <span id="verdict" class="verdict"></span>
                <span id="years" class="meta"></span>
            </div>
            <div class="bar"><div id="barFill" class="bar-fill"></div></div>
            <div class="cols">
                <div><h3 class="sec">Matched skills</h3><div id="matched" class="chips"></div></div>
                <div><h3 class="sec">Missing skills</h3><div id="missing" class="chips"></div></div>
            </div>
            <h3 class="sec">Rationale <span class="tag">demo only</span></h3>
            <p id="rationale"></p>
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
            body: JSON.stringify({ resume_text: text, target_role: role })
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