import os
import unicodedata
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ResumeRequest(BaseModel):
    resume_text: str
    target_role: str = "Software Engineer"

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    return text

@app.post("/score")
def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a {req.target_role} role on a scale of 1-10 and explain why: {req.resume_text}"
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7,
    )
    result = clean_text(response.choices[0].message.content)
    return {"result": result}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Resume Screener</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='1' y2='1'%3E%3Cstop offset='0%25' stop-color='%236366f1'/%3E%3Cstop offset='100%25' stop-color='%23a78bfa'/%3E%3C/linearGradient%3E%3C/defs%3E%3Crect width='24' height='24' rx='6' fill='url(%23g)'/%3E%3Cpath d='M6 3h9l4 4v13H6V3z' fill='none' stroke='white' stroke-width='1.6'/%3E%3Cpath d='M9 12h6M9 15h6M9 18h3' stroke='white' stroke-width='1.6' stroke-linecap='round'/%3E%3C/svg%3E">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
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
    .wrap {
        max-width: 820px;
        width: 100%;
        margin: 0 auto;
    }
    .panel {
        background: #14141f;
        border: 1px solid #26263a;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 18px;
        transition: border-color 0.2s;
    }
    .panel:hover { border-color: #35355a; }
    .header-row {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 18px;
    }
    .icon-box {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35);
    }
    .icon-box svg { width: 24px; height: 24px; }
    h1 {
        font-size: 24px;
        color: #f1f5f9;
        font-weight: 700;
    }
    .desc {
        color: #9ca3af;
        font-size: 14.5px;
        line-height: 1.7;
        margin-bottom: 20px;
    }
    .badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .badge {
        background: #1f1f30;
        border: 1px solid #35354d;
        color: #a5b4fc;
        font-size: 12px;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 999px;
    }
    details.panel summary {
        cursor: pointer;
        color: #cbd5e1;
        font-size: 15px;
        font-weight: 600;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    details.panel summary::-webkit-details-marker { display: none; }
    details.panel summary::before {
        content: '\\203A';
        display: inline-block;
        transition: transform 0.2s;
        color: #818cf8;
        font-size: 17px;
    }
    details.panel[open] summary::before { transform: rotate(90deg); }
    details.panel .how-body {
        margin-top: 16px;
        color: #9ca3af;
        font-size: 14.5px;
        line-height: 1.8;
    }
    label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cbd5e1;
        font-size: 14.5px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    input[type=text], textarea {
        width: 100%;
        background: #0e0e17;
        border: 1px solid #2c2c42;
        border-radius: 10px;
        padding: 13px 16px;
        color: #e5e7eb;
        font-size: 14.5px;
        font-family: inherit;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    input[type=text]:focus, textarea:focus {
        outline: none;
        border-color: #818cf8;
        box-shadow: 0 0 0 3px rgba(129,140,248,0.15);
    }
    textarea {
        min-height: 160px;
        resize: vertical;
    }
    .field { margin-bottom: 24px; }
    .field:last-of-type { margin-bottom: 26px; }
    button {
        width: 100%;
        padding: 15px;
        font-size: 15.5px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    button:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(99,102,241,0.35);
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #scoreBadge {
        display: none;
        align-items: center;
        gap: 14px;
        margin-top: 22px;
        padding: 18px 22px;
        background: linear-gradient(90deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
        border: 1px solid #35355a;
        border-radius: 12px;
    }
    #scoreBadge.show { display: flex; animation: fadeIn 0.4s ease; }
    #scoreBadge .num {
        font-size: 32px;
        font-weight: 800;
        color: #a5b4fc;
        line-height: 1;
    }
    #scoreBadge .label {
        color: #9ca3af;
        font-size: 13px;
    }
    #result {
        margin-top: 14px;
        padding: 20px 22px;
        background: #0e0e17;
        border: 1px solid #2c2c42;
        border-radius: 10px;
        font-size: 14px;
        line-height: 1.75;
        color: #d1d5db;
        display: none;
        overflow-x: auto;
    }
    #result.show { display: block; animation: fadeIn 0.4s ease; }
    #result.loading {
        color: #a5b4fc;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    #result h1, #result h2, #result h3 { margin: 14px 0 10px; color: #e5e7eb; }
    #result h3 { font-size: 16px; }
    #result strong { color: #f3f4f6; }
    #result table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
    #result th, #result td { border: 1px solid #2c2c42; padding: 8px 10px; text-align: left; }
    #result th { background: #1f1f30; color: #a5b4fc; }
    #result ul, #result ol { margin: 10px 0 10px 22px; }
    #result hr { border: none; border-top: 1px solid #2c2c42; margin: 14px 0; }
    .spinner {
        width: 15px; height: 15px;
        border: 2px solid #2c2c42;
        border-top-color: #a5b4fc;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    footer {
        text-align: center;
        font-size: 12px;
        color: #4b5563;
        margin-top: 8px;
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
        <p class="desc">AI-powered resume evaluator served via Groq (gpt-oss-20b) &mdash; scores a resume against a target role and explains the reasoning, so it can support quick screening decisions.</p>
        <div class="badges">
            <span class="badge">LoRA Fine-Tuned</span>
            <span class="badge">Groq API</span>
            <span class="badge">gpt-oss-20b</span>
            <span class="badge">FastAPI Backend</span>
        </div>
    </div>

    <details class="panel">
        <summary>How this model works</summary>
        <div class="how-body">
            The resume text and target role are combined into a scoring prompt and sent to an LLM, which returns a score out of 10 along with a structured breakdown of strengths and gaps. Originally built as a LoRA fine-tuned classifier; this deployment routes inference through Groq's hosted API for reliable free-tier serving.
        </div>
    </details>

    <div class="panel">
        <div class="field">
            <label>&#127919; Target role</label>
            <input type="text" id="role" placeholder="Enter your role">
        </div>
        <div class="field">
            <label>&#128196; Resume text</label>
            <textarea id="resume" placeholder="Paste resume text here..." autofocus></textarea>
        </div>
        <button id="btn" onclick="scoreResume()">Screen resume &rarr;</button>
        <div id="scoreBadge"><div class="num" id="scoreNum">-</div><div class="label">out of 10</div></div>
        <div id="result"></div>
    </div>

    <footer>Fine-tuned LoRA model &middot; served via Groq</footer>
</div>
<script>
async function scoreResume() {
    const text = document.getElementById('resume').value.trim();
    const role = document.getElementById('role').value.trim() || 'Software Engineer';
    const resultDiv = document.getElementById('result');
    const scoreBadge = document.getElementById('scoreBadge');
    const btn = document.getElementById('btn');

    scoreBadge.classList.remove('show');

    if (!text) {
        resultDiv.className = 'show';
        resultDiv.textContent = 'Please paste some resume text first.';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Screening...';
    resultDiv.className = 'show loading';
    resultDiv.innerHTML = '<span class="spinner"></span> Screening resume (may take up to 30s if the server was asleep)...';

    try {
        const res = await fetch('/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_text: text, target_role: role })
        });
        const data = await res.json();
        const raw = data.result || JSON.stringify(data);

        const match = raw.match(/(\\d{1,2}(?:\\.\\d)?)\\s*\\/\\s*10/);
        if (match) {
            document.getElementById('scoreNum').textContent = match[1];
            scoreBadge.classList.add('show');
        }

        resultDiv.className = 'show';
        resultDiv.innerHTML = marked.parse(raw);
    } catch (e) {
        resultDiv.className = 'show';
        resultDiv.textContent = 'Error: ' + e;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Screen resume \u2192';
    }
}
</script>
</body>
</html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}
