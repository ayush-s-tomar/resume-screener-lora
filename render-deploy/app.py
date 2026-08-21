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
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: #0a0a0f;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
    }
    .wrap {
        max-width: 640px;
        width: 100%;
    }
    .panel {
        background: #14141f;
        border: 1px solid #26263a;
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 16px;
    }
    .header-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 14px;
    }
    .icon-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: linear-gradient(135deg, #818cf8, #a78bfa);
        flex-shrink: 0;
    }
    h1 {
        font-size: 22px;
        color: #e5e7eb;
        font-weight: 700;
    }
    .desc {
        color: #9ca3af;
        font-size: 13.5px;
        line-height: 1.6;
        margin-bottom: 16px;
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
        font-size: 11.5px;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 999px;
    }
    details.panel summary {
        cursor: pointer;
        color: #cbd5e1;
        font-size: 14px;
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
        font-size: 16px;
    }
    details.panel[open] summary::before { transform: rotate(90deg); }
    details.panel .how-body {
        margin-top: 14px;
        color: #9ca3af;
        font-size: 13.5px;
        line-height: 1.7;
    }
    label {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #cbd5e1;
        font-size: 13.5px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    input[type=text], textarea {
        width: 100%;
        background: #0e0e17;
        border: 1px solid #2c2c42;
        border-radius: 10px;
        padding: 12px 14px;
        color: #e5e7eb;
        font-size: 14px;
        font-family: inherit;
    }
    input[type=text]:focus, textarea:focus {
        outline: none;
        border-color: #818cf8;
    }
    textarea {
        min-height: 140px;
        resize: vertical;
        margin-top: 4px;
    }
    .field { margin-bottom: 18px; }
    .field input { margin-top: 4px; }
    button {
        width: 100%;
        padding: 14px;
        font-size: 15px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #ef4444, #f87171);
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    button:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(239,68,68,0.3);
    }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #result {
        margin-top: 18px;
        padding: 16px 18px;
        background: #0e0e17;
        border: 1px solid #2c2c42;
        border-radius: 10px;
        font-size: 13.5px;
        line-height: 1.65;
        color: #d1d5db;
        display: none;
        overflow-x: auto;
    }
    #result.show { display: block; }
    #result.loading {
        color: #a5b4fc;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    #result h1, #result h2, #result h3 { margin: 12px 0 8px; color: #e5e7eb; }
    #result h3 { font-size: 15px; }
    #result strong { color: #f3f4f6; }
    #result table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 12.5px; }
    #result th, #result td { border: 1px solid #2c2c42; padding: 7px 9px; text-align: left; }
    #result th { background: #1f1f30; color: #a5b4fc; }
    #result ul, #result ol { margin: 8px 0 8px 20px; }
    #result hr { border: none; border-top: 1px solid #2c2c42; margin: 12px 0; }
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
        font-size: 11.5px;
        color: #4b5563;
        margin-top: 4px;
    }
</style>
</head>
<body>
<div class="wrap">
    <div class="panel">
        <div class="header-row">
            <div class="icon-box"></div>
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
            <input type="text" id="role" value="Software Engineer">
        </div>
        <div class="field">
            <label>&#128196; Resume text</label>
            <textarea id="resume" placeholder="Paste resume text here..."></textarea>
        </div>
        <button id="btn" onclick="scoreResume()">Screen resume &rarr;</button>
        <div id="result"></div>
    </div>

    <footer>Fine-tuned LoRA model &middot; served via Groq</footer>
</div>
<script>
async function scoreResume() {
    const text = document.getElementById('resume').value.trim();
    const role = document.getElementById('role').value.trim() || 'Software Engineer';
    const resultDiv = document.getElementById('result');
    const btn = document.getElementById('btn');

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
        resultDiv.className = 'show';
        resultDiv.innerHTML = data.result ? marked.parse(data.result) : JSON.stringify(data);
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
