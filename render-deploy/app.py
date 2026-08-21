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

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u202f", " ").replace("\u00a0", " ")
    return text

@app.post("/score")
def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a Software Engineer role on a scale of 1-10 and explain why: {req.resume_text}"
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
<title>Resume Screener AI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/9.1.6/marked.min.js"></script>
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 24px;
    }
    .card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.35);
        max-width: 680px;
        width: 100%;
        padding: 40px;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        font-size: 12px;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 12px;
        letter-spacing: 0.5px;
    }
    h1 {
        font-size: 28px;
        color: #1e1b4b;
        margin-bottom: 6px;
        font-weight: 800;
    }
    .subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 24px;
    }
    textarea {
        width: 100%;
        min-height: 160px;
        padding: 16px;
        font-size: 14px;
        font-family: 'Segoe UI', sans-serif;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        resize: vertical;
        transition: border-color 0.2s;
        color: #1e293b;
    }
    textarea:focus {
        outline: none;
        border-color: #6366f1;
    }
    button {
        width: 100%;
        margin-top: 16px;
        padding: 14px;
        font-size: 15px;
        font-weight: 700;
        color: white;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        border: none;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    button:hover:not(:disabled) {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(99,102,241,0.35);
    }
    button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
    #result {
        margin-top: 20px;
        padding: 18px 22px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.65;
        color: #1e293b;
        display: none;
        overflow-x: auto;
    }
    #result.show { display: block; }
    #result.loading {
        color: #6366f1;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    #result h1, #result h2, #result h3 {
        margin: 14px 0 8px;
        color: #1e1b4b;
    }
    #result h3 { font-size: 16px; }
    #result strong { color: #1e1b4b; }
    #result table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 13px;
    }
    #result th, #result td {
        border: 1px solid #e2e8f0;
        padding: 8px 10px;
        text-align: left;
    }
    #result th {
        background: #eef2ff;
        color: #4338ca;
    }
    #result ul, #result ol {
        margin: 8px 0 8px 20px;
    }
    #result hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 14px 0;
    }
    .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid #e2e8f0;
        border-top-color: #6366f1;
        border-radius: 50%;
        animation: spin 0.7s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    footer {
        margin-top: 24px;
        text-align: center;
        font-size: 12px;
        color: #94a3b8;
    }
</style>
</head>
<body>
    <div class="card">
        <span class="badge">AI POWERED</span>
        <h1>Resume Screener</h1>
        <p class="subtitle">Paste a resume below and get an instant AI-generated score out of 10, with reasoning.</p>
        <textarea id="resume" placeholder="Paste resume text here..."></textarea>
        <button id="btn" onclick="scoreResume()">Score Resume</button>
        <div id="result"></div>
        <footer>Fine-tuned LoRA model &middot; served via Groq</footer>
    </div>
<script>
async function scoreResume() {
    const text = document.getElementById('resume').value.trim();
    const resultDiv = document.getElementById('result');
    const btn = document.getElementById('btn');

    if (!text) {
        resultDiv.className = 'show';
        resultDiv.textContent = 'Please paste some resume text first.';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Scoring...';
    resultDiv.className = 'show loading';
    resultDiv.innerHTML = '<span class="spinner"></span> Scoring resume (may take up to 30s if the server was asleep)...';

    try {
        const res = await fetch('/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ resume_text: text })
        });
        const data = await res.json();
        resultDiv.className = 'show';
        resultDiv.innerHTML = data.result ? marked.parse(data.result) : JSON.stringify(data);
    } catch (e) {
        resultDiv.className = 'show';
        resultDiv.textContent = 'Error: ' + e;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Score Resume';
    }
}
</script>
</body>
</html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}
