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
        max_tokens=300,
        temperature=0.7,
    )
    result = clean_text(response.choices[0].message.content)
    return {"result": result}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Resume Screener</title>
        <style>
            body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }
            textarea { width: 100%; height: 150px; padding: 10px; font-size: 14px; }
            button { padding: 10px 20px; margin-top: 10px; cursor: pointer; }
            #result { margin-top: 20px; padding: 15px; background: #f4f4f4; border-radius: 6px; white-space: pre-wrap; }
        </style>
    </head>
    <body>
        <h2>Resume Screener</h2>
        <textarea id="resume" placeholder="Paste resume text here..."></textarea><br>
        <button onclick="scoreResume()">Score Resume</button>
        <div id="result"></div>
        <script>
            async function scoreResume() {
                const text = document.getElementById('resume').value;
                const resultDiv = document.getElementById('result');
                resultDiv.textContent = "Scoring... (may take up to 30s if server was asleep)";
                try {
                    const res = await fetch('/score', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ resume_text: text })
                    });
                    const data = await res.json();
                    resultDiv.textContent = data.result || JSON.stringify(data);
                } catch (e) {
                    resultDiv.textContent = "Error: " + e;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok"}
