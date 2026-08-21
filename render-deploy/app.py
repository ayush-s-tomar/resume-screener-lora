import os
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class ResumeRequest(BaseModel):
    resume_text: str

@app.post("/score")
def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a Software Engineer role on a scale of 1-10 and explain why: {req.resume_text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )
    return {"result": response.choices[0].message.content}

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
