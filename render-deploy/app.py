from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = FastAPI()

model_path = hf_hub_download(
    repo_id="Kus-hal/resume-screener-gguf",
    filename="resume-screener-q4_k_m.gguf"
)

llm = Llama(model_path=model_path, n_ctx=2048, n_threads=2)

class ResumeRequest(BaseModel):
    resume_text: str

@app.post("/score")
def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a Software Engineer role on a scale of 1-10 and explain why: {req.resume_text}"
    response = llm(prompt, max_tokens=200, stop=["<|im_end|>"])
    return {"result": response["choices"][0]["text"]}

@app.get("/")
def health():
    return {"status": "ok"}
