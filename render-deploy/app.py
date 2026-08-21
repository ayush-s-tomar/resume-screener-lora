from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

app = FastAPI()

model_path = hf_hub_download(
    repo_id="Kus-hal/resume-screener-gguf",
    filename="resume-screener-q4_k_m.gguf"
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=2,
    n_batch=64,
    n_ubatch=64,
)

class ResumeRequest(BaseModel):
    resume_text: str

def run_inference(prompt: str):
    response = llm(prompt, max_tokens=120, stop=["<|im_end|>"])
    return response["choices"][0]["text"]

@app.post("/score")
async def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a Software Engineer role on a scale of 1-10 and explain why: {req.resume_text}"
    result = await run_in_threadpool(run_inference, prompt)
    return {"result": result}

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
