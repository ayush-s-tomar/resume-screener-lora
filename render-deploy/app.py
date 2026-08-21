import sys
import traceback
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
    try:
        response = llm(prompt, max_tokens=120, stop=["<|im_end|>"])
        return response["choices"][0]["text"]
    except Exception:
        print("INFERENCE ERROR:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise

@app.post("/score")
async def score_resume(req: ResumeRequest):
    prompt = f"Score this resume for a Software Engineer role on a scale of 1-10 and explain why: {req.resume_text}"
    try:
        result = await run_in_threadpool(run_inference, prompt)
        return {"result": result}
    except Exception as e:
        print(f"ENDPOINT ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"error": str(e)}, 500

@app.api_route("/", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}
