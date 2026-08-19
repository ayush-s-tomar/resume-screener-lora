"""
FastAPI backend for the LoRA fine-tuned resume screener.
Loads Qwen2.5-0.5B-Instruct base model + LoRA adapter (merged at load time),
returns a structured, validated JSON verdict via a small JSON API.
The static/index.html frontend calls POST /api/screen.

Reliability note: the raw model output (score, verdict, matched/missing skills) is
NOT trusted as-is. It is validated and recomputed deterministically below, because
the fine-tuned model (0.5B params, 800 training examples) can hallucinate skill
names not present in the input and can produce a score/verdict that contradicts
its own skill list on out-of-distribution phrasing.
"""

import os
import re
import json
import gc

import torch
torch.set_num_threads(1)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

app = FastAPI(title="Resume Screener API")

_model = None
_tokenizer = None


def get_model():
    global _model, _tokenizer
    if _model is None:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, token=HF_TOKEN)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="cpu",
            token=HF_TOKEN,
        )
        peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = peft_model.merge_and_unload()
        del peft_model, base_model
        model.eval()
        gc.collect()
        _model, _tokenizer = model, tokenizer
    return _model, _tokenizer


def find_list_by_keyword(d: dict, keyword: str) -> list:
    for k, v in d.items():
        if keyword in k.lower() and isinstance(v, list):
            return v
    return []


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def validate_and_score(verdict: dict, resume_text: str) -> dict:
    resume_norm = normalize(resume_text)

    raw_matched = find_list_by_keyword(verdict, "match")
    raw_missing = find_list_by_keyword(verdict, "missing")

    matched = [s for s in raw_matched if normalize(s) in resume_norm]

    def looks_real(skill: str) -> bool:
        s = skill.strip()
        if len(s) < 2:
            return False
        if not re.search(r"[aeiouAEIOU]", s):
            return False
        return True

    missing = [
        s for s in raw_missing
        if normalize(s) not in resume_norm and looks_real(s)
    ]

    total = len(matched) + len(missing)
    match_ratio = (len(matched) / total) if total > 0 else 0.0
    score = round(match_ratio * 100)

    if score >= 70:
        verdict_label = "strong_match"
    elif score >= 45:
        verdict_label = "moderate_match"
    else:
        verdict_label = "weak_match"

    years = verdict.get("years_experience", None)
    if not isinstance(years, (int, float)) or years < 0 or years > 60:
        years = None

    return {
        "ats_score": score,
        "verdict": verdict_label,
        "years_experience": years,
        "matched_skills": matched,
        "missing_skills": missing,
    }


def screen_resume(resume_text: str, role: str, retry: bool = True) -> dict:
    model, tokenizer = get_model()

    prompt = f"Screen this resume for a {role} position and return a structured verdict."
    if retry:
        prompt += " Return ONLY a single valid JSON object, no other text, no truncation."
    prompt += f"\n\nResume: {resume_text}"

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs
    input_ids = input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens=300, do_sample=False)
    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        if retry:
            return screen_resume(resume_text, role, retry=False)
        return {"raw_output": raw, "parse_error": True}

    return validate_and_score(parsed, resume_text)


class ScreenRequest(BaseModel):
    role: str
    resume_text: str


@app.post("/api/screen")
def api_screen(req: ScreenRequest):
    return screen_resume(req.resume_text, req.role)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
