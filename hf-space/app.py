import json
import re

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER = "Kus-hal/resume-screener-lora"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
model = PeftModel.from_pretrained(_base, ADAPTER)
model.eval()

BACKEND = """Backend Engineer, 5 years of experience.
Skills: Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, REST APIs, AWS.
Experience: Built and scaled payment microservices handling 2M requests/day. Designed REST APIs with FastAPI and PostgreSQL, cut p95 latency by 40% with Redis caching, and ran services on Kubernetes with CI/CD.
Education: B.Tech in Computer Science."""
FRONTEND = """Frontend Developer, 2 years of experience.
Skills: JavaScript, React, HTML, CSS, Git, REST API integration.
Experience: Built responsive marketing pages and an internal admin dashboard in React for a 10-person startup. Fixed cross-browser layout bugs and integrated REST APIs. Raised the Lighthouse performance score of the main landing page from 62 to 85.
Education: B.Tech in Information Technology."""
AI = """AI Engineer, 3 years of experience.
Skills: Python, PyTorch, LangChain, LangGraph, RAG, Qdrant, pgvector, FastAPI, Docker, LLM evaluation, prompt engineering.
Experience: Built a retrieval-augmented support assistant over 50,000 documents using hybrid search and reranking, raising answer accuracy from 71% to 89% on an internal eval set. Fine-tuned a 0.5B model with LoRA to return structured JSON and served it behind a FastAPI service. Built multi-agent workflows with LangGraph and added tracing and automated evals to catch regressions.
Education: B.Tech in Computer Science."""
SOFTWARE = """Recent graduate applying for Software Developer roles.
Skills: Python, Java, SQL, Git, data structures and algorithms, basic HTML and CSS.
Experience: 3-month software internship fixing bugs and writing unit tests in a Java Spring service. Built a library management web app in Flask with SQLite and a command-line expense tracker. Solved 250+ practice problems on competitive programming sites.
Education: B.Tech in Computer Science."""
DEVOPS = """Systems Administrator moving into DevOps, 4 years of experience.
Skills: Linux, Bash, Docker, Jenkins, AWS (EC2, S3), Git, Nagios.
Experience: Administered 40 Linux servers and automated patching and backups with Bash scripts. Containerized three internal tools with Docker. Built Jenkins pipelines for nightly builds and cut manual deploy time from 2 hours to 20 minutes.
Education: B.Sc in Information Technology."""
MISMATCH = """Graphic Designer, 6 years of experience.
Skills: Photoshop, Illustrator, Figma, Brand Identity, Typography.
Experience: Designed branding and print campaigns for retail clients. Led a team of 3 designers on packaging projects.
Education: Bachelor of Fine Arts."""

EXAMPLES = [
    ["Backend Engineer", BACKEND],
    ["Frontend Engineer", FRONTEND],
    ["AI Engineer", AI],
    ["Software Developer", SOFTWARE],
    ["DevOps Engineer", DEVOPS],
    ["Backend Engineer", MISMATCH],
]


def _fmt(items):
    if not isinstance(items, list) or not items:
        return "none"
    return ", ".join(str(i) for i in items)


def screen(role, resume):
    role = (role or "").strip() or "Software Engineer"
    resume = (resume or "").strip()[:6000]
    if not resume:
        return "Please paste some resume text.", None
    prompt = (
        f"Screen this resume for a {role} position and return a structured verdict.\n\n"
        f"Resume: {resume}"
    )
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return "The model did not return JSON.", {"raw_output": text}
    try:
        v = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "The model returned malformed JSON.", {"raw_output": text}
    label = str(v.get("verdict", "?")).replace("_", " ")
    summary = (
        f"### {v.get('ats_score', '?')} / 100 - {label}\n\n"
        f"**Matched skills:** {_fmt(v.get('matched_skills'))}\n\n"
        f"**Missing skills:** {_fmt(v.get('missing_skills'))}\n\n"
        f"**Years of experience:** {v.get('years_experience', '?')}"
    )
    return summary, v


with gr.Blocks(title="Resume Screener - LoRA adapter") as demo:
    gr.Markdown(
        "# Resume Screener: the actual LoRA adapter\n"
        "This Space runs **Qwen2.5-0.5B-Instruct + the trained LoRA adapter** "
        "([Kus-hal/resume-screener-lora](https://huggingface.co/Kus-hal/resume-screener-lora)) on a free CPU, "
        "so expect roughly 20 to 60 seconds per resume. The faster, Groq-hosted demo is "
        "[here](https://resume-screener-finetune.onrender.com/). Scores follow the synthetic training rubric; "
        "evaluation and limitations are in the [GitHub repo](https://github.com/ayush-s-tomar/resume-screener-lora)."
    )
    role = gr.Textbox(label="Target role", value="Backend Engineer")
    resume = gr.Textbox(label="Resume text", lines=10, placeholder="Paste resume text here...")
    btn = gr.Button("Screen resume", variant="primary")
    summary = gr.Markdown()
    raw = gr.JSON(label="Model output (JSON)")
    gr.Examples(examples=EXAMPLES, inputs=[role, resume], label="Examples (click to load)")
    btn.click(screen, [role, resume], [summary, raw])

demo.queue(max_size=8).launch()