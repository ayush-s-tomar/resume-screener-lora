"""
Streamlit UI for the LoRA fine-tuned resume screener.
Loads Qwen2.5-0.5B-Instruct base model + LoRA adapter, returns a structured JSON verdict.
"""

import json
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./resume-screener-lora-adapter"

st.set_page_config(page_title="Resume Screener · LoRA fine-tuned", page_icon="📊", layout="centered")

HF_TOKEN = st.secrets.get("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a1f2e 0%, #0e1117 45%);
    }

    .hero {
        padding: 2.2rem 2rem 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(99,102,241,0.18) 0%, rgba(56,189,248,0.10) 100%);
        border: 1px solid rgba(148,163,255,0.18);
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        font-size: 2rem;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #a5b4fc, #67e8f9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0;
    }
    .badge-row { margin-top: 0.9rem; }
    .badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #c7d2fe;
        background: rgba(99,102,241,0.16);
        border: 1px solid rgba(129,140,248,0.35);
        padding: 3px 10px;
        border-radius: 999px;
        margin-right: 6px;
    }

    .verdict-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(16,185,129,0.04));
        border: 1px solid rgba(52,211,153,0.28);
        border-radius: 16px;
        padding: 1.6rem 1.7rem;
        margin-top: 1rem;
    }
    .verdict-card.moderate {
        background: linear-gradient(135deg, rgba(234,179,8,0.10), rgba(202,138,4,0.04));
        border: 1px solid rgba(250,204,21,0.28);
    }
    .verdict-card.weak {
        background: linear-gradient(135deg, rgba(239,68,68,0.10), rgba(220,38,38,0.04));
        border: 1px solid rgba(248,113,113,0.28);
    }

    .metric-row { display: flex; gap: 1rem; margin-bottom: 1.1rem; }
    .metric-box {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-box .label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }

    .score-bar-track {
        width: 100%;
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.4rem;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #34d399, #22d3ee);
    }

    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.4rem; }
    .chip {
        font-size: 0.78rem;
        padding: 4px 11px;
        border-radius: 999px;
        font-weight: 500;
    }
    .chip.matched {
        background: rgba(52,211,153,0.14);
        color: #6ee7b7;
        border: 1px solid rgba(52,211,153,0.3);
    }
    .chip.missing {
        background: rgba(248,113,113,0.12);
        color: #fca5a5;
        border: 1px solid rgba(248,113,113,0.28);
    }

    .subhead {
        font-size: 0.72rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1rem 0 0.4rem 0;
        font-weight: 600;
    }

    .footer-caption {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    .footer-caption a { color: #93c5fd; text-decoration: none; }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02);
    }

    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(239,68,68,0.28);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading fine-tuned model (first request only, ~30-60s)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="cpu", token=HF_TOKEN
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def screen_resume(model, tokenizer, resume_text: str, role: str) -> dict:
    prompt = f"Screen this resume for a {role} position and return a structured verdict.\n\nResume: {resume_text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs
    input_ids = input_ids.to(model.device)

    output = model.generate(input_ids, max_new_tokens=150, do_sample=False)
    raw = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📊 Resume Screener</h1>
        <p>LoRA fine-tuned Qwen2.5-0.5B-Instruct — outputs structured JSON verdicts instead of prose,
        so it can be piped straight into an ATS pipeline.</p>
        <div class="badge-row">
            <span class="badge">PEFT / LoRA</span>
            <span class="badge">Qwen2.5-0.5B</span>
            <span class="badge">Structured Output</span>
            <span class="badge">Trained on Colab T4</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How this model works"):
    st.markdown(
        """
        Base instruct models produce unstructured prose when asked to screen a resume —
        not something you can pipe into an ATS pipeline. This model was fine-tuned with
        LoRA (r=16, alpha=32) on 800 resume → structured-verdict examples so JSON output
        is the model's default behavior, not something coaxed out with prompting.
        """
    )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
with st.container(border=True):
    role = st.text_input("🧭 Target role", placeholder="e.g. Product Manager, Data Analyst, DevOps Engineer")
    resume_text = st.text_area(
        "📄 Resume text",
        height=180,
        placeholder="Paste resume text here (education, experience, skills)...",
    )
    run = st.button("Screen resume →", type="primary", disabled=not (role and resume_text), use_container_width=True)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
if run:
    model, tokenizer = load_model()
    with st.spinner("Generating verdict..."):
        verdict = screen_resume(model, tokenizer, resume_text, role)

    if verdict.get("parse_error"):
        st.warning("Model output wasn't valid JSON — showing raw output:")
        st.code(verdict["raw_output"])
    else:
        score = verdict.get("ats_score", 0)
        verdict_label = verdict.get("verdict", "unknown")
        tone = "moderate" if "moderate" in verdict_label else ("weak" if "weak" in verdict_label or "poor" in verdict_label else "")

        matched = verdict.get("matched_skills") or verdict.get("matched_stages") or []
        missing = verdict.get("missing_skills") or verdict.get("missing_stages") or []

        matched_html = ""
        if matched:
            chips = "".join(f'<span class="chip matched">✓ {s}</span>' for s in matched)
            matched_html = f'<div class="subhead">Matched skills</div><div class="chip-row">{chips}</div>'

        missing_html = ""
        if missing:
            chips = "".join(f'<span class="chip missing">✕ {s}</span>' for s in missing)
            missing_html = f'<div class="subhead">Missing skills</div><div class="chip-row">{chips}</div>'

        st.markdown(
            f"""
            <div class="verdict-card {tone}">
                <div class="metric-row">
                    <div class="metric-box">
                        <div class="label">ATS Score</div>
                        <div class="value">{score}</div>
                    </div>
                    <div class="metric-box">
                        <div class="label">Verdict</div>
                        <div class="value" style="font-size:1.1rem; text-transform:capitalize;">
                            {verdict_label.replace('_', ' ')}
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="label">Years Experience</div>
                        <div class="value">{verdict.get('years_experience', '—')}</div>
                    </div>
                </div>
                <div class="score-bar-track">
                    <div class="score-bar-fill" style="width:{min(score, 100)}%;"></div>
                </div>
                {matched_html}
                {missing_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Raw JSON output"):
            st.json(verdict)

st.markdown(
    """
    <div class="footer-caption">
        Model: <a href="https://github.com/ayush-s-tomar/resume-screener-lora" target="_blank">resume-screener-lora on GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)