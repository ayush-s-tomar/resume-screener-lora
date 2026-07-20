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

st.set_page_config(page_title="Resume Screener (LoRA fine-tuned)", page_icon="📄", layout="centered")


@st.cache_resource(show_spinner="Loading fine-tuned model (first request only, ~30-60s)...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype="auto", device_map="cpu"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def screen_resume(model, tokenizer, resume_text: str, role: str) -> dict:
    prompt = f"Screen this resume for a {role} position and return a structured verdict.\n\nResume: {resume_text}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    output = model.generate(inputs, max_new_tokens=150, do_sample=False)
    raw = tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_output": raw, "parse_error": True}


st.title("📄 Resume Screener")
st.caption("LoRA fine-tuned Qwen2.5-0.5B-Instruct — outputs structured JSON verdicts instead of prose.")

with st.expander("About this model"):
    st.markdown(
        """
        Base instruct models produce unstructured prose when asked to screen a resume —
        not something you can pipe into an ATS pipeline. This model was fine-tuned with
        LoRA (r=16, alpha=32) on 800 resume → structured-verdict examples so JSON output
        is the model's default behavior, not something coaxed out with prompting.

        **Stack:** PEFT/LoRA · Hugging Face Transformers · PyTorch · trained on Colab T4
        """
    )

role = st.text_input("Target role", placeholder="e.g. Backend Engineer")
resume_text = st.text_area(
    "Resume text",
    height=200,
    placeholder="Paste resume text here (education, experience, skills)...",
)

if st.button("Screen resume", type="primary", disabled=not (role and resume_text)):
    model, tokenizer = load_model()
    with st.spinner("Generating verdict..."):
        verdict = screen_resume(model, tokenizer, resume_text, role)

    if verdict.get("parse_error"):
        st.warning("Model output wasn't valid JSON — showing raw output:")
        st.code(verdict["raw_output"])
    else:
        st.success("Verdict generated")
        st.json(verdict)

        cols = st.columns(3)
        if "ats_score" in verdict:
            cols[0].metric("ATS Score", verdict["ats_score"])
        if "verdict" in verdict:
            cols[1].metric("Verdict", verdict["verdict"])
        if "years_experience" in verdict:
            cols[2].metric("Years Experience", verdict["years_experience"])

st.divider()
st.caption("Model: [resume-screener-lora on GitHub](https://github.com/ayush-s-tomar/resume-screener-lora)")
