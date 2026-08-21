# Hosted Demo Backend

This folder is a separate, simplified serving layer for the live demo link --
it is not the fine-tuned model itself. See the top-level README's
Deployment Note section for why this exists.

## What this is

A minimal FastAPI app that scores resumes via Groq's hosted API
(openai/gpt-oss-20b), with a simple dark-themed HTML frontend served at /.

## What this is NOT

This does not load or run the LoRA adapter in
../resume-screener-lora-adapter/. For the actual trained model, use
inference.py or app.py (Streamlit) at the repo root.

## Run locally

pip install -r requirements.txt
export GROQ_API_KEY=your_key_here    # Windows: $env:GROQ_API_KEY="your_key_here"
uvicorn app:app --host 0.0.0.0 --port 8000
# -> http://localhost:8000

## Why Groq instead of the local adapter

Render's free tier caps memory at 512MB, which isn't enough to reliably run
GGUF-quantized llama.cpp inference without OOM crashes or timeouts. Routing
through Groq's hosted API keeps the public demo fast and stable on free
hosting, at the cost of not literally running the fine-tuned weights in
production.
