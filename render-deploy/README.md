# Hosted Demo Backend

This folder is a separate, simplified serving layer for the live demo link --
it is not the fine-tuned model itself. See the top-level README's
Deployment Note section for why this exists.

## What this is

A minimal FastAPI app that scores resumes via Groq's hosted API
(openai/gpt-oss-20b), with a dark-themed HTML frontend served at /. It prompts the model for the same JSON schema the adapter was trained to emit (score out of 100, verdict, matched and missing skills) and includes six example resumes.

## What this is NOT

This does not load or run the LoRA adapter in
../resume-screener-lora-adapter/. For the actual trained model, use
inference.py at the repo root.

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

## Demo guards

The hosted demo shares one Groq quota, so `app.py` protects it:

- Identical requests are answered from an in-memory cache (the response has `cached: true` and an `X-Cache: hit` header) and cost no quota.
- Each visitor is limited to 8 new screenings per 10 minutes (`DEMO_RATE_LIMIT`, `DEMO_RATE_WINDOW_SECONDS`), and the whole demo to 150 new screenings per UTC day (`DEMO_DAILY_CAP`). Over the limit, the API answers HTTP 429 with a `Retry-After` header. The defaults are conservative guesses for Groq's free tier; adjust them in Render's environment settings.
- Request bodies over 64 KB are rejected with HTTP 413, and responses carry `X-Content-Type-Options` and `Referrer-Policy` headers.
- Logs record timing and cache hits but never the resume text; visitor addresses are logged only as a short hash.
- Counters and the cache live in memory, so they reset when the free instance sleeps and wakes. They protect against casual overuse, not a determined attacker.