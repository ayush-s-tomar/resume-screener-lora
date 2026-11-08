# CPU image for the trained LoRA adapter.
#   docker build -t resume-screener .
#   docker run --rm -v hf-cache:/app/.hf-cache resume-screener --role "Backend Engineer" --resume "Python, Docker, 4 years"
FROM python:3.12-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY inference.py .
COPY resume-screener-lora-adapter ./resume-screener-lora-adapter
ENTRYPOINT ["python", "inference.py"]