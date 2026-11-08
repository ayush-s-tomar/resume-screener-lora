# The hosted-demo backend (Groq-scored), runnable anywhere:
#   docker build -t resume-screener-demo render-deploy
#   docker run --rm -p 8000:8000 -e GROQ_API_KEY=your_key resume-screener-demo
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]