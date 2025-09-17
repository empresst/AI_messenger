# Python slim image
FROM python:3.11-slim

# System deps (just enough for numpy/spacy/faiss wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy requirements first (better layer cache)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLP data at build time (faster cold starts)
RUN python -m nltk.downloader -q wordnet punkt
RUN python -m spacy download en_core_web_sm

# Copy the app
COPY . .

# Render provides $PORT; listen on it
ENV PORT=8000
EXPOSE 8000

# Start the ASGI server (WebSockets OK)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "${PORT}"]
