# Privacy-first CV screening service image. Self-contained (no external UI coupling).
# Pin the base by digest at release time.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# OS + OCR packages. Poppler intentionally omitted (pypdfium2 renders pages).
# fonts-dejavu-core supplies the Unicode TTF used by later PDF reports.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-ara \
        tesseract-ocr-eng \
        fonts-dejavu-core \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/
COPY tests/ ./tests/

EXPOSE 8089
HEALTHCHECK --interval=30s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8089/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8089", "--workers", "1"]
