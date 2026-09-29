# GPU-ready base image validated against the public PyTorch image tags.
FROM pytorch/pytorch@sha256:4ec910830ec6c3b837d5621b6f995a31638d3b0824bf0ed4fcaeba7dd4f89b48
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY static ./static
COPY scripts ./scripts
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src
ENV DEXA_WEIGHTS=/app/artifacts
EXPOSE 8000
CMD ["uvicorn","dexa_ai.api:app","--host","0.0.0.0","--port","8000"]
