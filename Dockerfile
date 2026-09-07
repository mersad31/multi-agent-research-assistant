FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY api/ ./api/
COPY src/ ./src/

RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]