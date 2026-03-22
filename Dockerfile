FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY outputs/ outputs/

EXPOSE 8009

# Run with uvicor
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
