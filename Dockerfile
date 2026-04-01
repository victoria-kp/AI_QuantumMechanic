FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# Copy source code
COPY config.py .
COPY app.py .
COPY agent/ agent/
COPY tools/ tools/
COPY checkers/ checkers/

# Create outputs directory for generated plots
RUN mkdir -p outputs/figures outputs/logs

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
