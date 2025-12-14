FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN mkdir -p ./docs

RUN pip install --no-cache-dir -r requirements.txt

COPY . ./proxy

CMD ["uvicorn", "proxy.main:app", "--host", "0.0.0.0", "--port", "8080"]