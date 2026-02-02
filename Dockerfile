FROM python:3.13-slim

WORKDIR /app

# System deps (safe baseline; add more only if build errors)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "train.py"]