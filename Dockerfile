FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/requirements.txt /app/
RUN pip install --upgrade pip \
 && pip install -r /app/requirements.txt

COPY pyproject.toml /app/
COPY src/ /app/src/

CMD ["python3", "src/main.py"]