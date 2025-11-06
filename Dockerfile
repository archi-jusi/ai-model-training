# Use an ARM64-compatible Python base image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
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

# Upgrade pip
RUN pip install --upgrade pip

# Set work directory
WORKDIR /app

# Copy project files
COPY pyproject.toml /app/
COPY src/ /app/src/

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

# Set default command
CMD ["python3", "src/main.py"]
