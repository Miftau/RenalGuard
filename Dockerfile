# ============================
# HeartPredict - Dockerfile
# ============================

# Start from a lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first (for efficient Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into container
COPY . .

# Expose Flask port
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production
ENV FLASK_APP=app.py

# Create results directory if not present
RUN mkdir -p /app/static/results

# Command to run the app with Gunicorn (recommended for Fly.io)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "4"]
