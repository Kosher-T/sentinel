# Minimal Dockerfile for Development
# Relies on host-mounted venv for all dependencies
FROM python:3.12-slim

# Install system dependencies required by OpenCV and other libs in the venv
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# No COPY or RUN pip install commands needed