# MUNCHKIN FIX: Changed from 3.10-slim to 3.12-slim to match WSL venv
# This prevents C-extension (NumPy) mismatch errors.
FROM python:3.12-slim

# Install system-level dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# We do NOT run 'pip install' here. 
# Instead, the docker-compose.yml will mount your local WSL 'site-packages'.
ENV PYTHONPATH=/app:/usr/local/lib/python3.12/site-packages