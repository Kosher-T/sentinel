# Use a slim Python image to keep the base layer light
FROM python:3.10-slim

# Install system-level dependencies for OpenCV and image processing
# Updated for compatibility with modern Debian-based images
# Added --fix-missing and --no-install-recommends to handle flaky network (ISP interception)
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
# This effectively uses your local venv as the container's library.

ENV PYTHONPATH=/app