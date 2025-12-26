# Use the same version as your WSL Python
FROM python:3.10-slim

# Still need these system libs (usually small/cached)
# If you've built this once, Docker will cache these forever locally.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# We skip COPY requirements.txt and RUN pip install
# because we are mounting site-packages in docker-compose.yml

ENV PYTHONPATH=/app
# The rest of the files are mounted via volume