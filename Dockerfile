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

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose Streamlit port
EXPOSE 8501

ENTRYPOINT ["/entrypoint.sh"]