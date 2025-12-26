# Use a slim Python image with ML support
FROM python:3.10-slim

# Install system dependencies for OpenCV and SQLite
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# Note: You should create a requirements.txt with:
# tensorflow, keras, numpy, opencv-python, scikit-image, scikit-learn, pandas, streamlit, scipy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set PYTHONPATH to the current directory so internal imports work
ENV PYTHONPATH=/app

# Default command is overridden by docker-compose
CMD ["python", "sentinel_watch.py"]