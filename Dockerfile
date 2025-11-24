# Use the official TensorFlow CPU image (Lighter and compatible with GitHub Actions)
FROM tensorflow/tensorflow:2.15.0 as base

# 1. Install system dependencies for OpenCV (required for image processing)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the working directory inside the container
WORKDIR /app

# 3. Install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all source code (scripts, config, data folders if present)
COPY . /app

# 5. Create directories for output if they don't exist
RUN mkdir -p /app/incoming_data
RUN mkdir -p /app/status_output

# 6. The command to run when the container starts
ENTRYPOINT ["python", "monitoring_service.py"]