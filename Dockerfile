# Use a clean Python image to avoid conflicts with pre-installed libraries
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the entire package directory to preserve module structure
# This ensures detector_data_drift/ is a valid python package
COPY detector_data_drift/ ./detector_data_drift/

# 4. Create placeholder directories for volume mounting
RUN mkdir -p /app/incoming_data /app/status_output

# 5. Set the entrypoint
# We run the script using the module path so Python handles imports correctly
ENTRYPOINT ["python", "detector_data_drift/monitoring_service.py"]