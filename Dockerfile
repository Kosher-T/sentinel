# Use a clean Python image to avoid conflicts with pre-installed libraries
FROM python:3.9-slim

# 1. Install system dependencies for OpenCV (required for image processing)
# We update the package list and install the GL libraries needed for cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# (Assuming requirements.txt is in your root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX: Copy the script from its new sub-directory ---
# This takes the file from your local folder and puts it in /app/
COPY detector_data_drift/monitoring_service.py .

# Create placeholder directories for volume mounting
RUN mkdir -p /app/incoming_data /app/status_output

# Set the entrypoint to run the script
# Since we copied it directly into /app, we call it by name
ENTRYPOINT ["python", "monitoring_service.py"]