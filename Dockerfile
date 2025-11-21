# Use a lightweight Python version
FROM python:3.9-slim

# 1. Install system dependencies for OpenCV
# CRUCIAL FIX: Replacing libgL with the common runtime libraries libgl1 and libsm6
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

# 4. Copy your scripts AND your baseline memory
# We copy everything in the current folder to /app
COPY drift_detector.py .
COPY drift_analyzer.py .
COPY monitoring_service.py .

# IMPORTANT CHANGE: Copy baseline from the 'embeddings' subdirectory
COPY embeddings/baseline_embeddings.npy . 

# 5. Create a directory for incoming data (empty for now)
RUN mkdir /app/incoming_data

# 6. The command to run when the container starts
CMD ["python", "monitoring_service.py"]