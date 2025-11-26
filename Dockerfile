# Use a clean Python image to avoid conflicts with pre-installed libraries
FROM python:3.9-slim

# 1. Install system dependencies for OpenCV (required for image processing)
# We update the package list and install the GL libraries needed for cv2
RUN apt-get update && apt-get install -y \
    libgl1 \
    libsm6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Set up the working directory inside the container
WORKDIR /app

# 3. Upgrade pip to ensure we handle binary wheels correctly
RUN pip install --upgrade pip

# 4. Install Python libraries
# Since we are on a clean python image, this will install TensorFlow and Scikit-learn
# freshly, ensuring they play nice together.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all source code
COPY . /app

# 6. Create directories for output if they don't exist
RUN mkdir -p /app/incoming_data
RUN mkdir -p /app/status_output

# 7. The command to run when the container starts
ENTRYPOINT ["python", "monitoring_service.py"]