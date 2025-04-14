# Use a Python base image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed (e.g., for specific libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Install Python dependencies
# Using --no-cache-dir can reduce image size slightly
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and necessary data access paths
# (Gradio needs access to sample images for gallery display)
COPY ./app.py /app/app.py
# COPY ../sample_data /app/sample_data

# Expose the port Gradio runs on (default 7860)
EXPOSE 7860

# Command to run the Gradio application
# Use 0.0.0.0 to be accessible from outside the container
CMD ["python", "app.py"]
