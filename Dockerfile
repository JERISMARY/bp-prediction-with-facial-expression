# Use the official Python image
FROM python:3.11-slim

# Install system dependencies for librosa and soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Flask/Gunicorn will run on
ENV PORT 8080

# Run the web service on container startup
# Timeout is increased for librosa processing if needed
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
