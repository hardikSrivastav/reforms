# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix OpenAI and httpx compatibility issue
RUN pip install --no-cache-dir --force-reinstall "httpx<0.25.0" "openai>=1.3.0,<1.4.0"

# Add flower for monitoring
RUN pip install --no-cache-dir flower

# Copy the data pipeline code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Create a non-root user and switch to it
RUN useradd -m celeryuser
RUN chown -R celeryuser:celeryuser /app
USER celeryuser

# Remove the ENTRYPOINT
# Default command
CMD ["celery", "--help"] 