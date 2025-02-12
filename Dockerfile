# Use CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY .env .

# Create data directory for ChromaDB
RUN mkdir -p data/chroma_db

# Set environment variables
ENV MODEL_DEVICE=cuda \
    API_HOST=0.0.0.0 \
    API_PORT=8000

# Expose port
EXPOSE 8000

# Start the application
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]