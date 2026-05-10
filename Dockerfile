FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker layer caching)
COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

# Copy entire project
COPY . .

# Create necessary directories
RUN mkdir -p logs data chroma_db

# Expose Streamlit port (HF Spaces uses 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run the HF-compatible app directly
CMD ["streamlit", "run", "frontend/app_hf.py", "--server.port=7860", "--server.address=0.0.0.0"]