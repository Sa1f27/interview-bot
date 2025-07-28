# ✅ Use base image with Python
FROM python:3.10-slim

# ✅ Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ✅ Set working directory
WORKDIR /app

# ✅ Copy requirements file
COPY requirements.txt .

# ✅ Set pip to not use cache (smaller image)
ENV PIP_NO_CACHE_DIR=1

# ✅ Install basic Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# ✅ Install GPU-based PyTorch with retry logic
RUN for i in 1 2 3; do \
    pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && break || sleep 5; \
done

# ✅ (Optional CPU version if you don't need GPU)
# RUN pip install torch torchvision torchaudio

# ✅ Copy source code
COPY . .

# ✅ Expose the app's port (change to 8050 if needed)
# EXPOSE 8050

# ✅ Run the app (adjust if your main file is different)
CMD ["python", "app.py"]

# App/Dockerfile
# Multi-stage Docker build for production deployment

# ==================== STAGE 1: Base Python Environment ====================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    unixodbc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft ODBC Driver for SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# ==================== STAGE 2: Dependencies ====================
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# ==================== STAGE 3: Application ====================
FROM dependencies as application

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p daily_standup/audio \
    && mkdir -p daily_standup/temp \
    && mkdir -p daily_standup/reports \
    && mkdir -p static \
    && mkdir -p certs

# Set ownership to app user
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8070

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f https://localhost:8070/health || exit 1

# Default command
CMD ["python", "app.py"]