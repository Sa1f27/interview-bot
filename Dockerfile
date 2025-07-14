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
