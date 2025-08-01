#!/bin/bash

echo "?? Starting project setup..."

# === 1. Update and install system dependencies ===
echo "?? Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    openssl \
    net-tools \
    curl \
    build-essential

# === 2. Create Python virtual environment ===
echo "?? Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# === 3. Install Python packages ===
echo "?? Installing Python dependencies (Torch + CUDA 12.6)..."
pip install --upgrade pip
pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# === 4. Create required directories ===
echo "?? Creating project directories..."
mkdir -p daily_standup/audio \
         daily_standup/temp \
         daily_standup/reports \
         weekly_interview/audio \
         weekly_interview/temp \
         weekly_interview/reports \
         static \
         certs \
         env

# === 5. Generate self-signed SSL certs as cert.pem / key.pem and in TMPS/certs ===
CERT_PATH="./certs"
if [ ! -f "$CERT_PATH/cert.pem" ] || [ ! -f "$CERT_PATH/key.pem" ]; then
  echo "?? Generating self-signed SSL certificates..."
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$CERT_PATH/key.pem" \
    -out "$CERT_PATH/cert.pem" \
    -subj "/C=IN/ST=TS/L=Hyderabad/O=Lanciere/OU=Dev/CN=localhost"
  echo "? Certificates saved as: $CERT_PATH/cert.pem and $CERT_PATH/key.pem"
else
  echo "?? SSL certificates already exist. Skipping generation."
fi

# === 5. Generate self-signed SSL certs  ===
echo "?? Creating TMPS/certs and generating SSL certificates there..."

cd TMPS/
mkdir -p certs
cd certs

if [ ! -f "cert.pem" ] || [ ! -f "key.pem" ]; then
  echo "?? Generating self-signed SSL certificates in TMPS/certs..."
  openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout key.pem \
    -out cert.pem \
    -subj "/C=IN/ST=TS/L=Hyderabad/O=Lanciere/OU=Dev/CN=localhost"
  echo "? Certificates created: TMPS/certs/cert.pem and key.pem"
else
  echo "?? Certs already exist in TMPS/certs. Skipping generation."
fi

cd ../../  # Go back to project root

# === 6. Check if ports 8070 and 5173 are available ===
echo "?? Checking ports 8070 and 5173..."
for port in 8070 5173; do
  if lsof -i:$port >/dev/null 2>&1; then
    echo "?? Port $port is already in use."
  else
    echo "? Port $port is free."
  fi
done

# === 7. Create .env template for OpenAI & GROQ ===
ENV_PATH="./env/.env"
if [ ! -f "$ENV_PATH" ]; then
  echo "?? Creating .env template for API keys..."
  cat > "$ENV_PATH" <<EOL
# ?? Add your API keys here

OPENAI_API_KEY=your-openai-api-key
GROQ_API_KEY=your-groq-api-key
EOL
  echo "? .env file created at $ENV_PATH ? Please edit it and add your real keys."
else
  echo "?? .env file already exists. Skipping creation."
fi

echo "?? Setup completed successfully!"
