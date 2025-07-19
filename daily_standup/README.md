# Ultra-Fast Daily Standup System - Modular Architecture

This is a modularized version of the Daily Standup backend system, refactored for better maintainability and organization.

## 🏗️ Project Structure

```
daily_standup/
├── main.py              # FastAPI application entry point
├── .env                 # Environment variables (secrets)
├── requirements.txt     # Python dependencies
├── core/                # Core modules folder
│   ├── __init__.py
│   ├── config.py        # Configuration and settings
│   ├── database.py      # Database connections and operations
│   ├── ai_services.py   # AI/ML services (OpenAI, Groq, TTS, STT)
│   └── prompts.py       # AI prompt templates
├── audio/               # Audio file storage
├── temp/                # Temporary files
└── reports/             # Generated reports
```

## 🔧 Module Responsibilities

### `main.py`
- FastAPI application setup and routing
- WebSocket endpoints
- Session management orchestration
- PDF generation utilities

### `core/config.py`
- All non-sensitive configuration values
- Model names, voice settings, feature flags
- Directory paths and performance settings
- Development flags and toggles

### `core/database.py`
- MongoDB and SQL Server connection management
- Database operations (CRUD)
- Connection pooling and optimization
- Dummy data support for development

### `core/ai_services.py`
- OpenAI and Groq client management
- Audio processing (TTS/STT)
- Conversation management
- Summary chunk processing
- Session data models

### `core/prompts.py`
- All AI prompt templates
- Centralized prompt management
- Easily maintainable prompt library

## 🚀 Quick Setup

### 1. Environment Setup

Create `.env` file in the `daily_standup` directory:

```bash
# Database Configuration
MONGODB_URL=mongodb://username:password@host:port/database
MONGODB_HOST=192.168.48.201
MONGODB_PORT=27017
MONGODB_USERNAME=LanTech
MONGODB_PASSWORD=L@nc^ere@0012
MONGODB_DATABASE=Api-1
MONGODB_TRANSCRIPTS_COLLECTION=original-1
MONGODB_RESULTS_COLLECTION=daily_standup_results-1

# SQL Server Configuration
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_SERVER=183.82.108.211
SQL_DATABASE=SuperDB
SQL_USERNAME=Connectly
SQL_PASSWORD=LT@connect25
SQL_TIMEOUT=5

# AI Service API Keys
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Development Flags
USE_DUMMY_DATA=true
DEBUG_MODE=false
```

### 2. Install Dependencies

```bash
cd daily_standup
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# From the parent directory where your app.py is located
python app.py
```

## 🌟 Key Benefits of Modular Architecture

### ✅ Clean Separation of Concerns
- **Configuration**: All settings centralized in `core/config.py`
- **Database**: All DB operations in `core/database.py`
- **AI Services**: All AI/ML logic in `core/ai_services.py`
- **Prompts**: All AI prompts in `core/prompts.py`
- **Application**: Only routing and orchestration in `main.py`

### ✅ Easy Maintenance
- **Single Responsibility**: Each module has one clear purpose
- **Easy Testing**: Modules can be tested independently
- **Readable Code**: Clear imports and organization
- **Scalable**: Easy to add new features or modify existing ones

### ✅ Environment Management
- **Secure**: All secrets in `.env` file (not committed to git)
- **Flexible**: Easy to switch between development and production
- **Configurable**: Feature flags and settings clearly defined

### ✅ Development Features
- **Dummy Data**: Built-in support for development without DB
- **Debug Mode**: Easy debugging configuration
- **Hot Reload**: Clean module structure supports fast development

## 🔒 Security Notes

- Never commit the `.env` file to git
- Add `.env` to your `.gitignore` file
- Use environment-specific `.env` files for different deployments
- Rotate API keys regularly

## 🔧 Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run with dummy data (default)
USE_DUMMY_DATA=true python ../app.py

# Run with real database
USE_DUMMY_DATA=false python ../app.py

# Enable debug mode
DEBUG_MODE=true python ../app.py
```

## 📁 Import Structure

With this modular structure, imports are clean and organized:

```python
# In main.py
from .core import (
    config,
    DatabaseManager,
    SessionData,
    shared_clients,
    prompts
)

# Individual imports if needed
from .core.config import config
from .core.database import DatabaseManager
```

## 🚀 Performance Optimizations Maintained

All the original performance optimizations are preserved:
- ⚡ 800ms silence detection
- 🔄 Parallel processing pipeline
- 📊 Summary-based questioning
- 🪟 Sliding window conversation history
- 🎵 Ultra-fast TTS streaming
- 🧵 Thread pool optimization
- 🔄 Session synchronization
- ❌ NO FALLBACKS - Real error detection