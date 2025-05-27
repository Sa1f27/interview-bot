# FastAPI Multi-Module Interview & Testing System

A comprehensive AI-powered platform featuring voice-based interviews, mock tests, and daily standups. Built with FastAPI, this system provides an integrated solution for technical assessments and skill evaluation.

## 🚀 Features

### 🎤 Daily Standup Module
- **Voice-based testing** with AI-generated questions
- Real-time audio recording and transcription
- Adaptive questioning based on lecture summaries
- **Text-to-speech** responses with natural voice synthesis
- Live evaluation and feedback

### 📝 Weekend Mock Test Module  
- **Adaptive testing** for developers and non-developers
- Developer mode: Code writing, debugging, scenario-based questions
- Non-developer mode: Multiple choice questions with auto-evaluation
- **MongoDB integration** for dynamic content generation
- PDF export of test results
- Real-time scoring and analytics

### 💼 Weekly Interview Module
- **Three-round interview system**: Technical, Communication, HR
- Voice-based interaction with AI interviewers
- **Progressive difficulty** based on performance
- Comprehensive evaluation reports
- Real-time round transitions

## 🏗️ Project Structure

```
├── main.py                    # Main FastAPI app with sub-app mounting
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
├── README.md                 # This documentation
├── static/                   # Main landing page assets
│   └── index.html           # Project launcher interface
├── daily_standup/           # Voice testing module
│   ├── __init__.py
│   ├── main.py             # Standup API endpoints
│   └── frontend/           # Standup UI
│       └── index.html
├── weekend_mocktest/        # Mock test module  
│   ├── __init__.py
│   ├── main.py             # Test API endpoints
│   └── frontend/           # Test UI
│       └── index.html
└── weekly_interview/        # Interview module
    ├── __init__.py
    ├── main.py             # Interview API endpoints
    └── frontend/           # Interview UI
        └── index.html
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern async web framework
- **MongoDB** - Document database for content storage
- **LangChain** - LLM orchestration and prompt management
- **OpenAI GPT-4** - Question generation and evaluation
- **Groq API** - Fast audio transcription (Whisper)
- **Edge TTS** - Natural text-to-speech synthesis

### Frontend
- **HTML/CSS/JavaScript** - Modern responsive interfaces
- **Tailwind CSS** - Utility-first styling (weekend module)
- **Real-time audio** processing with Web Audio API
- **Progressive UI** with loading states and animations

### Audio Processing
- **sounddevice** - Real-time audio capture
- **scipy** - Audio file processing
- **FFmpeg** - Audio format conversion and speed adjustment

## 📋 Prerequisites

- Python 3.8+ installed on Windows
- MongoDB instance (local or remote)
- FFmpeg installed and added to Windows PATH
- OpenAI API key
- Groq API key
- Windows microphone access permissions

## ⚙️ Installation

1. **Clone the repository**
```cmd
git clone <repository-url>
cd fastapi-interview-system
```

2. **Create virtual environment (recommended)**
```cmd
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**
```cmd
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

Or set them in Windows Command Prompt:
```cmd
set OPENAI_API_KEY=your_openai_api_key_here
set GROQ_API_KEY=your_groq_api_key_here
```

Or set them in PowerShell:
```powershell
$env:OPENAI_API_KEY="your_openai_api_key_here"
$env:GROQ_API_KEY="your_groq_api_key_here"
```

5. **Install FFmpeg for Windows**

**Option A: Using Chocolatey (recommended)**
```cmd
# Install Chocolatey first if not installed
# Visit: https://chocolatey.org/install

choco install ffmpeg
```

**Option B: Manual Installation**
```cmd
# 1. Download FFmpeg from https://www.gyan.dev/ffmpeg/builds/
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to Windows PATH:
#    - Press Win + R, type "sysdm.cpl"
#    - Go to Advanced → Environment Variables
#    - Edit PATH and add C:\ffmpeg\bin
# 4. Restart Command Prompt and verify:
ffmpeg -version
```

6. **Verify microphone permissions**
```cmd
# Windows 10/11: Settings → Privacy → Microphone
# Ensure "Allow apps to access your microphone" is ON
```

## 🚀 Running the Application

1. **Activate virtual environment (if using)**
```cmd
venv\Scripts\activate
```

2. **Start the main application**
```cmd
python main.py
```

3. **Alternative: Run with uvicorn directly**
```cmd
uvicorn main:app --host 127.0.0.1 --port 8060 --reload
```

4. **Access the interface**
```
http://localhost:8060
```

5. **Individual module access**
- Daily Standup: `http://localhost:8060/daily_standup`
- Weekend Mock Test: `http://localhost:8060/weekend_mocktest`  
- Weekly Interview: `http://localhost:8060/weekly_interview`

## 📊 Database Configuration

The system requires MongoDB with specific collections:

```javascript
// Database: test
// Collection: drive
{
  "_id": ObjectId("..."),
  "summary": "Lecture content summary for question generation",
  "timestamp": ISODate("...")
}
```

**MongoDB Connection String:**
```
mongodb://sa:L%40nc%5Eere%400012@192.168.48.200:27017/?authSource=admin
```

## 🎯 Usage Guide

### Daily Standup Module
1. Click "Start Test" to begin voice-based assessment
2. Answer AI-generated questions verbally (10-second limit per response)
3. Receive real-time feedback and follow-up questions
4. View comprehensive evaluation report after completion

### Weekend Mock Test Module
1. Select user type: Developer or Non-Developer
2. Answer 10 adaptive questions (5 minutes each for dev, 2 minutes for non-dev)
3. Receive immediate scoring and detailed analytics
4. Export results as PDF for records

### Weekly Interview Module  
1. Click "Start Interview" for three-round assessment
2. Complete Technical, Communication, and HR rounds sequentially
3. Engage in natural voice conversations with AI interviewers
4. Receive comprehensive evaluation across all competency areas

## 🔧 Configuration

### Audio Settings
```python
SAMPLE_RATE = 16000           # Audio sample rate
SILENCE_THRESHOLD = 0.01      # Voice activity detection
SILENCE_DURATION = 2.0        # Silence before auto-stop
MAX_RECORDING_DURATION = 10.0 # Maximum recording time
TTS_SPEED = 1.2              # Text-to-speech playback speed
```

### Test Parameters
```python
TEST_DURATION_SEC = 120       # Daily standup time limit
INACTIVITY_TIMEOUT = 120      # Session timeout
```

## 🔌 API Endpoints

### Daily Standup (`/daily_standup/`)
- `GET /` - Serve standup interface
- `GET /start_test` - Initialize voice test session
- `POST /record_and_respond` - Process audio and generate response
- `GET /summary` - Get test evaluation

### Weekend Mock Test (`/weekend_mocktest/`)
- `GET /` - Serve test interface  
- `POST /start-test` - Begin mock test session
- `POST /submit-answer` - Submit question response
- `GET /export-results` - Download PDF results

### Weekly Interview (`/weekly_interview/`)
- `GET /` - Serve interview interface
- `GET /start_interview` - Initialize interview session
- `POST /record_and_respond` - Process audio interaction
- `GET /start_next_round` - Transition between rounds
- `GET /evaluate` - Generate final evaluation

## 🔍 Monitoring & Maintenance

### Health Checks
```cmd
curl http://localhost:8060/healthz
```

**Or using PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8060/healthz"
```

### Cleanup Operations
```cmd
# Clean expired sessions and audio files
curl http://localhost:8060/daily_standup/cleanup
curl http://localhost:8060/weekly_interview/cleanup
```

**Or using PowerShell:**
```powershell
Invoke-WebRequest -Uri "http://localhost:8060/daily_standup/cleanup"
Invoke-WebRequest -Uri "http://localhost:8060/weekly_interview/cleanup"
```

### Logs
- Application logs available via standard Python logging
- Audio files automatically cleaned after 1 hour
- Sessions expire after 2 hours of inactivity

## 🚨 Troubleshooting

### Common Windows Issues

**Audio Recording Problems:**
```cmd
# Test microphone access
python -c "import sounddevice as sd; print(sd.query_devices())"

# Check FFmpeg installation
ffmpeg -version

# Windows microphone permissions:
# Settings → Privacy & Security → Microphone → Allow apps to access microphone
```

**Virtual Environment Issues:**
```cmd
# If activation fails
venv\Scripts\activate.bat

# If still fails, recreate environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Port Already in Use:**
```cmd
# Find process using port 8060
netstat -ano | findstr :8060

# Kill process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or use different port
python main.py --port 8061
```

**TTS/Transcription Failures:**
```cmd
# Test API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY'))); print('Groq:', bool(os.getenv('GROQ_API_KEY')))"

# Check internet connectivity
ping api.openai.com
ping api.groq.com
```

**File Permission Errors:**
```cmd
# Run Command Prompt as Administrator if needed
# Or change audio directory permissions
icacls daily_standup\audio /grant Users:F
icacls weekly_interview\audio /grant Users:F
```

## 📈 Performance Notes

- **Concurrent Users**: Tested up to 10 simultaneous sessions
- **Audio Processing**: ~2-3 second latency for transcription
- **TTS Generation**: ~1-2 seconds for speech synthesis
- **Database Queries**: Optimized for quick content retrieval

## 🔐 Security Considerations

- Audio files are automatically cleaned up
- Sessions have built-in expiration
- API keys should be properly secured
- Consider implementing authentication for production use
