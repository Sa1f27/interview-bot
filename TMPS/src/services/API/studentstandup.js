// App/daily_standup/TMPS/src/services/API/studentstandup.js
// REALISTIC implementation with automatic voice detection - NO FALLBACKS!

import { assessmentApiRequest } from './index2';

// Automatic Voice Activity Detection
class VoiceActivityDetector {
  constructor() {
    this.isListening = false;
    this.silenceThreshold = 0.01; // Silence detection threshold
    this.silenceDelay = 2000; // 2 seconds of silence before stopping
    this.maxRecordingTime = 30000; // 30 seconds max
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.silenceTimer = null;
    this.recordingTimer = null;
    this.onSilenceDetected = null;
    this.onSpeechDetected = null;
  }

  async initialize(stream) {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.3;
      
      source.connect(this.analyser);
      
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      
      console.log('🎙️ Voice Activity Detector initialized');
      return true;
    } catch (error) {
      console.error('❌ VAD initialization failed:', error);
      return false;
    }
  }

  startListening() {
    if (this.isListening) return;
    
    this.isListening = true;
    console.log('👂 Started listening for voice activity');
    this.detectVoiceActivity();
    
    // Auto-stop after max recording time
    this.recordingTimer = setTimeout(() => {
      console.log('⏰ Max recording time reached');
      this.stopListening();
    }, this.maxRecordingTime);
  }

  detectVoiceActivity() {
    if (!this.isListening || !this.analyser) return;

    this.analyser.getByteFrequencyData(this.dataArray);
    
    // Calculate volume level
    const sum = this.dataArray.reduce((a, b) => a + b, 0);
    const average = sum / this.dataArray.length;
    const volume = average / 255;

    if (volume > this.silenceThreshold) {
      // Speech detected
      if (this.silenceTimer) {
        clearTimeout(this.silenceTimer);
        this.silenceTimer = null;
      }
      
      if (this.onSpeechDetected) {
        this.onSpeechDetected(volume);
      }
    } else {
      // Silence detected
      if (!this.silenceTimer) {
        this.silenceTimer = setTimeout(() => {
          console.log('🤫 Silence detected, stopping recording');
          this.stopListening();
        }, this.silenceDelay);
      }
    }

    // Continue monitoring
    requestAnimationFrame(() => this.detectVoiceActivity());
  }

  stopListening() {
    if (!this.isListening) return;
    
    this.isListening = false;
    
    if (this.silenceTimer) {
      clearTimeout(this.silenceTimer);
      this.silenceTimer = null;
    }
    
    if (this.recordingTimer) {
      clearTimeout(this.recordingTimer);
      this.recordingTimer = null;
    }
    
    if (this.onSilenceDetected) {
      this.onSilenceDetected();
    }
    
    console.log('🛑 Stopped listening for voice activity');
  }

  cleanup() {
    this.stopListening();
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
  }
}

// Realistic Audio Manager with Auto Voice Detection
class RealisticAudioManager {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.stream = null;
    this.isRecording = false;
    this.vad = new VoiceActivityDetector();
    this.onRecordingComplete = null;
    this.onSpeechStart = null;
    this.audioQueue = [];
    this.isPlayingAudio = false;
  }

  async startListening() {
    try {
      console.log('🎤 Starting realistic voice conversation...');
      
      // Get microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      // Initialize voice activity detection
      const vadInitialized = await this.vad.initialize(this.stream);
      if (!vadInitialized) {
        throw new Error('Voice activity detection failed to initialize');
      }
      
      // Setup media recorder
      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      this.audioChunks = [];
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };
      
      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        this.audioChunks = [];
        this.isRecording = false;
        
        if (this.onRecordingComplete) {
          this.onRecordingComplete(audioBlob);
        }
      };
      
      // Setup voice activity detection callbacks
      this.vad.onSpeechDetected = (volume) => {
        if (!this.isRecording && !this.isPlayingAudio) {
          console.log('🗣️ Speech detected, starting recording...');
          this.startRecording();
          
          if (this.onSpeechStart) {
            this.onSpeechStart();
          }
        }
      };
      
      this.vad.onSilenceDetected = () => {
        if (this.isRecording) {
          console.log('🤫 Silence detected, stopping recording...');
          this.stopRecording();
        }
      };
      
      // Start voice activity detection
      this.vad.startListening();
      
      console.log('✅ Ready for natural conversation');
      
    } catch (error) {
      console.error('❌ Failed to start listening:', error);
      throw new Error(`Microphone setup failed: ${error.message}`);
    }
  }

  startRecording() {
    if (this.isRecording || !this.mediaRecorder) return;
    
    this.mediaRecorder.start(100); // Record in small chunks
    this.isRecording = true;
    console.log('🔴 Recording started automatically');
  }

  stopRecording() {
    if (!this.isRecording || !this.mediaRecorder) return;
    
    this.mediaRecorder.stop();
    console.log('⏹️ Recording stopped automatically');
  }

  async playAudioBuffer(audioBuffer) {
    return new Promise((resolve, reject) => {
      try {
        this.isPlayingAudio = true;
        
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
          URL.revokeObjectURL(audioUrl);
          this.isPlayingAudio = false;
          console.log('🎵 Audio playback finished');
          resolve();
        };
        
        audio.onerror = (error) => {
          URL.revokeObjectURL(audioUrl);
          this.isPlayingAudio = false;
          console.error('❌ Audio playback failed:', error);
          reject(error);
        };
        
        audio.play().then(() => {
          console.log('🎵 Playing AI response...');
        }).catch(reject);
        
      } catch (error) {
        this.isPlayingAudio = false;
        reject(error);
      }
    });
  }

  async playAudioStream(audioChunks) {
    // Queue up audio chunks for seamless playback
    for (const chunk of audioChunks) {
      await this.playAudioBuffer(chunk);
    }
  }

  stopListening() {
    this.vad.cleanup();
    
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    this.isRecording = false;
    this.isPlayingAudio = false;
    
    console.log('🛑 Stopped listening');
  }
}

// Robust WebSocket with NO FALLBACKS - Fail Fast!
class RobustWebSocketManager {
  constructor() {
    this.websocket = null;
    this.eventHandlers = {};
    this.isConnected = false;
    this.sessionId = null;
  }

  setEventHandlers(handlers) {
    this.eventHandlers = handlers;
  }

  async connect(sessionId) {
    this.sessionId = sessionId;
    
    try {
      const wsUrl = `ws://192.168.48.57:8060/daily_standup/ws/${sessionId}`;
      console.log('🔌 Connecting to WebSocket:', wsUrl);
      
      this.websocket = new WebSocket(wsUrl);
      
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          this.websocket.close();
          reject(new Error('WebSocket connection timeout'));
        }, 10000);

        this.websocket.onopen = () => {
          clearTimeout(timeout);
          this.isConnected = true;
          this.setupEventHandlers();
          console.log('✅ WebSocket connected successfully');
          resolve();
        };

        this.websocket.onerror = (error) => {
          clearTimeout(timeout);
          console.error('❌ WebSocket connection failed:', error);
          reject(new Error('WebSocket connection failed - check if backend is running'));
        };

        this.websocket.onclose = (event) => {
          clearTimeout(timeout);
          if (event.code !== 1000) {
            reject(new Error(`WebSocket closed unexpectedly: ${event.code}`));
          }
        };
      });
      
    } catch (error) {
      console.error('❌ WebSocket setup failed:', error);
      throw new Error(`Connection failed: ${error.message}`);
    }
  }

  setupEventHandlers() {
    if (!this.websocket) return;

    this.websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('📨 Received:', data.type);
        
        if (this.eventHandlers.onMessage) {
          this.eventHandlers.onMessage(data);
        }
        
      } catch (error) {
        console.error('❌ Message parsing error:', error);
      }
    };
    
    this.websocket.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
      this.isConnected = false;
      
      if (this.eventHandlers.onError) {
        this.eventHandlers.onError(error);
      }
    };
    
    this.websocket.onclose = (event) => {
      console.log('🔌 WebSocket closed:', event.code, event.reason);
      this.isConnected = false;
      
      if (this.eventHandlers.onClose) {
        this.eventHandlers.onClose(event);
      }
    };
  }

  sendAudioData(audioBlob) {
    if (!this.isConnected || !this.websocket) {
      throw new Error('WebSocket not connected - cannot send audio');
    }

    const reader = new FileReader();
    reader.onload = () => {
      const base64Audio = reader.result.split(',')[1];
      const message = {
        type: 'audio_data',
        audio: base64Audio
      };
      this.websocket.send(JSON.stringify(message));
      console.log('📤 Audio sent to server');
    };
    reader.readAsDataURL(audioBlob);
  }

  disconnect() {
    if (this.websocket) {
      this.websocket.close(1000, 'Normal closure');
      this.websocket = null;
    }
    this.isConnected = false;
  }
}

// Main API Service - Realistic Conversation Flow
class RealisticStandupAPIService {
  constructor() {
    this.wsManager = new RobustWebSocketManager();
    this.audioManager = new RealisticAudioManager();
    this.currentSessionId = null;
    this.conversationState = 'idle'; // idle, listening, speaking, processing
    this.audioChunksBuffer = [];
  }

  async startStandup() {
    try {
      console.log('🚀 Starting realistic standup conversation...');
      
      // Get session from backend
      const response = await assessmentApiRequest('/daily_standup/start_test', {
        method: 'GET'
      });
      
      if (!response || !response.session_id) {
        throw new Error('Failed to create session - check backend server');
      }
      
      this.currentSessionId = response.session_id;
      console.log('✅ Session created:', this.currentSessionId);
      
      // Connect WebSocket - FAIL if this doesn't work
      await this.wsManager.connect(this.currentSessionId);
      
      // Setup WebSocket event handlers
      this.wsManager.setEventHandlers({
        onMessage: (data) => this.handleServerMessage(data),
        onError: (error) => this.handleConnectionError(error),
        onClose: (event) => this.handleConnectionClose(event)
      });
      
      // Setup automatic voice recording
      this.audioManager.onRecordingComplete = (audioBlob) => {
        this.handleAudioRecorded(audioBlob);
      };
      
      this.audioManager.onSpeechStart = () => {
        this.conversationState = 'listening';
        console.log('👂 User started speaking...');
      };
      
      // Start listening for user voice
      await this.audioManager.startListening();
      
      console.log('✅ Ready for natural conversation - speak naturally!');
      
      return {
        test_id: response.test_id,
        session_id: this.currentSessionId,
        status: 'ready'
      };
      
    } catch (error) {
      console.error('❌ Startup failed:', error);
      throw error; // NO FALLBACKS - Let the error bubble up!
    }
  }

  handleServerMessage(data) {
    switch (data.type) {
      case 'ai_response':
        console.log('🤖 AI said:', data.text);
        this.conversationState = 'speaking';
        this.audioChunksBuffer = [];
        break;
        
      case 'audio_chunk':
        // Collect audio chunks for seamless playback
        if (data.audio) {
          const binaryData = new Uint8Array(
            data.audio.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
          );
          this.audioChunksBuffer.push(binaryData);
        }
        break;
        
      case 'audio_end':
        console.log('🎵 Playing AI response...');
        this.playAIResponse();
        break;
        
      case 'conversation_end':
        console.log('🏁 Conversation completed');
        this.handleConversationEnd(data);
        break;
        
      case 'error':
        console.error('❌ Server error:', data.text);
        throw new Error(data.text);
        
      default:
        console.log('📨 Unknown message type:', data.type);
    }
  }

  async playAIResponse() {
    try {
      this.conversationState = 'speaking';
      
      // Play all collected audio chunks
      if (this.audioChunksBuffer.length > 0) {
        await this.audioManager.playAudioStream(this.audioChunksBuffer);
      }
      
      // Ready for next user input
      this.conversationState = 'idle';
      console.log('✅ Ready for your response - speak naturally');
      
    } catch (error) {
      console.error('❌ Audio playback failed:', error);
      this.conversationState = 'idle';
    }
  }

  async handleAudioRecorded(audioBlob) {
    try {
      this.conversationState = 'processing';
      console.log('📤 Sending your response to AI...');
      
      // Send audio to server via WebSocket
      this.wsManager.sendAudioData(audioBlob);
      
    } catch (error) {
      console.error('❌ Failed to send audio:', error);
      this.conversationState = 'idle';
      throw error;
    }
  }

  handleConnectionError(error) {
    console.error('❌ Connection error:', error);
    throw new Error('Connection lost - please refresh and try again');
  }

  handleConnectionClose(event) {
    if (event.code !== 1000) {
      console.error('❌ Connection closed unexpectedly:', event.code);
      throw new Error('Connection lost unexpectedly');
    }
  }

  handleConversationEnd(data) {
    this.conversationState = 'complete';
    console.log('✅ Standup completed successfully');
    
    // Stop listening
    this.audioManager.stopListening();
    
    // Return completion data
    return {
      evaluation: data.evaluation,
      score: data.score,
      summary: data.text
    };
  }

  getConversationState() {
    return this.conversationState;
  }

  disconnect() {
    this.audioManager.stopListening();
    this.wsManager.disconnect();
    this.conversationState = 'idle';
  }
}

// Create singleton instance
const realisticStandupAPI = new RealisticStandupAPIService();

// Export for compatibility
export const standupCallAPI = realisticStandupAPI;
export const dailyStandupAPI = realisticStandupAPI;
export default realisticStandupAPI;