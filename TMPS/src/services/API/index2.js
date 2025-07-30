// src/services/API/index2.js
// Complete update with SSL error handling for development

// Assessment API configuration with SSL bypass for development
const ASSESSMENT_API_BASE_URL = import.meta.env.VITE_ASSESSMENT_API_URL || 
                                import.meta.env.VITE_API_BASE_URL ||
                                'https://192.168.48.201:8070';

// Development SSL bypass configuration
const isDevelopment = import.meta.env.MODE === 'development' || import.meta.env.DEV;

// Enhanced error handling for SSL issues
const handleSSLErrors = (error) => {
  if (error.message.includes('SSL') || 
      error.message.includes('certificate') || 
      error.message.includes('ERR_SSL') ||
      error.message.includes('net::ERR_CERT')) {
    console.warn('?? SSL Certificate Error detected:', error.message);
    console.warn('?? To fix this:');
    console.warn('   1. Accept the certificate at: ' + ASSESSMENT_API_BASE_URL + '/weekly_interview/health');
    console.warn('   2. Or start browser with: --ignore-certificate-errors');
    console.warn('   3. Or recreate SSL certificate as instructed');
    
    return new Error(`SSL Certificate Error: Please accept the self-signed certificate by visiting ${ASSESSMENT_API_BASE_URL}/weekly_interview/health in your browser first.`);
  }
  return error;
};

// WebSocket URL configuration with SSL handling
const getWebSocketURL = () => {
  const baseURL = ASSESSMENT_API_BASE_URL;
  const protocol = baseURL.startsWith('https://') ? 'wss://' : 'ws://';
  const host = baseURL.replace(/^https?:\/\//, '');
  return `${protocol}${host}`;
};

// Get authentication token
const getAuthToken = () => {
  return localStorage.getItem('token') || 
         sessionStorage.getItem('token') || 
         localStorage.getItem('authToken') ||
         sessionStorage.getItem('authToken');
};

// Common headers with development SSL handling
const getAssessmentHeaders = (isFormData = false) => {
  const headers = {};
  
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  headers['Accept'] = 'application/json';
  
  // Add headers to help with SSL issues in development
  if (isDevelopment) {
    headers['Cache-Control'] = 'no-cache';
    headers['Pragma'] = 'no-cache';
  }
  
  return headers;
};

// Configuration
const DEFAULT_TIMEOUT = 30000;
const UPLOAD_TIMEOUT = 60000;
const WEBSOCKET_TIMEOUT = 300000;
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// Natural Interview Audio Configuration (unchanged)
const NATURAL_AUDIO_CONFIG = {
  SILENCE_THRESHOLD: 0.015,
  SILENCE_DURATION: 2000,
  MAX_RECORDING_TIME: 30000,
  SAMPLE_RATE: 44100,
  ECHO_CANCELLATION: true,
  NOISE_SUPPRESSION: true,
  AUTO_GAIN_CONTROL: true,
  MIN_SPEECH_DURATION: 1000
};

// Enhanced WebSocket connection manager with SSL error handling
class WebSocketManager {
  constructor() {
    this.connections = new Map();
  }

  connect(sessionId, onMessage, onError, onClose) {
    const wsURL = `${getWebSocketURL()}/weekly_interview/ws/${sessionId}`;
    console.log('?? Connecting to WebSocket:', wsURL);

    const ws = new WebSocket(wsURL);
    
    ws.onopen = () => {
      console.log('? WebSocket connected for session:', sessionId);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('?? WebSocket message received:', data);
        if (onMessage) onMessage(data);
      } catch (error) {
        console.error('? WebSocket message parse error:', error);
        if (onError) onError(error);
      }
    };

    ws.onerror = (error) => {
      console.error('? WebSocket error:', error);
      
      // Handle SSL-related WebSocket errors
      if (error.type === 'error') {
        const sslError = new Error('WebSocket SSL connection failed. Please accept the SSL certificate first.');
        if (onError) onError(handleSSLErrors(sslError));
      } else {
        if (onError) onError(error);
      }
    };

    ws.onclose = (event) => {
      console.log('?? WebSocket closed:', event.code, event.reason);
      this.connections.delete(sessionId);
      if (onClose) onClose(event);
    };

    this.connections.set(sessionId, ws);
    return ws;
  }

  send(sessionId, data) {
    const ws = this.connections.get(sessionId);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data));
      console.log('?? WebSocket message sent:', data);
      return true;
    } else {
      console.error('? WebSocket not connected for session:', sessionId);
      return false;
    }
  }

  disconnect(sessionId) {
    const ws = this.connections.get(sessionId);
    if (ws) {
      ws.close();
      this.connections.delete(sessionId);
      console.log('?? WebSocket disconnected for session:', sessionId);
    }
  }

  disconnectAll() {
    for (const [sessionId, ws] of this.connections) {
      ws.close();
    }
    this.connections.clear();
    console.log('?? All WebSocket connections closed');
  }
}

// Global WebSocket manager instance
const wsManager = new WebSocketManager();

// Enhanced audio recording (unchanged from your version)
export const recordAudio = async (duration = NATURAL_AUDIO_CONFIG.MAX_RECORDING_TIME) => {
  try {
    console.log('?? Starting natural interview audio recording...');
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: NATURAL_AUDIO_CONFIG.SAMPLE_RATE,
        channelCount: 1,
        echoCancellation: NATURAL_AUDIO_CONFIG.ECHO_CANCELLATION,
        noiseSuppression: NATURAL_AUDIO_CONFIG.NOISE_SUPPRESSION,
        autoGainControl: NATURAL_AUDIO_CONFIG.AUTO_GAIN_CONTROL
      } 
    });
    
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    const audioChunks = [];
    let silenceStart = null;
    let isRecording = true;
    let hasSpoken = false;
    let speechStartTime = null;
    
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);
    
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.8;
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    
    microphone.connect(analyser);
    
    return new Promise((resolve, reject) => {
      const checkAudioLevel = () => {
        if (!isRecording) return;
        
        analyser.getByteFrequencyData(dataArray);
        const averageLevel = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length;
        const normalizedLevel = averageLevel / 255;
        
        if (normalizedLevel > NATURAL_AUDIO_CONFIG.SILENCE_THRESHOLD) {
          if (!hasSpoken) {
            hasSpoken = true;
            speechStartTime = Date.now();
            console.log('??? User started speaking...');
          }
          silenceStart = null;
        } else if (hasSpoken && normalizedLevel <= NATURAL_AUDIO_CONFIG.SILENCE_THRESHOLD) {
          const speechDuration = Date.now() - speechStartTime;
          
          if (speechDuration >= NATURAL_AUDIO_CONFIG.MIN_SPEECH_DURATION) {
            if (silenceStart === null) {
              silenceStart = Date.now();
              console.log('?? Silence detected after speech, starting timer...');
            } else {
              const silenceElapsed = Date.now() - silenceStart;
              if (silenceElapsed >= NATURAL_AUDIO_CONFIG.SILENCE_DURATION) {
                console.log(`?? ${NATURAL_AUDIO_CONFIG.SILENCE_DURATION}ms of silence - natural pause detected`);
                stopRecording('natural_pause');
                return;
              }
            }
          }
        }
        
        requestAnimationFrame(checkAudioLevel);
      };
      
      const stopRecording = (reason) => {
        if (!isRecording) return;
        
        isRecording = false;
        console.log(`?? Stopping recording: ${reason}`);
        
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
        }
        
        stream.getTracks().forEach(track => track.stop());
        if (audioContext.state !== 'closed') {
          audioContext.close();
        }
      };
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log(`? Recording completed, size: ${audioBlob.size} bytes`);
        resolve(audioBlob);
      };
      
      mediaRecorder.onerror = (error) => {
        console.error('? MediaRecorder error:', error);
        stopRecording('error');
        reject(error);
      };
      
      mediaRecorder.start();
      console.log('?? Recording started, waiting for user to speak...');
      
      checkAudioLevel();
      
      setTimeout(() => {
        if (isRecording) {
          console.log('? Maximum duration reached');
          stopRecording('max_duration');
        }
      }, duration);
    });
    
  } catch (error) {
    console.error('? Failed to start natural audio recording:', error);
    throw new Error(`Natural audio recording failed: ${error.message}`);
  }
};

// Enhanced assessment API request function with SSL error handling
export const assessmentApiRequest = async (endpoint, options = {}) => {
  const url = `${ASSESSMENT_API_BASE_URL}${endpoint}`;
  
  const isFormData = options.body instanceof FormData;
  const timeout = options.timeout || (isFormData ? UPLOAD_TIMEOUT : DEFAULT_TIMEOUT);
  
  const config = {
    headers: getAssessmentHeaders(isFormData),
    ...options,
  };

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  config.signal = controller.signal;

  let lastError = null;
  
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`?? Assessment API Request (attempt ${attempt}):`, {
        url,
        method: config.method || 'GET',
        headers: config.headers,
        hasBody: !!config.body,
        timeout: timeout
      });

      const response = await fetch(url, config);
      
      console.log('?? Assessment API Response:', {
        status: response.status,
        statusText: response.statusText,
        url: response.url,
        ok: response.ok,
        attempt: attempt
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        let errorData;
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            errorData = await response.json();
          } else {
            errorData = await response.text();
          }
        } catch (e) {
          errorData = response.statusText;
        }
        
        console.error('? Assessment API Error Response:', {
          status: response.status,
          data: errorData,
          attempt: attempt
        });
        
        let errorMessage = `HTTP ${response.status}`;
        if (typeof errorData === 'object' && errorData.detail) {
          errorMessage += `: ${errorData.detail}`;
        } else if (typeof errorData === 'string') {
          errorMessage += `: ${errorData}`;
        } else {
          errorMessage += `: ${response.statusText}`;
        }
        
        const error = new Error(errorMessage);
        error.status = response.status;
        error.response = { data: errorData };
        
        if (response.status >= 400 && response.status < 500) {
          throw error;
        }
        
        lastError = error;
        
        if (attempt === MAX_RETRIES) {
          throw error;
        }
        
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
        continue;
      }
      
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const jsonData = await response.json();
        console.log('? Assessment API JSON Response:', jsonData);
        return jsonData;
      } else if (contentType && contentType.includes('application/pdf')) {
        const blob = await response.blob();
        console.log('?? Assessment API PDF Response:', blob.size, 'bytes');
        return blob;
      } else {
        const textData = await response.text();
        console.log('?? Assessment API Text Response:', textData);
        return textData;
      }
      
    } catch (error) {
      clearTimeout(timeoutId);
      
      console.error(`? Assessment API request failed (attempt ${attempt}):`, {
        url,
        error: error.message,
        name: error.name,
        attempt: attempt
      });
      
      // Handle SSL-specific errors
      const processedError = handleSSLErrors(error);
      lastError = processedError;
      
      if (error.name === 'AbortError') {
        const timeoutError = new Error(`Request timeout after ${timeout}ms`);
        timeoutError.name = 'TimeoutError';
        lastError = timeoutError;
      }
      
      if (attempt === MAX_RETRIES) {
        break;
      }
      
      // Don't retry SSL certificate errors
      if (error.message.includes('SSL') || error.message.includes('certificate')) {
        break;
      }
      
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
        continue;
      } else if (error.status && error.status >= 400 && error.status < 500) {
        break;
      }
      
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
    }
  }
  
  if (lastError) {
    if (lastError.message.includes('SSL Certificate Error')) {
      throw lastError; // Throw the enhanced SSL error message
    } else if (lastError.message.includes('Failed to fetch') || lastError.name === 'TypeError') {
      const networkError = new Error(`Network error: Cannot connect to ${ASSESSMENT_API_BASE_URL}. Please check your internet connection and verify the server is running.`);
      networkError.originalError = lastError;
      throw networkError;
    } else if (lastError.message.includes('CORS')) {
      const corsError = new Error(`CORS error: Server needs to allow requests from your domain. Please contact your administrator.`);
      corsError.originalError = lastError;
      throw corsError;
    } else if (lastError.name === 'TimeoutError') {
      const timeoutError = new Error(`Request timeout: Server took too long to respond. Please try again or contact support if the issue persists.`);
      timeoutError.originalError = lastError;
      throw timeoutError;
    } else {
      throw lastError;
    }
  }
  
  throw new Error('Unknown error occurred during API request');
};

// Enhanced connection test function with SSL guidance
export const testAPIConnection = async () => {
  try {
    console.log('?? Testing API connection to:', ASSESSMENT_API_BASE_URL);
    
    const healthEndpoints = [
      '/weekly_interview/health',
      '/health',
      '/'
    ];
    
    for (const endpoint of healthEndpoints) {
      try {
        console.log(`?? Trying endpoint: ${endpoint}`);
        const response = await assessmentApiRequest(endpoint, {
          method: 'GET',
          timeout: 5000
        });
        
        console.log('? API connection test successful:', response);
        return {
          status: 'success',
          message: 'API connection successful',
          response: response,
          baseUrl: ASSESSMENT_API_BASE_URL,
          endpoint: endpoint
        };
      } catch (endpointError) {
        console.log(`? Endpoint ${endpoint} failed:`, endpointError.message);
        continue;
      }
    }
    
    throw new Error('All health endpoints failed');
    
  } catch (error) {
    console.error('? API connection test failed:', error);
    
    // Provide specific guidance for SSL errors
    let message = error.message;
    if (error.message.includes('SSL Certificate Error')) {
      message += '\n\n?? SSL Fix Instructions:\n1. Visit ' + ASSESSMENT_API_BASE_URL + '/weekly_interview/health\n2. Accept the security warning\n3. Refresh this page';
    }
    
    return {
      status: 'failed',
      message: message,
      error: error,
      baseUrl: ASSESSMENT_API_BASE_URL
    };
  }
};

// Configuration validation with SSL check
export const validateAPIConfig = () => {
  const config = {
    baseUrl: ASSESSMENT_API_BASE_URL,
    wsUrl: getWebSocketURL(),
    hasToken: !!getAuthToken(),
    isDevelopment: isDevelopment,
    tokenSource: getAuthToken() ? 
      (localStorage.getItem('token') ? 'localStorage' : 
       sessionStorage.getItem('token') ? 'sessionStorage' : 
       localStorage.getItem('authToken') ? 'localStorage(authToken)' : 
       'sessionStorage(authToken)') : 'none'
  };
  
  console.log('?? API Configuration:', config);
  
  const issues = [];
  
  if (!config.baseUrl) {
    issues.push('API base URL not configured');
  }
  
  if (!config.baseUrl.startsWith('http')) {
    issues.push('API base URL should start with http:// or https://');
  }
  
  if (config.baseUrl.startsWith('https://') && isDevelopment) {
    issues.push('HTTPS in development - ensure SSL certificate is accepted');
  }
  
  return {
    isValid: issues.length === 0,
    issues: issues,
    config: config
  };
};

// Environment detection (unchanged)
export const getEnvironmentInfo = () => {
  return {
    mode: import.meta.env.MODE || 'production',
    isDevelopment: import.meta.env.DEV || false,
    isProduction: import.meta.env.PROD || true,
    baseUrl: ASSESSMENT_API_BASE_URL,
    wsUrl: getWebSocketURL(),
    hasViteConfig: !!(import.meta.env.VITE_ASSESSMENT_API_URL || import.meta.env.VITE_API_BASE_URL),
    envVars: {
      VITE_ASSESSMENT_API_URL: import.meta.env.VITE_ASSESSMENT_API_URL,
      VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
      NODE_ENV: import.meta.env.NODE_ENV
    }
  };
};

// Export everything
export { ASSESSMENT_API_BASE_URL, wsManager, getWebSocketURL };

export default {
  assessmentApiRequest,
  testAPIConnection,
  validateAPIConfig,
  getEnvironmentInfo,
  recordAudio,
  wsManager,
  ASSESSMENT_API_BASE_URL,
  getWebSocketURL,
  getAuthToken,
  getAssessmentHeaders,
  DEFAULT_TIMEOUT,
  UPLOAD_TIMEOUT,
  WEBSOCKET_TIMEOUT,
  MAX_RETRIES,
  RETRY_DELAY,
  NATURAL_AUDIO_CONFIG
};