// src/services/API/index2.js
// Assessment API configuration - Environment driven with WebSocket support

// Assessment API Base URL from environment - keeping your server IP
const ASSESSMENT_API_BASE_URL = import.meta.env.VITE_ASSESSMENT_API_URL || 
                                import.meta.env.VITE_API_BASE_URL ||
                                'https://192.168.48.201:8060';  // Your Linux server IP

// WebSocket URL configuration - supporting both HTTP and HTTPS
const getWebSocketURL = () => {
  const baseURL = ASSESSMENT_API_BASE_URL;
  // Use WSS for HTTPS, WS for HTTP
  const protocol = baseURL.startsWith('https://') ? 'wss://' : 'ws://';
  const host = baseURL.replace(/^https?:\/\//, '');
  return `${protocol}${host}`;
};

// Get authentication token from localStorage with fallback
const getAuthToken = () => {
  return localStorage.getItem('token') || 
         sessionStorage.getItem('token') || 
         localStorage.getItem('authToken') ||
         sessionStorage.getItem('authToken');
};

// Common headers for assessment API requests
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
  
  return headers;
};

// Network timeout configuration
const DEFAULT_TIMEOUT = 30000; // 30 seconds
const UPLOAD_TIMEOUT = 60000;  // 60 seconds for file uploads
const WEBSOCKET_TIMEOUT = 300000; // 5 minutes for WebSocket

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// WebSocket connection manager
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
      if (onError) onError(error);
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

// Generic assessment API request function with enhanced error handling
export const assessmentApiRequest = async (endpoint, options = {}) => {
  const url = `${ASSESSMENT_API_BASE_URL}${endpoint}`;
  
  const isFormData = options.body instanceof FormData;
  const timeout = options.timeout || (isFormData ? UPLOAD_TIMEOUT : DEFAULT_TIMEOUT);
  
  const config = {
    headers: getAssessmentHeaders(isFormData),
    ...options,
  };

  // Add timeout support
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  config.signal = controller.signal;

  let lastError = null;
  
  // Retry logic
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

      // Clear timeout on successful response
      clearTimeout(timeoutId);

      // Handle different response types
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
        
        // Create detailed error message
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
        
        // Don't retry client errors (4xx), only server errors (5xx) or network issues
        if (response.status >= 400 && response.status < 500) {
          throw error;
        }
        
        lastError = error;
        
        // If this is the last attempt, throw the error
        if (attempt === MAX_RETRIES) {
          throw error;
        }
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
        continue;
      }
      
      // Check if response has content
      const contentType = response.headers.get('content-type');
      if (contentType && contentType.includes('application/json')) {
        const jsonData = await response.json();
        console.log('? Assessment API JSON Response:', jsonData);
        return jsonData;
      } else if (contentType && contentType.includes('application/pdf')) {
        const blob = await response.blob();
        console.log('? Assessment API PDF Response:', blob.size, 'bytes');
        return blob;
      } else {
        const textData = await response.text();
        console.log('? Assessment API Text Response:', textData);
        return textData;
      }
      
    } catch (error) {
      clearTimeout(timeoutId);
      
      console.error(`?? Assessment API request failed (attempt ${attempt}):`, {
        url,
        error: error.message,
        name: error.name,
        attempt: attempt
      });
      
      lastError = error;
      
      // Handle specific error types
      if (error.name === 'AbortError') {
        const timeoutError = new Error(`Request timeout after ${timeout}ms`);
        timeoutError.name = 'TimeoutError';
        lastError = timeoutError;
      }
      
      // If this is the last attempt, throw the error
      if (attempt === MAX_RETRIES) {
        break;
      }
      
      // Don't retry on certain errors
      if (error.name === 'TypeError' && error.message.includes('Failed to fetch')) {
        // Network error - retry
        await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
        continue;
      } else if (error.status && error.status >= 400 && error.status < 500) {
        // Client error - don't retry
        break;
      }
      
      // Wait before retrying
      await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * attempt));
    }
  }
  
  // Provide more helpful error messages
  if (lastError) {
    if (lastError.message.includes('Failed to fetch') || lastError.name === 'TypeError') {
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

// Connection test function
export const testAPIConnection = async () => {
  try {
    console.log('?? Testing API connection...');
    
    const response = await assessmentApiRequest('/weekly_interview/health', {
      method: 'GET',
      timeout: 10000 // Shorter timeout for connection test
    });
    
    console.log('? API connection test successful:', response);
    return {
      status: 'success',
      message: 'API connection successful',
      response: response,
      baseUrl: ASSESSMENT_API_BASE_URL
    };
  } catch (error) {
    console.error('? API connection test failed:', error);
    return {
      status: 'failed',
      message: error.message,
      error: error,
      baseUrl: ASSESSMENT_API_BASE_URL
    };
  }
};

// Configuration validation
export const validateAPIConfig = () => {
  const config = {
    baseUrl: ASSESSMENT_API_BASE_URL,
    wsUrl: getWebSocketURL(),
    hasToken: !!getAuthToken(),
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
  
  return {
    isValid: issues.length === 0,
    issues: issues,
    config: config
  };
};

// Environment detection
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

// Audio recording utility
export const recordAudio = async (duration = 30000) => {
  try {
    console.log('?? Starting audio recording...');
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: 44100,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      } 
    });
    
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });
    
    const audioChunks = [];
    
    return new Promise((resolve, reject) => {
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log('? Audio recording completed, size:', audioBlob.size, 'bytes');
        
        // Stop all tracks to release microphone
        stream.getTracks().forEach(track => track.stop());
        
        resolve(audioBlob);
      };
      
      mediaRecorder.onerror = (error) => {
        console.error('? MediaRecorder error:', error);
        stream.getTracks().forEach(track => track.stop());
        reject(error);
      };
      
      mediaRecorder.start();
      console.log('?? Recording started...');
      
      // Auto-stop after duration
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          console.log('?? Auto-stopped recording after', duration, 'ms');
        }
      }, duration);
    });
    
  } catch (error) {
    console.error('? Failed to start audio recording:', error);
    throw new Error(`Audio recording failed: ${error.message}`);
  }
};

// Export the base URL and WebSocket manager for use in other modules
export { ASSESSMENT_API_BASE_URL, wsManager, getWebSocketURL };

// Default export with enhanced functionality
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
  
  // Configuration constants
  DEFAULT_TIMEOUT,
  UPLOAD_TIMEOUT,
  WEBSOCKET_TIMEOUT,
  MAX_RETRIES,
  RETRY_DELAY
};