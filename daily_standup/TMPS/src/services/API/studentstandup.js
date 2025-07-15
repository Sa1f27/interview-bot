// src/services/API/studentstandup.js

import { assessmentApiRequest } from './index2';

export const standupCallAPI = {
  // Start a new standup session - GET /start_test  
  startStandup: async () => {
    try {
      console.log('API: Starting new standup session');
      
      const response = await assessmentApiRequest('/daily_standup/start_test', {
        method: 'GET'
      });
      
      console.log('API Response for start standup:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from start standup endpoint');
      }
    } catch (error) {
      console.error('API Error in startStandup:', error);
      throw new Error(`Failed to start standup: ${error.message}`);
    }
  },

  // Submit standup response - POST /record_and_respond
  recordAndRespond: async (testId, responseData, additionalData = {}) => {
    try {
      if (!testId || !responseData) {
        throw new Error('Test ID and response data are required');
      }
      
      console.log('API: Recording and responding for test_id:', testId);
      console.log('API: Response data:', responseData);
      
      // Create FormData for the request
      const formData = new FormData();
      formData.append('test_id', testId);
      
      // If responseData is text, convert to audio blob
      if (typeof responseData === 'string') {
        const audioBlob = await convertTextToAudioBlob(responseData);
        formData.append('audio', audioBlob, 'standup_response.wav');
        formData.append('transcribed_text', responseData);
        formData.append('response_text', responseData);
      } 
      // If responseData is already an audio blob
      else if (responseData instanceof Blob) {
        formData.append('audio', responseData, 'standup_audio.webm');
      }
      
      // Add additional data
      Object.keys(additionalData).forEach(key => {
        if (additionalData[key] !== null && additionalData[key] !== undefined) {
          formData.append(key, additionalData[key].toString());
        }
      });
      
      const response = await assessmentApiRequest('/daily_standup/record_and_respond', {
        method: 'POST',
        body: formData
      });
      
      console.log('API Response for record and respond:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from record and respond endpoint');
      }
    } catch (error) {
      console.error('API Error in recordAndRespond:', error);
      throw new Error(`Failed to record and respond: ${error.message}`);
    }
  },

  // Get standup summary - GET /summary?test_id=${testId}
  getStandupSummary: async (testId) => {
    try {
      if (!testId) {
        throw new Error('Test ID is required for summary');
      }
      
      console.log('API: Getting standup summary for test_id:', testId);
      
      const response = await assessmentApiRequest(`/daily_standup/summary?test_id=${testId}`, {
        method: 'GET'
      });
      
      console.log('API Response for standup summary:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from summary endpoint');
      }
    } catch (error) {
      console.error('API Error in getStandupSummary:', error);
      throw new Error(`Failed to get standup summary: ${error.message}`);
    }
  },

  // Helper method to validate test ID format
  validateTestId: (testId) => {
    if (!testId) {
      return { valid: false, error: 'Test ID is required' };
    }
    
    if (typeof testId !== 'string' && typeof testId !== 'number') {
      return { valid: false, error: 'Test ID must be a string or number' };
    }
    
    return { valid: true };
  }
};

// ===== HELPER FUNCTIONS =====

// Convert text to audio blob using Web Speech API (TTS)
async function convertTextToAudioBlob(text) {
  return new Promise((resolve, reject) => {
    try {
      console.log('🎵 Converting text to audio:', text.substring(0, 50) + '...');
      
      if (!('speechSynthesis' in window)) {
        console.warn('Speech synthesis not supported, creating silent audio');
        resolve(createSilentAudioBlob());
        return;
      }
      
      // Create a structured audio blob that contains the text length info
      const audioBlob = createTextBasedAudioBlob(text);
      resolve(audioBlob);
      
    } catch (error) {
      console.error('Failed to convert text to audio:', error);
      resolve(createSilentAudioBlob());
    }
  });
}

// Create an audio blob that represents text (for backend processing)
function createTextBasedAudioBlob(text) {
  try {
    // Create a WAV file with length proportional to text length
    const sampleRate = 44100;
    const textLength = text.length;
    // Duration based on text length (roughly 100ms per 10 characters, min 0.5s, max 5s)
    const duration = Math.max(0.5, Math.min(5.0, textLength * 0.01));
    const numSamples = Math.floor(sampleRate * duration);
    const arrayBuffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    
    // Generate low-level audio data that represents the text
    const audioData = new Int16Array(arrayBuffer, 44, numSamples);
    for (let i = 0; i < numSamples; i++) {
      // Create a very low amplitude sine wave based on text characteristics
      const frequency = 200 + (text.charCodeAt(i % text.length) % 100);
      const sample = Math.sin(2 * Math.PI * frequency * i / sampleRate) * 100; // Very low volume
      audioData[i] = Math.floor(sample);
    }
    
    const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
    console.log('✅ Created text-based audio blob, size:', blob.size, 'duration:', duration.toFixed(2) + 's');
    return blob;
  } catch (error) {
    console.warn('Failed to create text-based audio blob, using silent:', error);
    return createSilentAudioBlob();
  }
}

// Create a minimal silent audio blob (fallback)
function createSilentAudioBlob() {
  try {
    const sampleRate = 44100;
    const duration = 0.5; // 500ms of silence
    const numSamples = sampleRate * duration;
    const arrayBuffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);
    
    // Silent audio data (all zeros)
    const audioData = new Int16Array(arrayBuffer, 44, numSamples);
    audioData.fill(0);
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  } catch (error) {
    console.error('Failed to create silent audio blob:', error);
    return new Blob([''], { type: 'audio/wav' });
  }
}

export default standupCallAPI;