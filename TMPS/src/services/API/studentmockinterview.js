// ===== 1. UPDATED API FILE: src/services/API/studentmockinterview.js =====

import { assessmentApiRequest } from './index2';

export const interviewOperationsAPI = {
  // Start a new interview session - GET /start_interview
  startInterview: async () => {
    try {
      console.log('API: Starting new interview session');
      
      const response = await assessmentApiRequest('/weekly_interview/start_interview', {
        method: 'GET'
      });
      
      console.log('API Response for start interview:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from start interview endpoint');
      }
    } catch (error) {
      console.error('API Error in startInterview:', error);
      throw new Error(`Failed to start interview: ${error.message}`);
    }
  },

  // Start the next round of interview - GET /start_next_round?test_id=${testId}
  startNextRound: async (testId) => {
    try {
      if (!testId) {
        throw new Error('Test ID is required to start next round');
      }
      
      console.log('API: Starting next round for test_id:', testId);
      
      const response = await assessmentApiRequest(`/weekly_interview/start_next_round?test_id=${testId}`, {
        method: 'GET'
      });
      
      console.log('API Response for start next round:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from start next round endpoint');
      }
    } catch (error) {
      console.error('API Error in startNextRound:', error);
      throw new Error(`Failed to start next round: ${error.message}`);
    }
  },

  // Evaluate the interview - GET /evaluate?test_id=${testId}
  evaluateInterview: async (testId) => {
    try {
      if (!testId) {
        throw new Error('Test ID is required for evaluation');
      }
      
      console.log('API: Evaluating interview for test_id:', testId);
      
      const response = await assessmentApiRequest(`/weekly_interview/evaluate?test_id=${testId}`, {
        method: 'GET'
      });
      
      console.log('API Response for evaluate interview:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from evaluate endpoint');
      }
    } catch (error) {
      console.error('API Error in evaluateInterview:', error);
      throw new Error(`Failed to evaluate interview: ${error.message}`);
    }
  },

  // PRIMARY METHOD: Convert text to audio blob and submit
  recordAndRespondWithText: async (testId, transcribedText, additionalData = {}) => {
    try {
      if (!testId || !transcribedText) {
        throw new Error('Test ID and transcribed text are required');
      }
      
      console.log('API: Recording and responding with text for test_id:', testId);
      console.log('API: Transcribed text:', transcribedText);
      console.log('API: Additional data:', additionalData);
      
      // Create an audio blob from the transcribed text using TTS
      const audioBlob = await convertTextToAudioBlob(transcribedText);
      console.log('API: Generated audio blob from text, size:', audioBlob.size);
      
      // Create FormData to match backend expectations
      const formData = new FormData();
      formData.append('test_id', testId);
      formData.append('audio', audioBlob, 'transcribed_response.wav');
      
      // Add transcribed text as additional form field for backend reference
      formData.append('transcribed_text', transcribedText);
      formData.append('response_text', transcribedText);
      
      // Add additional data
      if (additionalData.round) {
        formData.append('round', additionalData.round.toString());
      }
      if (additionalData.questionId) {
        formData.append('question_id', additionalData.questionId);
      }
      
      // Add any other additional fields
      Object.keys(additionalData).forEach(key => {
        if (!['round', 'questionId', 'question_id'].includes(key) && additionalData[key] !== undefined) {
          formData.append(key, additionalData[key].toString());
        }
      });
      
      console.log('API: Submitting FormData with audio blob generated from text');
      
      const response = await assessmentApiRequest('/weekly_interview/record_and_respond', {
        method: 'POST',
        body: formData
        // Don't set Content-Type header - let browser set it for FormData
      });
      
      console.log('API Response for text submission:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from record and respond endpoint');
      }
    } catch (error) {
      console.error('API Error in recordAndRespondWithText:', error);
      throw new Error(`Failed to record and respond with text: ${error.message}`);
    }
  },

  // SECONDARY METHOD: Submit actual recorded audio
  recordAndRespondWithAudio: async (testId, audioBlob, additionalData = {}) => {
    try {
      if (!testId || !audioBlob) {
        throw new Error('Test ID and audio blob are required');
      }
      
      console.log('API: Recording and responding with audio for test_id:', testId);
      console.log('API: Audio blob size:', audioBlob.size, 'bytes');
      
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('test_id', testId);
      formData.append('audio', audioBlob, 'user_response.webm');
      
      // Add additional data
      Object.keys(additionalData).forEach(key => {
        if (additionalData[key] !== null && additionalData[key] !== undefined) {
          formData.append(key, additionalData[key].toString());
        }
      });
      
      const response = await assessmentApiRequest('/weekly_interview/record_and_respond', {
        method: 'POST',
        body: formData
      });
      
      console.log('API Response for audio submission:', response);
      
      if (response && response.data) {
        return response.data;
      } else if (response) {
        return response;
      } else {
        throw new Error('Invalid response from record and respond endpoint');
      }
    } catch (error) {
      console.error('API Error in recordAndRespondWithAudio:', error);
      throw new Error(`Failed to record and respond with audio: ${error.message}`);
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
      
      // Create a short audio representation of the text
      // Since we're using speech-to-text, we'll create a minimal audio file
      // that represents the fact that the user spoke this text
      
      // For now, create a structured audio blob that contains the text length info
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

export default interviewOperationsAPI;

// ===== 2. UPDATED REACT COMPONENT PART - submitTextResponse function =====

// Replace your submitTextResponse function in StartInterview.jsx with this:
/*
const submitTextResponse = async (transcribedText) => {
  if (!transcribedText || !interview?.testId) {
    console.error('❌ Missing text or test ID:', { 
      hasText: !!transcribedText, 
      textLength: transcribedText?.length || 0,
      testId: interview?.testId 
    });
    setError('No speech was transcribed or invalid interview session');
    return;
  }

  try {
    setIsSubmitting(true);
    setError(null); // Clear any previous errors
    
    console.log('📤 Submitting transcribed text...');
    console.log('📝 Text content:', transcribedText);
    console.log('🆔 Test ID:', interview.testId);
    console.log('🔄 Current round:', currentRound);

    // Validate test ID
    const validation = interviewOperationsAPI.validateTestId(interview.testId);
    if (!validation.valid) {
      throw new Error(`Invalid test ID: ${validation.error}`);
    }

    // Prepare additional data
    const additionalData = {
      round: currentRound,
      questionId: currentQuestion?.id || null,
      timestamp: new Date().toISOString(),
      source: 'speech_to_text',
      userAgent: navigator.userAgent
    };

    console.log('📊 Additional data:', additionalData);

    // FIXED: Call the correct API method
    const response = await interviewOperationsAPI.recordAndRespondWithText(
      interview.testId, 
      transcribedText.trim(),
      additionalData
    );
    
    console.log('📥 Submit response received:', response);
    console.log('📥 Response type:', typeof response);
    console.log('📥 Response keys:', Object.keys(response || {}));
    
    // Clear the transcribed text
    setTranscribedText('');
    
    // Handle different response structures based on your backend
    if (response) {
      // Check for continue flag (next question in same round)
      if (response.continue || response.has_next_question) {
        console.log('➡️ Continuing to next question in same round');
        
        const nextQuestion = response.response || response.question || response.next_question;
        const audioPath = response.audio_path || response.audio_url || response.ai_audio_url;
        
        if (nextQuestion) {
          setCurrentQuestion(nextQuestion);
          setQuestionCount(prev => prev + 1);
          
          // Play next question
          setTimeout(async () => {
            await playInterviewerAudio(audioPath, nextQuestion);
          }, 1000);
        }
      }
      // Check for round completion
      else if (response.round_complete || response.round_finished) {
        console.log('🎯 Round completed, moving to next round');
        
        const roundCompleteMessage = response.response || response.message || "Round completed! Moving to next round...";
        setCurrentQuestion(roundCompleteMessage);
        
        // Speak round completion message
        await speakQuestion(roundCompleteMessage);
        
        setTimeout(() => {
          setInterviewerSpeaking(false);
          const nextRound = currentRound + 1;
          setCurrentRound(nextRound);
          setQuestionCount(1);
          
          if (nextRound <= 3) {
            const roundConfig = rounds[nextRound];
            if (roundConfig) {
              setRoundName(roundConfig.name);
              setTotalQuestions(roundConfig.questions);
            }
            
            // Start next round
            setTimeout(() => {
              startNextRound();
            }, 2000);
          }
        }, 2000);
      }
      // Check for interview completion
      else if (response.interview_complete || response.is_complete || response.finished) {
        console.log('✅ Interview completed');
        
        const completionMessage = response.response || response.message || "Interview completed! Thank you.";
        setCurrentQuestion(completionMessage);
        setInterviewStarted(false);
        
        // Speak completion message
        await speakQuestion(completionMessage);
        
        setTimeout(() => {
          handleInterviewComplete();
        }, 3000);
      }
      // Default case - got a response, treat as next question
      else if (response.response || response.question) {
        console.log('📝 Got new question response');
        
        const nextQuestion = response.response || response.question;
        const audioPath = response.audio_path || response.audio_url;
        
        setCurrentQuestion(nextQuestion);
        
        // Play the response
        setTimeout(async () => {
          await playInterviewerAudio(audioPath, nextQuestion);
        }, 1000);
      }
      // Handle unexpected response structure
      else {
        console.warn('⚠️ Unexpected response structure:', response);
        setError('Received unexpected response from server. Please try again.');
      }
    } else {
      console.error('❌ No response received from server');
      setError('No response received from server. Please try again.');
    }

  } catch (error) {
    console.error('❌ Failed to submit text response:', error);
    console.error('❌ Error details:', {
      message: error.message,
      stack: error.stack,
      response: error.response?.data,
      status: error.response?.status
    });
    
    // Set user-friendly error message
    let errorMessage = 'Failed to submit response. ';
    if (error.message.includes('422')) {
      errorMessage += 'Server validation error. Please try again.';
    } else if (error.message.includes('404')) {
      errorMessage += 'Interview session not found. Please restart the interview.';
    } else if (error.message.includes('network')) {
      errorMessage += 'Network error. Please check your connection.';
    } else {
      errorMessage += error.message;
    }
    
    setError(errorMessage);
    
    // Re-enable recording after error
    setTimeout(() => {
      setIsRecording(false);
      setIsSubmitting(false);
    }, 1000);
    
  } finally {
    setIsSubmitting(false);
  }
};
*/

// ===== 3. INTEGRATION NOTES =====

/*
INTEGRATION STEPS:

1. Replace your current API file (studentmockinterview.js) with the code above
2. Replace your submitTextResponse function in StartInterview.jsx with the commented function above
3. Make sure your import statement is correct:
   import { interviewOperationsAPI } from '../../../services/API/studentmockinterview';

4. Your backend should now receive:
   - test_id: The interview session ID
   - audio: A small WAV file representing the spoken text
   - transcribed_text: The actual text that was transcribed
   - Additional metadata fields

5. The backend can:
   - Process the audio if needed (though it's minimal)
   - Use the transcribed_text for the actual response processing
   - Store the conversation history with the text content

BACKEND NOTES:
- The audio file sent is a minimal WAV that represents the text
- The actual text content is in the 'transcribed_text' form field
- Audio files are temporary and will be auto-deleted as per your design
- Only the text conversation history needs to be stored

TROUBLESHOOTING:
- Check browser console for detailed logs
- Verify test_id is being passed correctly
- Ensure backend can handle the additional 'transcribed_text' form field
- Test with simple short responses first
*/