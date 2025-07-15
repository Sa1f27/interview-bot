import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Alert,
  Card,
  CardContent,
  IconButton,
  Avatar,
  LinearProgress,
  Chip,
  Fade,
  useTheme,
  alpha,
  styled,
  keyframes,
  Grid
} from '@mui/material';
import {
  Mic,
  VolumeUp,
  ArrowBack,
  Summarize,
  CheckCircle,
  RadioButtonChecked,
  GraphicEq,
  Timer,
  PlayArrow,
  Stop,
  RecordVoiceOver
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';

// Import the standup API
import { standupCallAPI } from '../../../services/API/studentstandup';

// ==================== CONFIGURATION ====================
// **FIXED: Set your specific backend server details**
const API_BASE_URL = 'https://192.168.48.42:8060';
console.log('🔗 Using API Base URL:', API_BASE_URL);

// ==================== STYLED COMPONENTS ====================

const pulse = keyframes`
  0% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.7);
  }
  70% {
    transform: scale(1.05);
    box-shadow: 0 0 0 10px rgba(244, 67, 54, 0);
  }
  100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(244, 67, 54, 0);
  }
`;

const recording = keyframes`
  0%, 100% { transform: scaleY(0.5); opacity: 0.5; }
  50% { transform: scaleY(1.5); opacity: 1; }
`;

const audioWave = keyframes`
  0%, 100% { transform: scaleY(0.3); }
  25% { transform: scaleY(0.8); }
  50% { transform: scaleY(1.2); }
  75% { transform: scaleY(0.6); }
`;

const blinkMic = keyframes`
  0%, 50% { 
    opacity: 1; 
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7);
  }
  25% { 
    opacity: 0.7; 
    transform: scale(1.1);
    box-shadow: 0 0 0 5px rgba(33, 150, 243, 0.4);
  }
  75% { 
    opacity: 0.9; 
    transform: scale(1.05);
    box-shadow: 0 0 0 8px rgba(33, 150, 243, 0.2);
  }
  100% { 
    opacity: 1; 
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(33, 150, 243, 0);
  }
`;

const speakingCardPulse = keyframes`
  0%, 100% { 
    background: linear-gradient(135deg, rgba(33, 150, 243, 0.1), rgba(33, 150, 243, 0.05));
    box-shadow: 0 4px 20px rgba(33, 150, 243, 0.1);
  }
  50% { 
    background: linear-gradient(135deg, rgba(33, 150, 243, 0.2), rgba(33, 150, 243, 0.1));
    box-shadow: 0 8px 25px rgba(33, 150, 243, 0.2);
  }
`;

const VoiceLevelIndicator = styled(Box)(({ theme, level }) => ({
  width: '100%',
  height: '8px',
  backgroundColor: alpha(theme.palette.success.main, 0.2),
  borderRadius: '4px',
  position: 'relative',
  overflow: 'hidden',
  '&::after': {
    content: '""',
    position: 'absolute',
    left: 0,
    top: 0,
    height: '100%',
    width: `${level * 100}%`,
    backgroundColor: level > 0.05 ? theme.palette.success.main : theme.palette.grey[400],
    borderRadius: '4px',
    transition: 'width 0.1s ease-out, background-color 0.2s ease',
  }
}));

const RecordingIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '6px',
  marginTop: theme.spacing(2),
  '& .bar': {
    width: '6px',
    height: '30px',
    backgroundColor: theme.palette.error.main,
    borderRadius: '3px',
    animation: `${recording} 1.2s ease-in-out infinite`,
    '&:nth-of-type(1)': { animationDelay: '0s' },
    '&:nth-of-type(2)': { animationDelay: '0.2s' },
    '&:nth-of-type(3)': { animationDelay: '0.4s' },
    '&:nth-of-type(4)': { animationDelay: '0.6s' },
    '&:nth-of-type(5)': { animationDelay: '0.8s' },
    '&:nth-of-type(6)': { animationDelay: '1s' },
  }
}));

const AudioWaveIndicator = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '4px',
  marginTop: theme.spacing(2),
  '& .wave': {
    width: '4px',
    height: '20px',
    backgroundColor: theme.palette.info.main,
    borderRadius: '2px',
    animation: `${audioWave} 1.5s ease-in-out infinite`,
    '&:nth-of-type(1)': { animationDelay: '0s' },
    '&:nth-of-type(2)': { animationDelay: '0.1s' },
    '&:nth-of-type(3)': { animationDelay: '0.2s' },
    '&:nth-of-type(4)': { animationDelay: '0.3s' },
    '&:nth-of-type(5)': { animationDelay: '0.4s' },
    '&:nth-of-type(6)': { animationDelay: '0.5s' },
    '&:nth-of-type(7)': { animationDelay: '0.6s' },
  }
}));

const InterviewerSpeakingCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  animation: `${speakingCardPulse} 2s ease-in-out infinite`,
  border: `2px solid ${theme.palette.info.main}`,
  overflow: 'visible',
  position: 'relative',
  '&::before': {
    content: '""',
    position: 'absolute',
    top: -2,
    left: -2,
    right: -2,
    bottom: -2,
    background: `linear-gradient(45deg, ${theme.palette.info.main}, ${theme.palette.info.light})`,
    borderRadius: 'inherit',
    zIndex: -1,
    opacity: 0.3,
    animation: `${pulse} 2s ease-in-out infinite`,
  }
}));

const BlinkingMicAvatar = styled(Avatar)(({ theme }) => ({
  width: 64,
  height: 64,
  backgroundColor: theme.palette.info.main,
  animation: `${blinkMic} 1.5s ease-in-out infinite`,
  border: `3px solid ${theme.palette.background.paper}`,
}));

const MainAvatar = styled(Avatar)(({ theme, status }) => ({
  width: 140,
  height: 140,
  margin: '0 auto',
  marginBottom: theme.spacing(3),
  fontSize: '3rem',
  boxShadow: theme.shadows[12],
  border: `4px solid ${alpha(theme.palette.background.paper, 0.8)}`,
  ...(status === 'recording' && {
    animation: `${pulse} 1.5s infinite`,
    backgroundColor: theme.palette.error.main,
    borderColor: theme.palette.error.light,
  }),
  ...(status === 'speaking' && {
    backgroundColor: theme.palette.info.main,
    borderColor: theme.palette.info.light,
  }),
  ...(status === 'complete' && {
    backgroundColor: theme.palette.success.main,
    borderColor: theme.palette.success.light,
  }),
  ...(status === 'processing' && {
    backgroundColor: theme.palette.warning.main,
    borderColor: theme.palette.warning.light,
  }),
}));

const TimerBox = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  backgroundColor: alpha(theme.palette.error.main, 0.1),
  border: `2px solid ${theme.palette.error.main}`,
  borderRadius: theme.spacing(2),
  textAlign: 'center',
  minWidth: '120px',
}));

// ==================== MAIN COMPONENT ====================

const StandupCallSession = () => {
  const { testId: urlTestId } = useParams();
  const navigate = useNavigate();
  const theme = useTheme();
  
  // ==================== STATE MANAGEMENT ====================
  
  const [testId, setTestId] = useState(null);
  const [status, setStatus] = useState('Ready to start your daily standup');
  const [recording, setRecording] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [error, setError] = useState(null);
  const [questionCount, setQuestionCount] = useState(1);
  const [totalQuestions] = useState(5);
  const [sessionComplete, setSessionComplete] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [voiceLevel, setVoiceLevel] = useState(0);
  const [aiSpeaking, setAiSpeaking] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState('');
  
  // ==================== REFS ====================
  
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const micSourceRef = useRef(null);
  const dataArrayRef = useRef(null);
  const silenceStartRef = useRef(null);
  const hasSpokenRef = useRef(false);
  const recordingTimerRef = useRef(null);
  const currentStreamRef = useRef(null);
  const testIdRef = useRef(null);
  const currentAudioRef = useRef(null);
  const ttsTimeoutRef = useRef(null);
  
  // ==================== CONSTANTS ====================
  
  const SILENCE_THRESHOLD = 0.01;
  const SILENCE_DURATION = 2000;
  const MAX_DURATION = 15000;

  // ==================== EFFECTS ====================
  
  useEffect(() => {
    setIsInitialized(true);
    setTestId(null);
    testIdRef.current = null;
    setStatus('Ready to start your daily standup');
    
    return () => {
      cleanup();
    };
  }, []);

  // ==================== MAIN FUNCTIONS ====================
  
  const startTest = async () => {
    if (isStarting) {
      console.log('Already starting, ignoring duplicate call');
      return;
    }

    try {
      setIsStarting(true);
      setStatus('Starting your standup session...');
      setError(null);
      setSessionComplete(false);
      setQuestionCount(1);
      
      console.log('Starting new standup test...');
      const response = await standupCallAPI.startStandup();
      console.log('Start test response:', response);
      
      if (!response || !response.test_id) {
        throw new Error('Invalid response from server - missing test_id');
      }
      
      console.log('📝 Setting testId to:', response.test_id);
      setTestId(response.test_id);
      testIdRef.current = response.test_id;
      console.log('📝 testId set successfully in both state and ref');
      
      const questionText = response.question || 'Please share your progress from yesterday.';
      setStatus(questionText);
      setCurrentQuestion(questionText);
      
      // **FIXED: Improved audio playback with better fallback**
      await playAudioWithFallback(response.audio_path, questionText);
      
      // **FIXED: Start recording after audio completes**
      console.log('⏳ Waiting 2 seconds before starting recording...');
      setTimeout(() => {
        if (!sessionComplete && (testIdRef.current || testId)) {
          console.log('🎤 Now starting recording');
          startRecording();
        }
      }, 2000);
      
    } catch (err) {
      console.error('Error starting test:', err);
      setError(`Failed to start test: ${err.message}`);
      setStatus('Ready to start your daily standup');
    } finally {
      setIsStarting(false);
    }
  };

  // **FIXED: Completely rewritten audio system with proper URL handling**
  const playAudioWithFallback = async (audioPath, fallbackText) => {
    console.log('🎵 playAudioWithFallback called:', { audioPath, fallbackText: fallbackText?.substring(0, 50) + '...' });
    
    // If no audio path provided, use TTS directly
    if (!audioPath) {
      console.log('📢 No audio path provided, using TTS fallback');
      return await speakTextOptimized(fallbackText);
    }
    
    try {
      console.log('🎵 Attempting backend audio first:', audioPath);
      await playBackendAudio(audioPath);
      console.log('✅ Backend audio played successfully');
    } catch (audioError) {
      console.warn('❌ Backend audio failed, falling back to TTS:', audioError.message);
      await speakTextOptimized(fallbackText);
      console.log('✅ TTS fallback completed');
    }
  };

  // **FIXED: Completely rewritten backend audio playback**
  const playBackendAudio = (audioPath) => {
    return new Promise((resolve, reject) => {
      console.log('🎵 Starting backend audio playback for:', audioPath);
      setAiSpeaking(true);
      
      // Clean up any existing audio
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current.src = '';
        currentAudioRef.current = null;
      }
      
      // **FIXED: Proper URL construction with multiple fallbacks**
      const constructAudioUrl = (path) => {
        console.log('🔗 Constructing URL for path:', path);
        
        // If it's already a full URL, use it
        if (path.startsWith('http://') || path.startsWith('https://')) {
          console.log('🔗 Using provided full URL:', path);
          return path;
        }
        
        // If it starts with /audio/, use API base + path
        if (path.startsWith('/audio/')) {
          const url = `${API_BASE_URL}${path}`;
          console.log('🔗 Constructed API URL:', url);
          return url;
        }
        
        // If it starts with ./audio/, replace with API base
        if (path.startsWith('./audio/')) {
          const url = path.replace('./audio/', `${API_BASE_URL}/audio/`);
          console.log('🔗 Replaced relative URL:', url);
          return url;
        }
        
        // If it's just a filename, assume it's in /audio/
        const url = `${API_BASE_URL}/audio/${path}`;
        console.log('🔗 Assumed filename, constructed URL:', url);
        return url;
      };
      
      const audioUrl = constructAudioUrl(audioPath);
      console.log('🔗 Final audio URL:', audioUrl);
      
      const audio = new Audio();
      currentAudioRef.current = audio;
      let hasResolved = false;
      let loadStartTime = Date.now();
      
      const cleanup = (success = false, errorMsg = '') => {
        if (!hasResolved) {
          hasResolved = true;
          setAiSpeaking(false);
          
          if (currentAudioRef.current) {
            currentAudioRef.current.pause();
            currentAudioRef.current.src = '';
            currentAudioRef.current = null;
          }
          
          const duration = Date.now() - loadStartTime;
          console.log(`🎵 Audio operation completed in ${duration}ms:`, success ? 'SUCCESS' : 'FAILED');
          
          if (success) {
            resolve();
          } else {
            reject(new Error(errorMsg || 'Audio playback failed'));
          }
        }
      };
      
      // **FIXED: More comprehensive event handling**
      audio.onloadstart = () => {
        console.log('📀 Audio loading started...');
      };
      
      audio.oncanplay = () => {
        console.log('📀 Audio can play, starting playback...');
        audio.play()
          .then(() => {
            console.log('🎵 Audio playback started successfully');
          })
          .catch((playError) => {
            console.warn('❌ Audio play() failed:', playError);
            cleanupWithTimeout(false, `Play error: ${playError.message}`);
          });
      };
      
      audio.onended = () => {
        console.log('✅ Backend audio completed successfully');
        cleanupWithTimeout(true);
      };
      
      audio.onerror = (e) => {
        const errorCode = e.target?.error?.code;
        const errorMessage = e.target?.error?.message;
        let detailedError = 'Unknown audio error';
        
        // Provide more specific error messages
        switch (errorCode) {
          case 1: // MEDIA_ERR_ABORTED
            detailedError = 'Audio loading was aborted';
            break;
          case 2: // MEDIA_ERR_NETWORK
            detailedError = 'Network error while loading audio';
            break;
          case 3: // MEDIA_ERR_DECODE
            detailedError = 'Audio decoding error';
            break;
          case 4: // MEDIA_ERR_SRC_NOT_SUPPORTED
            detailedError = 'Audio format not supported or file not found';
            break;
          default:
            detailedError = errorMessage || 'Audio error occurred';
        }
        
        console.warn('❌ Audio error:', { errorCode, errorMessage, detailedError });
        cleanupWithTimeout(false, detailedError);
      };
      
      audio.onabort = () => {
        console.warn('📀 Audio loading aborted');
        cleanupWithTimeout(false, 'Audio loading aborted');
      };
      
      audio.onstalled = () => {
        console.warn('📀 Audio loading stalled');
      };
      
      audio.onwaiting = () => {
        console.log('📀 Audio waiting for data...');
      };
      
      // **FIXED: Progressive timeout with multiple attempts**
      const timeoutId = setTimeout(() => {
        if (!hasResolved) {
          console.warn('⏰ Audio loading timeout after 8 seconds');
          cleanupWithTimeout(false, 'Audio loading timeout - file may not exist or network issue');
        }
      }, 8000); // 8 second timeout
      
      // Cleanup timeout when audio resolves
      const cleanupWithTimeout = (success, errorMsg) => {
        clearTimeout(timeoutId);
        cleanup(success, errorMsg);
      };
      
      // **FIXED: Set audio properties for better compatibility**
      audio.volume = 0.8;
      audio.preload = 'auto';
      audio.crossOrigin = 'anonymous'; // Handle CORS issues
      
      // **FIXED: Add error handling for URL validation**
      try {
        new URL(audioUrl); // Validate URL format
        audio.src = audioUrl;
        audio.load(); // Explicitly start loading
      } catch (urlError) {
        console.error('❌ Invalid audio URL:', urlError);
        cleanup(false, 'Invalid audio URL format');
      }
    });
  };

  // **FIXED: Optimized TTS with better error handling**
  const speakTextOptimized = (text) => {
    return new Promise((resolve) => {
      if (!('speechSynthesis' in window) || !text?.trim()) {
        console.warn('❌ TTS not available or no text provided');
        setAiSpeaking(false);
        resolve();
        return;
      }
      
      console.log('🗣 Starting TTS for:', text.substring(0, 50) + '...');
      setAiSpeaking(true);
      
      // **FIXED: Immediate cleanup and restart**
      window.speechSynthesis.cancel();
      
      const startSpeech = () => {
        // Quick check for ongoing speech
        if (window.speechSynthesis.speaking) {
          window.speechSynthesis.cancel();
          setTimeout(startSpeech, 200);
          return;
        }
        
        const utterance = new SpeechSynthesisUtterance(text);
        
        // **FIXED: Optimal TTS settings**
        utterance.rate = 0.9;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        // **FIXED: Better voice selection**
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => 
          v.lang.includes('en') && 
          (v.name.includes('Google') || v.name.includes('Microsoft') || v.name.includes('Natural'))
        ) || voices.find(v => v.lang.includes('en')) || voices[0];
        
        if (preferredVoice) {
          utterance.voice = preferredVoice;
          console.log('🎤 Using TTS voice:', preferredVoice.name);
        }
        
        let hasCompleted = false;
        
        const completeTTS = () => {
          if (!hasCompleted) {
            hasCompleted = true;
            setAiSpeaking(false);
            if (ttsTimeoutRef.current) {
              clearTimeout(ttsTimeoutRef.current);
              ttsTimeoutRef.current = null;
            }
            console.log('✅ TTS completed');
            resolve();
          }
        };
        
        utterance.onend = completeTTS;
        utterance.onerror = (event) => {
          console.error('❌ TTS error:', event.error);
          completeTTS();
        };
        
        // **FIXED: Smarter timeout based on text length**
        const estimatedDuration = Math.max(text.split(' ').length * 500, 3000); // 500ms per word, min 3s
        ttsTimeoutRef.current = setTimeout(() => {
          console.log('⏰ TTS timeout reached');
          window.speechSynthesis.cancel();
          completeTTS();
        }, estimatedDuration);
        
        try {
          window.speechSynthesis.speak(utterance);
          console.log('🎤 TTS started successfully');
        } catch (error) {
          console.error('❌ TTS speak() failed:', error);
          completeTTS();
        }
      };
      
      // **FIXED: Reduced initial delay**
      setTimeout(startSpeech, 300);
    });
  };

  const startRecording = async () => {
    if (recording) {
      console.log('Already recording, skipping...');
      return;
    }

    try {
      setError(null);
      console.log('Starting recording...');
      
      if (currentStreamRef.current) {
        currentStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000
        } 
      });
      
      currentStreamRef.current = stream;
      
      // Set up audio analysis
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
      analyserRef.current = audioContextRef.current.createAnalyser();
      micSourceRef.current = audioContextRef.current.createMediaStreamSource(stream);
      analyserRef.current.fftSize = 256;
      dataArrayRef.current = new Float32Array(analyserRef.current.frequencyBinCount);
      micSourceRef.current.connect(analyserRef.current);

      // Set up media recorder
      mediaRecorderRef.current = new MediaRecorder(stream, { 
        mimeType: 'audio/webm;codecs=opus' 
      });
      audioChunksRef.current = [];
      silenceStartRef.current = null;
      hasSpokenRef.current = false;

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        console.log('Recording stopped');
        
        if (currentStreamRef.current) {
          currentStreamRef.current.getTracks().forEach(track => track.stop());
          currentStreamRef.current = null;
        }
        
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          await audioContextRef.current.close();
        }
        
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        console.log('Audio blob size:', blob.size);
        
        if (blob.size < 1000) {
          console.warn('Audio too small, restarting recording...');
          setTimeout(() => {
            if (!sessionComplete && testId) {
              startRecording();
            }
          }, 1000);
          return;
        }
        
        await sendAudio(blob);
      };

      setRecording(true);
      setRecordingDuration(0);
      
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
      
      mediaRecorderRef.current.start(100);
      detectSilence();

      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          console.log('Stopping recording due to time limit');
          mediaRecorderRef.current.stop();
        }
      }, MAX_DURATION);
      
    } catch (err) {
      console.error('Microphone error:', err);
      setError(`Microphone error: ${err.message}`);
      setRecording(false);
    }
  };

  const detectSilence = () => {
    if (!analyserRef.current || !recording) return;
    
    analyserRef.current.getFloatTimeDomainData(dataArrayRef.current);
    
    const rms = Math.sqrt(
      dataArrayRef.current.reduce((acc, val) => acc + val * val, 0) / dataArrayRef.current.length
    );
    
    setVoiceLevel(Math.min(rms * 10, 1));
    
    const now = Date.now();

    if (rms < SILENCE_THRESHOLD) {
      if (!silenceStartRef.current) {
        silenceStartRef.current = now;
      } else if (hasSpokenRef.current && now - silenceStartRef.current > SILENCE_DURATION) {
        console.log('Auto-stopping due to silence');
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
        return;
      }
    } else {
      silenceStartRef.current = null;
      if (!hasSpokenRef.current && rms > SILENCE_THRESHOLD * 3) {
        hasSpokenRef.current = true;
        console.log('User started speaking');
      }
    }

    if (recording) {
      requestAnimationFrame(detectSilence);
    }
  };

  const sendAudio = async (blob) => {
    try {
      setRecording(false);
      setStatus('Processing your response...');
      
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      
      const currentTestId = testIdRef.current || testId;
      console.log('🔍 Using testId:', currentTestId);
      
      if (!currentTestId) {
        console.error('❌ No test ID available in sendAudio');
        throw new Error('No test ID available. Please restart the standup.');
      }
      
      console.log('Sending audio response for testId:', currentTestId);
      
      const response = await standupCallAPI.recordAndRespond(currentTestId, blob);
      console.log('Response received:', response);

      if (!response) {
        throw new Error('No response received from server');
      }

      const responseText = response.response || response.question || 'Processing...';
      setStatus(responseText);
      setCurrentQuestion(responseText);
      
      if (response.ended || response.complete) {
        setSessionComplete(true);
        
        const finalMessage = response.response || response.message || 'Thank you for completing the standup!';
        
        // **FIXED: Use smart audio with fallback for final message**
        await playAudioWithFallback(response.audio_path, finalMessage);
        
        setTimeout(getSummary, 2000);
        return;
      }

      setQuestionCount(prev => prev + 1);
      
      const nextQuestion = response.response || response.question;
      
      // **FIXED: Use smart audio with fallback for next question**
      await playAudioWithFallback(response.audio_path, nextQuestion);
      
      // **FIXED: Reduced delay from 8 to 2 seconds**
      console.log('⏳ Waiting 2 seconds before next recording...');
      setTimeout(() => {
        const currentTestId = testIdRef.current || testId;
        if (!sessionComplete && currentTestId) {
          console.log('🎤 Starting next recording');
          startRecording();
        }
      }, 2000);
      
    } catch (err) {
      console.error('Error processing response:', err);
      setError(`Error processing response: ${err.message}`);
      setStatus('Error occurred. Please try again.');
      
      setTimeout(() => {
        if (!sessionComplete) {
          setStatus('Ready to start your daily standup');
        }
      }, 3000);
    }
  };

  const getSummary = async () => {
    try {
      const currentTestId = testIdRef.current || testId;
      console.log('🔍 Getting standup summary for test ID:', currentTestId);
      
      if (!currentTestId) {
        console.error('❌ No test ID available for summary');
        setError('Cannot get summary - no test ID available');
        return;
      }
      
      const summary = await standupCallAPI.getStandupSummary(currentTestId);
      console.log('✅ Summary received:', summary);
      
      setEvaluation(summary);
      
      if (summary) {
        navigate(`/student/daily-standups/summary/${currentTestId}`, { 
          state: { 
            summary: summary.summary || summary,
            analytics: summary.analytics || {},
            pdf_url: summary.pdf_url
          } 
        });
      }
      
    } catch (err) {
      console.error('❌ Error getting summary:', err);
      setError('Failed to get summary, but your standup was recorded successfully.');
      setSessionComplete(true);
      setStatus('Standup completed successfully! Summary unavailable.');
    }
  };

  // **FIXED: Improved cleanup function**
  const cleanup = () => {
    console.log('🧹 Cleaning up resources...');
    
    if (recording && mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    
    if (currentStreamRef.current) {
      currentStreamRef.current.getTracks().forEach(track => track.stop());
      currentStreamRef.current = null;
    }
    
    // **FIXED: Cleanup TTS**
    if (window.speechSynthesis && window.speechSynthesis.speaking) {
      window.speechSynthesis.cancel();
    }
    
    // **FIXED: Cleanup backend audio**
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current.src = '';
      currentAudioRef.current = null;
    }
    
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }
    
    if (ttsTimeoutRef.current) {
      clearTimeout(ttsTimeoutRef.current);
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    
    setRecording(false);
    setAiSpeaking(false);
    setIsStarting(false);
  };

  const handleGoBack = () => {
    cleanup();
    navigate('/student/daily-standups');
  };

  // ==================== HELPER FUNCTIONS ====================
  
  const getSessionStatus = () => {
    if (sessionComplete) return 'complete';
    if (aiSpeaking) return 'speaking';
    if (recording) return 'recording';
    if (status.includes('Processing')) return 'processing';
    if (status.includes('Starting') || isStarting) return 'initializing';
    return 'ready';
  };

  const getStatusMessage = () => {
    const sessionStatus = getSessionStatus();
    switch (sessionStatus) {
      case 'complete':
        return 'Standup Complete! 🎉';
      case 'speaking':
        return '🤖 AI Speaking...';
      case 'recording':
        return '🎤 Recording Your Response';
      case 'processing':
        return 'Processing your response...';
      case 'initializing':
        return 'Starting your standup...';
      default:
        return 'Ready to start your daily standup';
    }
  };

  const getStatusIcon = () => {
    const sessionStatus = getSessionStatus();
    switch (sessionStatus) {
      case 'speaking':
        return <VolumeUp fontSize="inherit" />;
      case 'recording':
        return <GraphicEq fontSize="inherit" />;
      case 'processing':
        return <Timer fontSize="inherit" />;
      case 'complete':
        return <CheckCircle fontSize="inherit" />;
      default:
        return <Mic fontSize="inherit" />;
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // ==================== RENDER ====================
  
  const sessionStatus = getSessionStatus();

  return (
    <Fade in={true}>
      <Box sx={{ p: 3, minHeight: '100vh', backgroundColor: alpha(theme.palette.primary.main, 0.02) }}>
        {/* Header */}
        <Box 
          display="flex" 
          justifyContent="space-between" 
          alignItems="center" 
          mb={4}
          sx={{
            background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})`,
            p: 3,
            borderRadius: 3,
            backdropFilter: 'blur(10px)',
            border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`
          }}
        >
          <Box display="flex" alignItems="center" gap={2}>
            <IconButton onClick={handleGoBack} sx={{ mr: 1 }}>
              <ArrowBack />
            </IconButton>
            <Avatar 
              sx={{ 
                bgcolor: sessionComplete ? theme.palette.success.main : theme.palette.primary.main,
                width: 56,
                height: 56,
                boxShadow: theme.shadows[8]
              }}
            >
              {getStatusIcon()}
            </Avatar>
            <Box>
              <Typography 
                variant="h5" 
                component="h1"
                sx={{
                  background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  fontWeight: 'bold'
                }}
              >
                Daily Standup Session
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Question {questionCount} of {totalQuestions} {(testIdRef.current || testId) && `• Test ID: ${(testIdRef.current || testId).slice(-8)}`}
              </Typography>
            </Box>
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={sessionComplete ? "Complete" : "In Progress"}
              color={sessionComplete ? "success" : "primary"}
              icon={sessionComplete ? <CheckCircle /> : <RadioButtonChecked />}
              size="medium"
            />
          </Box>
        </Box>

        {/* Interviewer Speaking Card */}
        {aiSpeaking && (
          <Fade in={true}>
            <InterviewerSpeakingCard elevation={8}>
              <CardContent sx={{ p: 3 }}>
                <Box display="flex" alignItems="center" gap={3}>
                  <BlinkingMicAvatar>
                    <RecordVoiceOver fontSize="large" />
                  </BlinkingMicAvatar>
                  <Box flex={1}>
                    <Typography variant="h6" color="info.main" sx={{ fontWeight: 'bold', mb: 1 }}>
                      🎤 Interviewer Speaking
                    </Typography>
                    <Typography variant="body1" sx={{ fontSize: '1.1rem', lineHeight: 1.6 }}>
                      "{currentQuestion}"
                    </Typography>
                    <Box sx={{ mt: 2 }}>
                      <AudioWaveIndicator>
                        <div className="wave"></div>
                        <div className="wave"></div>
                        <div className="wave"></div>
                        <div className="wave"></div>
                        <div className="wave"></div>
                        <div className="wave"></div>
                        <div className="wave"></div>
                      </AudioWaveIndicator>
                    </Box>
                  </Box>
                </Box>
              </CardContent>
            </InterviewerSpeakingCard>
          </Fade>
        )}

        {/* Progress and Timer Row */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2, borderRadius: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Progress: {questionCount} / {totalQuestions}
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={(questionCount / totalQuestions) * 100} 
                sx={{ height: 10, borderRadius: 5 }}
              />
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            {recording && (
              <TimerBox elevation={3}>
                <Typography variant="h6" color="error" sx={{ fontWeight: 'bold', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}>
                  <Timer />
                  {formatTime(recordingDuration)}
                </Typography>
              </TimerBox>
            )}
          </Grid>
        </Grid>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
            {error}
          </Alert>
        )}

        {/* Main Interface */}
        <Card sx={{ borderRadius: 4, boxShadow: theme.shadows[16] }}>
          <CardContent sx={{ p: 6 }}>
            <Box textAlign="center">
              <Typography 
                variant="h3" 
                gutterBottom 
                sx={{ 
                  mb: 4,
                  fontWeight: 'bold',
                  background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                {getStatusMessage()}
              </Typography>
              
              {/* Main Visual Indicator */}
              <Box sx={{ mb: 4 }}>
                <MainAvatar status={sessionStatus}>
                  {getStatusIcon()}
                </MainAvatar>
                
                {/* Recording Visual Indicator */}
                {recording && (
                  <Box sx={{ mt: 2 }}>
                    <RecordingIndicator>
                      <div className="bar"></div>
                      <div className="bar"></div>
                      <div className="bar"></div>
                      <div className="bar"></div>
                      <div className="bar"></div>
                      <div className="bar"></div>
                    </RecordingIndicator>
                    
                    {/* Voice Activity Indicator */}
                    <Box sx={{ mt: 3, maxWidth: 300, mx: 'auto' }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        Voice Level:
                      </Typography>
                      <VoiceLevelIndicator level={voiceLevel} />
                      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                        <Typography variant="caption" color={voiceLevel > 0.05 ? 'success.main' : 'text.secondary'}>
                          {voiceLevel > 0.05 ? '🎤 Speaking' : (hasSpokenRef.current ? '🔇 Silent' : '🎙 Start speaking')}
                        </Typography>
                      </Box>
                    </Box>
                    
                    <Typography variant="h6" color="error" sx={{ mt: 2, fontWeight: 'bold' }}>
                      🎤 RECORDING • {formatTime(recordingDuration)}
                    </Typography>
                  </Box>
                )}
              </Box>

              {/* Status Indicators */}
              <Box sx={{ mb: 4, display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
                {aiSpeaking && (
                  <Chip 
                    label="AI Speaking..." 
                    color="info" 
                    icon={<VolumeUp />}
                    size="medium"
                  />
                )}
                {recording && (
                  <Chip 
                    label="Recording • Speak naturally"
                    color="error"
                    icon={<Mic />}
                    size="medium"
                  />
                )}
                {status.includes('Processing') && (
                  <Chip 
                    label="Processing..." 
                    color="warning" 
                    icon={<Timer />}
                    size="medium"
                  />
                )}
              </Box>

              {/* Instructions based on status */}
              <Box sx={{ mb: 4 }}>
                {/* Show start button when ready and not active */}
                {sessionStatus === 'ready' && !testId && (
                  <Box>
                    <Typography variant="h6" color="text.secondary" sx={{ mb: 3 }}>
                      Ready to start your daily standup?
                    </Typography>
                    <Button
                      variant="contained"
                      size="large"
                      onClick={startTest}
                      startIcon={<PlayArrow />}
                      disabled={isStarting || status.includes('Starting') || status.includes('Processing')}
                      sx={{ 
                        px: 4, 
                        py: 2, 
                        fontSize: '1.1rem',
                        borderRadius: 3,
                        boxShadow: theme.shadows[8]
                      }}
                    >
                      {isStarting ? 'Starting...' : 'Start Standup'}
                    </Button>
                  </Box>
                )}

                {/* Show specific messages for each state */}
                {sessionStatus === 'initializing' && (
                  <Typography variant="h6" color="warning.main" sx={{ fontWeight: 'medium' }}>
                    ⏳ Initializing your standup session...
                  </Typography>
                )}

                {sessionStatus === 'speaking' && (
                  <Typography variant="h6" color="info.main" sx={{ fontWeight: 'medium' }}>
                    🎧 Please listen carefully to the complete question. Recording will start automatically in a few seconds.
                  </Typography>
                )}

                {sessionStatus === 'recording' && (
                  <Typography variant="h6" color="error.main" sx={{ fontWeight: 'medium' }}>
                    🗣 Now you can speak your answer. Recording will stop automatically when you finish.
                  </Typography>
                )}

                {sessionStatus === 'processing' && (
                  <Typography variant="h6" color="warning.main" sx={{ fontWeight: 'medium' }}>
                    ⏳ Processing your response. Please wait...
                  </Typography>
                )}
              </Box>

              {/* Session Complete Actions */}
              {sessionComplete && (
                <Box sx={{ mt: 4 }}>
                  <Typography variant="h4" color="success.main" gutterBottom sx={{ fontWeight: 'bold' }}>
                    🎉 Standup Complete!
                  </Typography>
                  <Typography variant="h6" sx={{ mb: 4, color: 'text.secondary' }}>
                    Great job! Your responses have been recorded successfully.
                  </Typography>
                  <Button
                    variant="contained"
                    color="success"
                    onClick={getSummary}
                    startIcon={<Summarize />}
                    size="large"
                    sx={{ 
                      px: 4, 
                      py: 2, 
                      fontSize: '1.1rem',
                      borderRadius: 3,
                      boxShadow: theme.shadows[8]
                    }}
                  >
                    View Summary & Results
                  </Button>
                </Box>
              )}

              {/* Evaluation Display */}
              {evaluation && (
                <Box sx={{ mt: 4, p: 3, backgroundColor: alpha(theme.palette.success.main, 0.1), borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom sx={{ fontWeight: 'bold' }}>
                    📊 Evaluation Results
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2, whiteSpace: 'pre-wrap' }}>
                    {evaluation.summary || evaluation}
                  </Typography>
                  {evaluation.analytics && (
                    <Box>
                      <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 'bold' }}>
                        Analytics:
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                        {evaluation.analytics.num_questions && (
                          <Chip label={`Questions: ${evaluation.analytics.num_questions}`} size="small" />
                        )}
                        {evaluation.analytics.avg_response_length && (
                          <Chip label={`Avg Length: ${evaluation.analytics.avg_response_length}`} size="small" />
                        )}
                        {evaluation.analytics.concept_coverage && (
                          <Chip label={`Concepts: ${evaluation.analytics.concept_coverage}`} size="small" />
                        )}
                      </Box>
                    </Box>
                  )}
                  {evaluation.pdf_url && (
                    <Box sx={{ mt: 2 }}>
                      <Button
                        variant="outlined"
                        size="small"
                        href={evaluation.pdf_url}
                        download
                        sx={{ borderRadius: 2 }}
                      >
                        Download PDF Report
                      </Button>
                    </Box>
                  )}
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Fade>
  );
};

export default StandupCallSession;