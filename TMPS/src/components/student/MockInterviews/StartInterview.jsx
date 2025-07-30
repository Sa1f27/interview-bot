// FIXED: Fully Automated Speech-to-Speech Interview with Proper WebSocket Integration
// src/components/student/MockInterviews/StartInterview.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import {
  Container, Typography, Box, Card, CardContent, Button,
  Alert, CircularProgress, LinearProgress, Chip, IconButton,
  Paper
} from '@mui/material';
import { 
  Mic, MicOff, VolumeUp, VolumeOff, Stop, PlayArrow,
  ArrowBack, RadioButtonChecked, Circle, Pause
} from '@mui/icons-material';

const StartInterview = () => {
  const { sessionId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get data from navigation state or URL params
  const urlParams = new URLSearchParams(window.location.search);
  const testId = location.state?.testId || urlParams.get('testId');
  const studentName = location.state?.studentName || urlParams.get('studentName') || 'Student';
  
  // Connection states
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(true);
  
  // Interview states
  const [interviewStarted, setInterviewStarted] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [currentStage, setCurrentStage] = useState('greeting');
  const [questionNumber, setQuestionNumber] = useState(0);
  
  // Audio states
  const [isAIPlaying, setIsAIPlaying] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [silenceTimer, setSilenceTimer] = useState(0);
  const [microphoneReady, setMicrophoneReady] = useState(false);
  
  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const silenceTimeoutRef = useRef(null);
  const audioChunksRef = useRef([]);
  const keepAliveIntervalRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const shouldStartListeningRef = useRef(false);
  const speechDetectionRef = useRef(null);

  // FIXED: Auto-silence detection configuration - more responsive
  const SILENCE_THRESHOLD = 0.015; // Lowered threshold for better detection
  const SILENCE_DURATION = 2500;   // 2.5 seconds (more natural)
  const MAX_RECORDING_TIME = 30000; // 30 seconds max
  const MIN_SPEECH_TIME = 800;     // Must speak for at least 800ms

  useEffect(() => {
    console.log('??? Starting automated speech-to-speech interview');
    console.log('Session ID:', sessionId);
    console.log('Test ID:', testId);
    console.log('Student:', studentName);
    
    if (!sessionId) {
      setConnectionError('No session ID provided. Please start a new interview.');
      setIsConnecting(false);
      return;
    }
    
    // FIXED: Handle both session ID formats properly
    if (!sessionId || sessionId === 'undefined') {
      setConnectionError('Invalid session format. Please start a new interview.');
      setIsConnecting(false);
      return;
    }
    
    initializeInterview();
    
    return cleanup;
  }, [sessionId]);

  const initializeInterview = async () => {
    try {
      console.log('?? Initializing automated interview system...');
      
      // Initialize microphone first
      await setupMicrophone();
      
      // Then initialize WebSocket
      await initializeWebSocket();
      
    } catch (error) {
      console.error('? Interview initialization failed:', error);
      setConnectionError(`Initialization failed: ${error.message}`);
      setIsConnecting(false);
    }
  };

  const setupMicrophone = async () => {
    try {
      console.log('?? Setting up microphone for continuous listening...');
      
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 44100,
          channelCount: 1
        }
      });
      
      streamRef.current = stream;
      
      // FIXED: Proper audio context setup with user gesture
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        
        // Resume context if suspended
        if (audioContextRef.current.state === 'suspended') {
          await audioContextRef.current.resume();
        }
      }
      
      // Setup audio analysis for silence detection
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = 512;
      analyserRef.current.smoothingTimeConstant = 0.8;
      source.connect(analyserRef.current);
      
      setMicrophoneReady(true);
      console.log('? Microphone setup complete');
      
    } catch (error) {
      console.error('? Microphone setup failed:', error);
      throw new Error(`Microphone access failed: ${error.message}`);
    }
  };

  const initializeWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = `wss://192.168.48.201:8070/weekly_interview/ws/${sessionId}`;
        console.log('?? Connecting to WebSocket:', wsUrl);
        
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('? WebSocket connected - Interview ready!');
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionError(null);
          
          // Setup keepalive pings
          keepAliveIntervalRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'ping' }));
            }
          }, 30000);
          
          resolve();
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('?? WebSocket message:', data.type);
            
            handleWebSocketMessage(data);
            
          } catch (error) {
            console.error('? WebSocket message parse error:', error);
          }
        };
        
        ws.onerror = (error) => {
          console.error('? WebSocket error:', error);
          setConnectionError('Connection failed. Please refresh and try again.');
          reject(error);
        };
        
        ws.onclose = (event) => {
          console.log('?? WebSocket closed:', event.code);
          setIsConnected(false);
          
          if (keepAliveIntervalRef.current) {
            clearInterval(keepAliveIntervalRef.current);
          }
          
          if (event.code !== 1000 && event.code !== 1001) {
            setConnectionError('Connection lost. Please refresh to reconnect.');
          }
        };
        
        wsRef.current = ws;
        
      } catch (error) {
        console.error('? WebSocket initialization failed:', error);
        setConnectionError(`Connection failed: ${error.message}`);
        setIsConnecting(false);
        reject(error);
      }
    });
  };

  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'error':
        console.error('? Server error:', data.text);
        setConnectionError(data.text);
        break;
        
      case 'ai_response':
        console.log('?? AI Question:', data.text.substring(0, 50) + '...');
        setCurrentMessage(data.text);
        setCurrentStage(data.stage || 'unknown');
        setQuestionNumber(data.question_number || 0);
        setIsAIPlaying(true);
        
        // FIXED: Stop any current recording when AI starts speaking
        if (isRecording) {
          stopRecording();
        }
        break;
        
      case 'audio_chunk':
        // FIXED: Queue audio chunks for sequential playback
        if (data.audio) {
          audioQueueRef.current.push(data.audio);
          if (!isPlayingAudioRef.current) {
            processAudioQueue();
          }
        }
        break;
        
      case 'audio_end':
        console.log('?? AI finished speaking - Your turn!');
        setIsAIPlaying(false);
        
        // FIXED: Wait for all audio to finish, then start listening
        shouldStartListeningRef.current = true;
        
        // Check if audio queue is empty before starting listening
        setTimeout(() => {
          if (!isPlayingAudioRef.current && shouldStartListeningRef.current && interviewStarted) {
            startAutomaticListening();
            shouldStartListeningRef.current = false;
          }
        }, 300); // Small delay to ensure audio is fully processed
        break;
        
      case 'interview_complete':
        console.log('?? Interview completed!');
        setInterviewStarted(false);
        stopListening();
        
        // Navigate to results
        navigate(`/student/mock-interviews/results/${testId}`, {
          state: { evaluation: data }
        });
        break;
        
      case 'pong':
        // Keepalive response
        break;
        
      default:
        console.log('? Unknown message type:', data.type);
    }
  }, [interviewStarted, isRecording, testId, navigate]);

  // FIXED: Proper audio queue processing for smooth AI speech
  const processAudioQueue = async () => {
    if (isPlayingAudioRef.current || audioQueueRef.current.length === 0) {
      return;
    }
    
    isPlayingAudioRef.current = true;
    
    while (audioQueueRef.current.length > 0) {
      const hexAudio = audioQueueRef.current.shift();
      await playAudioChunk(hexAudio);
      
      // Small delay between chunks for smooth playback
      await new Promise(resolve => setTimeout(resolve, 50));
    }
    
    isPlayingAudioRef.current = false;
    
    // Check if we should start listening after all audio is done
    if (shouldStartListeningRef.current && !isAIPlaying && interviewStarted) {
      startAutomaticListening();
      shouldStartListeningRef.current = false;
    }
  };

  const playAudioChunk = async (hexAudio) => {
    return new Promise((resolve) => {
      try {
        if (!hexAudio) {
          resolve();
          return;
        }
        
        // FIXED: Proper hex to binary conversion
        const audioData = new Uint8Array(
          hexAudio.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
        );
        
        if (!audioContextRef.current) {
          resolve();
          return;
        }
        
        // FIXED: Create a copy of the array buffer for decoding
        const audioBuffer = audioData.buffer.slice();
        
        audioContextRef.current.decodeAudioData(
          audioBuffer,
          (decodedBuffer) => {
            const source = audioContextRef.current.createBufferSource();
            source.buffer = decodedBuffer;
            source.connect(audioContextRef.current.destination);
            
            source.onended = () => {
              resolve();
            };
            
            source.start();
          },
          (error) => {
            console.warn('?? Audio decode failed:', error);
            resolve();
          }
        );
        
      } catch (error) {
        console.warn('?? Audio playback failed:', error);
        resolve();
      }
    });
  };

  const startInterview = async () => {
    try {
      console.log('?? Starting automated interview conversation...');
      
      // FIXED: Ensure audio context is running before starting
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      setInterviewStarted(true);
      
      // The AI greeting should already be playing from WebSocket connection
      // We'll start listening automatically when AI finishes
      
    } catch (error) {
      console.error('? Failed to start interview:', error);
      setConnectionError(`Failed to start interview: ${error.message}`);
    }
  };

  const startAutomaticListening = async () => {
    try {
      if (isRecording || isAIPlaying || !microphoneReady) {
        console.log('?? Skip listening - conditions not met');
        return;
      }
      
      console.log('?? Starting automatic listening with silence detection...');
      
      audioChunksRef.current = [];
      
      const mediaRecorder = new MediaRecorder(streamRef.current, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };
      
      mediaRecorder.onstop = () => {
        handleRecordingComplete();
      };
      
      mediaRecorder.onerror = (error) => {
        console.error('? MediaRecorder error:', error);
        setIsRecording(false);
        setIsListening(false);
      };
      
      mediaRecorder.start(100); // Collect data every 100ms
      mediaRecorderRef.current = mediaRecorder;
      
      setIsRecording(true);
      setIsListening(true);
      
      // FIXED: Start improved silence detection
      startImprovedSilenceDetection();
      
      // Max recording time safety
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          console.log('? Max recording time reached');
          stopRecording();
        }
      }, MAX_RECORDING_TIME);
      
    } catch (error) {
      console.error('? Auto listening failed:', error);
      setIsRecording(false);
      setIsListening(false);
    }
  };

  // FIXED: Improved silence detection with better parameters
  const startImprovedSilenceDetection = () => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    let speechStartTime = null;
    let silenceStartTime = null;
    let hasSpokeEnough = false;
    let consecutiveSilenceFrames = 0;
    const minSilenceFrames = 25; // About 2.5 seconds at 10fps
    
    const detectSilence = () => {
      if (!isRecording) return;
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // FIXED: Better volume calculation using RMS
      const rms = Math.sqrt(
        dataArray.reduce((sum, value) => sum + value * value, 0) / dataArray.length
      );
      const normalizedVolume = rms / 255;
      
      setAudioLevel(normalizedVolume);
      
      const isSpeaking = normalizedVolume > SILENCE_THRESHOLD;
      
      if (isSpeaking) {
        // User is speaking
        if (!speechStartTime) {
          speechStartTime = Date.now();
          console.log('?? Speech detected');
        }
        
        silenceStartTime = null;
        consecutiveSilenceFrames = 0;
        setSilenceTimer(0);
        
        // Check if user has spoken long enough
        if (!hasSpokeEnough && (Date.now() - speechStartTime) > MIN_SPEECH_TIME) {
          hasSpokeEnough = true;
          console.log('? Minimum speech duration reached');
        }
        
      } else if (hasSpokeEnough) {
        // Silence detected after sufficient speech
        if (!silenceStartTime) {
          silenceStartTime = Date.now();
          console.log('?? Silence detected, starting timer...');
        }
        
        consecutiveSilenceFrames++;
        const silenceElapsed = Date.now() - silenceStartTime;
        setSilenceTimer(silenceElapsed);
        
        // FIXED: More stable silence detection
        if (consecutiveSilenceFrames >= minSilenceFrames && silenceElapsed >= SILENCE_DURATION) {
          console.log('?? Auto-stopping due to stable silence');
          stopRecording();
          return;
        }
      }
      
      // Continue monitoring at ~10fps for better performance
      setTimeout(detectSilence, 100);
    };
    
    detectSilence();
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    setIsRecording(false);
    setIsListening(false);
    setSilenceTimer(0);
    setAudioLevel(0);
  };

  const handleRecordingComplete = async () => {
    try {
      if (audioChunksRef.current.length === 0) {
        console.warn('?? No audio recorded');
        return;
      }
      
      console.log('?? Processing and sending audio...');
      
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      
      if (audioBlob.size < 100) {
        console.warn('?? Audio too small:', audioBlob.size, 'bytes');
        return;
      }
      
      // FIXED: Convert to base64 properly
      const base64Audio = await convertBlobToBase64(audioBlob);
      
      // Send to WebSocket
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const message = {
          type: 'audio_data',
          audio: base64Audio.split(',')[1] || base64Audio // Remove data:audio prefix if present
        };
        
        wsRef.current.send(JSON.stringify(message));
        console.log('?? Audio sent, waiting for AI response...');
      } else {
        console.error('? WebSocket not connected');
        setConnectionError('Connection lost. Please refresh.');
      }
      
    } catch (error) {
      console.error('? Audio processing failed:', error);
    }
  };

  const convertBlobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const stopInterview = () => {
    console.log('?? Stopping interview...');
    setInterviewStarted(false);
    stopListening();
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_interview' }));
    }
    
    // Navigate back to dashboard
    setTimeout(() => {
      navigate('/student/mock-interviews');
    }, 1000);
  };

  const stopListening = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
    
    setIsRecording(false);
    setIsListening(false);
    setSilenceTimer(0);
    setAudioLevel(0);
    shouldStartListeningRef.current = false;
  };

  const cleanup = () => {
    console.log('?? Cleaning up interview resources...');
    
    stopListening();
    
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    
    // Clear audio queue
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
    shouldStartListeningRef.current = false;
  };

  const getStageColor = (stage) => {
    const colors = {
      greeting: 'primary',
      technical: 'warning', 
      communication: 'info',
      hr: 'success'
    };
    return colors[stage] || 'default';
  };

  const getStageLabel = (stage) => {
    const labels = {
      greeting: 'Introduction',
      technical: 'Technical Assessment',
      communication: 'Communication Skills', 
      hr: 'Behavioral Questions'
    };
    return labels[stage] || 'Interview';
  };

  // Loading state
  if (isConnecting) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box textAlign="center">
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h5" gutterBottom>
            Initializing AI Interview System...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Setting up microphone and connecting to interview AI
          </Typography>
          {!microphoneReady && (
            <Typography variant="caption" display="block" sx={{ mt: 1 }}>
              Requesting microphone access...
            </Typography>
          )}
        </Box>
      </Container>
    );
  }

  // Error state
  if (connectionError) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="h6">Connection Error</Typography>
          <Typography variant="body2">{connectionError}</Typography>
        </Alert>
        
        <Box display="flex" gap={2} justifyContent="center">
          <Button
            variant="outlined"
            startIcon={<ArrowBack />}
            onClick={() => navigate('/student/mock-interviews')}
          >
            Back to Dashboard
          </Button>
          
          <Button
            variant="contained"
            onClick={() => window.location.reload()}
          >
            Retry
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box>
          <Typography variant="h4" fontWeight="bold" color="primary.main">
            ??? AI Interview Conversation
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            {studentName} • Fully Automated Speech-to-Speech
          </Typography>
        </Box>
        
        <Box display="flex" alignItems="center" gap={2}>
          <Chip
            label={`${getStageLabel(currentStage)} (Q${questionNumber || 1})`}
            color={getStageColor(currentStage)}
          />
          
          <Chip
            icon={isConnected ? <RadioButtonChecked /> : <Circle />}
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
          />
          
          <Chip
            icon={microphoneReady ? <Mic /> : <MicOff />}
            label={microphoneReady ? 'Mic Ready' : 'Mic Setup'}
            color={microphoneReady ? 'success' : 'warning'}
            variant="outlined"
          />
        </Box>
      </Box>

      {/* AI Message Display */}
      {currentMessage && (
        <Card sx={{ mb: 3, borderRadius: 2 }}>
          <CardContent>
            <Box display="flex" alignItems="flex-start" gap={2}>
              <VolumeUp color="primary" sx={{ mt: 0.5 }} />
              <Box flex={1}>
                <Typography variant="subtitle2" color="primary" gutterBottom>
                  AI Interviewer
                </Typography>
                <Typography variant="body1">
                  {currentMessage}
                </Typography>
                {isAIPlaying && (
                  <Box sx={{ mt: 2 }}>
                    <LinearProgress />
                    <Typography variant="caption" color="text.secondary">
                      AI is speaking... ({audioQueueRef.current.length} audio chunks queued)
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Interview Control */}
      <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 2 }}>
        {!interviewStarted ? (
          <Box>
            <Typography variant="h5" gutterBottom>
              Ready for Automated Interview?
            </Typography>
            <Typography variant="body1" color="text.secondary" paragraph>
              Click start and the AI will begin the interview. Speak naturally - 
              the system will automatically detect when you finish speaking (2.5s silence).
            </Typography>
            
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={startInterview}
              disabled={!isConnected || !microphoneReady}
              sx={{
                borderRadius: 3,
                py: 1.5,
                px: 4,
                fontSize: '1.1rem'
              }}
            >
              Start Automated Interview
            </Button>
            
            {!microphoneReady && (
              <Typography variant="caption" display="block" sx={{ mt: 1 }} color="warning.main">
                Please allow microphone access to continue
              </Typography>
            )}
          </Box>
        ) : (
          <Box>
            {/* Voice Activity Indicator */}
            <Box sx={{ mb: 3 }}>
              <Box display="flex" alignItems="center" justifyContent="center" gap={2} mb={2}>
                <Mic 
                  sx={{ 
                    fontSize: 40,
                    color: isListening ? 'success.main' : 'grey.400',
                    animation: isListening ? 'pulse 1s infinite' : 'none'
                  }} 
                />
                <Typography variant="h6">
                  {isAIPlaying ? '?? AI Speaking...' : 
                   isListening ? '?? Listening...' : 
                   '?? Standby'}
                </Typography>
              </Box>
              
              {/* Audio Level Meter */}
              {isListening && (
                <Box>
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(audioLevel * 100, 100)} 
                    color="success"
                    sx={{ height: 8, borderRadius: 4, mb: 1 }}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Audio Level: {Math.round(audioLevel * 100)}% | Threshold: {Math.round(SILENCE_THRESHOLD * 100)}%
                  </Typography>
                </Box>
              )}
              
              {/* Silence Timer */}
              {silenceTimer > 0 && (
                <Box sx={{ mt: 1 }}>
                  <LinearProgress 
                    variant="determinate" 
                    value={(silenceTimer / SILENCE_DURATION) * 100}
                    color="warning"
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                  <Typography variant="caption" color="warning.main">
                    Silence: {(silenceTimer / 1000).toFixed(1)}s / {SILENCE_DURATION / 1000}s
                  </Typography>
                </Box>
              )}
            </Box>
            
            {/* Manual Controls */}
            <Box display="flex" gap={2} justifyContent="center">
              <Button
                variant="outlined"
                startIcon={isListening ? <Pause /> : <Mic />}
                onClick={isListening ? stopListening : startAutomaticListening}
                disabled={isAIPlaying}
              >
                {isListening ? 'Pause Listening' : 'Start Speaking'}
              </Button>
              
              <Button
                variant="contained"
                color="error"
                startIcon={<Stop />}
                onClick={stopInterview}
              >
                End Interview
              </Button>
            </Box>
          </Box>
        )}
      </Paper>

      {/* Instructions */}
      <Card sx={{ mt: 3, bgcolor: 'grey.50', borderRadius: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            ?? How the Fully Automated Interview Works:
          </Typography>
          <Typography variant="body2" component="div">
            <strong>1. Fully Automated:</strong> Just click start - no recording buttons needed<br/>
            <strong>2. Natural Conversation:</strong> Speak normally after AI finishes talking<br/>
            <strong>3. Smart Silence Detection:</strong> 2.5 seconds of silence automatically submits your response<br/>
            <strong>4. Real-time Audio Processing:</strong> Seamless speech-to-speech like a real interview<br/>
            <strong>5. Audio Queue Management:</strong> Smooth AI speech with no overlapping<br/>
            <strong>6. Improved Stability:</strong> Better error handling and connection management
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default StartInterview;