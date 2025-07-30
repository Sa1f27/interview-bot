// FULLY AUTOMATED: Zero manual buttons - Complete speech-to-speech automation
// src/components/student/MockInterviews/StartInterview.jsx

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import {
  Container, Typography, Box, Card, CardContent, Button,
  Alert, CircularProgress, LinearProgress, Chip,
  Paper
} from '@mui/material';
import { 
  Mic, VolumeUp, Stop,
  ArrowBack, RadioButtonChecked, Circle
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
  
  // Audio states - FULLY AUTOMATED
  const [isAIPlaying, setIsAIPlaying] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [silenceTimer, setSilenceTimer] = useState(0);
  const [microphoneReady, setMicrophoneReady] = useState(false);
  const [conversationState, setConversationState] = useState('initializing'); // initializing, ai_speaking, listening, processing
  
  // Refs
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const audioChunksRef = useRef([]);
  const keepAliveIntervalRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingAudioRef = useRef(false);
  const speechDetectionRef = useRef(null);
  const autoListenTimeoutRef = useRef(null);

  // FULLY AUTOMATED: Silence detection configuration - optimized for natural conversation
  const SILENCE_THRESHOLD = 0.012; 
  const SILENCE_DURATION = 2000;   // 2 seconds for natural conversation flow
  const MAX_RECORDING_TIME = 30000;
  const MIN_SPEECH_TIME = 600;     // Minimum speech time before silence detection

  useEffect(() => {
    console.log('??? Initializing FULLY AUTOMATED interview system...');
    console.log('Session ID:', sessionId);
    console.log('Test ID:', testId);
    console.log('Student:', studentName);
    
    if (!sessionId) {
      setConnectionError('No session ID provided. Please start a new interview.');
      setIsConnecting(false);
      return;
    }
    
    // PREVENT DOUBLE INITIALIZATION in React StrictMode
    if (isConnecting && !isConnected) {
      initializeAutomatedInterview();
    }
    
    return cleanup;
  }, [sessionId]); // Only depend on sessionId

  const initializeAutomatedInterview = async () => {
    try {
      console.log('?? Initializing FULLY AUTOMATED interview system...');
      
      // PREVENT multiple initializations
      if (isConnected || connectionError) {
        console.log('?? Already initialized or has error, skipping');
        return;
      }
      
      setConversationState('initializing');
      
      // Initialize microphone first
      await setupMicrophone();
      
      // Then initialize WebSocket
      await initializeWebSocket();
      
      // Set ready state immediately after WebSocket connection
      setConversationState('ready');
      console.log('? System ready for automated interview');
      
    } catch (error) {
      console.error('? Automated interview initialization failed:', error);
      setConnectionError(`Initialization failed: ${error.message}`);
      setIsConnecting(false);
      setConversationState('error');
    }
  };

  const setupMicrophone = async () => {
    try {
      console.log('?? Setting up microphone for continuous automation...');
      
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
      
      // Setup audio context
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        
        if (audioContextRef.current.state === 'suspended') {
          await audioContextRef.current.resume();
        }
      }
      
      // Setup audio analysis for continuous monitoring
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      
      analyserRef.current.fftSize = 512;
      analyserRef.current.smoothingTimeConstant = 0.8;
      source.connect(analyserRef.current);
      
      setMicrophoneReady(true);
      console.log('? Microphone ready for automation');
      
    } catch (error) {
      console.error('? Microphone setup failed:', error);
      throw new Error(`Microphone access failed: ${error.message}`);
    }
  };

  const initializeWebSocket = () => {
    return new Promise((resolve, reject) => {
      try {
        // PREVENT DUPLICATE CONNECTIONS
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          console.log('? WebSocket already connected, using existing connection');
          resolve();
          return;
        }
        
        // Close any existing connection
        if (wsRef.current) {
          console.log('?? Closing existing WebSocket connection');
          wsRef.current.close();
          wsRef.current = null;
        }
        
        const wsUrl = `wss://192.168.48.201:8070/weekly_interview/ws/${sessionId}`;
        console.log('?? Connecting to automated WebSocket:', wsUrl);
        
        const ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
          console.log('? WebSocket connected for automation!');
          setIsConnected(true);
          setIsConnecting(false);
          setConnectionError(null);
          
          // Setup keepalive
          if (keepAliveIntervalRef.current) {
            clearInterval(keepAliveIntervalRef.current);
          }
          
          keepAliveIntervalRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: 'ping' }));
            }
          }, 30000);
          
          // AUTOMATICALLY START INTERVIEW - Interview will start when first ai_response arrives
          setInterviewStarted(true);
          
          resolve();
        };
        
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log('?? Automated WebSocket message:', data.type);
            handleAutomatedWebSocketMessage(data);
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

  const handleAutomatedWebSocketMessage = useCallback((data) => {
    console.log('?? WebSocket Message Details:', {
      type: data.type,
      conversationState,
      isAIPlaying,
      isListening,
      audioQueueLength: audioQueueRef.current.length
    });

    switch (data.type) {
      case 'error':
        console.error('? Server error:', data.text);
        setConnectionError(data.text);
        setConversationState('error');
        break;
        
      case 'ai_response':
        console.log('?? AI Response received:', data.text.substring(0, 100) + '...');
        
        // PREVENT DUPLICATE PROCESSING
        if (isAIPlaying && currentMessage === data.text) {
          console.log('?? Ignoring duplicate ai_response - same message already playing');
          return;
        }
        
        console.log('?? FORCE SETTING AI SPEAKING STATE IMMEDIATELY');
        
        // FORCE all states immediately - SYNCHRONOUS updates
        setCurrentMessage(data.text);
        setCurrentStage(data.stage || 'unknown');
        setQuestionNumber(data.question_number || 0);
        setIsAIPlaying(true);
        setConversationState('ai_speaking');
        
        // CLEAR audio queue
        audioQueueRef.current = [];
        isPlayingAudioRef.current = false;
        
        // STOP any recording
        stopAutomaticRecording();
        
        console.log('? AI SPEAKING STATE SET - READY FOR AUDIO');
        break;
        
      case 'audio_chunk':
        console.log(`?? Audio chunk received - CURRENT state check:`);
        console.log(`   conversationState: ${conversationState}`);
        console.log(`   isAIPlaying: ${isAIPlaying}`);
        console.log(`   currentMessage exists: ${!!currentMessage}`);
        
        // FORCE ACCEPT ALL AUDIO CHUNKS if we have a current message
        if (data.audio) {
          console.log('?? FORCING audio chunk acceptance');
          
          // Force set AI playing state if not already set
          if (!isAIPlaying || conversationState !== 'ai_speaking') {
            console.log('?? EMERGENCY: Force setting AI playing state');
            setIsAIPlaying(true);
            setConversationState('ai_speaking');
          }
          
          audioQueueRef.current.push(data.audio);
          console.log(`?? Audio chunk FORCED into queue (${audioQueueRef.current.length} total)`);
          
          // Start processing immediately
          if (!isPlayingAudioRef.current) {
            console.log('?? Starting FORCED audio playback');
            processAudioQueueAutomatically();
          }
        } else {
          console.log('? No audio data in chunk');
        }
        break;
        
      case 'audio_end':
        console.log('?? AI audio_end received');
        console.log(`   currentMessage exists: ${!!currentMessage}`);
        console.log(`   audioQueue length: ${audioQueueRef.current.length}`);
        console.log(`   isPlayingAudio: ${isPlayingAudioRef.current}`);
        
        // FORCE process audio_end if we have any current message
        if (currentMessage) {
          console.log('?? FORCING audio_end processing');
          
          // Wait for all audio chunks to finish playing
          const waitForAudioComplete = () => {
            if (isPlayingAudioRef.current || audioQueueRef.current.length > 0) {
              console.log('? Waiting for audio to complete...', {
                isPlaying: isPlayingAudioRef.current,
                queueLength: audioQueueRef.current.length
              });
              setTimeout(waitForAudioComplete, 200);
              return;
            }
            
            console.log('? All AI audio completed - starting user recording');
            setIsAIPlaying(false);
            setConversationState('listening');
            
            // Start recording after a brief pause
            setTimeout(() => {
              if (interviewStarted && !isListening) {
                console.log('?? Starting automatic recording after AI finished');
                startAutomaticRecording();
              }
            }, 500);
          };
          
          waitForAudioComplete();
        } else {
          console.log('?? No current message - ignoring audio_end');
        }
        break;
        
      case 'interview_complete':
        console.log('?? Interview completed automatically!');
        setInterviewStarted(false);
        setConversationState('complete');
        stopAutomaticRecording();
        
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
  }, [conversationState, isAIPlaying, isListening, interviewStarted, testId, navigate, currentMessage]);

  // FULLY AUTOMATED: Start interview without any manual buttons
  const startFullyAutomatedInterview = async () => {
    try {
      console.log('?? Starting FULLY AUTOMATED interview - NO MANUAL BUTTONS!');
      
      if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      setInterviewStarted(true);
      setConversationState('ai_speaking');
      
      console.log('? Fully automated interview started - AI will speak first automatically');
      
    } catch (error) {
      console.error('? Failed to start automated interview:', error);
      setConnectionError(`Failed to start interview: ${error.message}`);
    }
  };

  // AUTOMATED: Process AI audio queue for seamless playback
  const processAudioQueueAutomatically = async () => {
    if (isPlayingAudioRef.current) {
      console.log('?? Audio already playing, skipping queue processing');
      return;
    }
    
    if (audioQueueRef.current.length === 0) {
      console.log('?? No audio chunks to process');
      return;
    }
    
    isPlayingAudioRef.current = true;
    console.log(`?? Processing ${audioQueueRef.current.length} audio chunks automatically`);
    
    try {
      while (audioQueueRef.current.length > 0) {
        const hexAudio = audioQueueRef.current.shift();
        await playAudioChunkAutomatically(hexAudio);
        
        // Small delay between chunks for smooth playback
        await new Promise(resolve => setTimeout(resolve, 50));
      }
      
      console.log('? All audio chunks processed successfully');
    } catch (error) {
      console.error('? Error processing audio queue:', error);
    } finally {
      isPlayingAudioRef.current = false;
    }
  };

  const playAudioChunkAutomatically = async (hexAudio) => {
    return new Promise((resolve) => {
      try {
        if (!hexAudio || !audioContextRef.current) {
          console.log('?? No audio data or context');
          resolve();
          return;
        }
        
        console.log(`?? Playing audio chunk: ${hexAudio.length} hex chars`);
        
        // Convert hex to ArrayBuffer
        const audioData = new Uint8Array(
          hexAudio.match(/.{1,2}/g).map(byte => parseInt(byte, 16))
        );
        
        console.log(`?? Converted to ${audioData.length} bytes`);
        
        // Create a copy of the array buffer for decoding
        const audioBuffer = audioData.buffer.slice();
        
        audioContextRef.current.decodeAudioData(
          audioBuffer,
          (decodedBuffer) => {
            console.log(`?? Decoded buffer: ${decodedBuffer.duration}s`);
            
            const source = audioContextRef.current.createBufferSource();
            source.buffer = decodedBuffer;
            source.connect(audioContextRef.current.destination);
            
            source.onended = () => {
              console.log('?? Audio chunk finished playing');
              resolve();
            };
            
            source.start();
            console.log('?? Audio chunk started playing');
          },
          (error) => {
            console.error('? Audio decode failed:', error);
            resolve();
          }
        );
        
      } catch (error) {
        console.error('? Audio playback failed:', error);
        resolve();
      }
    });
  };

  // AUTOMATED: Start recording automatically when AI finishes
  const startAutomaticRecording = async () => {
    try {
      // Check if we should start recording
      if (isListening) {
        console.log('?? Already listening, skipping start recording');
        return;
      }
      
      if (isAIPlaying || isPlayingAudioRef.current) {
        console.log('?? AI still playing, cannot start recording');
        return;
      }
      
      if (!microphoneReady || !streamRef.current) {
        console.log('?? Microphone not ready for recording');
        return;
      }
      
      console.log('?? AUTOMATICALLY starting user recording - NO BUTTONS!');
      
      // Clear previous audio chunks
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
        console.log('?? Recording stopped automatically');
        handleAutomaticRecordingComplete();
      };
      
      mediaRecorder.onerror = (error) => {
        console.error('? MediaRecorder error:', error);
        setIsListening(false);
        setConversationState('error');
      };
      
      mediaRecorder.start(250); // Collect data every 250ms
      mediaRecorderRef.current = mediaRecorder;
      
      setIsListening(true);
      console.log('? Recording started, beginning silence detection');
      
      // Start AUTOMATIC silence detection
      startAutomaticSilenceDetection();
      
      // Safety timeout
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
          console.log('? Max recording time reached - auto-stopping');
          stopAutomaticRecording();
        }
      }, MAX_RECORDING_TIME);
      
    } catch (error) {
      console.error('? Automatic recording failed:', error);
      setIsListening(false);
      setConversationState('error');
    }
  };

  // AUTOMATED: Silence detection that automatically stops recording
  const startAutomaticSilenceDetection = () => {
    if (!analyserRef.current) return;
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    let speechStartTime = null;
    let silenceStartTime = null;
    let hasSpokeEnough = false;
    
    const detectSilenceAutomatically = () => {
      if (!isListening) return;
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Enhanced RMS calculation for better detection
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
          console.log('?? Speech detected automatically');
        }
        
        silenceStartTime = null;
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
          console.log('?? Silence detected - starting automatic timer...');
        }
        
        const silenceElapsed = Date.now() - silenceStartTime;
        setSilenceTimer(silenceElapsed);
        
        if (silenceElapsed >= SILENCE_DURATION) {
          console.log('?? AUTOMATICALLY stopping due to silence - NO MANUAL INTERVENTION!');
          stopAutomaticRecording();
          return;
        }
      }
      
      // Continue monitoring
      setTimeout(detectSilenceAutomatically, 100);
    };
    
    detectSilenceAutomatically();
  };

  // AUTOMATED: Stop recording automatically when silence is detected
  const stopAutomaticRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      console.log('?? Stopping automatic recording');
      mediaRecorderRef.current.stop();
    }
    
    setIsListening(false);
    setSilenceTimer(0);
    setAudioLevel(0);
  };

  // AUTOMATED: Handle recording completion and send to backend
  const handleAutomaticRecordingComplete = async () => {
    try {
      console.log('?? Handling automatic recording completion');
      
      if (audioChunksRef.current.length === 0) {
        console.warn('?? No audio recorded automatically - may need to try again');
        // Reset to listening state after a delay
        setTimeout(() => {
          if (conversationState === 'processing' && interviewStarted) {
            setConversationState('listening');
            startAutomaticRecording();
          }
        }, 1000);
        return;
      }
      
      console.log('?? AUTOMATICALLY processing and sending audio...');
      setConversationState('processing');
      
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      
      if (audioBlob.size < 100) {
        console.warn('?? Audio too small:', audioBlob.size, 'bytes - retrying');
        setTimeout(() => {
          if (conversationState === 'processing' && interviewStarted) {
            setConversationState('listening');
            startAutomaticRecording();
          }
        }, 1000);
        return;
      }
      
      console.log(`?? Processing audio blob: ${audioBlob.size} bytes`);
      
      // Convert to base64
      const base64Audio = await convertBlobToBase64(audioBlob);
      
      // Send to WebSocket automatically
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        const message = {
          type: 'audio_data',
          audio: base64Audio.split(',')[1] || base64Audio,
          size: audioBlob.size,
          timestamp: Date.now()
        };
        
        wsRef.current.send(JSON.stringify(message));
        console.log('?? Audio sent automatically, waiting for AI response...');
      } else {
        console.error('? WebSocket not connected - cannot send audio');
        setConversationState('error');
        setConnectionError('Connection lost. Please refresh.');
      }
      
    } catch (error) {
      console.error('? Automatic audio processing failed:', error);
      setConversationState('error');
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

  // MANUAL: Only emergency stop button (not part of automated flow)
  const emergencyStopInterview = () => {
    console.log('?? EMERGENCY: Stopping automated interview...');
    setInterviewStarted(false);
    setConversationState('stopped');
    stopAutomaticRecording();
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'end_interview' }));
    }
    
    setTimeout(() => {
      navigate('/student/mock-interviews');
    }, 1000);
  };

  const cleanup = () => {
    console.log('?? Cleaning up automated interview resources...');
    
    stopAutomaticRecording();
    
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    if (keepAliveIntervalRef.current) {
      clearInterval(keepAliveIntervalRef.current);
    }
    
    if (autoListenTimeoutRef.current) {
      clearTimeout(autoListenTimeoutRef.current);
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    
    audioQueueRef.current = [];
    isPlayingAudioRef.current = false;
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

  const getConversationStateLabel = () => {
    const labels = {
      initializing: 'Setting up automation...',
      ready: '?? Ready to Start',
      ai_speaking: '?? AI Speaking',
      listening: '?? Listening to You',
      processing: '?? Processing Response',
      complete: '? Complete',
      error: '? Error',
      stopped: '?? Stopped'
    };
    return labels[conversationState] || 'Unknown';
  };

  // Loading state
  if (isConnecting) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box textAlign="center">
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h5" gutterBottom>
            Initializing Fully Automated Interview...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Setting up microphone and AI connection - No manual buttons needed!
          </Typography>
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
            ??? Fully Automated AI Interview
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            {studentName} • Zero Manual Buttons - Completely Automated
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
            icon={<Mic />}
            label={getConversationStateLabel()}
            color={
              conversationState === 'ai_speaking' ? 'primary' :
              conversationState === 'listening' ? 'success' :
              conversationState === 'processing' ? 'warning' :
              'default'
            }
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
                      AI speaking automatically... ({audioQueueRef.current.length} chunks queued)
                    </Typography>
                  </Box>
                )}
              </Box>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Automated Status Display */}
      <Paper sx={{ p: 3, textAlign: 'center', borderRadius: 2 }}>
        <Box>
          {/* Conversation State Indicator */}
          <Box sx={{ mb: 3 }}>
            <Box display="flex" alignItems="center" justifyContent="center" gap={2} mb={2}>
              <Mic 
                sx={{ 
                  fontSize: 40,
                  color: isListening ? 'success.main' : 
                         isAIPlaying ? 'primary.main' : 'grey.400',
                  animation: isListening ? 'pulse 1s infinite' : 'none'
                }} 
              />
              <Typography variant="h6">
                {getConversationStateLabel()}
              </Typography>
            </Box>
            
            {/* Audio Level Meter - Only when listening */}
            {isListening && (
              <Box>
                <LinearProgress 
                  variant="determinate" 
                  value={Math.min(audioLevel * 100, 100)} 
                  color="success"
                  sx={{ height: 8, borderRadius: 4, mb: 1 }}
                />
                <Typography variant="caption" color="text.secondary">
                  Voice Level: {Math.round(audioLevel * 100)}% | Threshold: {Math.round(SILENCE_THRESHOLD * 100)}%
                </Typography>
              </Box>
            )}
            
            {/* Silence Timer - Only when detecting silence */}
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
          
          {/* Status Messages */}
          <Box sx={{ mb: 3 }}>
            {conversationState === 'initializing' && (
              <Typography variant="body1" color="text.secondary">
                ?? Setting up fully automated interview system...
              </Typography>
            )}
            
            {conversationState === 'ready' && (
              <Typography variant="body1" color="primary">
                ? System ready! Interview will start automatically when AI responds.
              </Typography>
            )}
            
            {conversationState === 'ai_speaking' && (
              <Typography variant="body1" color="primary">
                ?? AI is speaking - Listen carefully, recording will start automatically when finished
              </Typography>
            )}
            
            {conversationState === 'listening' && (
              <Typography variant="body1" color="success.main">
                ?? Recording automatically - Speak naturally, will stop on {SILENCE_DURATION/1000}s silence
              </Typography>
            )}
            
            {conversationState === 'processing' && (
              <Typography variant="body1" color="warning.main">
                ?? Processing your response - AI will speak next automatically
              </Typography>
            )}
            
            {!interviewStarted && conversationState === 'initializing' && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Interview will start automatically once everything is ready!
              </Typography>
            )}
          </Box>
          
          {/* Emergency Stop Only */}
          {interviewStarted && (
            <Button
              variant="contained"
              color="error"
              startIcon={<Stop />}
              onClick={emergencyStopInterview}
              sx={{ borderRadius: 2 }}
            >
              Emergency Stop
            </Button>
          )}
        </Box>
      </Paper>

      {/* Instructions */}
      <Card sx={{ mt: 3, bgcolor: 'grey.50', borderRadius: 2 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            ?? Fully Automated Interview - Zero Manual Buttons:
          </Typography>
          <Typography variant="body2" component="div">
            <strong>1. 100% Automated:</strong> No buttons to press - everything happens automatically<br/>
            <strong>2. AI Speaks First:</strong> Interview starts with AI greeting you automatically<br/>
            <strong>3. Auto Recording:</strong> When AI finishes, recording starts automatically<br/>
            <strong>4. Silence Detection:</strong> {SILENCE_DURATION/1000} seconds of silence automatically submits your answer<br/>
            <strong>5. Natural Flow:</strong> Continuous conversation like talking to a real interviewer<br/>
            <strong>6. Smart Processing:</strong> Audio sent in chunks for faster response times
          </Typography>
        </CardContent>
      </Card>
    </Container>
  );
};

export default StartInterview;