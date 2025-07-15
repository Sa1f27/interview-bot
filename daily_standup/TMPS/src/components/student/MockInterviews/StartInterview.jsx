import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Alert,
  Button,
  Container,
  LinearProgress,
  IconButton,
  useTheme,
  alpha,
  styled,
  CircularProgress,
  Avatar
} from '@mui/material';
import {
  ArrowBack,
  Assessment,
  PlayArrow,
  Timer as TimerIcon,
  Person,
  FiberManualRecord,
  CheckCircle
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { interviewOperationsAPI } from '../../../services/API/studentmockinterview';

// Styled components for better UI
const StyledCard = styled(Card)(({ theme }) => ({
  height: '100%',
  boxShadow: theme.shadows[4],
  borderRadius: theme.spacing(2),
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    boxShadow: theme.shadows[8],
    transform: 'translateY(-2px)'
  }
}));

const PulsingDot = styled(Box)(({ theme }) => ({
  width: 12,
  height: 12,
  borderRadius: '50%',
  backgroundColor: theme.palette.error.main,
  animation: 'pulse 1.5s infinite',
  '@keyframes pulse': {
    '0%': {
      transform: 'scale(0.95)',
      boxShadow: `0 0 0 0 ${alpha(theme.palette.error.main, 0.7)}`
    },
    '70%': {
      transform: 'scale(1)',
      boxShadow: `0 0 0 10px ${alpha(theme.palette.error.main, 0)}`
    },
    '100%': {
      transform: 'scale(0.95)',
      boxShadow: `0 0 0 0 ${alpha(theme.palette.error.main, 0)}`
    }
  }
}));

// Timer component
const InterviewTimer = ({ startTime, isActive }) => {
  const [elapsed, setElapsed] = useState(0);
  
  useEffect(() => {
    let interval = null;
    if (isActive && startTime) {
      interval = setInterval(() => {
        const now = new Date().getTime();
        const start = new Date(startTime).getTime();
        setElapsed(Math.floor((now - start) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isActive, startTime]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box display="flex" alignItems="center" gap={1}>
      <TimerIcon color="primary" />
      <Typography variant="h6" fontFamily="monospace" color="primary">
        {formatTime(elapsed)}
      </Typography>
    </Box>
  );
};

// Question Counter Component
const QuestionCounter = ({ current, total, round, roundName }) => {
  return (
    <Box display="flex" alignItems="center" gap={2}>
      <Chip 
        label={roundName} 
        color="primary" 
        variant="outlined" 
        size="small"
      />
      <Typography variant="body2" color="text.secondary">
        Question {current} of {total}
      </Typography>
      <LinearProgress 
        variant="determinate" 
        value={(current / total) * 100}
        sx={{ 
          width: 100, 
          height: 6, 
          borderRadius: 3,
          bgcolor: alpha('#1976d2', 0.1)
        }}
      />
    </Box>
  );
};

const StudentStartInterview = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const theme = useTheme();
  const audioRef = useRef(null);
  
  // Interview state
  const [interview, setInterview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [interviewStarted, setInterviewStarted] = useState(false);
  
  // Question state
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [questionCount, setQuestionCount] = useState(1);
  const [totalQuestions, setTotalQuestions] = useState(2);
  const [currentRound, setCurrentRound] = useState(1);
  const [roundName, setRoundName] = useState('Technical Round');
  const [interviewerSpeaking, setInterviewerSpeaking] = useState(false);
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  // Speech recognition state
  const [speechRecognition, setSpeechRecognition] = useState(null);
  const [transcribedText, setTranscribedText] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);

  // Timer state
  const [interviewStartTime, setInterviewStartTime] = useState(null);
  
  // Audio state
  const audioContextRef = useRef(null);
  const [defaultVoice, setDefaultVoice] = useState(null);

  // Round configuration
  const rounds = {
    1: { name: 'Technical Round', questions: 2 },
    2: { name: 'Communication Round', questions: 2 },
    3: { name: 'HR Round', questions: 2 }
  };

  useEffect(() => {
    initializeInterview();
    initializeVoice();
  }, [id]);

  // Initialize speech recognition
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;
      
      setSpeechRecognition(recognition);
      console.log('✅ Speech Recognition initialized');
    } else {
      console.warn('⚠️ Speech Recognition not supported');
    }
  }, []);

  // Test microphone on component mount
  useEffect(() => {
    const checkMicrophone = async () => {
      console.log('🔍 Checking microphone on component mount...');
      const micWorking = await testMicrophone();
      if (!micWorking) {
        setError('Microphone access is required for this interview. Please allow microphone permissions and refresh the page.');
      } else {
        console.log('✅ Microphone check passed');
      }
    };
    
    if (!interviewStarted) {
      checkMicrophone();
    }
  }, [interviewStarted]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      speechSynthesis.cancel();
    };
  }, []);

  // Speech-to-Text recording function with auto-stop
  const startSpeechRecording = async () => {
    try {
      console.log('🎤 STARTING SPEECH-TO-TEXT RECORDING...');
      
      if (!speechRecognition) {
        setError('Speech recognition not supported in this browser. Please use Chrome or Edge.');
        return;
      }

      setIsRecording(true);
      setIsTranscribing(true);
      setTranscribedText('');
      
      let finalTranscript = '';
      let interimTranscript = '';
      let silenceTimer = null;
      let lastSpeechTime = Date.now();

      // Configuration for auto-stop
      const SILENCE_DURATION = 2000; // 2 seconds of silence
      const MAX_RECORDING_TIME = 60000; // 60 seconds max

      speechRecognition.onstart = () => {
        console.log('🔊 Speech recognition started');
        lastSpeechTime = Date.now();
      };

      speechRecognition.onresult = (event) => {
        interimTranscript = '';
        let hasNewFinalResult = false;
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' ';
            hasNewFinalResult = true;
            console.log('📝 Final transcript chunk:', transcript);
          } else {
            interimTranscript += transcript;
          }
        }
        
        // Update last speech time when we get new results
        if (hasNewFinalResult || interimTranscript.trim()) {
          lastSpeechTime = Date.now();
          
          // Clear existing silence timer
          if (silenceTimer) {
            clearTimeout(silenceTimer);
          }
          
          // Set new silence timer only if we have some final transcript
          if (finalTranscript.trim()) {
            silenceTimer = setTimeout(() => {
              const timeSinceLastSpeech = Date.now() - lastSpeechTime;
              if (timeSinceLastSpeech >= SILENCE_DURATION && speechRecognition) {
                console.log('🔇 Auto-stopping due to silence');
                speechRecognition.stop();
              }
            }, SILENCE_DURATION);
          }
        }
        
        const currentText = finalTranscript + interimTranscript;
        setTranscribedText(currentText);
        console.log('🗣️ Current transcription:', currentText);
      };

      speechRecognition.onerror = (event) => {
        console.error('❌ Speech recognition error:', event.error);
        setIsRecording(false);
        setIsTranscribing(false);
        
        if (silenceTimer) {
          clearTimeout(silenceTimer);
        }
        
        if (event.error === 'no-speech') {
          setError('No speech detected. Please speak louder and try again.');
        } else if (event.error === 'audio-capture') {
          setError('Microphone not accessible. Please check permissions.');
        } else if (event.error === 'aborted') {
          console.log('Speech recognition was aborted (normal for auto-stop)');
          // Don't set error for aborted - this is normal
        } else {
          setError(`Speech recognition error: ${event.error}`);
        }
      };

      speechRecognition.onend = () => {
        console.log('⏹️ Speech recognition ended');
        console.log('📝 Final transcribed text:', finalTranscript.trim());
        
        setIsRecording(false);
        setIsTranscribing(false);
        
        if (silenceTimer) {
          clearTimeout(silenceTimer);
        }
        
        const finalText = finalTranscript.trim();
        if (finalText) {
          setTranscribedText(finalText);
          // Auto-submit the transcribed text
          setTimeout(() => {
            submitTextResponse(finalText);
          }, 500);
        } else {
          setError('No speech was detected. Please try speaking again.');
        }
      };

      // Start recognition
      speechRecognition.start();
      
      // Auto-stop after maximum time
      setTimeout(() => {
        if (speechRecognition && isRecording) {
          console.log('⏰ Auto-stopping speech recognition after maximum time');
          speechRecognition.stop();
        }
        if (silenceTimer) {
          clearTimeout(silenceTimer);
        }
      }, MAX_RECORDING_TIME);
      
    } catch (error) {
      console.error('❌ Failed to start speech recording:', error);
      setError('Failed to start speech recognition: ' + error.message);
      setIsRecording(false);
      setIsTranscribing(false);
    }
  };

  // Submit transcribed text to backend - FIXED VERSION
  // Replace your submitTextResponse function in StartInterview.jsx with this:

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

    // FIXED: Call the correct API method with proper error handling
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
        
        // Try to extract any meaningful response
        const possibleResponse = response.ai_response || response.next_question || response.message || JSON.stringify(response);
        
        if (possibleResponse && possibleResponse !== '{}') {
          console.log('📝 Using fallback response extraction');
          setCurrentQuestion(possibleResponse);
          
          setTimeout(async () => {
            await playInterviewerAudio(null, possibleResponse);
          }, 1000);
        } else {
          setError('Received unexpected response from server. Please try again.');
        }
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
    } else if (error.message.includes('network') || error.message.includes('NetworkError')) {
      errorMessage += 'Network error. Please check your connection.';
    } else if (error.message.includes('timeout')) {
      errorMessage += 'Request timed out. Please try again.';
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
  // Add microphone testing function
  const testMicrophone = async () => {
    try {
      console.log('Testing microphone...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('Microphone test successful');
      
      const audioTracks = stream.getAudioTracks();
      console.log('Available audio tracks:', audioTracks.length);
      
      if (audioTracks.length > 0) {
        console.log('Microphone capabilities:', audioTracks[0].getCapabilities());
        console.log('Microphone settings:', audioTracks[0].getSettings());
      }
      
      stream.getTracks().forEach(track => track.stop());
      return true;
    } catch (error) {
      console.error('Microphone test failed:', error);
      return false;
    }
  };

  const initializeVoice = () => {
    if ('speechSynthesis' in window) {
      const loadVoices = () => {
        const voices = speechSynthesis.getVoices();
        console.log('🎙️ Available voices:', voices.length);
        
        const englishVoices = voices.filter(voice => voice.lang.startsWith('en'));
        console.log('🇺🇸 English voices:', englishVoices.length);
        
        if (englishVoices.length > 0) {
          const preferredVoice = 
            englishVoices.find(voice => voice.name.includes('Google')) ||
            englishVoices.find(voice => voice.name.includes('Microsoft')) ||
            englishVoices.find(voice => voice.name.includes('Female')) ||
            englishVoices.find(voice => voice.name.includes('Samantha')) ||
            englishVoices.find(voice => voice.default) ||
            englishVoices[0];
          
          setDefaultVoice(preferredVoice);
          console.log('✅ Selected voice:', preferredVoice.name, preferredVoice.lang);
          
          setTimeout(() => {
            const testUtterance = new SpeechSynthesisUtterance("Voice test successful");
            testUtterance.voice = preferredVoice;
            testUtterance.volume = 1.0;
            testUtterance.rate = 0.8;
            testUtterance.onstart = () => console.log('🔊 Voice test started');
            testUtterance.onend = () => console.log('✅ Voice test completed');
            testUtterance.onerror = (e) => console.log('❌ Voice test failed:', e);
            speechSynthesis.speak(testUtterance);
          }, 1000);
        } else {
          console.warn('⚠️ No English voices found');
        }
      };
      
      loadVoices();
      speechSynthesis.addEventListener('voiceschanged', loadVoices);
      
      if (speechSynthesis.getVoices().length === 0) {
        console.log('🔄 Forcing voice loading...');
        speechSynthesis.speak(new SpeechSynthesisUtterance(''));
      }
    } else {
      console.error('❌ Speech synthesis not supported');
    }
  };

  const initializeInterview = async () => {
    if (!id) {
      setError('No interview ID provided');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      console.log('Initializing interview with ID:', id);
      
      const interviewData = {
        testId: id,
        status: 'ready',
        currentRound: 1,
        totalRounds: 3,
        startTime: new Date().toISOString(),
        isComplete: false
      };

      setInterview(interviewData);
      setInterviewStartTime(new Date().toISOString());
      setRoundName(rounds[1].name);
      setTotalQuestions(rounds[1].questions);
      
    } catch (error) {
      console.error('Failed to initialize interview:', error);
      setError('Failed to load interview data: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  // Force TTS for reliable audio
  const speakQuestion = (text) => {
    return new Promise((resolve) => {
      if (!('speechSynthesis' in window) || !text) {
        console.log('⚠️ Speech synthesis not available or no text');
        resolve();
        return;
      }
      
      console.log('🗣️ FORCE SPEAKING TEXT:', text.substring(0, 100) + '...');
      
      speechSynthesis.cancel();
      
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        
        if (defaultVoice) {
          utterance.voice = defaultVoice;
          console.log('🎙️ Using voice:', defaultVoice.name);
        } else {
          console.log('🎙️ Using default system voice');
        }
        
        utterance.rate = 0.8;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        
        utterance.onstart = () => {
          console.log('🔊 TTS STARTED SPEAKING');
        };
        
        utterance.onend = () => {
          console.log('✅ TTS FINISHED SPEAKING');
          resolve();
        };
        
        utterance.onerror = (e) => {
          console.log('❌ TTS ERROR:', e);
          resolve();
        };
        
        console.log('🚀 Starting TTS speech...');
        speechSynthesis.speak(utterance);
        
        setTimeout(() => {
          console.log('⏰ TTS backup timeout triggered');
          resolve();
        }, Math.max(text.length * 100, 3000));
        
      }, 250);
    });
  };

  // ALWAYS use TTS - force speech for reliability
  const playInterviewerAudio = async (audioPath, questionText = '') => {
    try {
      console.log('🗣️ PLAYING QUESTION');
      console.log('📁 Audio path:', audioPath);
      console.log('📝 Question text:', questionText);
      
      setInterviewerSpeaking(true);
      
      // ALWAYS use TTS for reliability
      if (questionText) {
        console.log('🔊 USING TEXT-TO-SPEECH (FORCED)');
        await speakQuestion(questionText);
      } else {
        console.warn('⚠️ NO QUESTION TEXT PROVIDED');
        await speakQuestion("Please answer the current question.");
      }
      
      setInterviewerSpeaking(false);
      console.log('✅ Question finished, starting recording immediately');
      
      // Start SPEECH-TO-TEXT recording IMMEDIATELY when question finishes
      setTimeout(() => {
        if (!isRecording && !transcribedText) {
          console.log('🎤 Auto-starting speech recording after question');
          startSpeechRecording();
        }
      }, 500);
      
    } catch (error) {
      console.error('❌ Audio playback failed:', error);
      setInterviewerSpeaking(false);
      
      setTimeout(() => {
        if (!isRecording && !transcribedText) {
          console.log('🎤 Starting speech recording despite audio failure');
          startSpeechRecording();
        }
      }, 1000);
    }
  };

  const startInterview = async () => {
    try {
      setLoading(true);
      console.log('Starting interview session...');
      
      const response = await interviewOperationsAPI.startInterview();
      console.log('Start interview response:', response);
      
      if (response.question) {
        setCurrentQuestion(response.question);
        setInterviewStarted(true);
        
        await playInterviewerAudio(response.audio_path, response.question);
        
        setInterview(prev => ({
          ...prev,
          ...response,
          status: 'in_progress'
        }));
      }
      
    } catch (error) {
      console.error('Failed to start interview:', error);
      setError('Failed to start interview: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const startNextRound = async () => {
    try {
      setLoading(true);
      console.log('🚀 Starting next round...');
      
      const response = await interviewOperationsAPI.startNextRound(interview.testId);
      
      if (response.question) {
        console.log('📝 Got next round question:', response.question);
        setCurrentQuestion(response.question);
        setInterviewStarted(true);
        setQuestionCount(1);
        
        setTimeout(async () => {
          await playInterviewerAudio(response.audio_path, response.question);
        }, 500);
      }
      
    } catch (error) {
      console.error('❌ Failed to start next round:', error);
      setError('Failed to start next round: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInterviewComplete = async () => {
    try {
      const evaluation = await interviewOperationsAPI.evaluateInterview(interview.testId);
      navigate(`/student/mock-interviews/results/${interview.testId}`, {
        state: { evaluation }
      });
    } catch (error) {
      console.error('Failed to get evaluation:', error);
      setError('Failed to get evaluation: ' + error.message);
    }
  };

  // Manual speech recording start for testing
  const startManualSpeechRecording = () => {
    console.log('🔴 Manual speech recording test started');
    startSpeechRecording();
  };

  if (loading && !interview) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" py={8}>
          <CircularProgress size={60} />
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3, borderRadius: 2 }}>
          {error}
        </Alert>
        <Button
          variant="outlined"
          startIcon={<ArrowBack />}
          onClick={() => navigate('/student/mock-interviews')}
        >
          Back to Mock Interviews
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ py: 4, minHeight: '100vh' }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={3}>
        <Box display="flex" alignItems="center">
          <IconButton onClick={() => navigate('/student/mock-interviews')} sx={{ mr: 1 }}>
            <ArrowBack />
          </IconButton>
          <Assessment sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h4" fontWeight="bold" color="primary.main">
            Interview Session
          </Typography>
        </Box>
        
        <Box display="flex" alignItems="center" gap={3}>
          <QuestionCounter 
            current={questionCount} 
            total={totalQuestions} 
            round={currentRound}
            roundName={roundName}
          />
          <InterviewTimer 
            startTime={interviewStartTime} 
            isActive={interviewStarted}
          />
        </Box>
      </Box>

      {/* Hidden audio element */}
      <audio 
        ref={audioRef} 
        style={{ display: 'none' }} 
        preload="metadata"
        controls={false}
        crossOrigin="anonymous"
      >
        <source src="" type="audio/mpeg" />
        <source src="" type="audio/wav" />
        <source src="" type="audio/ogg" />
        <source src="" type="audio/webm" />
        <source src="" type="audio/mp4" />
      </audio>

      {!interviewStarted && !loading ? (
        /* Start Interview Card */
        <StyledCard sx={{ textAlign: 'center', maxWidth: 600, mx: 'auto' }}>
          <CardContent sx={{ p: 6 }}>
            <Box display="flex" justifyContent="center" gap={2} mb={3}>
              <Avatar sx={{ bgcolor: 'primary.main', width: 60, height: 60 }}>
                <Person sx={{ fontSize: 30 }} />
              </Avatar>
              <Avatar sx={{ bgcolor: 'secondary.main', width: 60, height: 60 }}>
                <Person sx={{ fontSize: 30 }} />
              </Avatar>
            </Box>
            <Typography variant="h4" gutterBottom fontWeight="bold">
              Ready to Begin Your Interview?
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
              You'll be interviewed by our expert interviewer. The session consists of three rounds: 
              Technical, Communication, and HR. Each round has multiple questions.
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
              The interview will flow automatically. Listen carefully to each question and answer 
              naturally. Recording will start and stop automatically for each question.
            </Typography>
            
            {/* Enhanced Microphone Permission Notice */}
            <Box sx={{ mb: 4, p: 2, bgcolor: 'info.light', borderRadius: 2 }}>
              <Typography variant="body2" color="info.dark" gutterBottom>
                <strong>🎤 Audio & Microphone Setup:</strong>
              </Typography>
              <Typography variant="body2" color="info.dark" sx={{ mb: 2 }}>
                • Make sure your speakers/headphones are on and volume is up<br/>
                • Allow microphone access when prompted<br/>
                • Test both audio and microphone before starting
              </Typography>
              
              {/* Test Recording Button */}
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={startManualSpeechRecording}
                  disabled={isRecording || interviewStarted}
                >
                  {isRecording ? 'Recording Speech...' : 'Test Speech-to-Text'}
                </Button>
                
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => speakQuestion("Hello! This is a voice test. Can you hear me clearly?")}
                  disabled={interviewStarted}
                >
                  Test Speaker Voice
                </Button>
              </Box>
              
              {isRecording && (
                <Box sx={{ mt: 1 }}>
                  <Typography variant="caption" color="error" sx={{ display: 'block' }}>
                    🔴 Recording - Speak now to test speech-to-text
                  </Typography>
                  {transcribedText && (
                    <Typography variant="caption" color="primary" sx={{ display: 'block', mt: 1 }}>
                      📝 Transcribed: "{transcribedText}"
                    </Typography>
                  )}
                </Box>
              )}
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
              <strong>Note:</strong> Make sure you're in a quiet environment for the best interview experience.
            </Typography>
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={startInterview}
              disabled={loading}
              sx={{
                px: 6,
                py: 2,
                fontSize: '1.2rem',
                borderRadius: 3,
                boxShadow: 4
              }}
            >
              {loading ? 'Starting Interview...' : 'Start Interview'}
            </Button>
          </CardContent>
        </StyledCard>
      ) : (
        /* Main Interview Interface - Side by Side */
        <Grid container spacing={3} sx={{ height: 'calc(100vh - 200px)' }}>
          {/* Interviewer Side - Left */}
          <Grid item xs={6}>
            <StyledCard sx={{ height: '100%' }}>
              <CardHeader
                avatar={
                  <Avatar sx={{ bgcolor: 'primary.main', width: 50, height: 50 }}>
                    <Person sx={{ fontSize: 25 }} />
                  </Avatar>
                }
                title="Interviewer"
                subheader={
                  interviewerSpeaking 
                    ? "Asking question..." 
                    : "Listening to your response"
                }
                action={
                  interviewerSpeaking && (
                    <Box display="flex" alignItems="center" gap={1}>
                      <PulsingDot />
                      <Typography variant="caption" color="error">
                        Speaking
                      </Typography>
                    </Box>
                  )
                }
                sx={{ 
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  borderBottom: `1px solid ${theme.palette.divider}`
                }}
              />
              <CardContent sx={{ p: 4, height: 'calc(100% - 80px)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <Box textAlign="center">
                  {interviewerSpeaking ? (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'primary.main', mb: 3 }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Asking Question
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Please listen carefully...
                      </Typography>
                    </Box>
                  ) : (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'grey.500', mb: 3 }} />
                      <Typography variant="h5" color="text.secondary" gutterBottom>
                        Listening
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Waiting for your answer
                      </Typography>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>

          {/* Candidate Side - Right */}
          <Grid item xs={6}>
            <StyledCard sx={{ height: '100%' }}>
              <CardHeader
                avatar={
                  <Avatar sx={{ bgcolor: 'secondary.main', width: 50, height: 50 }}>
                    <Person sx={{ fontSize: 25 }} />
                  </Avatar>
                }
                title="You"
                subheader={
                  isRecording 
                    ? "Recording your answer..." 
                    : isSubmitting 
                    ? "Processing response..." 
                    : interviewerSpeaking 
                    ? "Listen to the question" 
                    : "Ready to answer"
                }
                action={
                  isRecording && (
                    <Box display="flex" alignItems="center" gap={1}>
                      <PulsingDot />
                      <Typography variant="caption" color="error">
                        Recording
                      </Typography>
                    </Box>
                  )
                }
                sx={{ 
                  bgcolor: alpha(theme.palette.secondary.main, 0.1),
                  borderBottom: `1px solid ${theme.palette.divider}`
                }}
              />
              <CardContent sx={{ p: 4, height: 'calc(100% - 80px)', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <Box textAlign="center">
                  {interviewerSpeaking && (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'primary.main', mb: 3 }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Listen to Question
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Recording will start automatically
                      </Typography>
                    </Box>
                  )}

                  {isRecording && (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'error.main', mb: 3 }} />
                      <FiberManualRecord sx={{ fontSize: 40, color: 'error.main', mb: 2 }} />
                      <Typography variant="h5" color="error" gutterBottom>
                        Recording Your Answer
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                        Speak naturally, recording will stop automatically when you finish
                      </Typography>
                      {transcribedText && (
                        <Box sx={{ mt: 2, p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                          <Typography variant="body2" color="white">
                            📝 "{transcribedText}"
                          </Typography>
                        </Box>
                      )}
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                        Recording will stop after 2 seconds of silence
                      </Typography>
                    </Box>
                  )}

                  {isSubmitting && (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'primary.main', mb: 3 }} />
                      <CircularProgress size={60} sx={{ mb: 2 }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Processing Response
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Analyzing your answer...
                      </Typography>
                    </Box>
                  )}

                  {!isRecording && !isSubmitting && !interviewerSpeaking && (
                    <Box>
                      <Person sx={{ fontSize: 120, color: 'success.main', mb: 3 }} />
                      <CheckCircle sx={{ fontSize: 40, color: 'success.main', mb: 2 }} />
                      <Typography variant="h5" color="success.main" gutterBottom>
                        Ready for Next Question
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Great job! Preparing next question...
                      </Typography>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      )}

      {/* Simple Status Bar */}
      <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={3}>
            <Typography variant="body2" color="text.secondary">
              <strong>Round:</strong> {roundName}
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="text.secondary">
              <strong>Status:</strong> {
                interviewerSpeaking ? 'Asking Question' :
                isRecording ? 'Recording' :
                isSubmitting ? 'Processing' :
                'Waiting'
              }
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="text.secondary">
              <strong>Progress:</strong> Round {currentRound} of 3
            </Typography>
          </Grid>
          <Grid item xs={3}>
            <Typography variant="body2" color="text.secondary">
              <strong>Question:</strong> {questionCount} of {totalQuestions}
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default StudentStartInterview;