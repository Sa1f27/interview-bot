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
  CheckCircle,
  Mic,
  MicOff,
  VolumeUp,
  Pause
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { wsManager, recordAudio } from '../../../services/API/index2';

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
  
  // Interview state
  const [sessionId, setSessionId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [interviewStarted, setInterviewStarted] = useState(false);
  
  // Question state
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [questionCount, setQuestionCount] = useState(1);
  const [totalQuestions, setTotalQuestions] = useState(6);
  const [currentRound, setCurrentRound] = useState(1);
  const [roundName, setRoundName] = useState('Technical Round');
  const [interviewerSpeaking, setInterviewerSpeaking] = useState(false);
  
  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [canRecord, setCanRecord] = useState(false);
  const [autoRecordingEnabled, setAutoRecordingEnabled] = useState(false);

  // WebSocket state
  const [wsConnected, setWsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Timer state
  const [interviewStartTime, setInterviewStartTime] = useState(null);

  // Audio playback state
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);

  // Round configuration
  const rounds = {
    1: { name: 'Technical Round', questions: 6 },
    2: { name: 'Communication Round', questions: 6 },
    3: { name: 'HR Round', questions: 6 }
  };

  useEffect(() => {
    setSessionId(id);
    setInterviewStartTime(new Date().toISOString());
    setRoundName(rounds[1].name);
    setTotalQuestions(rounds[1].questions);
    
    testMicrophone();
    
    if (id) {
      initializeWebSocket(id);
    }
    
    setLoading(false);
  }, [id]);

  useEffect(() => {
    return () => {
      if (sessionId) {
        wsManager.disconnect(sessionId);
      }
    };
  }, [sessionId]);

  const testMicrophone = async () => {
    try {
      console.log('🔍 Testing microphone...');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log('✅ Microphone test successful');
      
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length > 0) {
        console.log('Microphone capabilities:', audioTracks[0].getCapabilities());
      }
      
      stream.getTracks().forEach(track => track.stop());
      setCanRecord(true);
      return true;
    } catch (error) {
      console.error('❌ Microphone test failed:', error);
      setError('Microphone access is required for this interview. Please allow microphone permissions and refresh the page.');
      setCanRecord(false);
      return false;
    }
  };

  const initializeWebSocket = (sessionId) => {
    console.log('🔌 Initializing WebSocket for session:', sessionId);
    setConnectionStatus('connecting');
    
    const ws = wsManager.connect(
      sessionId,
      handleWebSocketMessage,
      handleWebSocketError,
      handleWebSocketClose
    );

    ws.addEventListener('open', () => {
      setWsConnected(true);
      setConnectionStatus('connected');
      console.log('✅ WebSocket connected successfully');
    });
  };

  const handleWebSocketMessage = (data) => {
    console.log('📨 WebSocket message received:', data);
    
    switch (data.type) {
      case 'ai_response':
        setCurrentQuestion(data.message);
        setInterviewerSpeaking(true);
        setIsSubmitting(false);
        
        if (data.stage) {
          const stageMap = { 'greeting': 1, 'technical': 1, 'communication': 2, 'hr': 3 };
          const newRound = stageMap[data.stage] || currentRound;
          if (newRound !== currentRound) {
            setCurrentRound(newRound);
            setRoundName(rounds[newRound]?.name || 'Unknown Round');
            setQuestionCount(1);
          }
        }
        break;
        
      case 'audio_data':
        if (data.audio) {
          playAudioData(data.audio);
        }
        break;
        
      case 'audio_end':
        setInterviewerSpeaking(false);
        setIsPlayingAudio(false);
        
        // AUTOMATIC RECORDING START - No button needed!
        if (autoRecordingEnabled) {
          console.log('🤖 AI finished speaking, auto-starting user recording...');
          setTimeout(() => {
            startAutomaticRecording();
          }, 1000); // 1 second delay for natural flow
        }
        break;
        
      case 'interview_complete':
        setInterviewStarted(false);
        setAutoRecordingEnabled(false);
        handleInterviewComplete(data.result);
        break;
        
      case 'error':
        setError(data.message);
        setIsSubmitting(false);
        setIsRecording(false);
        break;
        
      case 'timeout':
        setError('Interview session timed out due to inactivity.');
        break;
        
      default:
        console.log('Unknown WebSocket message type:', data.type);
    }
  };

  const handleWebSocketError = (error) => {
    console.error('❌ WebSocket error:', error);
    setWsConnected(false);
    setConnectionStatus('error');
    setError('Connection error. Please check your internet connection and try again.');
  };

  const handleWebSocketClose = (event) => {
    console.log('🔌 WebSocket closed:', event.code, event.reason);
    setWsConnected(false);
    setConnectionStatus('disconnected');
    
    if (event.code !== 1000 && interviewStarted) {
      setError('Connection lost. Please refresh the page to reconnect.');
    }
  };

  const playAudioData = async (audioBase64) => {
    try {
      setIsPlayingAudio(true);
      
      const audioData = atob(audioBase64);
      const audioArray = new Uint8Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
      }
      
      const audioBlob = new Blob([audioArray], { type: 'audio/webm' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const audio = new Audio(audioUrl);
      audio.onended = () => {
        setIsPlayingAudio(false);
        URL.revokeObjectURL(audioUrl);
      };
      audio.onerror = (error) => {
        console.error('❌ Audio playback error:', error);
        setIsPlayingAudio(false);
        URL.revokeObjectURL(audioUrl);
      };
      
      await audio.play();
      
    } catch (error) {
      console.error('❌ Failed to play audio:', error);
      setIsPlayingAudio(false);
    }
  };

  const startAutomaticRecording = async () => {
    if (!canRecord || isRecording || !autoRecordingEnabled) {
      return;
    }

    try {
      setIsRecording(true);
      setError(null);
      
      console.log('🎤 Starting automatic audio recording...');
      
      // Use the enhanced natural recording with silence detection
      const audioBlob = await recordAudio(30000);
      
      setIsRecording(false);
      setIsSubmitting(true);
      
      // Convert audio to base64 and send via WebSocket
      const reader = new FileReader();
      reader.onload = () => {
        const audioBase64 = reader.result.split(',')[1];
        
        const message = {
          type: 'audio_data',
          audio: audioBase64
        };
        
        if (wsManager.send(sessionId, message)) {
          console.log('📤 Audio sent via WebSocket');
        } else {
          setError('Failed to send audio. Please check your connection.');
          setIsSubmitting(false);
        }
      };
      
      reader.onerror = () => {
        setError('Failed to process audio recording.');
        setIsSubmitting(false);
      };
      
      reader.readAsDataURL(audioBlob);
      
    } catch (error) {
      console.error('❌ Recording failed:', error);
      setError('Failed to record audio: ' + error.message);
      setIsRecording(false);
      setIsSubmitting(false);
    }
  };

  const startInterview = async () => {
    if (!wsConnected) {
      setError('Not connected to server. Please refresh and try again.');
      return;
    }

    try {
      setLoading(true);
      setInterviewStarted(true);
      setAutoRecordingEnabled(true); // Enable automatic recording after this point
      
      console.log('🚀 Starting fully automatic interview session...');
      
      const message = {
        type: 'start_interview'
      };
      
      if (wsManager.send(sessionId, message)) {
        console.log('📤 Start interview message sent');
      } else {
        throw new Error('Failed to start interview via WebSocket');
      }
      
    } catch (error) {
      console.error('❌ Failed to start interview:', error);
      setError('Failed to start interview: ' + error.message);
      setInterviewStarted(false);
      setAutoRecordingEnabled(false);
    } finally {
      setLoading(false);
    }
  };

  const completeInterview = () => {
    const message = {
      type: 'complete_interview'
    };
    
    setAutoRecordingEnabled(false); // Disable automatic recording
    
    if (wsManager.send(sessionId, message)) {
      console.log('📤 Complete interview message sent');
    } else {
      setError('Failed to complete interview. Please try again.');
    }
  };

  const handleInterviewComplete = (result) => {
    try {
      console.log('✅ Interview completed:', result);
      setAutoRecordingEnabled(false);
      
      navigate(`/student/mock-interviews/results/${id}`, {
        state: { evaluation: result }
      });
      
    } catch (error) {
      console.error('❌ Failed to handle interview completion:', error);
      setError('Failed to complete interview: ' + error.message);
    }
  };

  if (loading && !sessionId) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" py={8}>
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ ml: 2 }}>
            Initializing interview session...
          </Typography>
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
        <Box display="flex" gap={2}>
          <Button
            variant="outlined"
            startIcon={<ArrowBack />}
            onClick={() => navigate('/student/mock-interviews')}
          >
            Back to Mock Interviews
          </Button>
          <Button
            variant="contained"
            onClick={() => window.location.reload()}
          >
            Retry Connection
          </Button>
        </Box>
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

      {/* Connection Status */}
      <Alert 
        severity={wsConnected ? 'success' : 'warning'} 
        sx={{ mb: 3 }}
        action={
          !wsConnected && (
            <Button 
              color="inherit" 
              size="small" 
              onClick={() => initializeWebSocket(sessionId)}
            >
              Reconnect
            </Button>
          )
        }
      >
        Connection Status: {connectionStatus} 
        {wsConnected && ' - Ready for interview'}
      </Alert>

      {/* Automatic Mode Indicator */}
      {autoRecordingEnabled && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            🤖 <strong>Automatic Mode Active:</strong> The interview will flow naturally. 
            Speak after the AI finishes, and recording will stop automatically when you pause for 2 seconds.
          </Typography>
        </Alert>
      )}

      {!interviewStarted && !loading ? (
        /* Start Interview Card - ONLY BUTTON IN ENTIRE INTERVIEW */
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
              This will be a fully automatic interview experience. After you press "Start Interview", 
              everything will flow naturally like a real conversation.
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
              <strong>How it works:</strong> The AI will ask questions, you speak naturally, 
              and recording stops automatically when you pause for 2 seconds.
            </Typography>
            
            {/* Enhanced Audio Setup Notice */}
            <Box sx={{ mb: 4, p: 2, bgcolor: 'info.light', borderRadius: 2 }}>
              <Typography variant="body2" color="info.dark" gutterBottom>
                <strong>🎤 Automatic Audio Setup:</strong>
              </Typography>
              <Typography variant="body2" color="info.dark" sx={{ mb: 2 }}>
                • Your speakers/headphones should be on and volume up<br/>
                • Microphone permissions will be requested<br/>
                • Recording starts and stops automatically<br/>
                • Speak naturally - pauses of 2+ seconds will trigger AI response
              </Typography>
              
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={testMicrophone}
                  disabled={interviewStarted}
                  startIcon={canRecord ? <Mic color="success" /> : <MicOff color="error" />}
                >
                  {canRecord ? 'Microphone Ready' : 'Test Microphone'}
                </Button>
              </Box>
            </Box>
            
            <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
              <strong>Note:</strong> This is the ONLY button you'll need to click. 
              Everything else happens automatically for a natural interview experience.
            </Typography>
            
            {/* THE ONLY BUTTON IN THE ENTIRE INTERVIEW */}
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={startInterview}
              disabled={loading || !wsConnected || !canRecord}
              sx={{
                px: 6,
                py: 2,
                fontSize: '1.2rem',
                borderRadius: 3,
                boxShadow: 4,
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                '&:hover': {
                  background: 'linear-gradient(45deg, #1976D2 30%, #2196F3 90%)',
                  transform: 'translateY(-2px)',
                  boxShadow: 6
                }
              }}
            >
              {loading ? 'Starting Interview...' : 'Start Interview'}
            </Button>
            
            {!wsConnected && (
              <Typography variant="caption" color="error" sx={{ display: 'block', mt: 2 }}>
                Please wait for connection to establish
              </Typography>
            )}
          </CardContent>
        </StyledCard>
      ) : (
        /* Main Interview Interface - Fully Automatic */
        <Grid container spacing={3} sx={{ height: 'calc(100vh - 250px)' }}>
          {/* Interviewer Side - Left */}
          <Grid item xs={6}>
            <StyledCard sx={{ height: '100%' }}>
              <CardHeader
                avatar={
                  <Avatar sx={{ bgcolor: 'primary.main', width: 50, height: 50 }}>
                    <Person sx={{ fontSize: 25 }} />
                  </Avatar>
                }
                title="AI Interviewer"
                subheader={
                  interviewerSpeaking || isPlayingAudio
                    ? "Speaking..." 
                    : "Listening to your response"
                }
                action={
                  (interviewerSpeaking || isPlayingAudio) && (
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
              <CardContent sx={{ p: 4, height: 'calc(100% - 80px)', display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {interviewerSpeaking || isPlayingAudio ? (
                    <Box textAlign="center">
                      <VolumeUp sx={{ fontSize: 120, color: 'primary.main', mb: 3 }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Speaking
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Please listen carefully...
                      </Typography>
                    </Box>
                  ) : (
                    <Box textAlign="center">
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
                
                {/* Current Question Display */}
                {currentQuestion && (
                  <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1, mt: 2 }}>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Current Question:
                    </Typography>
                    <Typography variant="body1">
                      {currentQuestion}
                    </Typography>
                  </Box>
                )}
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
                    : interviewerSpeaking || isPlayingAudio
                    ? "Listen to the question" 
                    : "Speak naturally - it's recording automatically"
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
                  {interviewerSpeaking || isPlayingAudio ? (
                    <Box>
                      <VolumeUp sx={{ fontSize: 120, color: 'primary.main', mb: 3 }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Listen to Question
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Recording will start automatically after the question
                      </Typography>
                    </Box>
                  ) : isRecording ? (
                    <Box>
                      <FiberManualRecord sx={{ fontSize: 120, color: 'error.main', mb: 3 }} />
                      <Typography variant="h5" color="error" gutterBottom>
                        Recording Your Answer
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
                        Speak naturally. Recording will stop automatically when you pause for 2 seconds.
                      </Typography>
                      <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                        <Typography variant="body2" color="text.secondary">
                          💡 Just speak naturally - no buttons needed!
                        </Typography>
                      </Box>
                    </Box>
                  ) : isSubmitting ? (
                    <Box>
                      <CircularProgress size={120} sx={{ mb: 3, color: 'primary.main' }} />
                      <Typography variant="h5" color="primary" gutterBottom>
                        Processing Response
                      </Typography>
                      <Typography variant="body1" color="text.secondary">
                        Analyzing your answer...
                      </Typography>
                    </Box>
                  ) : (
                    <Box>
                      <CheckCircle sx={{ fontSize: 120, color: 'success.main', mb: 3 }} />
                      <Typography variant="h5" color="success.main" gutterBottom>
                        Ready for Next Question
                      </Typography>
                      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                        Automatic recording is active. Speak when ready.
                      </Typography>
                      
                      {/* ONLY Emergency Complete Button - Hidden During Normal Flow */}
                      <Button
                        variant="outlined"
                        color="secondary"
                        onClick={completeInterview}
                        size="small"
                        sx={{ mt: 2 }}
                      >
                        Complete Interview Early
                      </Button>
                    </Box>
                  )}
                </Box>
              </CardContent>
            </StyledCard>
          </Grid>
        </Grid>
      )}

      {/* Status Bar */}
      <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Round:</strong> {roundName}
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Status:</strong> {
                interviewerSpeaking || isPlayingAudio ? 'AI Speaking' :
                isRecording ? 'Recording' :
                isSubmitting ? 'Processing' :
                autoRecordingEnabled ? 'Auto Mode' : 'Ready'
              }
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Progress:</strong> Round {currentRound} of 3
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Question:</strong> {questionCount} of {totalQuestions}
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Connection:</strong> {wsConnected ? '🟢 Connected' : '🔴 Disconnected'}
            </Typography>
          </Grid>
          <Grid item xs={2}>
            <Typography variant="body2" color="text.secondary">
              <strong>Auto Mode:</strong> {autoRecordingEnabled ? '🤖 Active' : '⏸️ Inactive'}
            </Typography>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default StudentStartInterview;