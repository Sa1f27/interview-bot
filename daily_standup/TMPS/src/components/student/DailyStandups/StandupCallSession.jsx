import React, { useState, useEffect, useRef, useCallback } from 'react';
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
  Grid,
  CircularProgress
} from '@mui/material';
import {
  Mic,
  VolumeUp,
  ArrowBack,
  Assignment,
  CheckCircle,
  RadioButtonChecked,
  GraphicEq,
  Timer,
  PlayArrow,
  Stop,
  RecordVoiceOver,
  Warning,
  Refresh
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { standupCallAPI } from '../../../services/API/studentstandup';

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

const aiSpeaking = keyframes`
  0%, 100% { 
    opacity: 1; 
    transform: scale(1);
  }
  50% { 
    opacity: 0.8; 
    transform: scale(1.05);
  }
`;

const voiceWave = keyframes`
  0%, 100% { transform: scaleY(0.5); }
  50% { transform: scaleY(1.5); }
`;

const MainAvatar = styled(Avatar)(({ theme, status }) => ({
  width: 160,
  height: 160,
  margin: '0 auto',
  marginBottom: theme.spacing(3),
  fontSize: '4rem',
  boxShadow: theme.shadows[12],
  border: `4px solid ${alpha(theme.palette.background.paper, 0.8)}`,
  transition: 'all 0.3s ease-in-out',
  ...(status === 'recording' && {
    animation: `${pulse} 1.5s infinite`,
    backgroundColor: theme.palette.error.main,
    borderColor: theme.palette.error.light,
  }),
  ...(status === 'ai_speaking' && {
    animation: `${aiSpeaking} 2s infinite`,
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

const VoiceWaveBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '4px',
  height: '40px',
  '& .wave': {
    width: '4px',
    height: '20px',
    backgroundColor: theme.palette.primary.main,
    borderRadius: '2px',
    animation: `${voiceWave} 0.8s ease-in-out infinite`,
    '&:nth-of-type(1)': { animationDelay: '0s' },
    '&:nth-of-type(2)': { animationDelay: '0.1s' },
    '&:nth-of-type(3)': { animationDelay: '0.2s' },
    '&:nth-of-type(4)': { animationDelay: '0.3s' },
    '&:nth-of-type(5)': { animationDelay: '0.4s' },
    '&:nth-of-type(6)': { animationDelay: '0.5s' },
    '&:nth-of-type(7)': { animationDelay: '0.6s' },
  }
}));

const PersonaCard = styled(Card)(({ theme, isActive }) => ({
  borderRadius: 20,
  overflow: 'hidden',
  background: isActive 
    ? `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary.main, 0.1)})` 
    : `linear-gradient(135deg, ${alpha(theme.palette.grey[100], 0.8)}, ${alpha(theme.palette.grey[50], 0.8)})`,
  border: isActive 
    ? `2px solid ${theme.palette.primary.main}` 
    : `1px solid ${alpha(theme.palette.grey[300], 0.5)}`,
  boxShadow: isActive ? theme.shadows[8] : theme.shadows[2],
  transition: 'all 0.3s ease-in-out',
  '&:hover': {
    transform: 'translateY(-2px)',
    boxShadow: theme.shadows[12]
  }
}));

// ==================== MAIN COMPONENT ====================

const StandupCallSession = () => {
  const { testId: urlTestId } = useParams();
  const navigate = useNavigate();
  const theme = useTheme();
  
  // ==================== STATE MANAGEMENT ====================
  
  const [sessionState, setSessionState] = useState('initializing'); // initializing, ready, connecting, ai_speaking, recording, processing, complete, error
  const [currentStage, setCurrentStage] = useState('greeting'); // greeting, conversation, complete
  const [testId, setTestId] = useState(null);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [progressValue, setProgressValue] = useState(0);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [evaluation, setEvaluation] = useState(null);
  const [score, setScore] = useState(null);
  
  // ==================== REFS ====================
  
  const sessionStartTime = useRef(null);
  const recordingStartTime = useRef(null);
  const durationTimerRef = useRef(null);
  const recordingTimerRef = useRef(null);
  const isInitialized = useRef(false);
  
  // ==================== EFFECTS ====================
  
  useEffect(() => {
    if (!isInitialized.current) {
      isInitialized.current = true;
      initializeSession();
    }
    
    return () => {
      cleanup();
    };
  }, []);
  
  useEffect(() => {
    // Start session duration timer
    if (sessionState === 'connecting' && !durationTimerRef.current) {
      sessionStartTime.current = Date.now();
      durationTimerRef.current = setInterval(() => {
        setSessionDuration(Math.floor((Date.now() - sessionStartTime.current) / 1000));
      }, 1000);
    }
    
    // Start recording timer
    if (sessionState === 'recording' && !recordingTimerRef.current) {
      recordingStartTime.current = Date.now();
      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration(Math.floor((Date.now() - recordingStartTime.current) / 1000));
      }, 1000);
    } else if (sessionState !== 'recording' && recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
      setRecordingDuration(0);
    }
  }, [sessionState]);
  
  // ==================== MAIN FUNCTIONS ====================
  
  const initializeSession = async () => {
    try {
      setSessionState('initializing');
      setError(null);
      
      console.log('🚀 Initializing standup session...');
      
      // Set up event handlers
      standupCallAPI.setEventHandlers({
        onMessage: handleWebSocketMessage,
        onError: handleWebSocketError,
        onClose: handleWebSocketClose,
        onAudioEnd: handleAudioEnd
      });
      
      setCurrentMessage('Starting your standup session...');
      setSessionState('ready');
      
    } catch (error) {
      console.error('❌ Session initialization error:', error);
      setError(error.message);
      setSessionState('error');
    }
  };
  
  const startStandup = async () => {
    try {
      setSessionState('connecting');
      setError(null);
      setCurrentMessage('Connecting to standup session...');
      
      console.log('📞 Starting standup...');
      
      // Start standup session
      const response = await standupCallAPI.startStandup();
      
      if (response && response.test_id) {
        setTestId(response.test_id);
        setIsConnected(true);
        setCurrentMessage('Connected! Waiting for interviewer...');
        setSessionState('ai_speaking');
        
        console.log('✅ Standup session started:', response.test_id);
      } else {
        throw new Error('Failed to start standup session');
      }
      
    } catch (error) {
      console.error('❌ Error starting standup:', error);
      setError(error.message);
      setSessionState('error');
    }
  };
  
  // ==================== WEBSOCKET HANDLERS ====================
  
  const handleWebSocketMessage = useCallback((data) => {
    console.log('📨 WebSocket message:', data.type, data.status);
    
    switch (data.type) {
      case 'ai_response':
        handleAIResponse(data);
        break;
        
      case 'audio_chunk':
        // Audio chunks are handled automatically by the API
        break;
        
      case 'audio_end':
        handleAudioEnd(data);
        break;
        
      case 'conversation_end':
        handleConversationEnd(data);
        break;
        
      case 'error':
        handleServerError(data);
        break;
        
      case 'clarification':
        handleClarification(data);
        break;
        
      default:
        console.warn('Unknown message type:', data.type);
    }
    
    // Update stage if provided
    if (data.status) {
      setCurrentStage(data.status);
    }
    
    // Update progress
    updateProgress(data.status);
    
  }, []);
  
  const handleAIResponse = (data) => {
    console.log('🤖 AI Response received:', data.text);
    
    setCurrentMessage(data.text);
    setSessionState('ai_speaking');
    
    // Add to conversation history
    addToConversationHistory('ai', data.text);
    
    // Update progress based on stage
    updateProgress(data.status);
  };
  
  const handleAudioEnd = (data) => {
    console.log('🎵 Audio ended, starting recording...');
    
    setTimeout(() => {
      if (sessionState !== 'complete' && sessionState !== 'error') {
        startRecording();
      }
    }, 1000); // 1 second delay after audio ends
  };
  
  const handleConversationEnd = (data) => {
    console.log('🏁 Conversation ended');
    
    setCurrentMessage(data.text);
    setEvaluation(data.evaluation);
    setScore(data.score);
    setSessionState('complete');
    setProgressValue(100);
    
    // Add final message to history
    addToConversationHistory('ai', data.text);
    
    // Stop all timers
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
    
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
  };
  
  const handleServerError = (data) => {
    console.error('❌ Server error:', data.text);
    setError(data.text);
    setSessionState('error');
  };
  
  const handleClarification = (data) => {
    console.log('❓ Clarification requested:', data.text);
    setCurrentMessage(data.text);
    setSessionState('ai_speaking');
    
    // Add to conversation history
    addToConversationHistory('ai', data.text);
    
    // Restart recording after clarification
    setTimeout(() => {
      if (sessionState !== 'complete' && sessionState !== 'error') {
        startRecording();
      }
    }, 2000);
  };
  
  const handleWebSocketError = (error) => {
    console.error('❌ WebSocket error:', error);
    setError('Connection error. Please try again.');
    setSessionState('error');
    setIsConnected(false);
  };
  
  const handleWebSocketClose = (event) => {
    console.log('🔌 WebSocket closed:', event.code, event.reason);
    setIsConnected(false);
    
    if (sessionState !== 'complete' && event.code !== 1000) {
      setError('Connection lost. Please try again.');
      setSessionState('error');
    }
  };
  
  // ==================== RECORDING FUNCTIONS ====================
  
  const startRecording = async () => {
    try {
      console.log('🎤 Starting recording...');
      
      setSessionState('recording');
      setCurrentMessage('Listening... Please speak your response');
      
      await standupCallAPI.startRecording();
      
    } catch (error) {
      console.error('❌ Recording error:', error);
      setError(`Recording failed: ${error.message}`);
      setSessionState('error');
    }
  };
  
  const stopRecording = () => {
    console.log('⏹️ Stopping recording...');
    
    standupCallAPI.stopRecording();
    setSessionState('processing');
    setCurrentMessage('Processing your response...');
  };
  
  // ==================== HELPER FUNCTIONS ====================
  
  const addToConversationHistory = (type, message) => {
    const timestamp = new Date().toLocaleTimeString();
    setConversationHistory(prev => [...prev, {
      type,
      message,
      timestamp,
      id: Date.now()
    }]);
  };
  
  const updateProgress = (status) => {
    const progressMap = {
      'greeting': 20,
      'conversation': 70,
      'complete': 100
    };
    
    const newProgress = progressMap[status] || progressValue;
    setProgressValue(newProgress);
  };
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  const getStatusMessage = () => {
    switch (sessionState) {
      case 'initializing':
        return 'Initializing system...';
      case 'ready':
        return 'Ready to start your standup';
      case 'connecting':
        return 'Connecting to interviewer...';
      case 'ai_speaking':
        return 'Interviewer is speaking...';
      case 'recording':
        return 'Recording your response...';
      case 'processing':
        return 'Processing your response...';
      case 'complete':
        return 'Standup complete!';
      case 'error':
        return 'Error occurred';
      default:
        return 'Loading...';
    }
  };
  
  const getStatusIcon = () => {
    switch (sessionState) {
      case 'ai_speaking':
        return <VolumeUp fontSize="inherit" />;
      case 'recording':
        return <Mic fontSize="inherit" />;
      case 'processing':
        return <Timer fontSize="inherit" />;
      case 'complete':
        return <CheckCircle fontSize="inherit" />;
      case 'error':
        return <Warning fontSize="inherit" />;
      default:
        return <RadioButtonChecked fontSize="inherit" />;
    }
  };
  
  const cleanup = () => {
    console.log('🧹 Cleaning up session...');
    
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
    
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
      recordingTimerRef.current = null;
    }
    
    standupCallAPI.disconnect();
  };
  
  const handleGoBack = () => {
    cleanup();
    navigate('/student/daily-standups');
  };
  
  const handleViewSummary = async () => {
    try {
      if (testId) {
        const summary = await standupCallAPI.getStandupSummary(testId);
        navigate(`/student/daily-standups/summary/${testId}`, {
          state: { summary, evaluation, score }
        });
      }
    } catch (error) {
      console.error('❌ Error getting summary:', error);
    }
  };
  
  const handleRetry = () => {
    setError(null);
    setSessionState('ready');
    setCurrentMessage('');
    setConversationHistory([]);
    setProgressValue(0);
    setSessionDuration(0);
    setRecordingDuration(0);
    setEvaluation(null);
    setScore(null);
    setTestId(null);
    setIsConnected(false);
  };
  
  // ==================== RENDER ====================
  
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
                bgcolor: sessionState === 'complete' ? theme.palette.success.main : theme.palette.primary.main,
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
                {formatTime(sessionDuration)} {testId && `• ${testId.slice(-8)}`}
              </Typography>
            </Box>
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={isConnected ? "Connected" : "Disconnected"}
              color={isConnected ? "success" : "default"}
              icon={isConnected ? <CheckCircle /> : <RadioButtonChecked />}
              size="medium"
            />
          </Box>
        </Box>

        {/* Progress Bar */}
        <Paper sx={{ p: 2, mb: 3, borderRadius: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="text.secondary">
              Session Progress
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {progressValue}%
            </Typography>
          </Box>
          <LinearProgress 
            variant="determinate" 
            value={progressValue} 
            sx={{ height: 8, borderRadius: 4 }}
          />
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3, borderRadius: 2 }}
            action={
              <Button color="inherit" size="small" onClick={handleRetry}>
                <Refresh sx={{ mr: 1 }} />
                Retry
              </Button>
            }
          >
            {error}
          </Alert>
        )}

        {/* Main Interface */}
        <PersonaCard isActive={sessionState !== 'ready' && sessionState !== 'error'} elevation={8}>
          <CardContent sx={{ p: 6 }}>
            <Box textAlign="center">
              
              {/* Main Avatar */}
              <MainAvatar status={sessionState === 'ai_speaking' ? 'ai_speaking' : sessionState}>
                {getStatusIcon()}
              </MainAvatar>
              
              {/* Status Message */}
              <Typography 
                variant="h4" 
                gutterBottom 
                sx={{ 
                  mb: 3,
                  fontWeight: 'bold',
                  background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                {getStatusMessage()}
              </Typography>
              
              {/* Current Message */}
              {currentMessage && (
                <Box sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
                  <Typography 
                    variant="h6" 
                    sx={{ 
                      mb: 2,
                      fontStyle: 'italic',
                      color: theme.palette.text.secondary,
                      lineHeight: 1.6
                    }}
                  >
                    "{currentMessage}"
                  </Typography>
                </Box>
              )}
              
              {/* Visual Indicators */}
              <Box sx={{ mb: 4 }}>
                {sessionState === 'ai_speaking' && (
                  <Box>
                    <VoiceWaveBox>
                      <div className="wave"></div>
                      <div className="wave"></div>
                      <div className="wave"></div>
                      <div className="wave"></div>
                      <div className="wave"></div>
                      <div className="wave"></div>
                      <div className="wave"></div>
                    </VoiceWaveBox>
                    <Typography variant="body2" color="info.main" sx={{ mt: 1 }}>
                      🎧 AI is speaking...
                    </Typography>
                  </Box>
                )}
                
                {sessionState === 'recording' && (
                  <Box>
                    <Typography variant="h5" color="error" sx={{ mb: 2, fontWeight: 'bold' }}>
                      🎤 RECORDING • {formatTime(recordingDuration)}
                    </Typography>
                    <Typography variant="body1" color="text.secondary">
                      Speak clearly and naturally. Recording will stop automatically when you finish.
                    </Typography>
                  </Box>
                )}
                
                {sessionState === 'processing' && (
                  <Box>
                    <CircularProgress sx={{ mb: 2 }} />
                    <Typography variant="body1" color="warning.main">
                      Processing your response...
                    </Typography>
                  </Box>
                )}
              </Box>
              
              {/* Action Buttons */}
              <Box sx={{ mb: 4 }}>
                {sessionState === 'ready' && (
                  <Button
                    variant="contained"
                    size="large"
                    onClick={startStandup}
                    startIcon={<PlayArrow />}
                    sx={{ 
                      px: 4, 
                      py: 2, 
                      fontSize: '1.1rem',
                      borderRadius: 3,
                      background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                      boxShadow: theme.shadows[8]
                    }}
                  >
                    Start Standup
                  </Button>
                )}
                
                {sessionState === 'recording' && (
                  <Button
                    variant="contained"
                    color="error"
                    size="large"
                    onClick={stopRecording}
                    startIcon={<Stop />}
                    sx={{ 
                      px: 4, 
                      py: 2, 
                      fontSize: '1.1rem',
                      borderRadius: 3,
                      boxShadow: theme.shadows[8]
                    }}
                  >
                    Stop Recording
                  </Button>
                )}
                
                {sessionState === 'complete' && (
                  <Button
                    variant="contained"
                    color="success"
                    size="large"
                    onClick={handleViewSummary}
                    startIcon={<Assignment />}
                    sx={{ 
                      px: 4, 
                      py: 2, 
                      fontSize: '1.1rem',
                      borderRadius: 3,
                      boxShadow: theme.shadows[8]
                    }}
                  >
                    View Summary
                  </Button>
                )}
              </Box>
              
              {/* Evaluation Display */}
              {evaluation && score && (
                <Box 
                  sx={{ 
                    mt: 4, 
                    p: 3, 
                    backgroundColor: alpha(theme.palette.success.main, 0.1),
                    borderRadius: 3,
                    border: `1px solid ${alpha(theme.palette.success.main, 0.2)}`
                  }}
                >
                  <Typography variant="h6" color="success.main" gutterBottom sx={{ fontWeight: 'bold' }}>
                    📊 Session Evaluation
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2, whiteSpace: 'pre-wrap', textAlign: 'left' }}>
                    {evaluation}
                  </Typography>
                  <Chip 
                    label={`Score: ${score}/10`}
                    color="success"
                    size="large"
                    sx={{ fontWeight: 'bold' }}
                  />
                </Box>
              )}
              
              {/* Instructions */}
              <Box 
                sx={{ 
                  mt: 4, 
                  p: 3, 
                  backgroundColor: alpha(theme.palette.info.main, 0.05),
                  borderRadius: 2,
                  border: `1px solid ${alpha(theme.palette.info.main, 0.2)}`
                }}
              >
                <Typography variant="h6" color="info.main" gutterBottom>
                  💡 Tips for a Great Standup:
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Speak Clearly:</strong> Use a clear, natural speaking voice
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Be Specific:</strong> Mention specific tasks and challenges
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={4}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Stay Focused:</strong> Keep responses relevant and concise
                    </Typography>
                  </Grid>
                </Grid>
              </Box>
            </Box>
          </CardContent>
        </PersonaCard>
        
        {/* Conversation History */}
        {conversationHistory.length > 0 && (
          <Paper sx={{ mt: 3, p: 3, borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>
              Conversation History
            </Typography>
            <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
              {conversationHistory.map((item) => (
                <Box key={item.id} sx={{ mb: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    {item.timestamp} • {item.type === 'ai' ? 'Interviewer' : 'You'}
                  </Typography>
                  <Typography variant="body1" sx={{ fontStyle: item.type === 'ai' ? 'italic' : 'normal' }}>
                    {item.message}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        )}
      </Box>
    </Fade>
  );
};

export default StandupCallSession;