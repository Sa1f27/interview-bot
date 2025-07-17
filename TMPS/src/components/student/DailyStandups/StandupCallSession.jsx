import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert,
  Card,
  CardContent,
  IconButton,
  Button,
  Avatar,
  LinearProgress,
  Chip,
  Fade,
  useTheme,
  alpha,
  styled,
  keyframes
} from '@mui/material';
import {
  Mic,
  VolumeUp,
  ArrowBack,
  CheckCircle,
  RadioButtonChecked,
  Timer,
  PlayArrow,
  Warning,
  RecordVoiceOver
} from '@mui/icons-material';
import { useParams, useNavigate } from 'react-router-dom';
import { standupCallAPI } from '../../../services/API/studentstandup';

// ==================== STYLED COMPONENTS ====================

const pulse = keyframes`
  0% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.7);
  }
  70% {
    transform: scale(1.05);
    box-shadow: 0 0 0 20px rgba(76, 175, 80, 0);
  }
  100% {
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(76, 175, 80, 0);
  }
`;

const speaking = keyframes`
  0%, 100% { 
    opacity: 1; 
    transform: scale(1);
  }
  50% { 
    opacity: 0.8; 
    transform: scale(1.1);
  }
`;

const listening = keyframes`
  0%, 100% { 
    transform: scale(1);
    box-shadow: 0 0 0 0 rgba(244, 67, 54, 0.5);
  }
  50% { 
    transform: scale(1.02);
    box-shadow: 0 0 0 15px rgba(244, 67, 54, 0);
  }
`;

const MainAvatar = styled(Avatar)(({ theme, status }) => ({
  width: 200,
  height: 200,
  margin: '0 auto',
  marginBottom: theme.spacing(3),
  fontSize: '5rem',
  boxShadow: theme.shadows[16],
  border: `4px solid ${alpha(theme.palette.background.paper, 0.8)}`,
  transition: 'all 0.3s ease-in-out',
  ...(status === 'listening' && {
    animation: `${listening} 2s infinite`,
    backgroundColor: theme.palette.error.main,
    borderColor: theme.palette.error.light,
  }),
  ...(status === 'speaking' && {
    animation: `${speaking} 1.5s infinite`,
    backgroundColor: theme.palette.info.main,
    borderColor: theme.palette.info.light,
  }),
  ...(status === 'idle' && {
    animation: `${pulse} 3s infinite`,
    backgroundColor: theme.palette.success.main,
    borderColor: theme.palette.success.light,
  }),
  ...(status === 'complete' && {
    backgroundColor: theme.palette.primary.main,
    borderColor: theme.palette.primary.light,
  }),
}));

const StatusCard = styled(Card, {
  shouldForwardProp: (prop) => prop !== 'isActive',
})(({ theme, isActive }) => ({
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
  const { testId } = useParams();
  const navigate = useNavigate();
  const theme = useTheme();
  
  // ==================== STATE MANAGEMENT ====================
  
  const [sessionState, setSessionState] = useState('initializing'); 
  const [error, setError] = useState(null);
  const [currentMessage, setCurrentMessage] = useState('');
  const [sessionDuration, setSessionDuration] = useState(0);
  const [conversationCount, setConversationCount] = useState(0);
  const [testIdState, setTestIdState] = useState(testId);
  const [isConnected, setIsConnected] = useState(false);
  
  // ==================== REFS ====================
  
  const sessionStartTime = useRef(null);
  const durationTimerRef = useRef(null);
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
    if (sessionState === 'connecting' && !durationTimerRef.current) {
      sessionStartTime.current = Date.now();
      durationTimerRef.current = setInterval(() => {
        setSessionDuration(Math.floor((Date.now() - sessionStartTime.current) / 1000));
      }, 1000);
    }
  }, [sessionState]);
  
  // ==================== MAIN FUNCTIONS ====================
  
  const initializeSession = async () => {
    try {
      setSessionState('initializing');
      setError(null);
      setCurrentMessage('Preparing your standup session...');
      
      console.log('🚀 Initializing standup session...');
      
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setSessionState('ready');
      setCurrentMessage('Ready to start your conversation');
      
    } catch (error) {
      console.error('❌ Session initialization error:', error);
      setError(error.message);
      setSessionState('error');
    }
  };
  
  const startConversation = async () => {
    try {
      setSessionState('connecting');
      setError(null);
      setCurrentMessage('Connecting to your interviewer...');
      
      console.log('🚀 Starting conversation...');
      
      // Just start the standup - WebSocket will connect automatically
      const response = await standupCallAPI.startStandup();
      
      if (response && response.session_id) {
        setTestIdState(response.test_id);
        setIsConnected(true);
        setSessionState('idle');
        setCurrentMessage('Connected! AI will speak automatically...');
        setConversationCount(0);
        
        console.log('✅ Conversation started - WebSocket should be connected');
      } else {
        throw new Error('Failed to start conversation');
      }
      
    } catch (error) {
      console.error('❌ Error starting conversation:', error);
      setError(`Backend connection failed: ${error.message}`);
      setSessionState('error');
    }
  };
  
  // ==================== WEBSOCKET HANDLERS ====================
  
  const handleWebSocketMessage = (data) => {
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
        
      default:
        console.log('Unknown message type:', data.type);
    }
  };
  
  const handleAIResponse = (data) => {
    console.log('🤖 AI Response received:', data.text);
    
    setCurrentMessage(data.text);
    setSessionState('speaking');
    setConversationCount(prev => prev + 1);
  };
  
  const handleAudioEnd = (data) => {
    console.log('🎵 Audio ended, conversation will continue automatically...');
    
    // Automatically transition to listening state
    setTimeout(() => {
      if (sessionState !== 'complete' && sessionState !== 'error') {
        setSessionState('idle');
        setCurrentMessage('Your turn - speak naturally, I\'ll detect when you\'re done');
      }
    }, 1000);
  };
  
  const handleConversationEnd = (data) => {
    console.log('🏁 Conversation ended');
    
    setCurrentMessage(data.text || 'Standup completed successfully!');
    setSessionState('complete');
    
    // Auto-navigate to summary after delay
    setTimeout(() => {
      if (testIdState) {
        navigate(`/student/daily-standups/summary/${testIdState}`);
      }
    }, 3000);
  };
  
  const handleServerError = (data) => {
    console.error('❌ Server error:', data.text);
    setError(data.text);
    setSessionState('error');
  };
  
  const handleWebSocketError = (error) => {
    console.error('❌ WebSocket error:', error);
    setError('Connection error. Please check if backend is running.');
    setSessionState('error');
    setIsConnected(false);
  };
  
  const handleWebSocketClose = (event) => {
    console.log('🔌 WebSocket closed:', event.code, event.reason);
    setIsConnected(false);
    
    if (sessionState !== 'complete' && event.code !== 1000) {
      setError('Connection lost. Please refresh and try again.');
      setSessionState('error');
    }
  };
  
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };
  
  const getStatusMessage = () => {
    switch (sessionState) {
      case 'initializing':
        return 'Initializing...';
      case 'ready':
        return 'Ready to start';
      case 'connecting':
        return 'Connecting...';
      case 'idle':
        return 'Listening for your voice';
      case 'listening':
        return 'Recording your response';
      case 'speaking':
        return 'AI is responding';
      case 'processing':
        return 'Processing your input';
      case 'complete':
        return 'Standup completed!';
      case 'error':
        return 'Connection Error';
      default:
        return 'Loading...';
    }
  };
  
  const getStatusIcon = () => {
    switch (sessionState) {
      case 'listening':
        return <Mic fontSize="inherit" />;
      case 'speaking':
        return <VolumeUp fontSize="inherit" />;
      case 'processing':
        return <Timer fontSize="inherit" />;
      case 'complete':
        return <CheckCircle fontSize="inherit" />;
      case 'error':
        return <Warning fontSize="inherit" />;
      case 'idle':
        return <RecordVoiceOver fontSize="inherit" />;
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
    
    standupCallAPI.disconnect();
  };
  
  const handleGoBack = () => {
    cleanup();
    navigate('/student/daily-standups');
  };
  
  const handleRetry = () => {
    setError(null);
    setSessionState('ready');
    setCurrentMessage('');
    setConversationCount(0);
    setSessionDuration(0);
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
                {formatTime(sessionDuration)} {testIdState && `• ${testIdState.slice(-8)}`}
              </Typography>
            </Box>
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            <Chip 
              label={getStatusMessage()}
              color={
                sessionState === 'complete' ? "success" :
                sessionState === 'error' ? "error" :
                sessionState === 'listening' ? "warning" :
                sessionState === 'speaking' ? "info" :
                "primary"
              }
              icon={getStatusIcon()}
              size="medium"
            />
          </Box>
        </Box>

        {/* Error Display */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3, borderRadius: 2 }}
            action={
              <Button color="inherit" size="small" onClick={handleRetry}>
                Try Again
              </Button>
            }
          >
            <strong>Backend Connection Failed:</strong> {error}
            <br />
            <Typography variant="body2" sx={{ mt: 1 }}>
              Check if the backend server is running properly on port 8060
            </Typography>
          </Alert>
        )}

        {/* Main Interface */}
        <StatusCard isActive={sessionState !== 'ready' && sessionState !== 'error'} elevation={8}>
          <CardContent sx={{ p: 6, textAlign: 'center' }}>
            
            {/* Main Avatar */}
            <MainAvatar status={sessionState}>
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
            
            {/* Current Message - REAL AI responses only */}
            {currentMessage && sessionState !== 'ready' && (
              <Box sx={{ mb: 4, maxWidth: 600, mx: 'auto' }}>
                <Typography 
                  variant="h6" 
                  sx={{ 
                    mb: 2,
                    fontStyle: sessionState === 'speaking' ? 'italic' : 'normal',
                    color: sessionState === 'speaking' ? theme.palette.info.main : theme.palette.text.primary,
                    lineHeight: 1.6
                  }}
                >
                  {sessionState === 'speaking' ? `"${currentMessage}"` : currentMessage}
                </Typography>
              </Box>
            )}
            
            {/* Visual Indicators - Automatic states only */}
            <Box sx={{ mb: 4 }}>
              {sessionState === 'speaking' && (
                <Box>
                  <Typography variant="h5" color="info.main" sx={{ mb: 2, fontWeight: 'bold' }}>
                    🎧 AI SPEAKING
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Listen to my response... I'll know when you start talking
                  </Typography>
                </Box>
              )}
              
              {sessionState === 'idle' && isConnected && (
                <Box>
                  <Typography variant="h5" color="success.main" sx={{ mb: 2, fontWeight: 'bold' }}>
                    👂 LISTENING
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Speak naturally - I'll automatically detect when you're done talking
                  </Typography>
                </Box>
              )}
              
              {sessionState === 'connecting' && (
                <Box>
                  <Typography variant="h5" color="primary.main" sx={{ mb: 2, fontWeight: 'bold' }}>
                    🔄 CONNECTING
                  </Typography>
                  <Typography variant="body1" color="text.secondary">
                    Establishing connection to AI interviewer...
                  </Typography>
                </Box>
              )}
            </Box>
            
            {/* Action Buttons - Start only */}
            <Box sx={{ mb: 4 }}>
              {sessionState === 'ready' && (
                <Button
                  variant="contained"
                  size="large"
                  onClick={startConversation}
                  startIcon={<PlayArrow />}
                  sx={{ 
                    px: 4, 
                    py: 2, 
                    fontSize: '1.2rem',
                    borderRadius: 3,
                    background: `linear-gradient(45deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`,
                    boxShadow: theme.shadows[8],
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: theme.shadows[12]
                    }
                  }}
                >
                  Start Natural Conversation
                </Button>
              )}
              
              {sessionState === 'complete' && (
                <Typography variant="h6" color="success.main" sx={{ fontWeight: 'bold' }}>
                  ✅ Redirecting to summary...
                </Typography>
              )}
            </Box>
            
            {/* Instructions - Completely automatic */}
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
                💡 Completely Automatic Conversation:
              </Typography>
              <Box sx={{ textAlign: 'left', maxWidth: 500, mx: 'auto' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  <strong>🤖 AI speaks first:</strong> Real AI-generated responses, not static text
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  <strong>🎙️ Voice detection:</strong> Automatically starts recording when you speak
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  <strong>🤫 Silence detection:</strong> Stops recording when you pause (2 seconds)
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  <strong>⚡ Zero buttons:</strong> No manual controls - just speak naturally!
                </Typography>
              </Box>
            </Box>
          </CardContent>
        </StatusCard>
      </Box>
    </Fade>
  );
};

export default StandupCallSession;