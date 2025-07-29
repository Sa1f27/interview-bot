import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Paper,
  Typography,
  Button,
  Container,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Snackbar,
  CircularProgress,
  Card,
  CardContent,
  Stack,
  useTheme,
  alpha,
  Chip,
  Collapse,
  IconButton,
  Grid
} from '@mui/material';
import {
  PlayArrow,
  Psychology,
  Code,
  Chat,
  Assessment,
  Timer,
  Lightbulb,
  Warning,
  Error as ErrorIcon,
  Refresh,
  ExpandMore,
  ExpandLess,
  CheckCircle,
  Cancel,
  History,
  TrendingUp
} from '@mui/icons-material';

import { interviewOperationsAPI } from '../../../services/API/studentmockinterview';
import { weeklyInterviewsAPI } from '../../../services/API/mockinterviews';

// Enhanced System Status Component
const SystemStatusCard = ({ onRetry }) => {
  const [status, setStatus] = useState('checking');
  const [healthData, setHealthData] = useState(null);
  const [showDetails, setShowDetails] = useState(false);
  const theme = useTheme();

  const checkSystemHealth = async () => {
    try {
      setStatus('checking');
      
      const response = await fetch('/weekly_interview/health');
      const data = await response.json();
      
      setHealthData(data);
      
      if (response.ok && data.status === 'healthy') {
        setStatus('healthy');
      } else {
        setStatus('unhealthy');
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setStatus('error');
      setHealthData({ error: error.message });
    }
  };

  useEffect(() => {
    checkSystemHealth();
    
    // Check health every 30 seconds
    const interval = setInterval(checkSystemHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case 'healthy': return 'success';
      case 'unhealthy': return 'warning';
      case 'error': return 'error';
      default: return 'info';
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'healthy': return <CheckCircle />;
      case 'unhealthy': return <Warning />;
      case 'error': return <ErrorIcon />;
      default: return <CircularProgress size={20} />;
    }
  };

  const getStatusMessage = () => {
    switch (status) {
      case 'healthy': return 'System Ready - All services operational';
      case 'unhealthy': return 'System Degraded - Limited functionality';
      case 'error': return 'System Unavailable - Please try again later';
      default: return 'Checking System Status...';
    }
  };

  return (
    <Card sx={{ mb: 3, border: `2px solid ${theme.palette[getStatusColor()].main}` }}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box display="flex" alignItems="center" gap={2}>
            {getStatusIcon()}
            <Typography variant="h6" color={getStatusColor()}>
              {getStatusMessage()}
            </Typography>
            {healthData?.timestamp && (
              <Typography variant="caption" color="text.secondary">
                Last checked: {new Date(healthData.timestamp * 1000).toLocaleTimeString()}
              </Typography>
            )}
          </Box>
          
          <Box display="flex" alignItems="center" gap={1}>
            <IconButton 
              onClick={checkSystemHealth}
              disabled={status === 'checking'}
              size="small"
            >
              <Refresh />
            </IconButton>
            <IconButton 
              onClick={() => setShowDetails(!showDetails)}
              size="small"
            >
              {showDetails ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
        </Box>

        <Collapse in={showDetails}>
          <Box mt={2}>
            {healthData?.database_status && (
              <Stack spacing={1}>
                <Typography variant="subtitle2" color="text.secondary">
                  Database Status:
                </Typography>
                {Object.entries(healthData.database_status).map(([db, isHealthy]) => (
                  <Box key={db} display="flex" alignItems="center" gap={2}>
                    <Chip 
                      label={db.toUpperCase()}
                      size="small"
                      color={isHealthy ? 'success' : 'error'}
                      variant="outlined"
                    />
                    <Typography variant="caption">
                      {isHealthy ? 'Connected' : 'Connection Failed'}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            )}
            
            {healthData?.features && (
              <Box mt={2}>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Available Features:
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {Object.entries(healthData.features).map(([feature, enabled]) => (
                    <Chip 
                      key={feature}
                      label={feature.replace(/_/g, ' ')}
                      size="small"
                      color={enabled ? 'primary' : 'default'}
                      variant={enabled ? 'filled' : 'outlined'}
                    />
                  ))}
                </Stack>
              </Box>
            )}
            
            {healthData?.error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                <Typography variant="body2">
                  <strong>Error Details:</strong> {healthData.error}
                </Typography>
              </Alert>
            )}
            
            {status !== 'healthy' && (
              <Box mt={2}>
                <Button 
                  variant="outlined" 
                  onClick={onRetry}
                  startIcon={<Refresh />}
                  size="small"
                >
                  Retry Interview Start
                </Button>
              </Box>
            )}
          </Box>
        </Collapse>
      </CardContent>
    </Card>
  );
};

// Interview Statistics Component
const InterviewStatsCard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const statsData = await weeklyInterviewsAPI.getStats();
      setStats(statsData);
    } catch (error) {
      console.error('Failed to load stats:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Card sx={{ mb: 3 }}>
        <CardContent sx={{ textAlign: 'center', py: 3 }}>
          <CircularProgress size={40} />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Loading interview statistics...
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (!stats || stats.total === 0) {
    return null; // Don't show stats if no interviews yet
  }

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom color="primary">
          Your Interview Progress
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="primary">
                {stats.total}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Total Interviews
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="success.main">
                {stats.averageOverallScore || 'N/A'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Average Score
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="info.main">
                {stats.averageTechnicalScore || 'N/A'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Technical Avg
              </Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h4" color="warning.main">
                {stats.averageCommunicationScore || 'N/A'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Communication Avg
              </Typography>
            </Box>
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );
};

// Enhanced Interview Start Dialog Component
const StartInterviewDialog = ({ open, onClose, onStartInterview, loading, error, systemStatus }) => {
  const theme = useTheme();
  
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={2}>
          <PlayArrow color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h5">Start New Mock Interview</Typography>
        </Box>
      </DialogTitle>
      <DialogContent>
        <Box sx={{ py: 2 }}>
          {/* System Status Check */}
          {systemStatus !== 'healthy' && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                System Status: {systemStatus}
              </Typography>
              <Typography variant="body2">
                The interview system may not be fully operational. You can still try to start an interview, 
                but you may experience issues.
              </Typography>
            </Alert>
          )}
          
          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Failed to Start Interview
              </Typography>
              <Typography variant="body2">
                {error}
              </Typography>
            </Alert>
          )}
          
          <Typography variant="body1" sx={{ mb: 3, fontSize: '1.1rem' }}>
            Are you ready to start your mock interview session?
          </Typography>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Before you begin, please ensure:
            </Typography>
            <Box component="ul" sx={{ mt: 1, pl: 2, mb: 0 }}>
              <li>You have a stable internet connection</li>
              <li>Your microphone is working properly and permissions are granted</li>
              <li>You're in a quiet environment</li>
              <li>You have enough time to complete the interview (45-60 minutes)</li>
              <li>You're using a supported browser (Chrome, Firefox, Safari)</li>
              <li>Your speakers/headphones are working for audio feedback</li>
            </Box>
          </Alert>

          <Box sx={{ bgcolor: 'grey.50', p: 3, borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom color="primary">
              What to expect:
            </Typography>
            <Stack spacing={2}>
              <Box display="flex" alignItems="center" gap={2}>
                <Psychology color="primary" />
                <Typography variant="body2">
                  <strong>Technical Round:</strong> Questions based on your recent project work and 7-day summaries
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Chat color="primary" />
                <Typography variant="body2">
                  <strong>Communication Round:</strong> Presentation and explanation skills assessment
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Assessment color="primary" />
                <Typography variant="body2">
                  <strong>HR Round:</strong> Behavioral questions and cultural fit evaluation
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Timer color="primary" />
                <Typography variant="body2">
                  <strong>Duration:</strong> Approximately 45-60 minutes with real-time audio interaction
                </Typography>
              </Box>
            </Stack>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ p: 3 }}>
        <Button 
          onClick={onClose} 
          disabled={loading}
          size="large"
        >
          Cancel
        </Button>
        <Button 
          onClick={onStartInterview} 
          variant="contained" 
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
          size="large"
          sx={{
            px: 4,
            background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
            '&:hover': {
              background: 'linear-gradient(45deg, #1976D2 30%, #2196F3 90%)'
            }
          }}
        >
          {loading ? 'Starting Interview...' : 'Start Interview'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// Enhanced Error Display Component
const ErrorDisplay = ({ error, onRetry, onDismiss }) => {
  const [showDetails, setShowDetails] = useState(false);
  
  const getErrorType = (errorMessage) => {
    const msg = errorMessage.toLowerCase();
    if (msg.includes('network') || msg.includes('fetch') || msg.includes('connection')) {
      return { type: 'network', icon: <Warning color="warning" /> };
    } else if (msg.includes('timeout')) {
      return { type: 'timeout', icon: <Timer color="error" /> };
    } else if (msg.includes('500') || msg.includes('internal server')) {
      return { type: 'server', icon: <ErrorIcon color="error" /> };
    } else {
      return { type: 'unknown', icon: <ErrorIcon color="error" /> };
    }
  };

  const { type, icon } = getErrorType(error);

  const getErrorSolution = (errorType) => {
    switch (errorType) {
      case 'network':
        return [
          'Check your internet connection',
          'Try refreshing the page',
          'Disable VPN if you\'re using one',
          'Try a different browser'
        ];
      case 'timeout':
        return [
          'The server is taking too long to respond',
          'Wait a few moments and try again',
          'Check if the server is running',
          'Contact support if the issue persists'
        ];
      case 'server':
        return [
          'The interview system is currently unavailable',
          'This is likely a temporary issue',
          'Wait a few minutes and try again',
          'Contact support if the problem continues'
        ];
      default:
        return [
          'An unexpected error occurred',
          'Try refreshing the page',
          'Clear your browser cache',
          'Contact support with the error details'
        ];
    }
  };

  return (
    <Alert 
      severity="error" 
      sx={{ mb: 3 }}
      action={
        <Box>
          <IconButton 
            onClick={() => setShowDetails(!showDetails)}
            size="small"
            color="inherit"
          >
            {showDetails ? <ExpandLess /> : <ExpandMore />}
          </IconButton>
          {onDismiss && (
            <IconButton onClick={onDismiss} size="small" color="inherit">
              <Cancel />
            </IconButton>
          )}
        </Box>
      }
    >
      <Box display="flex" alignItems="center" gap={2}>
        {icon}
        <Typography variant="subtitle1">
          Unable to start interview
        </Typography>
      </Box>
      
      <Collapse in={showDetails}>
        <Box mt={2}>
          <Typography variant="body2" gutterBottom>
            <strong>Error:</strong> {error}
          </Typography>
          
          <Typography variant="subtitle2" sx={{ mt: 2, mb: 1 }}>
            Troubleshooting steps:
          </Typography>
          <Box component="ul" sx={{ pl: 2, mb: 2 }}>
            {getErrorSolution(type).map((step, index) => (
              <li key={index}>
                <Typography variant="body2">{step}</Typography>
              </li>
            ))}
          </Box>
          
          {onRetry && (
            <Button 
              variant="outlined" 
              onClick={onRetry}
              startIcon={<Refresh />}
              size="small"
            >
              Try Again
            </Button>
          )}
        </Box>
      </Collapse>
    </Alert>
  );
};

const StudentMockInterviews = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  
  // State for interview operations
  const [startInterviewDialog, setStartInterviewDialog] = useState(false);
  const [startingInterview, setStartingInterview] = useState(false);
  const [systemStatus, setSystemStatus] = useState('checking');
  const [error, setError] = useState(null);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  // Enhanced error handling state
  const [retryCount, setRetryCount] = useState(0);
  const [lastAttempt, setLastAttempt] = useState(null);

  useEffect(() => {
    // Check system status on component mount
    checkSystemStatus();
  }, []);

  const checkSystemStatus = async () => {
    try {
      const response = await fetch('/weekly_interview/health');
      const data = await response.json();
      
      if (response.ok && data.status === 'healthy') {
        setSystemStatus('healthy');
      } else {
        setSystemStatus('degraded');
      }
    } catch (error) {
      console.error('System status check failed:', error);
      setSystemStatus('unavailable');
    }
  };

  const handleStartNewInterview = () => {
    setError(null);
    setStartInterviewDialog(true);
  };

  // FIXED: Navigation to correct route with sessionId parameter
  const handleConfirmStartInterview = async () => {
    try {
      setStartingInterview(true);
      setError(null);
      setLastAttempt(new Date().toISOString());
      
      console.log('?? Starting new interview...');
      
      // Add timeout to the interview start request
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Request timeout - server took too long to respond')), 30000)
      );
      
      const interviewPromise = interviewOperationsAPI.startInterview();
      
      const response = await Promise.race([interviewPromise, timeoutPromise]);
      
      console.log('? Interview started successfully:', response);
      
      // Extract session_id from response
      const sessionId = response.session_id;
      
      if (!sessionId) {
        throw new Error('No session ID received from server. Please try again.');
      }
      
      setSnackbar({
        open: true,
        message: `Interview started successfully! Session ID: ${sessionId}`,
        severity: 'success'
      });

      // Close the dialog
      setStartInterviewDialog(false);
      setRetryCount(0); // Reset retry count on success
      
      // FIXED: Navigate to the correct route with sessionId parameter
      navigate(`/student/mock-interviews/session/${sessionId}`, {
        state: { 
          sessionData: response,
          isNewSession: true,
          sessionId: sessionId,
          testId: response.test_id
        }
      });
      
    } catch (error) {
      console.error('? Failed to start interview:', error);
      
      // Enhanced error processing
      let errorMessage = 'An unexpected error occurred';
      
      if (error.message.includes('HTTP 500')) {
        errorMessage = 'Server error: The interview system is currently unavailable. Please try again in a few minutes.';
      } else if (error.message.includes('HTTP 503')) {
        errorMessage = 'Service unavailable: The interview system is temporarily down for maintenance.';
      } else if (error.message.includes('timeout')) {
        errorMessage = 'Request timeout: The server is taking too long to respond. Please check your connection and try again.';
      } else if (error.message.includes('network') || error.message.includes('fetch')) {
        errorMessage = 'Network error: Please check your internet connection and try again.';
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      setRetryCount(prev => prev + 1);
      
      setSnackbar({
        open: true,
        message: `Failed to start interview: ${errorMessage}`,
        severity: 'error'
      });
    } finally {
      setStartingInterview(false);
    }
  };

  const handleRetry = () => {
    setError(null);
    checkSystemStatus();
    
    // If dialog is open, try starting interview again
    if (startInterviewDialog) {
      handleConfirmStartInterview();
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  const handleDismissError = () => {
    setError(null);
  };

  const handleViewPreviousResults = () => {
    // Navigate to a results history page (you can implement this later)
    navigate('/student/mock-interviews/history');
  };

  return (
    <Container maxWidth="lg" sx={{ py: 6 }}>
      {/* Header */}
      <Box textAlign="center" mb={6}>
        <Typography 
          variant="h3" 
          component="h1" 
          fontWeight={700}
          color="primary"
          gutterBottom
        >
          Mock Interview Center
        </Typography>
        <Typography 
          variant="h6" 
          color="text.secondary" 
          sx={{ maxWidth: 600, mx: 'auto' }}
        >
          Practice your interview skills with our AI-powered mock interview system featuring real-time audio interaction and comprehensive evaluation
        </Typography>
      </Box>

      {/* System Status */}
      <SystemStatusCard onRetry={handleRetry} />

      {/* Interview Statistics */}
      <InterviewStatsCard />

      {/* Error Display */}
      {error && (
        <ErrorDisplay 
          error={error}
          onRetry={handleRetry}
          onDismiss={handleDismissError}
        />
      )}

      {/* Retry Information */}
      {retryCount > 0 && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <Typography variant="body2">
            Attempt #{retryCount + 1}
            {lastAttempt && (
              <span> • Last attempt: {new Date(lastAttempt).toLocaleTimeString()}</span>
            )}
          </Typography>
        </Alert>
      )}

      {/* Main Content */}
      <Box display="flex" justifyContent="center" mb={6}>
        <Card 
          elevation={8}
          sx={{ 
            maxWidth: 500, 
            width: '100%',
            background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
            border: `3px solid ${theme.palette.primary.main}`,
            borderRadius: 4
          }}
        >
          <CardContent sx={{ p: 6, textAlign: 'center' }}>
            <Box 
              sx={{ 
                width: 120, 
                height: 120, 
                borderRadius: '50%', 
                background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                mx: 'auto',
                mb: 4,
                boxShadow: `0 8px 24px ${alpha(theme.palette.primary.main, 0.3)}`
              }}
            >
              <Psychology sx={{ fontSize: 60, color: 'white' }} />
            </Box>
            
            <Typography variant="h4" fontWeight={700} color="primary" gutterBottom>
              Ready to Practice?
            </Typography>
            
            <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
              Test your skills with our comprehensive mock interview system featuring real-time audio interaction and AI-powered evaluation.
            </Typography>
            
            <Stack spacing={2}>
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                onClick={handleStartNewInterview}
                disabled={startingInterview || systemStatus === 'unavailable'}
                sx={{
                  px: 6,
                  py: 2,
                  fontSize: '1.2rem',
                  fontWeight: 600,
                  borderRadius: 3,
                  background: systemStatus === 'healthy' 
                    ? 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)'
                    : 'linear-gradient(45deg, #757575 30%, #9E9E9E 90%)',
                  boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                  '&:hover': {
                    background: systemStatus === 'healthy'
                      ? 'linear-gradient(45deg, #1976D2 30%, #2196F3 90%)'
                      : 'linear-gradient(45deg, #616161 30%, #757575 90%)',
                    transform: systemStatus === 'healthy' ? 'translateY(-2px)' : 'none',
                    boxShadow: systemStatus === 'healthy' 
                      ? `0 8px 25px ${alpha(theme.palette.primary.main, 0.5)}`
                      : 'none'
                  },
                  transition: 'all 0.3s ease-in-out'
                }}
              >
                {systemStatus === 'unavailable' ? 'System Unavailable' : 'Start New Interview'}
              </Button>
            </Stack>

            {systemStatus !== 'healthy' && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
                {systemStatus === 'degraded' && 'System is running with limited functionality'}
                {systemStatus === 'unavailable' && 'Please try again later or contact support'}
              </Typography>
            )}
          </CardContent>
        </Card>
      </Box>

      {/* Features Section */}
      <Box>
        <Typography 
          variant="h5" 
          fontWeight={600} 
          textAlign="center" 
          color="text.primary"
          mb={4}
        >
          Interview Features
        </Typography>
        
        <Stack 
          direction={{ xs: 'column', md: 'row' }} 
          spacing={3}
          justifyContent="center"
        >
          <Card elevation={2} sx={{ flex: 1, maxWidth: 300 }}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Code color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Technical Assessment
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Questions based on your recent project work and 7-day summaries with intelligent fragment processing
              </Typography>
            </CardContent>
          </Card>
          
          <Card elevation={2} sx={{ flex: 1, maxWidth: 300 }}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Chat color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Real-time Audio Interaction
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Natural conversation flow with ultra-fast TTS streaming and automatic turn management
              </Typography>
            </CardContent>
          </Card>
          
          <Card elevation={2} sx={{ flex: 1, maxWidth: 300 }}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Assessment color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Comprehensive Evaluation
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Detailed scoring across technical, communication, and behavioral dimensions with PDF reports
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      </Box>

      {/* Tips Section */}
      <Alert 
        severity="info" 
        icon={<Lightbulb />}
        sx={{ 
          mt: 6, 
          borderRadius: 3,
          '& .MuiAlert-message': {
            width: '100%'
          }
        }}
      >
        <Typography variant="h6" gutterBottom>
          Pro Tips for Success:
        </Typography>
        <Box component="ul" sx={{ mt: 1, pl: 2, mb: 0 }}>
          <li>Speak clearly and at a moderate pace for better transcription accuracy</li>
          <li>Think out loud to show your problem-solving process</li>
          <li>Listen carefully to the AI interviewer's questions</li>
          <li>Practice regularly to build confidence</li>
          <li>Ensure stable internet and quiet environment</li>
          <li>Grant microphone permissions when prompted</li>
          <li>Use headphones to prevent audio feedback</li>
        </Box>
      </Alert>

      {/* Start Interview Dialog */}
      <StartInterviewDialog
        open={startInterviewDialog}
        onClose={() => {
          setStartInterviewDialog(false);
          setError(null);
        }}
        onStartInterview={handleConfirmStartInterview}
        loading={startingInterview}
        error={error}
        systemStatus={systemStatus}
      />

      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseSnackbar} 
          severity={snackbar.severity} 
          sx={{ width: '100%' }}
          variant="filled"
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default StudentMockInterviews;