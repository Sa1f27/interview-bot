// EMERGENCY FIX - Replace your entire MockInterviews.jsx with this

import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Container, Typography, Button, Box, Card, CardContent, 
  Alert, CircularProgress, Grid, Chip 
} from '@mui/material';
import { PlayArrow, Mic, Assessment } from '@mui/icons-material';

const StudentMockInterviews = () => {
  const navigate = useNavigate();
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState(null);
  const [systemReady, setSystemReady] = useState(false);

  useEffect(() => {
    checkSystemReady();
  }, []);

  const checkSystemReady = async () => {
    try {
      console.log('?? Checking if interview system is ready...');
      
      const response = await fetch('https://192.168.48.201:8070/weekly_interview/health', {
        method: 'GET',
        headers: { 'Accept': 'application/json' }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'healthy') {
          setSystemReady(true);
          console.log('? Interview system ready!');
        } else {
          throw new Error(`System status: ${data.status}`);
        }
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
      
    } catch (error) {
      console.error('? System check failed:', error);
      setError(`System not ready: ${error.message}`);
      setSystemReady(false);
    }
  };

  const startInterview = async () => {
    try {
      setIsStarting(true);
      setError(null);
      
      console.log('?? EMERGENCY FIX: Starting FRESH AI interview...');
      
      // FORCE clear any old session data
      localStorage.removeItem('currentSessionId');
      sessionStorage.clear();
      
      // Get fresh session data
      const response = await fetch('https://192.168.48.201:8070/weekly_interview/start_interview', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const sessionData = await response.json();
      
      console.log('?? FRESH SESSION DATA:');
      console.log('Full response:', sessionData);
      console.log('NEW session_id:', sessionData.session_id);
      console.log('NEW test_id:', sessionData.test_id);
      
      const freshSessionId = sessionData.session_id;
      const freshTestId = sessionData.test_id;
      
      if (!freshSessionId) {
        throw new Error('Backend did not return session_id');
      }
      
      console.log('?? FORCING NAVIGATION TO FRESH SESSION:', freshSessionId);
      
      // FORCE navigate to the NEW session
      window.location.href = `/student/mock-interviews/session/${freshSessionId}?testId=${freshTestId}&studentName=${encodeURIComponent(sessionData.student_name || 'Student')}`;
      
    } catch (error) {
      console.error('? EMERGENCY: Failed to start interview:', error);
      setError(`Failed to start interview: ${error.message}`);
    } finally {
      setIsStarting(false);
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box textAlign="center" mb={4}>
        <Mic sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
        <Typography variant="h3" component="h1" gutterBottom fontWeight="bold">
          AI Mock Interview
        </Typography>
        <Typography variant="h6" color="text.secondary">
          Practice your interview skills with AI-powered conversations
        </Typography>
      </Box>

      {/* System Status */}
      <Box mb={4}>
        {systemReady ? (
          <Alert severity="success" sx={{ borderRadius: 2 }}>
            <strong>System Ready!</strong> Interview AI is online and ready to chat.
          </Alert>
        ) : error ? (
          <Alert severity="error" sx={{ borderRadius: 2 }}>
            <strong>System Issue:</strong> {error}
            <Button 
              size="small" 
              onClick={checkSystemReady} 
              sx={{ ml: 2 }}
            >
              Retry
            </Button>
          </Alert>
        ) : (
          <Alert severity="info" sx={{ borderRadius: 2 }}>
            <CircularProgress size={20} sx={{ mr: 1 }} />
            Checking interview system...
          </Alert>
        )}
      </Box>

      {/* Main Interview Card */}
      <Card sx={{ borderRadius: 3, boxShadow: 3 }}>
        <CardContent sx={{ p: 4, textAlign: 'center' }}>
          <Assessment sx={{ fontSize: 50, color: 'primary.main', mb: 2 }} />
          
          <Typography variant="h4" gutterBottom>
            Ready for Your Interview?
          </Typography>
          
          <Typography variant="body1" color="text.secondary" paragraph>
            This AI interviewer will conduct a realistic mock interview with you. 
            It includes technical questions, communication assessment, and behavioral evaluation.
          </Typography>

          <Box sx={{ my: 3 }}>
            <Grid container spacing={2} justifyContent="center">
              <Grid item>
                <Chip 
                  label="?? Voice Interaction" 
                  variant="outlined" 
                  color="primary" 
                />
              </Grid>
              <Grid item>
                <Chip 
                  label="?? AI-Powered Questions" 
                  variant="outlined" 
                  color="primary" 
                />
              </Grid>
              <Grid item>
                <Chip 
                  label="?? Instant Feedback" 
                  variant="outlined" 
                  color="primary" 
                />
              </Grid>
            </Grid>
          </Box>

          <Button
            variant="contained"
            size="large"
            startIcon={isStarting ? <CircularProgress size={20} color="inherit" /> : <PlayArrow />}
            onClick={startInterview}
            disabled={!systemReady || isStarting}
            sx={{
              borderRadius: 3,
              py: 1.5,
              px: 4,
              fontSize: '1.1rem',
              fontWeight: 'bold'
            }}
          >
            {isStarting ? 'Starting Interview...' : 'Start AI Interview'}
          </Button>

          {systemReady && (
            <Typography variant="caption" display="block" sx={{ mt: 2, color: 'text.secondary' }}>
              The interview will take approximately 30-45 minutes
            </Typography>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default StudentMockInterviews;