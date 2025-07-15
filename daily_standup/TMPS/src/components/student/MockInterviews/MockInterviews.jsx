import React, { useState } from 'react';
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
  alpha
} from '@mui/material';
import {
  PlayArrow,
  Psychology,
  Code,
  Chat,
  Assessment,
  Timer,
  Lightbulb
} from '@mui/icons-material';

// Import the API
import { interviewOperationsAPI } from '../../../services/API/studentmockinterview';

// Interview Start Dialog Component
const StartInterviewDialog = ({ open, onClose, onStartInterview, loading }) => {
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
          <Typography variant="body1" sx={{ mb: 3, fontSize: '1.1rem' }}>
            Are you ready to start your mock interview session?
          </Typography>
          
          <Alert severity="info" sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Before you begin, please ensure:
            </Typography>
            <Box component="ul" sx={{ mt: 1, pl: 2, mb: 0 }}>
              <li>You have a stable internet connection</li>
              <li>Your microphone is working properly</li>
              <li>You're in a quiet environment</li>
              <li>You have enough time to complete the interview</li>
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
                  Technical questions based on your expertise level
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Chat color="primary" />
                <Typography variant="body2">
                  Communication and behavioral questions
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Timer color="primary" />
                <Typography variant="body2">
                  Approximately 45-60 minutes duration
                </Typography>
              </Box>
              <Box display="flex" alignItems="center" gap={2}>
                <Assessment color="primary" />
                <Typography variant="body2">
                  Real-time feedback and scoring
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

const StudentMockInterviews = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  
  // State for interview operations
  const [startInterviewDialog, setStartInterviewDialog] = useState(false);
  const [startingInterview, setStartingInterview] = useState(false);
  const [snackbar, setSnackbar] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  // Interview operation handlers
  const handleStartNewInterview = () => {
    setStartInterviewDialog(true);
  };

  // const handleConfirmStartInterview = async () => {
  //   try {
  //     setStartingInterview(true);
      
  //     console.log('Starting new interview...');
  //     const response = await interviewOperationsAPI.startInterview();
      
  //     console.log('Interview started successfully:', response);
      
  //     // Transform the response data
  //     const interviewData = interviewOperationsAPI.transformInterviewOperationData(response);
      
  //     setSnackbar({
  //       open: true,
  //       message: `Interview started successfully! Test ID: ${interviewData.testId}`,
  //       severity: 'success'
  //     });

  //     // Close the dialog
  //     setStartInterviewDialog(false);
      
  //     // Navigate to the interview session page
  //     if (interviewData.testId) {
  //       navigate(`/student/interview-session/${interviewData.testId}`);
  //     }
      
  //   } catch (error) {
  //     console.error('Failed to start interview:', error);
  //     setSnackbar({
  //       open: true,
  //       message: `Failed to start interview: ${error.message}`,
  //       severity: 'error'
  //     });
  //   } finally {
  //     setStartingInterview(false);
  //   }
  // };
// Update your handleConfirmStartInterview function in StudentMockInterviews.jsx:

const handleConfirmStartInterview = async () => {
  try {
    setStartingInterview(true);
    
    console.log('Starting new interview...');
    const response = await interviewOperationsAPI.startInterview();
    
    console.log('Interview started successfully:', response);
    
    // Check if we got a test_id directly from response
    const testId = response.test_id || response.testId;
    
    if (!testId) {
      throw new Error('No test ID received from server');
    }
    
    setSnackbar({
      open: true,
      message: `Interview started successfully! Test ID: ${testId}`,
      severity: 'success'
    });

    // Close the dialog
    setStartInterviewDialog(false);
    
    // Navigate to the interview session page using the CORRECT route pattern
    // Change from: `/student/mock-interviews/${testId}`
    // To: `/student/mock-interviews/view/${testId}`
    console.log('Navigating to:', `/student/mock-interviews/view/${testId}`);
    navigate(`/student/mock-interviews/view/${testId}`);
    
  } catch (error) {
    console.error('Failed to start interview:', error);
    setSnackbar({
      open: true,
      message: `Failed to start interview: ${error.message}`,
      severity: 'error'
    });
  } finally {
    setStartingInterview(false);
  }
};
  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
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
          Practice your interview skills with our AI-powered mock interview system
        </Typography>
      </Box>

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
              Test your skills with our comprehensive mock interview system designed to help you succeed.
            </Typography>
            
            <Button
              variant="contained"
              size="large"
              startIcon={<PlayArrow />}
              onClick={handleStartNewInterview}
              disabled={startingInterview}
              sx={{
                px: 6,
                py: 2,
                fontSize: '1.2rem',
                fontWeight: 600,
                borderRadius: 3,
                background: 'linear-gradient(45deg, #2196F3 30%, #21CBF3 90%)',
                boxShadow: `0 6px 20px ${alpha(theme.palette.primary.main, 0.4)}`,
                '&:hover': {
                  background: 'linear-gradient(45deg, #1976D2 30%, #2196F3 90%)',
                  transform: 'translateY(-2px)',
                  boxShadow: `0 8px 25px ${alpha(theme.palette.primary.main, 0.5)}`
                },
                transition: 'all 0.3s ease-in-out'
              }}
            >
              Start New Interview
            </Button>
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
                Technical Questions
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Practice coding problems, algorithms, and technical concepts
              </Typography>
            </CardContent>
          </Card>
          
          <Card elevation={2} sx={{ flex: 1, maxWidth: 300 }}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Chat color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Behavioral Questions
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Improve your communication and soft skills
              </Typography>
            </CardContent>
          </Card>
          
          <Card elevation={2} sx={{ flex: 1, maxWidth: 300 }}>
            <CardContent sx={{ textAlign: 'center', py: 3 }}>
              <Assessment color="primary" sx={{ fontSize: 48, mb: 2 }} />
              <Typography variant="h6" fontWeight={600} gutterBottom>
                Real-time Feedback
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Get instant scores and detailed performance analysis
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
          <li>Speak clearly and at a moderate pace</li>
          <li>Think out loud to show your problem-solving process</li>
          <li>Ask clarifying questions when needed</li>
          <li>Practice regularly to build confidence</li>
        </Box>
      </Alert>

      {/* Start Interview Dialog */}
      <StartInterviewDialog
        open={startInterviewDialog}
        onClose={() => setStartInterviewDialog(false)}
        onStartInterview={handleConfirmStartInterview}
        loading={startingInterview}
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