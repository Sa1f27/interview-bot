// src/components/student/MockTest/DeveloperTest.jsx
import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Container,
  LinearProgress,
  Paper,
  Chip,
  useTheme,
  alpha,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Code as CodeIcon,
  Timer as TimerIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  NavigateNext as NavigateNextIcon,
  Send as SendIcon
} from '@mui/icons-material';
import { mockTestAPI } from '../../../services/API/studentmocktest';

const DeveloperTest = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const theme = useTheme();
  
  // Get test data from navigation state (from start test response)
  const testData = location.state?.testData;
  const apiResponse = testData?.raw || testData; // The actual API response
  const testId = apiResponse?.test_id || testData?.testId || location.state?.testId;
  const sessionId = apiResponse?.session_id || testData?.sessionId || location.state?.sessionId;
  const userType = apiResponse?.user_type || testData?.userType || location.state?.userType || 'dev';
  const totalQuestions = apiResponse?.total_questions || testData?.totalQuestions || location.state?.totalQuestions || 2;
  const timeLimit = apiResponse?.time_limit || testData?.timeLimit || location.state?.timeLimit || 300; // seconds per question

  // Current question data - extract from the correct location
  const [currentQuestion, setCurrentQuestion] = useState({
    question: apiResponse?.question_html || '',
    questionNumber: apiResponse?.question_number || 1,
    options: apiResponse?.options || null,
    rawQuestion: apiResponse?.question_html || ''
  });
  
  const [answer, setAnswer] = useState('');
  const [timeLeft, setTimeLeft] = useState(timeLimit);
  const [showSubmitDialog, setShowSubmitDialog] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [testCompleted, setTestCompleted] = useState(false);
  const [results, setResults] = useState(null);

  const progress = (currentQuestion.questionNumber / totalQuestions) * 100;

  // Redirect if no test data
  useEffect(() => {
    console.log('Component loaded with data:', {
      testData,
      apiResponse: testData?.raw,
      testId,
      currentQuestion
    });
    
    if (!testData && !testId) {
      console.warn('No test data found, redirecting to test start');
      navigate('/student/mock-tests/start');
    }
  }, [testData, testId, navigate]);

  // Timer effect
  useEffect(() => {
    if (testCompleted || loading) return;

    const timer = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) {
          handleSubmitAnswer(true); // Auto-submit when time runs out
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [testCompleted, loading, currentQuestion.questionNumber]);

  // Reset timer when question changes
  useEffect(() => {
    setTimeLeft(timeLimit);
  }, [currentQuestion.questionNumber, timeLimit]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleAnswerChange = (event) => {
    setAnswer(event.target.value);
  };

  const handleSubmitAnswer = async (isAutoSubmit = false) => {
    if (!testId) {
      setError('Test ID is missing. Cannot submit answer.');
      return;
    }

    if (!isAutoSubmit && !answer.trim()) {
      setError('Please provide an answer before submitting.');
      return;
    }

    setSubmitting(true);
    setError('');

    try {
      console.log('Submitting answer:', {
        testId,
        questionNumber: currentQuestion.questionNumber,
        answer: answer.trim()
      });

      const response = await mockTestAPI.submitAnswerWithData(
        testId,
        currentQuestion.questionNumber,
        answer.trim() || 'No answer provided'
      );

      console.log('Submit response:', response);

      if (response.testCompleted) {
        // Test is finished
        setTestCompleted(true);
        setResults(response);
        
        // Navigate to results page
        navigate('/student/mock-tests/results', { 
          state: { 
            results: {
              ...response,
              testId: testId,
              userType: userType,
              totalQuestions: totalQuestions
            }, 
            testType: 'developer',
            testData
          } 
        });
      } else {
        // Move to next question
        const nextQuestion = response.nextQuestion;
        setCurrentQuestion({
          question: nextQuestion.questionHtml,
          questionNumber: nextQuestion.questionNumber,
          options: nextQuestion.options,
          rawQuestion: nextQuestion.questionHtml
        });
        setAnswer(''); // Clear answer for next question
      }
    } catch (error) {
      console.error('Failed to submit answer:', error);
      setError(`Failed to submit answer: ${error.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return theme.palette.success.main;
      case 'Medium': return theme.palette.warning.main;
      case 'Hard': return theme.palette.error.main;
      default: return theme.palette.info.main;
    }
  };

  const getQuestionPlaceholder = () => {
    if (userType === 'dev') {
      return '// Write your solution here\nfunction solution() {\n  // Your code here\n}\n\n// Explanation:\n// Time Complexity: \n// Space Complexity: \n// Approach: ';
    } else {
      return 'Enter your answer here...';
    }
  };

  // Loading state
  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography variant="h6" sx={{ mt: 2 }}>
          Processing your answer...
        </Typography>
      </Container>
    );
  }

  // Error state if no question data
  if (!currentQuestion.question && !loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <Alert severity="error" sx={{ mb: 4 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>No Question Data Available</Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            The test question couldn't be loaded. This might be due to:
          </Typography>
          <ul style={{ textAlign: 'left', marginBottom: '16px' }}>
            <li>Test wasn't started properly</li>
            <li>Session data is missing or corrupted</li>
            <li>API response structure has changed</li>
          </ul>
          <Typography variant="body2" component="pre" sx={{ 
            fontSize: '0.8rem', 
            backgroundColor: 'rgba(0,0,0,0.1)', 
            p: 1, 
            borderRadius: 1,
            textAlign: 'left'
          }}>
            {JSON.stringify({ 
              testId, 
              hasTestData: !!testData, 
              apiResponse: testData?.raw,
              currentQuestion,
              rawQuestionHtml: testData?.raw?.question_html
            }, null, 2)}
          </Typography>
        </Alert>
        
        <Button 
          variant="contained" 
          onClick={() => navigate('/student/mock-tests/start')}
        >
          Start New Test
        </Button>
      </Container>
    );
  }

  // Test completed state
  if (testCompleted && results) {
    return (
      <Container maxWidth="lg" sx={{ py: 4, textAlign: 'center' }}>
        <Typography variant="h4" sx={{ mb: 4, color: theme.palette.success.main }}>
          🎉 Test Completed!
        </Typography>
        <Typography variant="h6" sx={{ mb: 2 }}>
          Your Score: {results.score} / {results.total_questions}
        </Typography>
        <Typography variant="body1" sx={{ mb: 4 }}>
          Redirecting to detailed results...
        </Typography>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 4 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      {/* Header with Progress */}
      <Paper elevation={2} sx={{ p: 3, mb: 4, borderRadius: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CodeIcon sx={{ fontSize: 32, color: theme.palette.primary.main }} />
            <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
              Developer Assessment
            </Typography>
            {testId && (
              <Chip 
                label={`Test ID: ${testId.slice(0, 8)}...`} 
                size="small" 
                variant="outlined" 
                color="primary"
              />
            )}
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <Chip
              icon={<TimerIcon />}
              label={formatTime(timeLeft)}
              color={timeLeft < 60 ? 'error' : 'primary'}
              sx={{ fontSize: '1rem', fontWeight: 'bold' }}
            />
            <Typography variant="body1" color="text.secondary">
              Question {currentQuestion.questionNumber} of {totalQuestions}
            </Typography>
          </Box>
        </Box>
        
        <LinearProgress 
          variant="determinate" 
          value={progress} 
          sx={{ 
            height: 8, 
            borderRadius: 4,
            backgroundColor: alpha(theme.palette.primary.main, 0.1),
            '& .MuiLinearProgress-bar': {
              borderRadius: 4,
              background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary.main})`
            }
          }} 
        />
      </Paper>

      {/* Question Card */}
      <Card elevation={3} sx={{ mb: 4, borderRadius: 3 }}>
        <CardContent sx={{ p: 4 }}>
          {/* Question Header */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
            <Box>
              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Chip 
                  label={userType === 'dev' ? 'Programming' : 'Multiple Choice'} 
                  variant="outlined"
                  sx={{ fontWeight: 'bold' }}
                />
                <Chip 
                  label="Medium"
                  sx={{ 
                    backgroundColor: getDifficultyColor('Medium'),
                    color: 'white',
                    fontWeight: 'bold'
                  }}
                />
                <Chip 
                  label="10 points"
                  color="primary"
                  variant="outlined"
                />
              </Box>
            </Box>
            {answer.trim() && (
              <CheckCircleIcon sx={{ color: theme.palette.success.main, fontSize: 32 }} />
            )}
          </Box>

          {/* Question */}
          <Box 
            sx={{ 
              mb: 3, 
              lineHeight: 1.6,
              color: theme.palette.text.primary,
              fontWeight: 'medium',
              '& h2': { fontSize: '1.25rem', fontWeight: 'bold', mb: 2 },
              '& p': { mb: 1 },
              '& pre': { 
                backgroundColor: alpha(theme.palette.primary.main, 0.05),
                p: 2,
                borderRadius: 1,
                overflow: 'auto'
              },
              '& code': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                padding: '2px 6px',
                borderRadius: 1,
                fontFamily: 'Monaco, Consolas, "Courier New", monospace'
              }
            }}
            dangerouslySetInnerHTML={{ __html: currentQuestion.question }}
          />

          {/* Options for non-dev users */}
          {userType === 'non_dev' && currentQuestion.options && (
            <Box sx={{ mb: 3 }}>
              <Typography variant="h6" sx={{ mb: 2 }}>Select your answer:</Typography>
              {currentQuestion.options.map((option, index) => (
                <Box key={index} sx={{ mb: 1 }}>
                  <Button
                    variant={answer === index.toString() ? 'contained' : 'outlined'}
                    onClick={() => setAnswer(index.toString())}
                    sx={{ 
                      width: '100%', 
                      textAlign: 'left', 
                      justifyContent: 'flex-start',
                      py: 1.5,
                      px: 2
                    }}
                  >
                    {String.fromCharCode(65 + index)}. {option}
                  </Button>
                </Box>
              ))}
            </Box>
          )}

          {/* Answer Field for dev users */}
          {userType === 'dev' && (
            <TextField
              multiline
              rows={12}
              fullWidth
              placeholder={getQuestionPlaceholder()}
              value={answer}
              onChange={handleAnswerChange}
              sx={{
                '& .MuiOutlinedInput-root': {
                  fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                  fontSize: '0.9rem',
                  lineHeight: 1.5,
                  backgroundColor: alpha(theme.palette.primary.main, 0.02),
                  '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.04),
                  }
                }
              }}
            />
          )}
        </CardContent>
      </Card>

      {/* Submit Button */}
      <Box sx={{ display: 'flex', justifyContent: 'center' }}>
        {currentQuestion.questionNumber === totalQuestions ? (
          <Button
            variant="contained"
            size="large"
            startIcon={submitting ? <CircularProgress size={16} color="inherit" /> : <SendIcon />}
            onClick={() => setShowSubmitDialog(true)}
            disabled={submitting || (!answer.trim() && userType === 'dev')}
            sx={{ 
              py: 1.5, 
              px: 4,
              background: `linear-gradient(45deg, ${theme.palette.success.main}, ${theme.palette.success.dark})`,
              '&:hover': {
                background: `linear-gradient(45deg, ${theme.palette.success.dark}, ${theme.palette.success.main})`,
              }
            }}
          >
            {submitting ? 'Submitting...' : 'Finish Test'}
          </Button>
        ) : (
          <Button
            variant="contained"
            size="large"
            endIcon={submitting ? <CircularProgress size={16} color="inherit" /> : <NavigateNextIcon />}
            onClick={() => handleSubmitAnswer()}
            disabled={submitting || (!answer.trim() && userType === 'dev')}
            sx={{ py: 1.5, px: 4 }}
          >
            {submitting ? 'Submitting...' : 'Next Question'}
          </Button>
        )}
      </Box>

      {/* Submit Confirmation Dialog */}
      <Dialog open={showSubmitDialog} onClose={() => setShowSubmitDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WarningIcon color="warning" />
          Finish Test
        </DialogTitle>
        <DialogContent>
          <Typography variant="body1" sx={{ mb: 2 }}>
            Are you sure you want to finish the test? This is the last question.
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Time remaining: {formatTime(timeLeft)}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSubmitDialog(false)} disabled={submitting}>
            Continue
          </Button>
          <Button 
            onClick={() => {
              setShowSubmitDialog(false);
              handleSubmitAnswer();
            }} 
            variant="contained" 
            color="success"
            startIcon={submitting ? <CircularProgress size={16} color="inherit" /> : <SendIcon />}
            disabled={submitting}
          >
            {submitting ? 'Submitting...' : 'Finish Test'}
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default DeveloperTest;