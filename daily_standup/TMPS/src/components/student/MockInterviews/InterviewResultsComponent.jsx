// Create this file: src/components/student/MockInterviews/InterviewResults.jsx

import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Container,
  Card,
  CardContent,
  CardHeader,
  Grid,
  Chip,
  Button,
  Alert,
  CircularProgress,
  Divider,
  LinearProgress,
  IconButton
} from '@mui/material';
import {
  ArrowBack,
  Download,
  Assessment,
  CheckCircle,
  Warning,
  Error as ErrorIcon
} from '@mui/icons-material';
import { useParams, useNavigate, useLocation } from 'react-router-dom';

const InterviewResults = () => {
  const { testId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    console.log('🔍 InterviewResults - testId:', testId);
    console.log('🔍 InterviewResults - location.state:', location.state);
    
    // Get evaluation data from navigation state or localStorage
    const evaluationData = location.state?.evaluation || 
                          JSON.parse(localStorage.getItem(`interview_results_${testId}`) || 'null');
    
    console.log('📊 Evaluation data:', evaluationData);
    
    if (evaluationData) {
      setResults(evaluationData);
      setLoading(false);
      
      // Store in localStorage for refresh protection
      localStorage.setItem(`interview_results_${testId}`, JSON.stringify(evaluationData));
      console.log('✅ Results loaded successfully');
    } else {
      console.warn('⚠️ No evaluation data found');
      setError('No results data found. Please try taking the interview again.');
      setLoading(false);
    }
  }, [testId, location.state]);

  const getScoreColor = (score, maxScore = 10) => {
    const percentage = (score / maxScore) * 100;
    if (percentage >= 80) return 'success';
    if (percentage >= 60) return 'warning';
    return 'error';
  };

  const getScoreIcon = (score, maxScore = 10) => {
    const percentage = (score / maxScore) * 100;
    if (percentage >= 80) return <CheckCircle color="success" />;
    if (percentage >= 60) return <Warning color="warning" />;
    return <ErrorIcon color="error" />;
  };

  const handleDownloadPDF = () => {
    if (results?.pdf_url) {
      console.log('📥 Downloading PDF:', results.pdf_url);
      
      // Create a temporary link to download the PDF
      const link = document.createElement('a');
      link.href = results.pdf_url;
      link.download = `interview_results_${testId}.pdf`;
      link.target = '_blank'; // Open in new tab as fallback
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } else {
      console.warn('⚠️ No PDF URL available');
      alert('PDF download is not available for this interview.');
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ ml: 2 }}>
            Loading your interview results...
          </Typography>
        </Box>
      </Container>
    );
  }

  if (error) {
    return (
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
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
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={4}>
        <Box display="flex" alignItems="center">
          <IconButton onClick={() => navigate('/student/mock-interviews')} sx={{ mr: 1 }}>
            <ArrowBack />
          </IconButton>
          <Assessment sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h4" fontWeight="bold" color="primary.main">
            Interview Results
          </Typography>
        </Box>
        
        {results?.pdf_url && (
          <Button
            variant="contained"
            startIcon={<Download />}
            onClick={handleDownloadPDF}
            sx={{ borderRadius: 2 }}
          >
            Download PDF Report
          </Button>
        )}
      </Box>

      {/* Overall Summary */}
      <Card sx={{ mb: 3, borderRadius: 2 }}>
        <CardHeader
          title="Interview Summary"
          subheader={`Test ID: ${testId}`}
          sx={{ bgcolor: 'primary.light', color: 'white' }}
        />
        <CardContent sx={{ p: 3 }}>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
            {results?.evaluation || 'No evaluation summary available.'}
          </Typography>
        </CardContent>
      </Card>

      {/* Scores Section */}
      {results?.scores && typeof results.scores === 'object' && Object.keys(results.scores).length > 0 && (
        <Card sx={{ mb: 3, borderRadius: 2 }}>
          <CardHeader title="Round-wise Scores" />
          <CardContent>
            <Grid container spacing={3}>
              {Object.entries(results.scores).map(([round, score]) => {
                // Handle different score formats
                let numericScore = 0;
                if (typeof score === 'number') {
                  numericScore = score;
                } else if (typeof score === 'string') {
                  numericScore = parseInt(score) || 0;
                } else if (typeof score === 'object' && score?.score) {
                  numericScore = parseInt(score.score) || 0;
                }
                
                const maxScore = 10;
                const percentage = (numericScore / maxScore) * 100;
                
                return (
                  <Grid item xs={12} md={4} key={round}>
                    <Box
                      sx={{
                        p: 2,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 2,
                        textAlign: 'center'
                      }}
                    >
                      <Box display="flex" alignItems="center" justifyContent="center" mb={1}>
                        {getScoreIcon(numericScore)}
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          {round.replace(/_/g, ' ').toUpperCase()}
                        </Typography>
                      </Box>
                      
                      <Typography variant="h4" color={getScoreColor(numericScore)} gutterBottom>
                        {numericScore}/{maxScore}
                      </Typography>
                      
                      <LinearProgress
                        variant="determinate"
                        value={percentage}
                        color={getScoreColor(numericScore)}
                        sx={{
                          height: 8,
                          borderRadius: 4,
                          mb: 1
                        }}
                      />
                      
                      <Typography variant="body2" color="text.secondary">
                        {percentage.toFixed(0)}%
                      </Typography>
                    </Box>
                  </Grid>
                );
              })}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Analytics Section */}
      {results?.analytics && typeof results.analytics === 'object' && Object.keys(results.analytics).length > 0 && (
        <Card sx={{ mb: 3, borderRadius: 2 }}>
          <CardHeader title="Performance Analytics" />
          <CardContent>
            <Grid container spacing={2}>
              {Object.entries(results.analytics).map(([key, value]) => {
                // Safely handle different value types
                let displayValue = '';
                if (typeof value === 'number') {
                  displayValue = value.toFixed(2);
                } else if (typeof value === 'string') {
                  displayValue = value;
                } else if (typeof value === 'boolean') {
                  displayValue = value ? 'Yes' : 'No';
                } else if (value === null || value === undefined) {
                  displayValue = 'N/A';
                } else {
                  displayValue = String(value);
                }
                
                return (
                  <Grid item xs={12} sm={6} md={4} key={key}>
                    <Box sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                        {key.replace(/_/g, ' ').toUpperCase()}
                      </Typography>
                      <Typography variant="h6">
                        {displayValue}
                      </Typography>
                    </Box>
                  </Grid>
                );
              })}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Debug Information (remove in production) */}
    

      {/* Actions */}
      <Box display="flex" gap={2} justifyContent="center" mt={4}>
        <Button
          variant="outlined"
          onClick={() => navigate('/student/mock-interviews')}
          sx={{ borderRadius: 2 }}
        >
          Take Another Interview
        </Button>
        
        <Button
          variant="contained"
          onClick={() => navigate('/student/dashboard')}
          sx={{ borderRadius: 2 }}
        >
          Back to Dashboard
        </Button>
      </Box>
    </Container>
  );
};

export default InterviewResults;