// EMERGENCY FIX - Replace your entire StartInterview.jsx with this

import React, { useState, useEffect, useRef } from 'react';
import { useParams, useLocation, useNavigate } from 'react-router-dom';
import {
  Container, Typography, Box, Card, CardContent, Button,
  Alert, CircularProgress, IconButton
} from '@mui/material';
import { Mic, MicOff, ArrowBack } from '@mui/icons-material';

const StartInterview = () => {
  const { sessionId } = useParams(); // This should be the UUID now
  const location = useLocation();
  const navigate = useNavigate();
  
  // Get data from navigation state
  const testId = location.state?.testId;
  const studentName = location.state?.studentName || 'Student';
  const emergencyDebug = location.state?.emergencyDebug;
  
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState(null);
  const [isConnecting, setIsConnecting] = useState(true);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  
  const wsRef = useRef(null);

  useEffect(() => {
    // Get testId and studentName from URL params as fallback
    const urlParams = new URLSearchParams(window.location.search);
    const urlTestId = urlParams.get('testId');
    const urlStudentName = urlParams.get('studentName');
    
    const finalTestId = testId || urlTestId;
    const finalStudentName = studentName || urlStudentName || 'Student';
    
    console.log('?? EMERGENCY DEBUG START:');
    console.log('- Route sessionId:', sessionId);
    console.log('- State testId:', testId);
    console.log('- URL testId:', urlTestId);
    console.log('- Final testId:', finalTestId);
    console.log('- Final studentName:', finalStudentName);
    console.log('- Current URL:', window.location.href);
    
    if (!sessionId) {
      console.error('? EMERGENCY: No sessionId in route');
      setConnectionError('No session ID in route. Navigation failed.');
      setIsConnecting(false);
      return;
    }
    
    // Check if this is an old session ID format
    if (sessionId.startsWith('interview_')) {
      console.error('? EMERGENCY: Using old test_id format as session_id!');
      setConnectionError(`Wrong session format: ${sessionId}. This should be a UUID. Please start a new interview.`);
      setIsConnecting(false);
      return;
    }
    
    console.log('?? USING CORRECT UUID SESSION ID:', sessionId);
    initializeWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sessionId]);

  const initializeWebSocket = () => {
    try {
      const wsUrl = `wss://192.168.48.201:8070/weekly_interview/ws/${sessionId}`;
      
      console.log('?? EMERGENCY WEBSOCKET CONNECTION TO:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('?? EMERGENCY: WebSocket CONNECTED!');
        setIsConnected(true);
        setIsConnecting(false);
        setConnectionError(null);
        
        // Send a ping to keep connection alive
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
            console.log('?? Sent ping to keep connection alive');
          }
        }, 1000);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('?? EMERGENCY: WebSocket message:', data);
          
          if (data.type === 'error') {
            console.error('?? EMERGENCY: WebSocket error from server:', data.text);
            setConnectionError(`Server error: ${data.text}`);
            return;
          }
          
          if (data.type === 'pong') {
            console.log('?? Received pong - connection healthy');
            return;
          }
          
          if (data.type === 'ai_response') {
            setCurrentMessage(data.text);
          }
          
          if (data.type === 'audio_chunk') {
            console.log('?? Received audio chunk');
          }
          
          if (data.type === 'audio_end') {
            console.log('?? Audio playback complete');
          }
          
          if (data.type === 'interview_complete') {
            console.log('?? EMERGENCY: Interview completed, navigating to results with testId:', testId);
            navigate(`/student/mock-interviews/results/${testId}`, {
              state: { evaluation: data }
            });
          }
          
        } catch (error) {
          console.error('?? EMERGENCY: Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onerror = (error) => {
        console.error('?? EMERGENCY: WebSocket error:', error);
        setConnectionError('WebSocket connection failed');
      };
      
      ws.onclose = (event) => {
        console.log('?? EMERGENCY: WebSocket closed:', event.code, event.reason);
        setIsConnected(false);
        
        // Only show error for unexpected closures
        if (event.code !== 1000 && event.code !== 1001) {
          setConnectionError(`Connection closed unexpectedly (code: ${event.code})`);
          
          // Try to reconnect after 3 seconds
          setTimeout(() => {
            console.log('?? Attempting to reconnect...');
            initializeWebSocket();
          }, 3000);
        }
      };
      
      wsRef.current = ws;
      
      // Keep connection alive with periodic pings
      const pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
          console.log('?? Sent keepalive ping');
        } else {
          clearInterval(pingInterval);
        }
      }, 30000); // Ping every 30 seconds
      
    } catch (error) {
      console.error('?? EMERGENCY: WebSocket init failed:', error);
      setConnectionError(`Connection failed: ${error.message}`);
      setIsConnecting(false);
    }
  };

  const convertBlobToBase64 = (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          // Remove the data URL prefix (data:audio/webm;base64,)
          const base64 = reader.result.split(',')[1];
          resolve(base64);
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  };

  const toggleRecording = async () => {
    if (isRecording) {
      stopRecording();
    } else {
      await startRealRecording();
    }
  };

  const startRealRecording = async () => {
    try {
      console.log('?? Starting REAL audio recording...');
      
      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      const audioChunks = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
          console.log('?? Audio chunk:', event.data.size, 'bytes');
        }
      };
      
      mediaRecorder.onstop = async () => {
        console.log('?? Recording stopped, processing...');
        
        if (audioChunks.length === 0) {
          console.warn('?? No audio data recorded');
          return;
        }
        
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log('?? Audio blob size:', audioBlob.size, 'bytes');
        
        if (audioBlob.size < 100) {
          console.warn('?? Audio too small:', audioBlob.size, 'bytes');
          return;
        }
        
        try {
          // Convert to base64 using FileReader (handles large files better)
          const base64Audio = await convertBlobToBase64(audioBlob);
          
          console.log('?? Sending audio:', base64Audio.length, 'chars');
          
          // Send to WebSocket with connection check
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const message = {
              type: 'audio_data',
              audio: base64Audio
            };
            
            console.log('?? Sending audio data...');
            wsRef.current.send(JSON.stringify(message));
            console.log('?? Real audio sent to WebSocket successfully');
          } else {
            console.error('? WebSocket not connected, cannot send audio');
            console.log('WebSocket state:', wsRef.current?.readyState);
            
            // Try to reconnect
            if (wsRef.current?.readyState === WebSocket.CLOSED) {
              console.log('?? Attempting to reconnect before sending audio...');
              initializeWebSocket();
            }
          }
        } catch (error) {
          console.error('? Audio conversion failed:', error);
        }
        
        // Cleanup
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      
      // Auto-stop after 10 seconds
      setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
          mediaRecorder.stop();
          setIsRecording(false);
          console.log('?? Auto-stopped after 10 seconds');
        }
      }, 10000);
      
    } catch (error) {
      console.error('? Recording failed:', error);
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    console.log('?? Manual stop recording');
  };

  if (isConnecting) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Box textAlign="center">
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h5" gutterBottom>
            ?? EMERGENCY: Connecting...
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Session ID: {sessionId}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Test ID: {testId}
          </Typography>
        </Box>
      </Container>
    );
  }

  if (connectionError) {
    return (
      <Container maxWidth="md" sx={{ py: 4 }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="h6" gutterBottom>?? EMERGENCY CONNECTION ERROR</Typography>
          <Typography variant="body2">{connectionError}</Typography>
          <Typography variant="body2" sx={{ mt: 1 }}>
            Session ID: {sessionId}
          </Typography>
          <Typography variant="body2">
            Test ID: {testId}
          </Typography>
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
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Typography variant="h4" gutterBottom>
        ?? EMERGENCY: Interview Session Connected!
      </Typography>
      
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="body2">Session ID: {sessionId}</Typography>
          <Typography variant="body2">Test ID: {testId}</Typography>
          <Typography variant="body2">Student: {studentName}</Typography>
          <Typography variant="body2">Connected: {isConnected ? '? YES' : '? NO'}</Typography>
        </CardContent>
      </Card>

      {currentMessage && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>Interviewer:</Typography>
            <Typography variant="body1">{currentMessage}</Typography>
          </CardContent>
        </Card>
      )}

      <Box textAlign="center">
        <IconButton
          onClick={toggleRecording}
          disabled={!isConnected}
          sx={{
            width: 80,
            height: 80,
            bgcolor: isRecording ? 'error.main' : 'primary.main',
            color: 'white',
            mb: 2
          }}
        >
          {isRecording ? <MicOff sx={{ fontSize: 40 }} /> : <Mic sx={{ fontSize: 40 }} />}
        </IconButton>
        
        <Typography variant="body2">
          {isRecording ? '?? RECORDING (speak now, 10s max)' : '?? Click to record your response'}
        </Typography>
      </Box>
    </Container>
  );
};

export default StartInterview;