// src/services/api/mockTest.js
import { assessmentApiRequest } from './index2';

export const mockTestAPI = {
  // Start a new mock test - POST /weekend_mocktest/api/test/start
  startTest: async (testConfig = {}) => {
    try {
      console.log('API: Starting mock test with config:', testConfig);
      
      const response = await assessmentApiRequest('/weekend_mocktest/api/test/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(testConfig)
      });
      
      console.log('API Response for start test:', response);
      
      return response;
    } catch (error) {
      console.error('API Error in startTest:', error);
      throw new Error(`Failed to start test: ${error.message}`);
    }
  },

  // Submit a single answer - POST /weekend_mocktest/api/test/submit
  submitAnswer: async (answerData) => {
    try {
      if (!answerData) {
        throw new Error('Answer data is required');
      }
      
      console.log('API: Submitting answer:', answerData);
      
      const response = await assessmentApiRequest('/weekend_mocktest/api/test/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(answerData)
      });
      
      console.log('API Response for submit answer:', response);
      
      return response;
    } catch (error) {
      console.error('API Error in submitAnswer:', error);
      throw new Error(`Failed to submit answer: ${error.message}`);
    }
  },

  // Get test results - GET /weekend_mocktest/api/test/results/{test_id}
  getTestResults: async (testId) => {
    try {
      if (!testId) {
        throw new Error('Test ID is required');
      }
      
      console.log('API: Getting test results for:', testId);
      
      const response = await assessmentApiRequest(`/weekend_mocktest/api/test/results/${testId}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      console.log('API Response for test results:', response);
      
      return response;
    } catch (error) {
      console.error('API Error in getTestResults:', error);
      throw new Error(`Failed to get test results: ${error.message}`);
    }
  },

  // Download PDF results - GET /weekend_mocktest/api/test/pdf/{test_id}
  downloadResultsPDF: async (testId) => {
    try {
      if (!testId) {
        throw new Error('Test ID is required');
      }
      
      console.log('API: Downloading PDF for test:', testId);
      
      // First get the base URL from assessmentApiRequest or use fallback
      let baseUrl = '';
      try {
        // Try to extract base URL from your existing API setup
        // You might need to adjust this based on your index2.js implementation
        baseUrl = process.env.REACT_APP_API_BASE_URL || window.location.origin;
      } catch (e) {
        baseUrl = window.location.origin;
      }
      
      const url = `${baseUrl}/weekend_mocktest/api/test/pdf/${testId}`;
      console.log('PDF download URL:', url);
      
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/pdf',
          // Add any auth headers that assessmentApiRequest normally adds
          // 'Authorization': 'Bearer ' + token, // if needed
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return response.blob();
    } catch (error) {
      console.error('API Error in downloadResultsPDF:', error);
      throw new Error(`Failed to download PDF: ${error.message}`);
    }
  },

  // Transform test start response data to match your backend structure
  transformStartTestData: (apiData) => {
    if (!apiData) return null;
    
    return {
      // Test identifiers
      testId: apiData.test_id,
      sessionId: apiData.session_id,
      
      // Test configuration
      testType: apiData.test_type || 'developer',
      duration: (apiData.time_limit || 300) / 60, // Convert seconds to minutes for display
      totalQuestions: apiData.total_questions || 2,
      userType: apiData.user_type,
      
      // Current question data (your backend sends one question at a time)
      currentQuestion: {
        question: apiData.question_html || '',
        questionNumber: apiData.question_number || 1,
        options: apiData.options || null,
        timeLimit: apiData.time_limit || 300
      },
      
      // Test metadata
      startTime: Date.now(),
      instructions: '',
      
      // Keep original data for reference
      raw: apiData
    };
  },

  // Transform test submission response data
  transformSubmissionData: (apiData) => {
    if (!apiData) return null;
    
    if (apiData.test_completed) {
      return {
        testCompleted: true,
        score: apiData.score,
        totalQuestions: apiData.total_questions,
        analytics: apiData.analytics,
        pdfAvailable: apiData.pdf_available,
        raw: apiData
      };
    } else {
      return {
        testCompleted: false,
        nextQuestion: {
          questionNumber: apiData.next_question.question_number,
          totalQuestions: apiData.next_question.total_questions,
          questionHtml: apiData.next_question.question_html,
          options: apiData.next_question.options,
          timeLimit: apiData.next_question.time_limit
        },
        raw: apiData
      };
    }
  },

  // Helper method to prepare test start request
  prepareStartTestRequest: (options = {}) => {
    const defaultConfig = {
      user_type: 'dev' // This matches your backend exactly
    };
    
    // Ensure user_type is always 'dev' or 'non_dev' (matching your backend)
    let userType = options.user_type || options.userType || defaultConfig.user_type;
    
    // Convert frontend values to API values if needed
    if (userType === 'developer') {
      userType = 'dev';
    } else if (userType === 'non-developer') {
      userType = 'non_dev';
    }
    
    // Validate user_type to match your backend validation
    if (!['dev', 'non_dev'].includes(userType)) {
      console.warn(`Invalid user_type: ${userType}, defaulting to 'dev'`);
      userType = 'dev';
    }
    
    return {
      user_type: userType
    };
  },

  // Start test with prepared configuration
  startTestWithConfig: async (options = {}) => {
    try {
      const config = mockTestAPI.prepareStartTestRequest(options);
      console.log('Prepared config for API:', config);
      const response = await mockTestAPI.startTest(config);
      return mockTestAPI.transformStartTestData(response);
    } catch (error) {
      console.error('Error starting test with config:', error);
      throw error;
    }
  },

  // Submit answer with proper formatting
  submitAnswerWithData: async (testId, questionNumber, answer) => {
    try {
      const answerData = {
        test_id: testId,
        question_number: questionNumber,
        answer: answer
      };
      
      console.log('Sending answer data:', answerData);
      
      const response = await mockTestAPI.submitAnswer(answerData);
      return mockTestAPI.transformSubmissionData(response);
    } catch (error) {
      console.error('Error submitting answer with data:', error);
      throw error;
    }
  },

  // Validate test configuration before starting
  validateTestConfig: (config) => {
    const errors = [];
    
    const userType = config.user_type || config.userType;
    if (!userType) {
      errors.push('User type is required');
    } else if (!['dev', 'non_dev'].includes(userType)) {
      errors.push('User type must be "dev" or "non_dev"');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  },

  // Validate answer submission data
  validateAnswerData: (testId, questionNumber, answer) => {
    const errors = [];
    
    if (!testId) {
      errors.push('Test ID is required');
    }
    
    if (!questionNumber || questionNumber < 1) {
      errors.push('Valid question number is required');
    }
    
    if (answer === undefined || answer === null) {
      errors.push('Answer is required');
    }
    
    return {
      isValid: errors.length === 0,
      errors
    };
  },

  // Health check
  healthCheck: async () => {
    try {
      const response = await assessmentApiRequest('/weekend_mocktest/api/health', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      return response;
    } catch (error) {
      console.error('API Error in healthCheck:', error);
      throw new Error(`Health check failed: ${error.message}`);
    }
  },

  // Debug method to test API connection
  testAPIConnection: async () => {
    try {
      console.log('Testing API connection...');
      const health = await mockTestAPI.healthCheck();
      console.log('API Health Check Response:', health);
      return health;
    } catch (error) {
      console.error('API Connection Test Failed:', error);
      return { error: error.message, status: 'failed' };
    }
  },

  // Get all tests (optional - for debugging/admin purposes)
  getAllTests: async () => {
    try {
      console.log('API: Getting all tests');
      
      const response = await assessmentApiRequest('/weekend_mocktest/api/tests', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      
      console.log('API Response for all tests:', response);
      return response;
    } catch (error) {
      console.error('API Error in getAllTests:', error);
      throw new Error(`Failed to get all tests: ${error.message}`);
    }
  }
};