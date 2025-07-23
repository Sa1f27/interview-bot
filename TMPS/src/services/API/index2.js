// src/services/API/index2.js
// Assessment API configuration for Daily Standups, Mock Interviews, and Mock Tests

// Assessment API Base URL
const ASSESSMENT_API_BASE_URL = import.meta.env.VITE_ASSESSMENT_API_URL || 'https://192.168.48.201:8060';

// Get authentication token from localStorage
const getAuthToken = () => {
  return localStorage.getItem('token') || sessionStorage.getItem('token');
};

// Common headers for assessment API requests
const getAssessmentHeaders = (isFormData = false) => {
  const headers = {};
  
  if (!isFormData) {
    headers['Content-Type'] = 'application/json';
  }
  
  const token = getAuthToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  return headers;
};

// Generic assessment API request function
export const assessmentApiRequest = async (endpoint, options = {}) => {
  const url = `${ASSESSMENT_API_BASE_URL}${endpoint}`;
  
  const isFormData = options.body instanceof FormData;
  
  const config = {
    headers: getAssessmentHeaders(isFormData),
    ...options,
  };

  try {
    console.log('🔗 Assessment API Request:', {
      url,
      method: config.method || 'GET',
      headers: config.headers,
      hasBody: !!config.body
    });

    const response = await fetch(url, config);
    
    console.log('📡 Assessment API Response:', {
      status: response.status,
      statusText: response.statusText,
      url: response.url,
      ok: response.ok
    });

    // Handle different response types
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.text();
      } catch (e) {
        errorData = response.statusText;
        
      }
      console.error('❌ Assessment API Error Response:', errorData);
      throw new Error(`HTTP ${response.status}: ${errorData || response.statusText}`);
    }
    
    // Check if response has content
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      const jsonData = await response.json();
      console.log('✅ Assessment API JSON Response:', jsonData);
      return jsonData;
    } else {
      const textData = await response.text();
      console.log('✅ Assessment API Text Response:', textData);
      return textData;
    }
    
  } catch (error) {
    console.error('💥 Assessment API request failed:', {
      url,
      error: error.message,
      stack: error.stack
    });
    
    // Provide more helpful error messages
    if (error.message.includes('Failed to fetch')) {
      throw new Error(`Network error: Cannot connect to ${ASSESSMENT_API_BASE_URL}. Please check if the server is running.`);
    } else if (error.message.includes('CORS')) {
      throw new Error(`CORS error: Server needs to allow requests from your frontend domain.`);
    } else {
      throw error;
    }
  }
};

// Export the base URL for use in other modules
export { ASSESSMENT_API_BASE_URL };

// Default export
export default {
  assessmentApiRequest,
  ASSESSMENT_API_BASE_URL,
  getAuthToken,
  getAssessmentHeaders
};