import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || 'nsx-dev-key-change-in-prod';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Automatically inject API key header for authenticated requests
apiClient.interceptors.request.use((config) => {
  config.headers['x-api-key'] = API_KEY;
  return config;
});

export default apiClient;
