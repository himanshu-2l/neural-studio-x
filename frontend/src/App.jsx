import React, { useState } from 'react';
import Login from './components/Login';
import Dashboard from './components/Dashboard';

function App() {
  const [user, setUser] = useState(() => {
    return localStorage.getItem('nsx_auth_user') || '';
  });

  const handleLogin = (username) => {
    localStorage.setItem('nsx_auth_user', username);
    setUser(username);
  };

  const handleLogout = () => {
    localStorage.removeItem('nsx_auth_user');
    setUser('');
  };

  if (!user) {
    return <Login onLogin={handleLogin} />;
  }

  return <Dashboard username={user} onLogout={handleLogout} />;
}

export default App;
