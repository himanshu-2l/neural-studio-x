import React, { useState } from 'react';
import { Shield, Key, Eye, EyeOff, Terminal } from 'lucide-react';

export default function Login({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    // Verified credentials list
    const validUsers = {
      himanshu: 'neural2026',
      guest: 'guest123'
    };

    setTimeout(() => {
      if (validUsers[username] && validUsers[username] === password) {
        onLogin(username);
      } else {
        setError('Incorrect credentials. Please try again.');
        setLoading(false);
      }
    }, 600);
  };

  const handleQuickLogin = () => {
    setLoading(true);
    setTimeout(() => {
      onLogin('guest');
    }, 300);
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-[#09090b] px-4 overflow-hidden">
      {/* Background Neon Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-500/10 rounded-full blur-[120px] animate-pulse-slow"></div>
      <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-violet-600/10 rounded-full blur-[120px] animate-pulse-slow"></div>

      <div className="w-full max-w-md z-10">
        {/* Brand Banner */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-tr from-cyan-500 to-violet-500 shadow-[0_0_20px_rgba(6,182,212,0.3)] mb-4">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-blue-500 to-violet-500 bg-clip-text text-transparent">
            Neural Studio X
          </h1>
          <p className="text-zinc-400 mt-2 text-sm">
            Production-Grade ML Experimentation Suite
          </p>
        </div>

        {/* Card */}
        <div className="bg-[#0c0c0f]/80 backdrop-blur-xl border border-zinc-800/80 rounded-2xl p-8 shadow-2xl">
          <h2 className="text-xl font-bold text-zinc-100 mb-6">
            Sign In
          </h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Username
              </label>
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value.toLowerCase())}
                placeholder="e.g. himanshu"
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg px-4 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 transition-all"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-[#09090b] border border-zinc-800 rounded-lg pl-4 pr-10 py-3 text-sm text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-3.5 text-zinc-500 hover:text-zinc-300 focus:outline-none"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {error && (
              <div className="text-xs text-rose-400 bg-rose-950/20 border border-rose-900/30 rounded-lg p-3">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white rounded-lg py-3 font-semibold text-sm hover:from-cyan-400 hover:to-blue-500 shadow-md hover:shadow-cyan-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Authenticating...' : 'Sign In'}
            </button>
          </form>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-zinc-800/80"></div>
            </div>
            <div className="relative flex justify-center text-xs">
              <span className="bg-[#0c0c0f] px-3 text-zinc-500 uppercase tracking-wider">
                Or
              </span>
            </div>
          </div>

          {/* Quick Access */}
          <button
            onClick={handleQuickLogin}
            disabled={loading}
            className="w-full bg-zinc-900 border border-zinc-800 text-zinc-300 rounded-lg py-3 font-semibold text-sm hover:bg-zinc-800 transition-all disabled:opacity-50 flex items-center justify-center gap-2"
          >
            <Key className="w-4 h-4 text-cyan-400" />
            Quick Demo Login (guest)
          </button>
        </div>

        {/* Credentials hints */}
        <div className="mt-6 bg-[#0c0c0f]/40 border border-zinc-800/30 rounded-xl p-4 flex items-start gap-3">
          <Terminal className="w-4 h-4 text-zinc-500 mt-0.5" />
          <div className="text-xs text-zinc-500 space-y-1">
            <p className="font-semibold text-zinc-400">Demo Accounts Available:</p>
            <p>Admin: <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-400">himanshu</code> / <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-400">neural2026</code></p>
            <p>Guest: <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-400">guest</code> / <code className="bg-zinc-900 px-1 py-0.5 rounded text-zinc-400">guest123</code></p>
          </div>
        </div>
      </div>
    </div>
  );
}
