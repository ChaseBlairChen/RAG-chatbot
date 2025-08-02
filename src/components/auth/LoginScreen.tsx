// components/auth/LoginScreen.tsx
import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { TEST_ACCOUNTS, DEFAULT_BACKEND_URL } from '../../utils/constants';

export const LoginScreen: React.FC = () => {
  const { login } = useAuth();
  const [loginForm, setLoginForm] = useState({ username: '', password: '' });
  const [loginError, setLoginError] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoggingIn(true);
    setLoginError('');

    const result = await login(loginForm.username, loginForm.password);
    
    if (!result.success) {
      setLoginError(result.error || 'Login failed');
    } else {
      setLoginForm({ username: '', password: '' });
    }
    
    setIsLoggingIn(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-stone-50 via-neutral-50 to-stone-100 flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        {/* Logo/Header */}
        <div className="text-center mb-8">
          <div className="w-20 h-20 bg-gradient-to-br from-stone-200 to-stone-300 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg">
            <svg className="w-10 h-10 text-stone-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-stone-900 mb-2">Legally</h1>
          <p className="text-stone-600">— powered by AI</p>
        </div>

        {/* Login Form */}
        <div className="bg-white rounded-2xl shadow-2xl p-8">
          <form onSubmit={handleLogin} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <input
                type="text"
                value={loginForm.username}
                onChange={(e) => setLoginForm(prev => ({ ...prev, username: e.target.value }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your username"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm(prev => ({ ...prev, password: e.target.value }))}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                placeholder="Enter your password"
                required
              />
            </div>

            {loginError && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-red-800 text-sm">{loginError}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={isLoggingIn}
              className="w-full bg-stone-800 text-white py-3 px-4 rounded-lg hover:bg-stone-900 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium"
            >
              {isLoggingIn ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                  Logging in...
                </div>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Test Accounts Info */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Test Accounts:</h3>
            <div className="space-y-2 text-xs">
              {TEST_ACCOUNTS.map((account, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="font-medium">{account.username}</span>
                  <span className="text-gray-600">{account.password}</span>
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    account.subscription_tier === 'premium' ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'
                  }`}>
                    {account.subscription_tier}
                    {account.subscription_tier === 'premium' && <span className="ml-1">🔗</span>}
                  </span>
                </div>
              ))}
              <button
                type="button"
                onClick={() => setLoginForm({ username: 'demo', password: 'demo123' })}
                className="text-stone-700 hover:text-stone-900 text-xs font-medium"
              >
                Quick Login as Demo User (Free)
              </button>
            </div>
            <div className="mt-3 p-2 bg-stone-100 rounded text-xs text-stone-700">
              <strong>🔗 Premium users</strong> have access to external legal databases (LexisNexis, Westlaw)
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-6 text-stone-500 text-sm">
          <p>Legally — powered by AI</p>
          <p className="text-xs mt-1">Secure Legal Document Analysis Platform</p>
          <p className="text-xs mt-1">Connected to: {DEFAULT_BACKEND_URL}</p>
        </div>
      </div>
    </div>
  );
};
