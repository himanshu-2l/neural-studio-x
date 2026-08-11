import React, { useState, useEffect } from 'react';
import DataExplorer from './DataExplorer';
import ModelTrainer from './ModelTrainer';
import InferenceLab from './InferenceLab';
import AutoML from './AutoML';
import Explainability from './Explainability';
import Experiments from './Experiments';
import DriftMonitor from './DriftMonitor';
import DeployAPI from './DeployAPI';
import apiClient from '../apiClient';
import { 
  Shield, LogOut, LayoutDashboard, Settings, Eye, 
  Trophy, BrainCircuit, History, Server, Activity 
} from 'lucide-react';

export default function Dashboard({ username, onLogout }) {
  const [activeTab, setActiveTab] = useState('eda');
  const [stats, setStats] = useState({ total_experiments: 0, total_predictions: 0, best_rmsle: '—' });

  const fetchStats = async () => {
    try {
      const response = await apiClient.get('/health');
      const dbStats = response.data.db_stats || {};
      setStats({
        total_experiments: dbStats.total_experiments || 0,
        total_predictions: dbStats.total_predictions || 0,
        best_rmsle: dbStats.best_rmsle ? dbStats.best_rmsle.toFixed(4) : '—'
      });
    } catch (err) {
      console.error('Failed to load stats', err);
    }
  };

  useEffect(() => {
    fetchStats();
    const interval = setInterval(fetchStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'eda', label: 'Data Explorer', icon: LayoutDashboard, component: DataExplorer },
    { id: 'trainer', label: 'Model Trainer', icon: Settings, component: ModelTrainer },
    { id: 'inference', label: 'Inference Lab', icon: Eye, component: InferenceLab },
    { id: 'automl', label: 'AutoML', icon: Trophy, component: AutoML },
    { id: 'explainability', label: 'Explainability', icon: BrainCircuit, component: Explainability },
    { id: 'experiments', label: 'Experiments', icon: History, component: Experiments },
    { id: 'drift', label: 'Drift Monitor', icon: Shield, component: DriftMonitor },
    { id: 'deploy', label: 'Deploy & API', icon: Server, component: DeployAPI }
  ];

  const ActiveComponent = tabs.find(t => t.id === activeTab)?.component || DataExplorer;

  return (
    <div className="min-h-screen flex bg-[#09090b] text-[#fafafa] font-sans">
      
      {/* ── Sidebar ── */}
      <aside className="w-64 bg-[#0c0c0f] border-r border-zinc-800 flex flex-col justify-between p-5 flex-shrink-0 z-30">
        <div className="space-y-6">
          
          {/* User Profile Card */}
          <div className="flex items-center gap-3 bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-tr from-cyan-500 to-violet-500 flex items-center justify-center font-bold text-white uppercase shadow-md shadow-cyan-500/10">
              {username ? username.substring(0, 2) : 'GU'}
            </div>
            <div>
              <p className="text-xs font-bold text-zinc-100 capitalize">{username}</p>
              <p className="text-[10px] text-zinc-500">ML Architect</p>
            </div>
          </div>

          <div className="h-px bg-zinc-800/80"></div>

          {/* Navigation Links */}
          <nav className="space-y-1">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isSelected = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    isSelected 
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-bold' 
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/40 border border-transparent'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Sidebar Footer Info */}
        <div className="space-y-4 pt-4 border-t border-zinc-800/80">
          <div className="text-[10px] text-zinc-500 uppercase tracking-widest font-semibold mb-2">Engine Status</div>
          <div className="space-y-2 font-mono text-[10px] text-zinc-400">
            <div className="flex justify-between">
              <span>Scikit-Learn</span>
              <span className="text-emerald-400 font-bold">READY</span>
            </div>
            <div className="flex justify-between">
              <span>FastAPI</span>
              <span className="text-cyan-400 font-bold">ONLINE</span>
            </div>
          </div>

          <button
            onClick={onLogout}
            className="w-full flex items-center justify-center gap-2 bg-zinc-900/80 border border-zinc-800 hover:bg-zinc-800 text-zinc-400 hover:text-zinc-200 text-xs font-medium py-2.5 rounded-lg transition-colors"
          >
            <LogOut className="w-3.5 h-3.5" />
            Logout
          </button>
        </div>
      </aside>

      {/* ── Main Content Area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        
        {/* Topbar Brand Strip */}
        <header className="h-16 border-b border-zinc-800 bg-[#0c0c0f]/80 backdrop-blur-md flex items-center justify-between px-8 z-20">
          <div className="flex items-center gap-3">
            <span className="text-xs text-zinc-500 uppercase tracking-widest font-semibold">Workspace</span>
            <span className="text-xs bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5 font-mono text-cyan-400 font-bold">
              🏠 House Prices Regression
            </span>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="flex items-center gap-1.5 text-zinc-400 bg-zinc-900 border border-zinc-800 rounded-full px-3 py-1 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
              Live:8000
            </span>
            <span className="text-zinc-500 font-mono">
              Best RMSLE: <b className="text-emerald-400">{stats.best_rmsle}</b>
            </span>
          </div>
        </header>

        {/* Tab Content mount */}
        <main className="flex-1 overflow-y-auto p-8 max-w-[1400px] w-full mx-auto">
          <ActiveComponent username={username} />
        </main>
      </div>

    </div>
  );
}
