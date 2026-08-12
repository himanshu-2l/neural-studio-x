import React, { useState, useEffect, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import apiClient from '../apiClient';
import { History, TrendingDown, ShieldAlert, Rocket, Download } from 'lucide-react';

export default function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const fetchExperiments = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await apiClient.get('/experiments');
      const exps = response.data.experiments || [];
      // Sort by created_at descending
      exps.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
      setExperiments(exps);
    } catch (err) {
      setError('Failed to load experiments registry from SQLite.');
    } finally {
      setLoading(false);
    }
  };

  const handlePromote = async (runId) => {
    try {
      await apiClient.post(`/experiments/${runId}/promote`);
      fetchExperiments();
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to promote model to production.');
    }
  };

  useEffect(() => {
    fetchExperiments();
  }, []);

  const stats = useMemo(() => {
    if (experiments.length === 0) return { total: 0, best: '—', bestAlgo: '—' };
    const bestExp = [...experiments].sort((a, b) => a.mean_rmsle - b.mean_rmsle)[0];
    return {
      total: experiments.length,
      best: bestExp.mean_rmsle.toFixed(4),
      bestAlgo: bestExp.algorithm.replace('Regressor', '')
    };
  }, [experiments]);

  const trendOption = useMemo(() => {
    if (experiments.length === 0) return null;
    const chronological = [...experiments].sort((a, b) => new Date(a.created_at) - new Date(b.created_at));
    
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis' },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        type: 'category',
        data: chronological.map(e => e.run_id),
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' }
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [
        {
          name: 'Mean RMSLE',
          type: 'line',
          data: chronological.map(e => e.mean_rmsle),
          smooth: true,
          lineStyle: { color: '#06b6d4' },
          itemStyle: { color: '#06b6d4' },
          symbolSize: 6
        }
      ]
    };
  }, [experiments]);

  return (
    <div className="space-y-6">
      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
          <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Run History</p>
          <h3 className="text-2xl font-bold text-zinc-100 mt-1">{stats.total} runs</h3>
        </div>

        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
          <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Best RMSLE Champion</p>
          <h3 className="text-2xl font-bold text-emerald-400 mt-1">{stats.best}</h3>
        </div>

        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
          <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Champion Engine</p>
          <h3 className="text-2xl font-bold text-cyan-400 mt-1">{stats.bestAlgo}</h3>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Run table */}
        <div className="lg:col-span-2 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <History className="w-4 h-4 text-cyan-400" />
            Runs Database Registry
          </h3>

          {error && (
            <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-950/20 border border-rose-900/30 rounded-lg p-3">
              <ShieldAlert className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}

          {loading && experiments.length === 0 ? (
            <div className="py-20 flex justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-cyan-400"></div>
            </div>
          ) : experiments.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-800 text-[10px] text-zinc-500 uppercase tracking-wider font-semibold font-sans">
                    <th className="py-2.5 px-2">Deployment Tag</th>
                    <th className="py-2.5 px-2">Run ID</th>
                    <th className="py-2.5 px-2">Algorithm</th>
                    <th className="py-2.5 px-2 font-mono">Folds</th>
                    <th className="py-2.5 px-2 text-right">RMSLE</th>
                    <th className="py-2.5 px-2 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((row) => (
                    <tr key={row.run_id} className="border-b border-zinc-800/40 text-xs text-zinc-300">
                      <td className="py-3 px-2">
                        <div className="flex flex-wrap gap-1.5">
                          {row.is_champion === 1 && (
                            <span className="bg-yellow-950/30 border border-yellow-800/40 text-yellow-400 text-[10px] px-1.5 py-0.5 rounded font-bold">
                              🏆 CHAMPION
                            </span>
                          )}
                          {row.is_production === 1 && (
                            <span className="bg-emerald-950/30 border border-emerald-800/40 text-emerald-400 text-[10px] px-1.5 py-0.5 rounded font-bold">
                              🚀 PRODUCTION
                            </span>
                          )}
                          {row.is_champion === 0 && row.is_production === 0 && (
                            <span className="text-zinc-600 text-[10px]">Registry</span>
                          )}
                        </div>
                      </td>
                      <td className="py-3 px-2 font-mono text-zinc-500">{row.run_id}</td>
                      <td className="py-3 px-2 font-semibold">{row.algorithm.replace('Regressor', '')}</td>
                      <td className="py-3 px-2 font-mono">{row.cv_folds}</td>
                      <td className="py-3 px-2 text-right text-emerald-400 font-mono font-bold">{row.mean_rmsle.toFixed(4)}</td>
                      <td className="py-3 px-2 text-right">
                        <div className="flex justify-end gap-2">
                          {row.is_production === 0 && (
                            <button
                              onClick={() => handlePromote(row.run_id)}
                              title="Promote to Production"
                              className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 hover:bg-emerald-500/20 px-2 py-1 rounded text-[10px] font-bold flex items-center gap-1 transition-colors cursor-pointer"
                            >
                              <Rocket className="w-3 h-3" />
                              Promote
                            </button>
                          )}
                          <a
                            href={`http://localhost:8000/models/${row.algorithm}/download`}
                            title="Download PKL File"
                            className="bg-zinc-800/80 border border-zinc-700 text-zinc-300 hover:bg-zinc-700 px-2 py-1 rounded text-[10px] font-bold flex items-center gap-1 transition-colors"
                          >
                            <Download className="w-3 h-3" />
                            Download
                          </a>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-center py-20 text-xs text-zinc-500">
              No experiments found in SQLite db. Train a model to register.
            </div>
          )}
        </div>

        {/* Trend Line Chart */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4 flex flex-col justify-between">
          <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <TrendingDown className="w-4 h-4 text-violet-400" />
            Accuracy Performance Progress
          </h3>

          {experiments.length > 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <ReactECharts option={trendOption} style={{ height: '240px', width: '100%' }} />
            </div>
          ) : (
            <div className="text-center py-20 text-xs text-zinc-500">
              No trend data available.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
