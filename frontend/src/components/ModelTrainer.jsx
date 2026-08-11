import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import apiClient from '../apiClient';
import { Play, Settings, AlertCircle, CheckCircle, BarChart } from 'lucide-react';

export default function ModelTrainer({ username }) {
  const [algo, setAlgo] = useState('GradientBoostingRegressor');
  const [folds, setFolds] = useState(5);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const handleTrain = async () => {
    setLoading(true);
    setError('');
    setResult(null);
    setProgress(15);

    // Simulate progress phases
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 90) {
          clearInterval(interval);
          return 90;
        }
        return p + 15;
      });
    }, 450);

    try {
      const response = await apiClient.post('/train', {
        algorithm: algo,
        cv_folds: folds,
        username: username || 'guest'
      });
      clearInterval(interval);
      setProgress(100);
      setTimeout(() => {
        setResult(response.data);
        setLoading(false);
      }, 200);
    } catch (err) {
      clearInterval(interval);
      setLoading(false);
      setError(err.response?.data?.detail || 'Failed to trigger training run.');
    }
  };

  const chartOption = useMemo(() => {
    if (!result || !result.scores) return null;
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        type: 'category',
        data: result.scores.map((_, i) => `Fold ${i + 1}`),
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
          name: 'RMSLE',
          type: 'bar',
          data: result.scores,
          barWidth: '55%',
          itemStyle: {
            color: '#a78bfa',
            borderRadius: [4, 4, 0, 0]
          }
        }
      ]
    };
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Config card */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center">
              <Settings className="w-4 h-4 text-cyan-400" />
            </div>
            <h3 className="text-base font-bold text-zinc-100">Training Config</h3>
          </div>

          <div className="space-y-5">
            <div>
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Algorithm</label>
              <select
                value={algo}
                onChange={(e) => setAlgo(e.target.value)}
                disabled={loading}
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg p-3 text-sm text-zinc-300 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                <option value="GradientBoostingRegressor">Gradient Boosting Regressor</option>
                <option value="RandomForestRegressor">Random Forest Regressor</option>
                <option value="Ridge">Ridge Regression</option>
              </select>
            </div>

            <div>
              <div className="flex justify-between items-center mb-2">
                <label className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">K-Fold Splits</label>
                <span className="text-xs text-cyan-400 font-mono font-bold">{folds} Folds</span>
              </div>
              <input
                type="range"
                min="3"
                max="10"
                value={folds}
                onChange={(e) => setFolds(parseInt(e.target.value))}
                disabled={loading}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            {error && (
              <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-950/20 border border-rose-900/30 rounded-lg p-3">
                <AlertCircle className="w-4 h-4" />
                <span>{error}</span>
              </div>
            )}

            <button
              onClick={handleTrain}
              disabled={loading}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-lg py-3 font-semibold text-sm shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-50"
            >
              <Play className="w-4 h-4 fill-current" />
              {loading ? 'Executing Training Pipeline...' : 'Start Training Run'}
            </button>
          </div>
        </div>

        {/* Results/Monitor card */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm flex flex-col justify-between">
          <div className="space-y-6">
            <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
              <BarChart className="w-4 h-4 text-violet-400" />
              Performance Output
            </h3>

            {/* Loading / progress */}
            {loading && (
              <div className="space-y-4 py-8">
                <div className="flex justify-between items-center text-xs text-zinc-400">
                  <span>Executing K-Fold CV pipeline...</span>
                  <span>{progress}%</span>
                </div>
                <div className="w-full bg-zinc-800 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-gradient-to-r from-cyan-500 to-violet-500 h-1.5 transition-all duration-300" style={{ width: `${progress}%` }}></div>
                </div>
              </div>
            )}

            {/* Results display */}
            {result && (
              <div className="space-y-6">
                <div className="flex items-start gap-2.5 text-xs text-emerald-400 bg-emerald-950/20 border border-emerald-900/30 rounded-lg p-3.5">
                  <CheckCircle className="w-4 h-4 mt-0.5" />
                  <div>
                    <p className="font-semibold">Pipeline Execution Success</p>
                    <p className="text-zinc-500 mt-0.5">Model serialized as <code className="text-zinc-400">model_{result.algorithm.toLowerCase()}.pkl</code></p>
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-[#09090b] border border-zinc-800 rounded-lg p-4">
                    <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Mean RMSLE</p>
                    <p className="text-xl font-bold text-emerald-400 mt-1">{result.mean_rmsle.toFixed(4)}</p>
                  </div>
                  <div className="bg-[#09090b] border border-zinc-800 rounded-lg p-4">
                    <p className="text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Cross-Val Std</p>
                    <p className="text-xl font-bold text-zinc-200 mt-1">± {result.std_rmsle.toFixed(4)}</p>
                  </div>
                </div>

                {/* Chart */}
                <div className="pt-2">
                  <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-3">Individual Fold Scores</p>
                  <ReactECharts option={chartOption} style={{ height: '180px' }} />
                </div>
              </div>
            )}

            {/* Initial state */}
            {!loading && !result && (
              <div className="flex flex-col items-center justify-center text-center py-16 space-y-3">
                <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-lg">
                  ⚡
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-zinc-300">No Active Performance Data</h4>
                  <p className="text-xs text-zinc-500 max-w-xs mt-1">Configure your parameters and trigger a training run to load CV statistics.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
