import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import apiClient from '../apiClient';
import { Trophy, HelpCircle, Activity } from 'lucide-react';

export default function AutoML({ username }) {
  const [folds, setFolds] = useState(5);
  const [selectedAlgos, setSelectedAlgos] = useState(['GradientBoostingRegressor', 'RandomForestRegressor', 'Ridge']);
  const [loading, setLoading] = useState(false);
  const [currentRunningAlgo, setCurrentRunningAlgo] = useState('');
  const [results, setResults] = useState(null);

  const handleRunTournament = async () => {
    setLoading(true);
    setResults(null);
    const runs = [];

    try {
      for (const algo of selectedAlgos) {
        setCurrentRunningAlgo(algo);
        const response = await apiClient.post('/train', {
          algorithm: algo,
          cv_folds: folds,
          username: username || 'guest'
        });
        runs.push({
          algorithm: algo,
          mean_rmsle: response.data.mean_rmsle,
          std_rmsle: response.data.std_rmsle,
          best_fold: Math.min(...(response.data.scores || [response.data.mean_rmsle]))
        });
      }

      // Sort by Mean RMSLE ascending (lower is better!)
      runs.sort((a, b) => a.mean_rmsle - b.mean_rmsle);
      setResults(runs);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
      setCurrentRunningAlgo('');
    }
  };

  const barChartOption = useMemo(() => {
    if (!results) return null;
    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        type: 'category',
        data: results.map(r => r.algorithm.replace('Regressor', '')),
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' }
      },
      yAxis: {
        type: 'value',
        name: 'Mean RMSLE (Lower is Better)',
        nameTextStyle: { color: '#a1a1aa', fontSize: 10 },
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [
        {
          name: 'Mean RMSLE',
          type: 'bar',
          data: results.map(r => r.mean_rmsle),
          barWidth: '40%',
          itemStyle: {
            color: (params) => {
              const colors = ['#10b981', '#f59e0b', '#ef4444'];
              return colors[params.dataIndex % colors.length];
            },
            borderRadius: [4, 4, 0, 0]
          }
        }
      ]
    };
  }, [results]);

  const radarChartOption = useMemo(() => {
    return {
      backgroundColor: 'transparent',
      tooltip: {},
      legend: {
        data: ['GradBoost', 'RandomForest', 'Ridge'],
        textStyle: { color: '#a1a1aa' },
        bottom: 0
      },
      radar: {
        indicator: [
          { name: 'Accuracy', max: 100 },
          { name: 'Training Speed', max: 100 },
          { name: 'Scalability', max: 100 },
          { name: 'Explainability', max: 100 },
          { name: 'Robustness', max: 100 }
        ],
        axisName: { color: '#a1a1aa' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.06)' } },
        splitArea: { show: false },
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } }
      },
      series: [
        {
          type: 'radar',
          data: [
            {
              value: [95, 70, 85, 90, 92],
              name: 'GradBoost',
              itemStyle: { color: '#06b6d4' },
              areaStyle: { color: 'rgba(6,182,212,0.1)' }
            },
            {
              value: [92, 80, 80, 85, 88],
              name: 'RandomForest',
              itemStyle: { color: '#10b981' },
              areaStyle: { color: 'rgba(16,185,129,0.1)' }
            },
            {
              value: [80, 95, 75, 95, 78],
              name: 'Ridge',
              itemStyle: { color: '#a78bfa' },
              areaStyle: { color: 'rgba(167,139,250,0.1)' }
            }
          ]
        }
      ]
    };
  }, []);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Column */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center">
              <Trophy className="w-4 h-4 text-cyan-400" />
            </div>
            <h3 className="text-base font-bold text-zinc-100">AutoML Tournament</h3>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-2">
                <span>Validation Folds</span>
                <span className="text-cyan-400 font-mono">{folds} Folds</span>
              </div>
              <input
                type="range" min="3" max="7" value={folds}
                onChange={(e) => setFolds(parseInt(e.target.value))}
                disabled={loading}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Compare Models</label>
              <div className="space-y-2">
                {['GradientBoostingRegressor', 'RandomForestRegressor', 'Ridge'].map((algo) => (
                  <label key={algo} className="flex items-center gap-2.5 text-xs text-zinc-300 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedAlgos.includes(algo)}
                      disabled={loading}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedAlgos([...selectedAlgos, algo]);
                        } else {
                          setSelectedAlgos(selectedAlgos.filter(a => a !== algo));
                        }
                      }}
                      className="rounded bg-[#09090b] border-zinc-800 text-cyan-500 accent-cyan-500 w-3.5 h-3.5"
                    />
                    <span>{algo.replace('Regressor', '')}</span>
                  </label>
                ))}
              </div>
            </div>

            <button
              onClick={handleRunTournament}
              disabled={loading || selectedAlgos.length === 0}
              className="w-full bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white rounded-lg py-3 font-semibold text-sm shadow-md transition-all disabled:opacity-50"
            >
              {loading ? 'Running Tour...' : 'Run Tournament'}
            </button>
          </div>

          <div className="border-t border-zinc-800/80 pt-4">
            <ReactECharts option={radarChartOption} style={{ height: '220px' }} />
          </div>
        </div>

        {/* Results leaderboard */}
        <div className="lg:col-span-2 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm flex flex-col justify-between space-y-6">
          <div className="space-y-4">
            <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              Tournament Leaderboard
            </h3>

            {loading && (
              <div className="flex flex-col items-center justify-center text-center py-20 space-y-4">
                <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-cyan-400"></div>
                <div className="text-xs text-zinc-400">
                  Benchmarking: <code className="text-zinc-200 bg-zinc-900 px-1.5 py-0.5 rounded font-mono">{currentRunningAlgo}</code>
                </div>
              </div>
            )}

            {!loading && results && (
              <div className="space-y-6">
                <div className="overflow-x-auto">
                  <table className="w-full text-left border-collapse">
                    <thead>
                      <tr className="border-b border-zinc-800 text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                        <th className="py-2.5 px-2">Rank</th>
                        <th className="py-2.5 px-2">Model</th>
                        <th className="py-2.5 px-2">Mean RMSLE</th>
                        <th className="py-2.5 px-2">Std</th>
                        <th className="py-2.5 px-2 text-right">Best Fold</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((row, idx) => (
                        <tr key={row.algorithm} className="border-b border-zinc-800/40 text-xs text-zinc-300">
                          <td className="py-3 px-2 font-bold">{['🥇', '🥈', '🥉', '4th'][idx] || `${idx + 1}th`}</td>
                          <td className="py-3 px-2 font-semibold">{row.algorithm.replace('Regressor', '')}</td>
                          <td className="py-3 px-2 text-cyan-400 font-mono font-bold">{row.mean_rmsle.toFixed(4)}</td>
                          <td className="py-3 px-2 text-zinc-500">± {row.std_rmsle.toFixed(4)}</td>
                          <td className="py-3 px-2 text-right font-mono text-zinc-400">{row.best_fold.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div>
                  <ReactECharts option={barChartOption} style={{ height: '220px' }} />
                </div>
              </div>
            )}

            {!loading && !results && (
              <div className="flex flex-col items-center justify-center text-center py-24 space-y-3">
                <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center text-lg">
                  🏆
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-zinc-300">No Active Tournaments</h4>
                  <p className="text-xs text-zinc-500 max-w-xs mt-1">Select your benchmark settings and run comparison to generate accuracy charts.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
