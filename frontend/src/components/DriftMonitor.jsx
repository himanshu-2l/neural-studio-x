import React, { useState, useEffect, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import apiClient from '../apiClient';
import { getHouseData } from '../utils';
import { ShieldAlert, ShieldCheck, Database, BarChart2 } from 'lucide-react';

export default function DriftMonitor() {
  const [driftData, setDriftData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [activeFeature, setActiveFeature] = useState('GrLivArea');

  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const [driftRes, predRes] = await Promise.all([
        apiClient.get('/drift'),
        apiClient.get('/predictions')
      ]);
      setDriftData(driftRes.data);
      setPredictions(predRes.data.predictions || []);
    } catch (err) {
      setError('Failed to fetch data quality metrics from database.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const refData = useMemo(() => getHouseData(), []);

  const parsedPredictions = useMemo(() => {
    return predictions.map(p => {
      try {
        return JSON.parse(p.input_features);
      } catch {
        return null;
      }
    }).filter(Boolean);
  }, [predictions]);

  // Comparative distribution histograms
  const chartOption = useMemo(() => {
    if (parsedPredictions.length === 0 || !refData) return null;

    const refValues = refData.map(d => d[activeFeature]);
    const predValues = parsedPredictions.map(d => d[activeFeature]).filter(v => v !== undefined);

    if (refValues.length === 0 || predValues.length === 0) return null;

    const min = Math.min(...refValues, ...predValues);
    const max = Math.max(...refValues, ...predValues);
    const step = (max - min) / 20;

    const refBins = Array(20).fill(0);
    const predBins = Array(20).fill(0);
    const labels = Array(20).fill(0).map((_, i) => (min + i * step).toFixed(1));

    refValues.forEach(v => {
      const idx = Math.min(Math.floor((v - min) / step), 19);
      refBins[idx]++;
    });

    predValues.forEach(v => {
      const idx = Math.min(Math.floor((v - min) / step), 19);
      predBins[idx]++;
    });

    // Normalize counts to probability density
    const totalRef = refValues.length;
    const totalPred = predValues.length;
    const refDensity = refBins.map(c => c / totalRef);
    const predDensity = predBins.map(c => c / totalPred);

    return {
      backgroundColor: 'transparent',
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { data: ['Reference (Train)', 'Production (Live)'], textStyle: { color: '#a1a1aa' } },
      grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' }
      },
      yAxis: {
        type: 'value',
        name: 'Density',
        nameTextStyle: { color: '#a1a1aa' },
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [
        {
          name: 'Reference (Train)',
          type: 'bar',
          data: refDensity,
          barGap: 0,
          itemStyle: { color: 'rgba(6, 182, 212, 0.65)', borderRadius: [2, 2, 0, 0] }
        },
        {
          name: 'Production (Live)',
          type: 'bar',
          data: predDensity,
          itemStyle: { color: 'rgba(248, 113, 113, 0.65)', borderRadius: [2, 2, 0, 0] }
        }
      ]
    };
  }, [refData, parsedPredictions, activeFeature]);

  const metricsList = useMemo(() => {
    if (!driftData || !driftData.metrics) return [];
    return Object.keys(driftData.metrics).map(col => ({
      feature: col,
      ...driftData.metrics[col]
    }));
  }, [driftData]);

  const driftedCount = useMemo(() => {
    return metricsList.filter(m => m.drift_detected).length;
  }, [metricsList]);

  return (
    <div className="space-y-6">
      {loading && !driftData ? (
        <div className="py-20 flex justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-cyan-400"></div>
        </div>
      ) : driftData?.status === 'insufficient_data' ? (
        <div className="flex flex-col items-center justify-center text-center py-20 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-8 space-y-4">
          <Database className="w-12 h-12 text-zinc-500" />
          <h3 className="text-sm font-semibold text-zinc-300">Insufficient Data for Drift Analysis</h3>
          <p className="text-xs text-zinc-500 max-w-sm">
            Kolmogorov-Smirnov statistical tests require at least 5 logged predictions to compare distributions. Currently logged: <code className="text-cyan-400 bg-zinc-900 px-1 py-0.5 rounded font-mono font-bold">{driftData.target_count}</code>.
          </p>
        </div>
      ) : driftData ? (
        <div className="space-y-6">
          {/* Summary stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className={`border rounded-xl p-5 shadow-sm bg-[#0c0c0f] flex items-center justify-between ${driftedCount > 0 ? 'border-rose-500/30' : 'border-zinc-800'}`}>
              <div>
                <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">System Stability</p>
                <h3 className={`text-2xl font-bold mt-1 flex items-center gap-2 ${driftedCount > 0 ? 'text-rose-400' : 'text-emerald-400'}`}>
                  {driftedCount > 0 ? <ShieldAlert className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
                  {driftedCount > 0 ? 'Drift Detected' : 'Healthy Status'}
                </h3>
              </div>
            </div>
            <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Reference Set Size</p>
              <h3 className="text-2xl font-bold text-zinc-100 mt-1">{refData.length.toLocaleString()} samples</h3>
            </div>
            <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
              <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Production queries</p>
              <h3 className="text-2xl font-bold text-zinc-100 mt-1">{driftData.target_count} requests</h3>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Breakdown Table */}
            <div className="lg:col-span-2 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
              <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                Kolmogorov-Smirnov Breakdown
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-zinc-800 text-[10px] text-zinc-500 uppercase tracking-wider font-semibold font-sans">
                      <th className="py-2.5 px-2">Feature</th>
                      <th className="py-2.5 px-2">KS Statistic</th>
                      <th className="py-2.5 px-2">p-value</th>
                      <th className="py-2.5 px-2">Baseline Mean</th>
                      <th className="py-2.5 px-2">Production Mean</th>
                      <th className="py-2.5 px-2 text-right">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metricsList.map((m) => (
                      <tr key={m.feature} className="border-b border-zinc-800/40 text-xs text-zinc-300">
                        <td className="py-3 px-2 font-semibold">{m.feature}</td>
                        <td className="py-3 px-2 font-mono">{m.statistic.toFixed(4)}</td>
                        <td className="py-3 px-2 font-mono">{m.p_value.toFixed(4)}</td>
                        <td className="py-3 px-2">{m.ref_mean.toFixed(1)}</td>
                        <td className="py-3 px-2">{m.tgt_mean.toFixed(1)}</td>
                        <td className="py-3 px-2 text-right font-bold">
                          <span className={`px-2 py-0.5 rounded-full text-[10px] ${m.drift_detected ? 'bg-rose-950/20 border border-rose-900/30 text-rose-400' : 'bg-emerald-950/20 border border-emerald-900/30 text-emerald-400'}`}>
                            {m.drift_detected ? '❌ Drifted' : '✅ Stable'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Visualize shift */}
            <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
              <h3 className="text-base font-bold text-zinc-100">Visualize Shifts</h3>
              
              <select
                value={activeFeature}
                onChange={(e) => setActiveFeature(e.target.value)}
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none"
              >
                <option value="GrLivArea">GrLivArea</option>
                <option value="OverallQual">OverallQual</option>
                <option value="TotalBsmtSF">TotalBsmtSF</option>
                <option value="YearBuilt">YearBuilt</option>
              </select>

              {chartOption ? (
                <ReactECharts option={chartOption} style={{ height: '240px' }} />
              ) : (
                <div className="text-center py-20 text-xs text-zinc-500">
                  No chart data available.
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
