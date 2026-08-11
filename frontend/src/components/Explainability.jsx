import React, { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { Shield, Sparkles, BrainCircuit } from 'lucide-react';

export default function Explainability() {
  const data = useMemo(() => [
    { name: 'TotalSF', value: 0.45, pct: 45 },
    { name: 'OverallQual', value: 0.28, pct: 28 },
    { name: 'TotalBath', value: 0.12, pct: 12 },
    { name: 'HouseAge', value: 0.08, pct: 8 },
    { name: 'YearBuilt', value: 0.04, pct: 4 },
    { name: 'GrLivArea', value: 0.02, pct: 2 },
    { name: 'TotalBsmtSF', value: 0.01, pct: 1 }
  ].sort((a, b) => a.value - b.value), []);

  const chartOption = useMemo(() => {
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      grid: { left: '3%', right: '8%', bottom: '3%', top: '3%', containLabel: true },
      xAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      yAxis: {
        type: 'category',
        data: data.map(d => d.name),
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [
        {
          name: 'Relative Importance',
          type: 'bar',
          data: data.map(d => d.value),
          itemStyle: {
            color: '#a78bfa',
            borderRadius: [0, 4, 4, 0]
          }
        }
      ]
    };
  }, [data]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart Column */}
        <div className="lg:col-span-2 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center">
              <BrainCircuit className="w-4 h-4 text-cyan-400" />
            </div>
            <h3 className="text-base font-bold text-zinc-100">Relative Feature Importances</h3>
          </div>

          <ReactECharts option={chartOption} style={{ height: '300px' }} />
        </div>

        {/* Feature Explanations Panel */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
          <h3 className="text-base font-bold text-zinc-100 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-violet-400" />
            Top Contributors
          </h3>

          <div className="space-y-4">
            {data.slice().reverse().slice(0, 5).map((f) => (
              <div key={f.name} className="space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-semibold text-zinc-200">{f.name}</span>
                  <span className="text-cyan-400 font-mono font-bold">{f.pct}%</span>
                </div>
                <div className="w-full bg-zinc-800/80 rounded-full h-1.5 overflow-hidden">
                  <div className="bg-gradient-to-r from-cyan-500 to-violet-500 h-1.5" style={{ width: `${f.pct}%` }}></div>
                </div>
              </div>
            ))}
          </div>

          <div className="border-t border-zinc-800/80 pt-4 flex gap-3 text-xs text-zinc-500 items-start">
            <Shield className="w-5 h-5 text-zinc-600 flex-shrink-0 mt-0.5" />
            <p>
              Importances represent the fractional contribution of each engineered feature's decision boundaries across tree nodes in cross-validation splits.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
