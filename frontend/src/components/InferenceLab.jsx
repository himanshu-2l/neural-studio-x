import React, { useState, useEffect, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import apiClient from '../apiClient';
import { Eye, ShieldAlert, Sparkles, TrendingUp } from 'lucide-react';

export default function InferenceLab({ username }) {
  const [grLivArea, setGrLivArea] = useState(1850);
  const [overallQual, setOverallQual] = useState(7);
  const [totalBsmtSF, setTotalBsmtSF] = useState(1050);
  const [yearBuilt, setYearBuilt] = useState(2005);
  const [fullBath, setFullBath] = useState(2);
  const [halfBath, setHalfBath] = useState(1);
  const [algo, setAlgo] = useState('GradientBoostingRegressor');

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Call API for live prediction when inputs change
  useEffect(() => {
    let active = true;
    const fetchPrediction = async () => {
      setLoading(true);
      setError('');
      try {
        const response = await apiClient.post('/predict', {
          GrLivArea: grLivArea,
          OverallQual: overallQual,
          TotalBsmtSF: totalBsmtSF,
          YearBuilt: yearBuilt,
          FullBath: fullBath,
          HalfBath: halfBath,
          algorithm: algo,
          username: username || 'guest'
        });
        if (active) {
          setPrediction(response.data);
          setLoading(false);
        }
      } catch (err) {
        if (active) {
          setLoading(false);
          // If model doesn't exist, we can fallback to mock calculation or prompt training
          if (err.response?.status === 404) {
            setError('Target model is not trained yet. Please train it first in the Model Trainer tab.');
          } else {
            setError(err.response?.data?.detail || 'Failed to fetch price prediction.');
          }
        }
      }
    };

    // Debounce the slider updates slightly
    const delayDebounce = setTimeout(() => {
      fetchPrediction();
    }, 250);

    return () => {
      active = false;
      clearTimeout(delayDebounce);
    };
  }, [grLivArea, overallQual, totalBsmtSF, yearBuilt, fullBath, halfBath, algo, username]);

  // Feature contribution calculations for visual breakdown chart
  const contributionOption = useMemo(() => {
    const categories = ['Living Area', 'Quality Factor', 'Basement SF', 'House Age Factor', 'Bathrooms'];
    // Approximated linear breakdown logic matching backend coefficient math
    const values = [
      grLivArea * 65,
      overallQual * 16000,
      totalBsmtSF * 42,
      (yearBuilt - 1940) * 560,
      (fullBath + 0.5 * halfBath) * 7500
    ];

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params) => {
          return `
            <div class="text-xs text-zinc-950 font-sans">
              <b>${params[0].name}:</b> $${params[0].value.toLocaleString()}
            </div>
          `;
        }
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
        data: categories,
        axisLine: { lineStyle: { color: '#3f3f46' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [
        {
          name: 'Impact ($)',
          type: 'bar',
          data: values,
          itemStyle: {
            color: '#06b6d4',
            borderRadius: [0, 4, 4, 0]
          }
        }
      ]
    };
  }, [grLivArea, overallQual, totalBsmtSF, yearBuilt, fullBath, halfBath]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Sliders input column */}
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-5">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center">
              <Eye className="w-4 h-4 text-cyan-400" />
            </div>
            <h3 className="text-base font-bold text-zinc-100">Live Feature Inputs</h3>
          </div>

          <div className="space-y-4">
            <div>
              <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                <span>Living Area</span>
                <span className="text-cyan-400 font-mono">{grLivArea} sqft</span>
              </div>
              <input
                type="range" min="500" max="5000" step="50" value={grLivArea}
                onChange={(e) => setGrLivArea(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                <span>Material Quality</span>
                <span className="text-cyan-400 font-mono">{overallQual}/10</span>
              </div>
              <input
                type="range" min="1" max="10" step="1" value={overallQual}
                onChange={(e) => setOverallQual(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                <span>Basement Area</span>
                <span className="text-cyan-400 font-mono">{totalBsmtSF} sqft</span>
              </div>
              <input
                type="range" min="0" max="3500" step="50" value={totalBsmtSF}
                onChange={(e) => setTotalBsmtSF(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div>
              <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                <span>Year Built</span>
                <span className="text-cyan-400 font-mono">{yearBuilt}</span>
              </div>
              <input
                type="range" min="1920" max="2025" step="1" value={yearBuilt}
                onChange={(e) => setYearBuilt(parseInt(e.target.value))}
                className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                  <span>Full Baths</span>
                  <span className="text-cyan-400 font-mono">{fullBath}</span>
                </div>
                <input
                  type="range" min="1" max="4" step="1" value={fullBath}
                  onChange={(e) => setFullBath(parseInt(e.target.value))}
                  className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>

              <div>
                <div className="flex justify-between items-center text-xs text-zinc-500 font-semibold mb-1.5">
                  <span>Half Baths</span>
                  <span className="text-cyan-400 font-mono">{halfBath}</span>
                </div>
                <input
                  type="range" min="0" max="2" step="1" value={halfBath}
                  onChange={(e) => setHalfBath(parseInt(e.target.value))}
                  className="w-full h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold text-zinc-500 uppercase tracking-wider mb-2">Model Engine</label>
              <select
                value={algo}
                onChange={(e) => setAlgo(e.target.value)}
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg p-3 text-sm text-zinc-300 focus:outline-none"
              >
                <option value="Production">🚀 Active Production Model</option>
                <option value="GradientBoostingRegressor">Gradient Boosting Regressor</option>
                <option value="RandomForestRegressor">Random Forest Regressor</option>
                <option value="Ridge">Ridge Regression</option>
              </select>
            </div>
          </div>
        </div>

        {/* Prediction Outputs */}
        <div className="space-y-6">
          {/* Main Prediction Display */}
          <div className="bg-gradient-to-tr from-cyan-950/10 via-zinc-900 to-violet-950/10 border border-cyan-500/25 rounded-2xl p-7 shadow-lg text-center space-y-4 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-3 text-cyan-500/20">
              <Sparkles className="w-8 h-8" />
            </div>

            <p className="text-xs font-bold text-zinc-400 uppercase tracking-widest">
              Live Projected Sale Price
            </p>

            {loading && !prediction ? (
              <div className="py-6 flex justify-center">
                <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-cyan-400"></div>
              </div>
            ) : prediction ? (
              <div className="space-y-2">
                <h2 className="text-5xl font-black tracking-tight bg-gradient-to-r from-cyan-400 to-emerald-400 bg-clip-text text-transparent">
                  ${Math.round(prediction.predicted_price).toLocaleString()}
                </h2>
                <p className="text-xs text-zinc-500">
                  Confidence range: <b>${Math.round(prediction.lower_bound).toLocaleString()}</b> — <b>${Math.round(prediction.upper_bound).toLocaleString()}</b>
                </p>
                <p className="text-[10px] text-zinc-600 font-mono pt-1">
                  Engine: {prediction.algorithm}
                </p>
              </div>
            ) : null}

            {error && (
              <div className="flex items-start gap-2.5 text-xs text-rose-400 bg-rose-950/20 border border-rose-900/30 rounded-lg p-4 text-left">
                <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-500" />
                <div>
                  <p className="font-semibold">Inference Model Mismatch</p>
                  <p className="text-zinc-500 mt-1">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Feature Impact Chart */}
          <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
            <h4 className="text-xs font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              Weight Impact Breakdowns
            </h4>
            <ReactECharts option={contributionOption} style={{ height: '200px' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
