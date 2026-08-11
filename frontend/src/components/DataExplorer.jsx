import React, { useState, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { getHouseData, buildFeatures } from '../utils';
import { LayoutGrid, BarChart2, Hash, Percent, Search, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react';

export default function DataExplorer() {
  const rawData = useMemo(() => getHouseData(), []);
  const feData = useMemo(() => rawData.map(buildFeatures), [rawData]);

  const [searchTerm, setSearchTerm] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [activeFeature, setActiveFeature] = useState('GrLivArea');
  const [showLog, setShowLog] = useState(false);
  const pageSize = 12;

  // Search & Filter
  const filteredData = useMemo(() => {
    return feData.filter(row => 
      row.Neighborhood.toLowerCase().includes(searchTerm.toLowerCase()) ||
      row.id.toString().includes(searchTerm)
    );
  }, [feData, searchTerm]);

  // Pagination
  const totalPages = Math.ceil(filteredData.length / pageSize);
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * pageSize;
    return filteredData.slice(start, start + pageSize);
  }, [filteredData, currentPage]);

  const stats = useMemo(() => {
    return {
      rows: rawData.length,
      cols: Object.keys(rawData[0]).length,
      numeric: Object.keys(rawData[0]).filter(k => typeof rawData[0][k] === 'number').length,
      missing: '0.0%'
    };
  }, [rawData]);

  // Histogram calculation
  const chartOption = useMemo(() => {
    const values = feData.map(d => showLog ? Math.log1p(d[activeFeature]) : d[activeFeature]);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const step = (max - min) / 25;
    
    const bins = Array(25).fill(0);
    const labels = Array(25).fill(0).map((_, i) => (min + i * step).toFixed(1));

    values.forEach(v => {
      const idx = Math.min(Math.floor((v - min) / step), 24);
      bins[idx]++;
    });

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' }
      },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
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
          name: 'Frequency',
          type: 'bar',
          data: bins,
          barWidth: '85%',
          itemStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: '#06b6d4' },
                { offset: 1, color: '#3b82f6' }
              ]
            },
            borderRadius: [4, 4, 0, 0]
          }
        }
      ]
    };
  }, [feData, activeFeature, showLog]);

  // Scatter plot configuration
  const scatterOption = useMemo(() => {
    const data = feData.map(d => [d.GrLivArea, d.SalePrice, d.OverallQual, d.Neighborhood]);
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: (params) => {
          return `
            <div class="text-xs text-zinc-950 font-sans">
              <b>ID:</b> ${params.dataIndex + 1461}<br/>
              <b>Area:</b> ${params.data[0]} sqft<br/>
              <b>Price:</b> $${params.data[1].toLocaleString()}<br/>
              <b>Qual:</b> ${params.data[2]}/10
            </div>
          `;
        }
      },
      grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
      xAxis: {
        name: 'Area (sqft)',
        nameTextStyle: { color: '#a1a1aa' },
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      yAxis: {
        name: 'Sale Price ($)',
        nameTextStyle: { color: '#a1a1aa' },
        axisLine: { lineStyle: { color: '#3f3f46' } },
        splitLine: { lineStyle: { color: '#18181b' } },
        axisLabel: { color: '#a1a1aa' }
      },
      series: [{
        type: 'scatter',
        data: data,
        symbolSize: (val) => val[2] * 2 + 3,
        itemStyle: {
          color: {
            type: 'radial',
            x: 0.4, y: 0.4, r: 1,
            colorStops: [
              { offset: 0, color: '#a78bfa' },
              { offset: 1, color: '#7c3aed' }
            ]
          },
          opacity: 0.75
        }
      }]
    };
  }, [feData]);

  return (
    <div className="space-y-6">
      {/* Overview stats grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 flex items-center justify-between shadow-sm">
          <div>
            <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Total Rows</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{stats.rows.toLocaleString()}</h3>
          </div>
          <LayoutGrid className="w-8 h-8 text-cyan-400 opacity-80" />
        </div>

        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 flex items-center justify-between shadow-sm">
          <div>
            <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Features</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{stats.cols}</h3>
          </div>
          <Hash className="w-8 h-8 text-blue-400 opacity-80" />
        </div>

        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 flex items-center justify-between shadow-sm">
          <div>
            <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Numeric Cols</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{stats.numeric}</h3>
          </div>
          <BarChart2 className="w-8 h-8 text-violet-400 opacity-80" />
        </div>

        <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 flex items-center justify-between shadow-sm">
          <div>
            <p className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Missing cells</p>
            <h3 className="text-2xl font-bold text-zinc-100 mt-1">{stats.missing}</h3>
          </div>
          <Percent className="w-8 h-8 text-emerald-400 opacity-80" />
        </div>
      </div>

      {/* Main body split layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Table list */}
        <div className="lg:col-span-2 bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
            <h3 className="text-lg font-bold text-zinc-100">Dataset Preview</h3>
            
            {/* Search */}
            <div className="relative w-full sm:w-64">
              <input
                type="text"
                placeholder="Search Neighborhood/ID..."
                value={searchTerm}
                onChange={(e) => { setSearchTerm(e.target.value); setCurrentPage(1); }}
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg pl-9 pr-4 py-2 text-xs text-zinc-200 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
              <Search className="w-4 h-4 text-zinc-600 absolute left-3 top-2.5" />
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-zinc-800 text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">
                  <th className="py-3 px-2">ID</th>
                  <th className="py-3 px-2">GrLivArea</th>
                  <th className="py-3 px-2">Quality</th>
                  <th className="py-3 px-2">BsmtSF</th>
                  <th className="py-3 px-2">Year</th>
                  <th className="py-3 px-2">Neighborhood</th>
                  <th className="py-3 px-2 text-right">SalePrice</th>
                </tr>
              </thead>
              <tbody>
                {paginatedData.map((row) => (
                  <tr key={row.id} className="border-b border-zinc-800/40 text-xs text-zinc-300 hover:bg-zinc-800/20 transition-colors">
                    <td className="py-3 px-2 font-mono text-zinc-500">{row.id}</td>
                    <td className="py-3 px-2">{row.GrLivArea} sqft</td>
                    <td className="py-3 px-2">
                      <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 rounded-full bg-cyan-400"></div>
                        {row.OverallQual}/10
                      </div>
                    </td>
                    <td className="py-3 px-2">{row.TotalBsmtSF}</td>
                    <td className="py-3 px-2">{row.YearBuilt}</td>
                    <td className="py-3 px-2">{row.Neighborhood}</td>
                    <td className="py-3 px-2 text-right font-semibold text-cyan-400">${row.SalePrice.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex justify-between items-center pt-3 border-t border-zinc-800 text-xs text-zinc-500">
              <span>Showing Page {currentPage} of {totalPages}</span>
              <div className="flex gap-1">
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(1)}
                  className="p-1 border border-zinc-800 rounded bg-[#09090b] hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <ChevronsLeft className="w-4 h-4" />
                </button>
                <button
                  disabled={currentPage === 1}
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  className="p-1 border border-zinc-800 rounded bg-[#09090b] hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30"
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                <button
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                  className="p-1 border border-zinc-800 rounded bg-[#09090b] hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30"
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  disabled={currentPage === totalPages}
                  onClick={() => setCurrentPage(totalPages)}
                  className="p-1 border border-zinc-800 rounded bg-[#09090b] hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  <ChevronsRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Charts & Interactive Panels */}
        <div className="space-y-6">
          {/* Distribution card */}
          <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
            <div className="flex justify-between items-center mb-4">
              <h4 className="text-sm font-bold text-zinc-200">Distribution Panel</h4>
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-zinc-500 uppercase tracking-wider font-semibold">Log</label>
                <input
                  type="checkbox"
                  checked={showLog}
                  onChange={() => setShowLog(!showLog)}
                  className="w-3.5 h-3.5 rounded accent-cyan-500 bg-[#09090b]"
                />
              </div>
            </div>
            
            <div className="flex gap-2 mb-4">
              <select
                value={activeFeature}
                onChange={(e) => setActiveFeature(e.target.value)}
                className="w-full bg-[#09090b] border border-zinc-800 rounded-lg p-2 text-xs text-zinc-200 focus:outline-none"
              >
                <option value="GrLivArea">Living Area (GrLivArea)</option>
                <option value="OverallQual">Material Quality (OverallQual)</option>
                <option value="TotalBsmtSF">Basement Area (TotalBsmtSF)</option>
                <option value="YearBuilt">Year Built (YearBuilt)</option>
                <option value="SalePrice">Sale Price (SalePrice)</option>
              </select>
            </div>

            <ReactECharts option={chartOption} style={{ height: '200px' }} />
          </div>

          {/* Scatterplot Card */}
          <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-5 shadow-sm">
            <h4 className="text-sm font-bold text-zinc-200 mb-4">Area vs Pricing</h4>
            <ReactECharts option={scatterOption} style={{ height: '220px' }} />
          </div>
        </div>
      </div>
    </div>
  );
}
