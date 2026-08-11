import React from 'react';
import { Terminal, Shield, BookOpen, ExternalLink } from 'lucide-react';

export default function DeployAPI() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Docker and backend deployment info */}
      <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-cyan-500/10 flex items-center justify-center">
            <Terminal className="w-4 h-4 text-cyan-400" />
          </div>
          <h3 className="text-base font-bold text-zinc-100">Docker Infrastructure</h3>
        </div>

        <div className="space-y-4">
          <p className="text-xs text-zinc-400 leading-relaxed">
            Execute the following container orchestration commands to run the unified React + FastAPI stack locally:
          </p>
          <pre className="bg-[#09090b] border border-zinc-800 rounded-lg p-4 font-mono text-xs text-cyan-400 overflow-x-auto leading-relaxed">
{`# Build unified image
docker build -t neural-studio-x .

# Start stack as background daemon
docker compose up -d

# Verify container statuses
docker compose ps

# Stop and purge networks
docker compose down -v`}
          </pre>
        </div>
      </div>

      {/* FastAPI curl specs */}
      <div className="bg-[#0c0c0f] border border-zinc-800 rounded-xl p-6 shadow-sm space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-violet-500/10 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-violet-400" />
          </div>
          <h3 className="text-base font-bold text-zinc-100">API Documentation</h3>
        </div>

        <div className="space-y-4">
          <p className="text-xs text-zinc-400 leading-relaxed">
            Standard HTTP headers and curl configurations for triggering remote model calls:
          </p>
          <pre className="bg-[#09090b] border border-zinc-800 rounded-lg p-4 font-mono text-xs text-cyan-400 overflow-x-auto leading-relaxed">
{`# Health check verification
curl http://localhost:8000/health

# Inference execution (api key required)
curl -X POST http://localhost:8000/predict \\
  -H "x-api-key: nsx-dev-key-change-in-prod" \\
  -H "Content-Type: application/json" \\
  -d '{
    "GrLivArea": 1850,
    "OverallQual": 7,
    "TotalBsmtSF": 1050,
    "YearBuilt": 2005,
    "FullBath": 2,
    "algorithm": "Ridge"
  }'

# Registry runs query
curl http://localhost:8000/experiments`}
          </pre>
        </div>
      </div>
    </div>
  );
}
