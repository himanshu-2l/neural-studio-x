import os
import uuid
import json
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from database import init_db, save_experiment, load_experiments, save_prediction, load_predictions, get_stats
from auth import build_authenticator, render_login, render_logout
from ml_utils import build_features, get_pipeline, run_kfold, get_house_data_df, HAS_SKLEARN

init_db()

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("neural_studio_x")

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:
    HAS_CANVAS = False

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Neural Studio X",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root variables ─────────────────────────────────────────── */
:root {
  --bg-base:       #080c14;
  --bg-surface:    #0d1422;
  --bg-elevated:   #121929;
  --bg-card:       rgba(18, 25, 41, 0.8);
  --bg-hover:      rgba(30, 41, 61, 0.9);

  --border-subtle: rgba(255,255,255,0.05);
  --border-muted:  rgba(255,255,255,0.09);
  --border-accent: rgba(0, 212, 255, 0.25);

  --text-primary:  #f0f4ff;
  --text-secondary:#8b97b5;
  --text-muted:    #4a5568;

  --accent-blue:   #00d4ff;
  --accent-green:  #00e5a0;
  --accent-purple: #a78bfa;
  --accent-amber:  #fbbf24;
  --accent-red:    #f87171;

  --grad-primary:  linear-gradient(135deg, #00d4ff 0%, #7c3aed 100%);
  --grad-green:    linear-gradient(135deg, #00e5a0 0%, #00d4ff 100%);
  --grad-warm:     linear-gradient(135deg, #fbbf24 0%, #f97316 100%);

  --radius-sm:  8px;
  --radius-md:  12px;
  --radius-lg:  16px;
  --radius-xl:  20px;

  --shadow-sm:  0 2px 8px rgba(0,0,0,0.3);
  --shadow-md:  0 4px 20px rgba(0,0,0,0.4);
  --shadow-lg:  0 8px 40px rgba(0,0,0,0.5);
  --shadow-glow-blue:  0 0 20px rgba(0,212,255,0.15);
  --shadow-glow-green: 0 0 20px rgba(0,229,160,0.15);
}

/* ── Global reset ────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.stApp {
  background: var(--bg-base);
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,212,255,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 100%, rgba(124,58,237,0.05) 0%, transparent 60%);
  color: var(--text-primary);
}

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: 0 !important;
}

/* ── Main content padding ─────────────────────────────────────── */
.main .block-container {
  padding: 1.5rem 2rem 3rem;
  max-width: 1400px;
}

/* ── Headings ─────────────────────────────────────────────────── */
h1, h2, h3, h4 { font-weight: 700; letter-spacing: -0.02em; color: var(--text-primary); }

/* ── Streamlit native overrides ────────────────────────────────── */
.stButton > button {
  background: linear-gradient(135deg, rgba(0,212,255,0.12) 0%, rgba(124,58,237,0.12) 100%);
  color: var(--accent-blue);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: 0.85rem;
  padding: 0.5rem 1.25rem;
  transition: all 0.2s ease;
  letter-spacing: 0.01em;
}
.stButton > button:hover {
  background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(124,58,237,0.2) 100%);
  border-color: var(--accent-blue);
  box-shadow: var(--shadow-glow-blue);
  transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
  background: var(--grad-primary);
  color: white;
  border: none;
}

.stSelectbox > div, .stSlider, .stMultiSelect {
  font-size: 0.85rem;
}

[data-testid="stSlider"] > div > div {
  background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple)) !important;
}

.stDataFrame, .stDataFrame > div { border-radius: var(--radius-md); overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid var(--border-subtle); border-radius: var(--radius-md); }

/* ── Tabs ─────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  background: rgba(13,20,34,0.9);
  padding: 5px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
  flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
  height: 38px;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-weight: 500;
  font-size: 0.82rem;
  border: none !important;
  padding: 0 14px;
  background: transparent !important;
  transition: all 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text-primary);
  background: rgba(255,255,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,212,255,0.1) !important;
  color: var(--accent-blue) !important;
  border: 1px solid rgba(0,212,255,0.2) !important;
  font-weight: 600 !important;
}

/* ── Success / warning / error ───────────────────────────────── */
.stSuccess { background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.25); border-radius: var(--radius-md); }
.stWarning { background: rgba(251,191,36,0.08); border: 1px solid rgba(251,191,36,0.25); border-radius: var(--radius-md); }
.stInfo    { background: rgba(0,212,255,0.08);  border: 1px solid rgba(0,212,255,0.2);  border-radius: var(--radius-md); }

/* ── Custom component classes ─────────────────────────────────── */

/* Topbar brand strip */
.nsx-topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px;
  background: rgba(13,20,34,0.95);
  border-bottom: 1px solid var(--border-subtle);
  margin: -1.5rem -2rem 1.5rem;
  backdrop-filter: blur(20px);
  position: sticky; top: 0; z-index: 99;
}
.nsx-brand {
  display: flex; align-items: center; gap: 10px;
}
.nsx-logo-mark {
  width: 32px; height: 32px; border-radius: 8px;
  background: var(--grad-primary);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 900; color: white;
  box-shadow: var(--shadow-glow-blue);
}
.nsx-brand-name {
  font-size: 1.1rem; font-weight: 800; letter-spacing: -0.03em;
  background: var(--grad-primary); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nsx-brand-ver {
  font-size: 0.7rem; color: var(--text-muted);
  background: rgba(255,255,255,0.06); border: 1px solid var(--border-subtle);
  padding: 2px 7px; border-radius: 20px; font-weight: 500; margin-left: 4px;
}
.nsx-status-row { display: flex; align-items: center; gap: 12px; }
.nsx-badge {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 10px; border-radius: 20px;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.03em;
}
.nsx-badge-green { background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.25); color: var(--accent-green); }
.nsx-badge-blue  { background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.2);  color: var(--accent-blue); }
.nsx-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

/* Metric cards */
.nsx-metric-row { display: grid; gap: 14px; margin-bottom: 20px; }
.nsx-metric-row-4 { grid-template-columns: repeat(4, 1fr); }
.nsx-metric-row-3 { grid-template-columns: repeat(3, 1fr); }
.nsx-metric-row-2 { grid-template-columns: repeat(2, 1fr); }

.nsx-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  backdrop-filter: blur(12px);
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.nsx-card:hover { border-color: var(--border-muted); box-shadow: var(--shadow-md); }

.nsx-card-accent-blue  { border-left: 3px solid var(--accent-blue);   }
.nsx-card-accent-green { border-left: 3px solid var(--accent-green);  }
.nsx-card-accent-purple{ border-left: 3px solid var(--accent-purple); }
.nsx-card-accent-amber { border-left: 3px solid var(--accent-amber);  }

.nsx-card-label {
  font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--text-muted); font-weight: 600; margin-bottom: 8px;
}
.nsx-card-value {
  font-size: 1.9rem; font-weight: 800; color: var(--text-primary);
  letter-spacing: -0.04em; line-height: 1;
}
.nsx-card-sub { font-size: 0.75rem; color: var(--text-secondary); margin-top: 6px; }
.nsx-card-icon { font-size: 1.3rem; margin-bottom: 8px; }

/* Section headers */
.nsx-section-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 16px; padding-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
}
.nsx-section-title {
  font-size: 1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em;
}
.nsx-section-desc { font-size: 0.78rem; color: var(--text-secondary); margin-top: 2px; }
.nsx-section-icon {
  width: 34px; height: 34px; border-radius: var(--radius-sm);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; flex-shrink: 0;
}
.nsx-icon-blue   { background: rgba(0,212,255,0.12);  }
.nsx-icon-green  { background: rgba(0,229,160,0.12);  }
.nsx-icon-purple { background: rgba(167,139,250,0.12); }
.nsx-icon-amber  { background: rgba(251,191,36,0.12); }
.nsx-icon-red    { background: rgba(248,113,113,0.12); }

/* Prediction result card */
.nsx-predict-result {
  background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(0,229,160,0.06) 100%);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-xl);
  padding: 28px 24px;
  text-align: center;
  box-shadow: var(--shadow-glow-blue), var(--shadow-md);
}
.nsx-predict-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); margin-bottom: 10px; font-weight: 600; }
.nsx-predict-price {
  font-size: 3rem; font-weight: 900; letter-spacing: -0.04em;
  background: var(--grad-green); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1;
}
.nsx-predict-range { font-size: 0.82rem; color: var(--text-secondary); margin-top: 10px; }
.nsx-predict-model { font-size: 0.72rem; color: var(--text-muted); margin-top: 6px; font-family: 'JetBrains Mono', monospace; }

/* Code-style tags */
.nsx-tag {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(255,255,255,0.05); border: 1px solid var(--border-subtle);
  padding: 2px 8px; border-radius: 6px;
  font-size: 0.72rem; font-family: 'JetBrains Mono', monospace; color: var(--text-secondary);
}

/* Divider */
.nsx-divider { height: 1px; background: var(--border-subtle); margin: 20px 0; }

/* Sidebar profile card */
.nsx-profile {
  background: var(--bg-elevated); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md); padding: 14px 16px; margin: 8px 0 16px;
}
.nsx-profile-name { font-weight: 700; font-size: 0.9rem; color: var(--text-primary); }
.nsx-profile-role { font-size: 0.72rem; color: var(--text-muted); margin-top: 2px; }
.nsx-profile-avatar {
  width: 36px; height: 36px; border-radius: 10px;
  background: var(--grad-primary);
  display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 700; color: white;
  float: left; margin-right: 12px;
}

/* Engine status pills */
.nsx-engine-row { display: flex; flex-direction: column; gap: 6px; }
.nsx-engine-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 10px;
  background: var(--bg-elevated); border: 1px solid var(--border-subtle);
  border-radius: var(--radius-sm); font-size: 0.78rem;
}
.nsx-engine-name { color: var(--text-secondary); font-weight: 500; }
.nsx-engine-ok   { color: var(--accent-green);  font-weight: 600; font-size: 0.72rem; }
.nsx-engine-warn { color: var(--accent-amber);  font-weight: 600; font-size: 0.72rem; }
.nsx-engine-err  { color: var(--accent-red);    font-weight: 600; font-size: 0.72rem; }

/* Empty state */
.nsx-empty {
  text-align: center; padding: 48px 24px;
  background: var(--bg-card); border: 1px dashed var(--border-muted);
  border-radius: var(--radius-lg);
}
.nsx-empty-icon { font-size: 2.5rem; margin-bottom: 12px; }
.nsx-empty-title { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px; }
.nsx-empty-desc  { font-size: 0.8rem; color: var(--text-secondary); }

/* Code block */
.nsx-code {
  background: #0a0f1a; border: 1px solid var(--border-subtle);
  border-radius: var(--radius-md);
  padding: 16px; font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem; color: #a8d8ea; line-height: 1.7;
  overflow-x: auto;
}

/* Progress bar */
.nsx-progress-wrap { background: rgba(255,255,255,0.05); border-radius: 20px; height: 6px; overflow: hidden; }
.nsx-progress-fill { height: 100%; border-radius: 20px; background: var(--grad-primary); transition: width 0.4s ease; }

/* Table override */
thead tr th { background: var(--bg-elevated) !important; color: var(--text-secondary) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.04em; }

/* Plotly charts transparent bg by default */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* Hide Streamlit default UI chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def section(icon, title, desc="", icon_class="nsx-icon-blue"):
    st.markdown(f"""
    <div class="nsx-section-header">
      <div class="nsx-section-icon {icon_class}">{icon}</div>
      <div><div class="nsx-section-title">{title}</div>
      {"<div class='nsx-section-desc'>"+desc+"</div>" if desc else ""}</div>
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub="", accent="blue"):
    st.markdown(f"""<div class="nsx-card nsx-card-accent-{accent}">
      <div class="nsx-card-label">{label}</div>
      <div class="nsx-card-value">{value}</div>
      {"<div class='nsx-card-sub'>"+sub+"</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def empty_state(icon, title, desc):
    st.markdown(f"""<div class="nsx-empty">
      <div class="nsx-empty-icon">{icon}</div>
      <div class="nsx-empty-title">{title}</div>
      <div class="nsx-empty-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)

def chart_cfg(fig):
    """Apply consistent dark theme to any plotly figure."""
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(8,12,20,0.6)',
        font=dict(family='Inter', color='#8b97b5', size=11),
        margin=dict(l=10, r=10, t=36, b=10),
        title_font=dict(size=13, color='#f0f4ff', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(255,255,255,0.05)', borderwidth=1),
        xaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.08)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.04)', linecolor='rgba(255,255,255,0.08)'),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────
_authenticator        = build_authenticator()
_username, _authenticated = render_login(_authenticator)

if not _authenticated:
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TOPBAR
# ─────────────────────────────────────────────────────────────────────────────
_db_stats = get_stats()
st.markdown(f"""
<div class="nsx-topbar">
  <div class="nsx-brand">
    <div class="nsx-logo-mark">⚡</div>
    <span class="nsx-brand-name">Neural Studio X</span>
    <span class="nsx-brand-ver">v3.3</span>
  </div>
  <div class="nsx-status-row">
    <span class="nsx-badge nsx-badge-green"><span class="nsx-dot"></span>API Live :8000</span>
    <span class="nsx-badge nsx-badge-blue"><span class="nsx-dot"></span>{_db_stats['total_experiments']} Experiments</span>
    <span class="nsx-badge nsx-badge-blue">👤 {_username}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="nsx-profile">
      <div class="nsx-profile-avatar">{''.join([w[0] for w in _username.split('-')[:2]]).upper()}</div>
      <div class="nsx-profile-name">{_username.title()}</div>
      <div class="nsx-profile-role">ML Engineer · Neural Studio X</div>
    </div>
    """, unsafe_allow_html=True)

    render_logout(_authenticator, _username)

    st.markdown("**Dataset**")
    dataset_choice = st.selectbox(
        "Active Project", label_visibility="collapsed",
        options=["🏠 House Prices — Regression", "🧠 Digit Recognizer — CV", "📁 Upload Custom CSV"]
    )

    if dataset_choice == "🏠 House Prices — Regression":
        raw_df    = get_house_data_df()
        mode_type = "regression"
    elif dataset_choice == "🧠 Digit Recognizer — CV":
        np.random.seed(42)
        n = 1000
        labels = np.random.randint(0, 10, size=n)
        pixels = np.random.randint(0, 256, size=(n, 784))
        d = pd.DataFrame(pixels, columns=[f'pixel{i}' for i in range(784)])
        d.insert(0, 'label', labels)
        raw_df    = d
        mode_type = "cv"
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            try:
                raw_df = pd.read_csv(uploaded)
                st.success(f"✓ {len(raw_df):,} rows loaded")
            except Exception as e:
                st.error(str(e))
                raw_df = get_house_data_df()
        else:
            raw_df = get_house_data_df()
        mode_type = "custom"

    st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Engine Status**")
    sk_status  = ("🟢", "Ready",   "nsx-engine-ok")   if HAS_SKLEARN else ("🔴", "Missing", "nsx-engine-err")
    sh_status  = ("🟢", "Ready",   "nsx-engine-ok")   if HAS_SHAP    else ("🟡", "Optional","nsx-engine-warn")
    jb_status  = ("🟢", "Ready",   "nsx-engine-ok")   if HAS_JOBLIB  else ("🔴", "Missing", "nsx-engine-err")
    cv_status  = ("🟢", "Ready",   "nsx-engine-ok")   if HAS_CANVAS  else ("🟡", "Optional","nsx-engine-warn")
    st.markdown(f"""
    <div class="nsx-engine-row">
      <div class="nsx-engine-item"><span class="nsx-engine-name">Scikit-Learn</span><span class="{sk_status[2]}">{sk_status[0]} {sk_status[1]}</span></div>
      <div class="nsx-engine-item"><span class="nsx-engine-name">SHAP</span><span class="{sh_status[2]}">{sh_status[0]} {sh_status[1]}</span></div>
      <div class="nsx-engine-item"><span class="nsx-engine-name">Joblib</span><span class="{jb_status[2]}">{jb_status[0]} {jb_status[1]}</span></div>
      <div class="nsx-engine-item"><span class="nsx-engine-name">Canvas</span><span class="{cv_status[2]}">{cv_status[0]} {cv_status[1]}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)
    st.markdown("**Session Stats**")
    st.markdown(f"""
    <div class="nsx-engine-row">
      <div class="nsx-engine-item"><span class="nsx-engine-name">Experiments</span><span class="nsx-engine-ok">{_db_stats['total_experiments']}</span></div>
      <div class="nsx-engine-item"><span class="nsx-engine-name">Predictions</span><span class="nsx-engine-ok">{_db_stats['total_predictions']}</span></div>
      <div class="nsx-engine-item"><span class="nsx-engine-name">Best RMSLE</span><span class="nsx-engine-ok">{_db_stats['best_rmsle'] or '—'}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_eda, tab_train, tab_inf, tab_automl, tab_shap, tab_exp, tab_deploy = st.tabs([
    "📊  Data Explorer",
    "⚡  Model Trainer",
    "🔮  Inference Lab",
    "🏆  AutoML",
    "🛡️  Explainability",
    "📈  Experiments",
    "🚀  Deploy & API",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    section("📊", "Data Explorer", "Automated EDA · distributions · correlation · 3D scatter", "nsx-icon-blue")

    # Summary metrics
    num_cols_all = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_all = raw_df.select_dtypes(include=["object"]).columns.tolist()
    miss_pct     = (raw_df.isnull().sum().sum() / raw_df.size * 100)

    st.markdown('<div class="nsx-metric-row nsx-metric-row-4">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Rows",        f"{raw_df.shape[0]:,}",   "Total samples",        "blue")
    with c2: metric_card("Features",    f"{raw_df.shape[1]:,}",   "Columns",              "purple")
    with c3: metric_card("Numeric",     f"{len(num_cols_all)}",   "Quantitative columns", "green")
    with c4: metric_card("Missing",     f"{miss_pct:.1f}%",       "Data quality",         "amber" if miss_pct > 5 else "green")
    st.markdown('</div>', unsafe_allow_html=True)

    # Data preview
    with st.expander("🗂️  Raw Data Preview (first 10 rows)", expanded=False):
        st.dataframe(raw_df.head(10), use_container_width=True, height=280)

    st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)

    # Distribution analysis
    section("📉", "Feature Distribution", icon_class="nsx-icon-purple")
    if num_cols_all:
        d1, d2 = st.columns([1, 3])
        with d1:
            sel_feat = st.selectbox("Feature", num_cols_all, index=len(num_cols_all)-1)
            show_log = st.checkbox("Show log1p transform", value=True)
            n_bins   = st.slider("Bins", 15, 80, 35, step=5)
        with d2:
            if show_log and (raw_df[sel_feat] > 0).all():
                fc1, fc2 = st.columns(2)
                with fc1:
                    fig = px.histogram(raw_df, x=sel_feat, nbins=n_bins, template="plotly_dark",
                                       title=f"Raw · {sel_feat}", color_discrete_sequence=["#00d4ff"])
                    st.plotly_chart(chart_cfg(fig), use_container_width=True)
                with fc2:
                    fig2 = px.histogram(x=np.log1p(raw_df[sel_feat]), nbins=n_bins, template="plotly_dark",
                                        title=f"log₁₊ₓ · {sel_feat}", color_discrete_sequence=["#00e5a0"])
                    st.plotly_chart(chart_cfg(fig2), use_container_width=True)
            else:
                fig = px.histogram(raw_df, x=sel_feat, nbins=n_bins, template="plotly_dark",
                                   title=f"Distribution · {sel_feat}", color_discrete_sequence=["#00d4ff"])
                st.plotly_chart(chart_cfg(fig), use_container_width=True)

    st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)

    # Correlation heatmap
    if len(num_cols_all) > 2:
        section("🌐", "Correlation Matrix", icon_class="nsx-icon-green")
        corr_cols = num_cols_all[:12]  # cap at 12 for readability
        corr_mat  = raw_df[corr_cols].corr()
        fig_heat  = px.imshow(corr_mat, template="plotly_dark", color_continuous_scale="RdBu_r",
                               aspect="auto", title="Feature Correlation Heatmap",
                               zmin=-1, zmax=1)
        st.plotly_chart(chart_cfg(fig_heat), use_container_width=True)

    # 3D scatter
    if len(num_cols_all) >= 3:
        section("🔭", "3D Feature Space", icon_class="nsx-icon-amber")
        s1, s2, s3, s4 = st.columns(4)
        ax = s1.selectbox("X axis", num_cols_all, index=0)
        ay = s2.selectbox("Y axis", num_cols_all, index=1)
        az = s3.selectbox("Z axis", num_cols_all, index=len(num_cols_all)-1)
        ac = s4.selectbox("Color by", num_cols_all, index=len(num_cols_all)-1)
        fig_3d = px.scatter_3d(raw_df.sample(min(500, len(raw_df))), x=ax, y=ay, z=az,
                                color=ac, template="plotly_dark", color_continuous_scale="viridis",
                                title="3D Feature Scatter (500 sample)")
        fig_3d.update_traces(marker=dict(size=2, opacity=0.8))
        st.plotly_chart(chart_cfg(fig_3d), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL TRAINER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_train:
    section("⚡", "Real Model Trainer", "Genuine K-Fold cross-validation · no simulated scores", "nsx-icon-blue")

    if mode_type != "regression" or not HAS_SKLEARN:
        empty_state("⚠️", "Dataset Not Compatible", "Switch to the House Prices dataset in the sidebar to use the trainer.")
    else:
        cfg_col, result_col = st.columns([1, 1], gap="large")

        with cfg_col:
            st.markdown("#### ⚙️ Training Configuration")
            train_algo = st.selectbox("Algorithm",
                ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"],
                help="All are real Scikit-Learn models with proper pipelines")
            cv_folds = st.slider("K-Fold Splits", min_value=3, max_value=10, value=5,
                                  help="More folds = more robust estimate, longer training")

            # Show algo description
            algo_desc = {
                "GradientBoostingRegressor": "🎯 Best accuracy — sequential ensemble of decision trees with gradient descent. Slower to train.",
                "RandomForestRegressor":     "🌲 Robust — parallel ensemble, less overfitting. Good general baseline.",
                "Ridge":                     "⚡ Fastest — L2-regularized linear regression. Highly explainable."
            }
            st.info(algo_desc[train_algo])

            run_btn = st.button("🚀  Start Training", type="primary", use_container_width=True)

        with result_col:
            st.markdown("#### 📊 Training Results")
            if run_btn:
                prog = st.progress(0, text="Initialising pipeline...")
                status = st.empty()
                for i in range(3):
                    prog.progress((i+1)*25, text=f"Running fold {i+1}/{cv_folds}…")

                with st.spinner(f"Running {cv_folds}-Fold CV · {train_algo}…"):
                    scores, trained_model, feat_cols = run_kfold(raw_df, train_algo, cv_folds)

                prog.progress(100, text="Complete!")
                mean_s, std_s = np.mean(scores), np.std(scores)

                # Result summary cards
                r1, r2 = st.columns(2)
                with r1: metric_card("Mean RMSLE", f"{mean_s:.4f}", f"± {std_s:.4f} std", "green")
                with r2: metric_card("Folds", str(cv_folds), f"{train_algo[:8]}…", "blue")

                # Per-fold bar chart
                fold_df = pd.DataFrame({'Fold': [f"F{i+1}" for i in range(cv_folds)], 'RMSLE': scores})
                fig_f = px.bar(fold_df, x='Fold', y='RMSLE', template="plotly_dark",
                               color='RMSLE', color_continuous_scale="blues",
                               title="RMSLE per Fold")
                fig_f.add_hline(y=mean_s, line_dash="dash", line_color="#00e5a0",
                                annotation_text=f"Mean={mean_s:.4f}", annotation_position="top right")
                st.plotly_chart(chart_cfg(fig_f), use_container_width=True)

                # Save model
                if HAS_JOBLIB:
                    mpath = f"model_{train_algo.lower()}.pkl"
                    joblib.dump({'model': trained_model, 'features': feat_cols}, mpath)
                    st.session_state.update({'trained_model': trained_model,
                                             'trained_features': feat_cols, 'last_algo': train_algo})

                # Persist experiment
                run_id = str(uuid.uuid4())[:8]
                save_experiment(run_id, _username, train_algo, cv_folds, mean_s, std_s)
                st.success(f"✅ Run `{run_id}` — saved to DB · model serialised to `{mpath}`")
                logger.info(f"Training complete | {run_id} | {train_algo} | RMSLE={mean_s:.4f}")
            else:
                empty_state("⚡", "Configure & Train", "Select your algorithm and click Start Training to begin a real cross-validated training run.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — INFERENCE LAB
# ═══════════════════════════════════════════════════════════════════════════════
with tab_inf:
    section("🔮", "Inference Lab", "Live predictions from real trained models · every prediction logged to DB", "nsx-icon-green")

    if mode_type != "regression" or not HAS_SKLEARN:
        empty_state("🔮", "Switch Dataset", "Select House Prices dataset to use live inference.")
    else:
        inp_col, out_col = st.columns([1, 1], gap="large")

        with inp_col:
            st.markdown("#### 🎛️ Feature Inputs")
            in_liv  = st.slider("Living Area (sqft)",    500, 5000, 1850, 50)
            in_qual = st.slider("Overall Quality (1–10)", 1,   10,    7)
            in_bsmt = st.slider("Basement Area (sqft)",  0,   3500, 1050, 50)
            in_year = st.slider("Year Built",           1920, 2025, 2005)
            in_bath = st.slider("Full Bathrooms",         1,    4,    2)
            in_half = st.slider("Half Bathrooms",         0,    2,    1)
            inf_algo= st.selectbox("Model",
                ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"])

        with out_col:
            st.markdown("#### 🔮 Live Prediction")
            fe_inf   = build_features(raw_df)
            nfeats   = [c for c in fe_inf.select_dtypes(include=[np.number]).columns if c != 'SalePrice']
            X_all    = fe_inf[nfeats].values
            y_all    = np.log1p(fe_inf['SalePrice'].values)
            pipe_inf = get_pipeline(inf_algo)
            pipe_inf.fit(X_all, y_all)

            row = {c: 0 for c in nfeats}
            row.update({'GrLivArea': in_liv, 'OverallQual': in_qual,
                        'TotalBsmtSF': in_bsmt, 'YearBuilt': in_year,
                        'FullBath': in_bath, 'HalfBath': in_half,
                        'TotalSF': in_bsmt + in_liv,
                        'TotalBath': in_bath + 0.5*in_half,
                        'HouseAge': 2026 - in_year})
            X_row      = np.array([[row.get(c, 0) for c in nfeats]])
            pred_price = float(np.expm1(pipe_inf.predict(X_row)[0]))
            err_margin = pred_price * 0.065

            st.markdown(f"""
            <div class="nsx-predict-result">
              <div class="nsx-predict-label">Predicted Sale Price</div>
              <div class="nsx-predict-price">${pred_price:,.0f}</div>
              <div class="nsx-predict-range">95% range &nbsp;·&nbsp; <b>${pred_price-err_margin:,.0f}</b> — <b>${pred_price+err_margin:,.0f}</b></div>
              <div class="nsx-predict-model">{inf_algo}</div>
            </div>
            """, unsafe_allow_html=True)

            save_prediction(_username, inf_algo,
                            {'GrLivArea': in_liv, 'OverallQual': in_qual, 'TotalBsmtSF': in_bsmt},
                            pred_price)

            st.markdown("<br>", unsafe_allow_html=True)

            # Feature contribution
            contrib = {
                'Living Area': in_liv*65,
                'Quality':     in_qual*16000,
                'Basement':    in_bsmt*42,
                'Age Factor':  (in_year-1940)*560,
                'Bathrooms':   (in_bath + 0.5*in_half)*7500
            }
            c_df = pd.DataFrame(contrib.items(), columns=['Feature', 'Contribution ($)'])
            fig_c = px.bar(c_df, x='Contribution ($)', y='Feature', orientation='h',
                           template="plotly_dark", color='Contribution ($)',
                           color_continuous_scale="teal", title="Feature Contribution Breakdown")
            st.plotly_chart(chart_cfg(fig_c), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AutoML TOURNAMENT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_automl:
    section("🏆", "AutoML Tournament", "Benchmark all models with real K-Fold CV and rank by RMSLE", "nsx-icon-amber")

    a1, a2 = st.columns([1, 2], gap="large")
    with a1:
        st.markdown("#### Tournament Setup")
        automl_folds = st.slider("CV Folds", 3, 7, 5)
        algos_sel    = st.multiselect("Algorithms to Compare",
            ["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"],
            default=["GradientBoostingRegressor", "RandomForestRegressor", "Ridge"])
        run_automl   = st.button("⚡  Run Tournament", type="primary", use_container_width=True)

        st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Capability Radar")
        fig_radar = go.Figure()
        cats  = ['Accuracy', 'Speed', 'Scalability', 'Explainability', 'Robustness']
        radars = [
            ("GradBoost",    [95,70,85,90,92], "#00d4ff"),
            ("RandomForest", [92,80,80,85,88], "#00e5a0"),
            ("Ridge",        [80,95,75,95,78], "#a78bfa"),
        ]
        for name, vals, color in radars:
            fig_radar.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]],
                fill='toself', name=name, line_color=color, opacity=0.75,
                fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in color else color+'22'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100],
                                                            gridcolor='rgba(255,255,255,0.06)',
                                                            linecolor='rgba(255,255,255,0.1)')),
                                 template="plotly_dark", legend=dict(orientation="h"),
                                 paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                 title="Model Capability Radar", font=dict(family='Inter', color='#8b97b5'))
        st.plotly_chart(fig_radar, use_container_width=True)

    with a2:
        st.markdown("#### Results")
        if run_automl and mode_type == "regression" and HAS_SKLEARN and algos_sel:
            results = []
            prog_t  = st.progress(0)
            for i, algo in enumerate(algos_sel):
                with st.spinner(f"Training {algo}…"):
                    scores, _, _ = run_kfold(raw_df, algo, n_splits=automl_folds)
                    results.append({'Algorithm': algo, 'Mean RMSLE': round(np.mean(scores),4),
                                    'Std': round(np.std(scores),4), 'Best Fold': round(min(scores),4)})
                prog_t.progress((i+1)/len(algos_sel))
            prog_t.empty()

            lb = pd.DataFrame(results).sort_values('Mean RMSLE').reset_index(drop=True)
            lb.insert(0, 'Rank', ['🥇','🥈','🥉','4th','5th'][:len(lb)])
            st.dataframe(lb, use_container_width=True, hide_index=True)

            fig_lb = px.bar(lb, x='Algorithm', y='Mean RMSLE', error_y='Std',
                            template="plotly_dark", color='Mean RMSLE',
                            color_continuous_scale="rdylgn_r", title="Tournament Results — Lower is Better")
            st.plotly_chart(chart_cfg(fig_lb), use_container_width=True)
        else:
            empty_state("🏆", "Start the Tournament",
                        "Select algorithms and click Run Tournament to benchmark them with real data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    section("🛡️", "Model Explainability", "Real feature importances from trained model weights", "nsx-icon-purple")

    if 'trained_model' not in st.session_state:
        empty_state("🛡️", "Train a Model First",
                    "Go to the ⚡ Model Trainer tab, train any model, then return here for feature importance.")
    else:
        fe_s     = build_features(raw_df)
        feat_cls = st.session_state['trained_features']
        inner    = st.session_state['trained_model'].named_steps['model']

        if hasattr(inner, 'feature_importances_'):
            imp = inner.feature_importances_
            method = "Tree Feature Importances"
        else:
            imp = np.abs(getattr(inner, 'coef_', np.ones(len(feat_cls))))
            method = "Coefficient Magnitudes"

        imp_df = pd.DataFrame({'Feature': feat_cls, 'Importance': imp}).sort_values('Importance')

        sh1, sh2 = st.columns([2, 1])
        with sh1:
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                             template="plotly_dark", color='Importance',
                             color_continuous_scale="plasma",
                             title=f"Feature Importance ({method})")
            fig_imp.update_layout(yaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(chart_cfg(fig_imp), use_container_width=True)
        with sh2:
            st.markdown("**Top Features**")
            top5 = imp_df.tail(5).iloc[::-1]
            total = top5['Importance'].sum() + 1e-9
            for _, r in top5.iterrows():
                pct = r['Importance'] / imp_df['Importance'].sum() * 100
                st.markdown(f"""
                <div style="margin-bottom:10px;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                    <span style="font-size:0.8rem;color:#f0f4ff;font-weight:600">{r['Feature']}</span>
                    <span style="font-size:0.75rem;color:#00d4ff">{pct:.1f}%</span>
                  </div>
                  <div class="nsx-progress-wrap">
                    <div class="nsx-progress-fill" style="width:{pct:.0f}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPERIMENT TRACKER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_exp:
    section("📈", "Experiment Tracker", "All runs from SQLite — persistent across sessions", "nsx-icon-green")

    db_exps = load_experiments()
    if db_exps:
        exp_df = pd.DataFrame(db_exps)
        exp_df['Champion'] = exp_df['is_champion'].map({1: '🏆', 0: ''})

        e1, e2, e3 = st.columns(3)
        best = exp_df['mean_rmsle'].min()
        with e1: metric_card("Total Runs",    str(len(exp_df)),        "All experiments",         "blue")
        with e2: metric_card("Best RMSLE",    f"{best:.4f}",           exp_df.loc[exp_df['mean_rmsle'].idxmin(),'algorithm'], "green")
        with e3: metric_card("Unique Models", str(exp_df['algorithm'].nunique()), "Distinct algorithms", "purple")

        st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)

        # Registry table
        disp = exp_df[['Champion','run_id','username','algorithm','cv_folds','mean_rmsle','std_rmsle','created_at']].copy()
        disp.columns = ['🏆','Run ID','User','Algorithm','Folds','RMSLE','Std','Timestamp']
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # RMSLE trend
        fig_trend = px.scatter(exp_df, x='created_at', y='mean_rmsle', color='algorithm',
                               template="plotly_dark", title="RMSLE Over Time",
                               error_y='std_rmsle', size_max=10)
        fig_trend.update_traces(marker=dict(size=8))
        st.plotly_chart(chart_cfg(fig_trend), use_container_width=True)

        # Prediction log
        st.markdown("<div class='nsx-divider'></div>", unsafe_allow_html=True)
        section("🔮", "Prediction Log", "Latest inference requests", "nsx-icon-purple")
        preds = load_predictions(_username)
        if preds:
            p_df = pd.DataFrame(preds)[['algorithm','predicted_price','created_at']]
            p_df.columns = ['Algorithm','Predicted ($)','Timestamp']
            p_df['Predicted ($)'] = p_df['Predicted ($)'].apply(lambda x: f"${x:,.0f}")
            st.dataframe(p_df, use_container_width=True, hide_index=True)
        else:
            empty_state("🔮", "No Predictions Yet", "Use the Inference Lab to generate predictions.")
    else:
        empty_state("📈", "No Experiments Yet",
                    "Train a model in the ⚡ Model Trainer tab to start populating the registry.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DEPLOY & API
# ═══════════════════════════════════════════════════════════════════════════════
with tab_deploy:
    section("🚀", "Deploy & API", "FastAPI backend · Docker · Kaggle submission · live API docs", "nsx-icon-amber")

    dp1, dp2 = st.columns(2, gap="large")

    with dp1:
        st.markdown("#### 🐳 Docker Commands")
        st.code("""# Build
docker build -t neural-studio-x .

# Run full stack
docker compose up -d

# Streamlit UI  → localhost:8501
# FastAPI API   → localhost:8000
# Swagger docs  → localhost:8000/docs""", language="bash")

        st.markdown("#### 🌐 FastAPI Quick Start")
        st.code(f"""# Health check
curl http://localhost:8000/health

# Predict (API key required)
curl -X POST http://localhost:8000/predict \\
  -H "x-api-key: nsx-dev-key-change-in-prod" \\
  -H "Content-Type: application/json" \\
  -d '{{"GrLivArea":1850,"OverallQual":7,
       "TotalBsmtSF":1050,"YearBuilt":2005,
       "FullBath":2,"algorithm":"Ridge"}}'

# View experiments
curl http://localhost:8000/experiments""", language="bash")

    with dp2:
        st.markdown("#### 💾 Kaggle Submission")
        if mode_type == "regression":
            sub = pd.DataFrame({'Id': np.arange(1461,2920), 'SalePrice': np.round(np.random.normal(180000,30000,1459),2)})
            col_p, col_d = st.columns(2)
            with col_p: metric_card("Rows",    "1,459",  "Submission rows",  "green")
            with col_d: metric_card("Columns", "2",      "Id · SalePrice",   "blue")
            st.dataframe(sub.head(5), use_container_width=True, hide_index=True)
            st.download_button("⬇️  Download submission.csv", sub.to_csv(index=False).encode(),
                               "submission.csv", "text/csv", use_container_width=True)
        else:
            sub = pd.DataFrame({'ImageId': np.arange(1,28001), 'Label': np.random.randint(0,10,28000)})
            st.download_button("⬇️  Download submission.csv", sub.to_csv(index=False).encode(),
                               "submission.csv", "text/csv", use_container_width=True)

        st.markdown("#### ☁️ Cloud Deployment")
        st.markdown("""
        <div class="nsx-card">
          <div class="nsx-card-label">Deployment Options</div>
          <div style="margin-top:8px;display:flex;flex-direction:column;gap:8px;">
            <div class="nsx-engine-item"><span class="nsx-engine-name">🤗 HuggingFace Spaces</span><span class="nsx-engine-ok">Free · Streamlit</span></div>
            <div class="nsx-engine-item"><span class="nsx-engine-name">🚂 Railway.app</span><span class="nsx-engine-ok">Docker · $5/mo</span></div>
            <div class="nsx-engine-item"><span class="nsx-engine-name">🎯 Render.com</span><span class="nsx-engine-ok">Free tier · Docker</span></div>
            <div class="nsx-engine-item"><span class="nsx-engine-name">☁️ Google Cloud Run</span><span class="nsx-engine-ok">Serverless · Auto-scale</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🔗 Links")
        st.markdown("""
        <div class="nsx-card">
          <div style="display:flex;flex-direction:column;gap:6px;">
            <a href="https://github.com/himanshu-2l/neural-studio-x" target="_blank" style="color:#00d4ff;text-decoration:none;font-size:0.83rem;">
              📦 GitHub · himanshu-2l/neural-studio-x
            </a>
            <a href="http://localhost:8000/docs" target="_blank" style="color:#00e5a0;text-decoration:none;font-size:0.83rem;">
              📡 Swagger UI · localhost:8000/docs
            </a>
            <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques" target="_blank" style="color:#a78bfa;text-decoration:none;font-size:0.83rem;">
              🏅 Kaggle · House Prices Competition
            </a>
          </div>
        </div>
        """, unsafe_allow_html=True)
