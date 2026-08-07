# 🧠 Neural Studio X (v3.0) — Commercial-Grade AI & Data Science Web Suite

[![GitHub Repo](https://img.shields.io/badge/GitHub-neural--studio--x-00f2fe?logo=github)](https://github.com/himanshu-2l/neural-studio-x)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b?logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e?logo=scikit-learn)](https://scikit-learn.org)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker)](https://www.docker.com)

**Neural Studio X** is an interactive, full-stack Data Science & Machine Learning Platform built with **Streamlit**, **PyTorch**, **Plotly**, **Scikit-Learn**, and **Pandas**.

It provides automated 3D Exploratory Data Analysis (EDA), interactive PyTorch Neural Networks, real-time in-browser model training, AutoML algorithm tournaments, live inference playgrounds, MLflow-style experiment tracking, and 1-click Kaggle submission validation.

---

## 🌟 Key Features & Modules

```
                    ┌──────────────────────────────────────────────┐
                    │            NEURAL STUDIO X (v3.0)            │
                    └──────────────────────┬───────────────────────┘
                                           │
 ┌─────────────────┬─────────────────┬─────┴─────┬─────────────────┬─────────────────┐
 │                 │                 │           │                 │                 │
┌▼────────┐   ┌────▼────┐       ┌────▼────┐ ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
│   EDA   │   │ CLEANER │       │ INFERENCE│ │ PyTorch │       │ AutoML  │       │ EXPERIMENT│
│ & 3D    │   │ & IQR   │       │ PAD     │ │ CANVAS  │       │ RADAR   │       │ TRACKER │
└─────────┘   └─────────┘       └─────────┘ └─────────┘       └─────────┘       └─────────┘
```

### 📊 1. Automated EDA & 3D Analytics Engine
- **Universal Dataset Support**: Built-in profiler for Kaggle House Prices, Digit Recognizer, and custom uploaded CSV files.
- **Target Distribution Normalization**: Interactive comparison of raw target distributions vs log-transformed $\log(1 + x)$ (`np.log1p`).
- **Interactive 3D Scatter Plot**: Dynamic 3D Plotly visualizer mapping living area, basement square footage, overall quality, and target variables.

### ⚙️ 2. Intelligent Feature Engineering Workshop
- **Domain Feature Generation**: Interactive creation of combined square footage (`TotalSF`), total bathrooms (`TotalBath`), and property age (`HouseAge`).
- **Interactive Feature Creator**: Select any two numeric columns to generate custom arithmetic interactions (`+`, `-`, `*`, `/`).

### 🧹 3. Automated Data Cleaner & Outlier Sanitizer
- **IQR Statistical Outlier Detection**: Dynamic threshold slider to inspect and flag statistical anomalies outside calculated bounds.
- **Live Outlier Clipping**: 1-click button to sanitize extreme values in numeric features.
- **Before vs After Distribution Boxplots**: Interactive Plotly boxplot comparison showing distribution normalization.

### 🔮 4. Live Model Inference Playground ("Predictor Pad")
- **Real-Time Input Controls**: Adjust sliders for Living Area, Quality, Basement SF, Year Built, and Bathroom count.
- **Dynamic Price Prediction**: Generates real-time model predictions with confidence ranges (e.g., `$214,500 ± $12,500`).
- **Feature Contribution Waterfall**: Visual breakdown showing estimated price impact per feature.

### 🎨 5. PyTorch Vision Lab 2.0 (Interactive Canvas & Softmax Inspector)
- **HTML5 Mouse Drawing Canvas**: Draw any handwritten digit (0–9) directly in your browser with your mouse.
- **Softmax Prediction Confidence Visualizer**: Live probability distribution bar charts across digit classes (0–9).

### ⚡ 6. Live In-Browser Model Trainer & Cross-Validation
- **K-Fold CV Execution**: Run K-Fold Cross-Validation live in your browser and track fold-by-fold validation RMSLE/Accuracy scores.

### 🏆 7. AutoML Algorithm Tournament & Radar Profiler
- **Multi-Algorithm Tournament**: Compare **Gradient Boosting**, **Random Forest**, **ExtraTrees**, **PyTorch CNN**, and **Ridge Regression**.
- **Multi-Metric Polar Radar Chart**: Plotly Radar chart comparing models across Accuracy, Training Speed, Scalability, Explainability, and Robustness.

### 🛡️ 8. SHAP Model Explainability & Feature Impact
- **White-Box Transparency**: Global SHAP feature importance rankings demonstrating positive and negative impacts on predictions.

### 📈 9. Experiment Tracking & Model Registry (MLflow-Style)
- **Model Run History**: Logs historical runs, architectures, hyperparameter configurations, and tags the **"🏆 CHAMPION MODEL"**.

### 🚀 10. Kaggle Submission Generator & REST API Export
- **Compliance Engine**: Enforces exact row count compliance (`1459` test rows for House Prices, `1000` for Digit Recognizer).
- **FastAPI Code Generator**: Auto-generates production backend code to serve trained models via REST API endpoints.

---

## 📁 Repository Structure

```
neural-studio-x/
├── app.py              # Main Multi-Module Streamlit Application Suite
├── requirements.txt    # Application Dependencies
├── Dockerfile          # Production Containerization Config
├── README.md           # Project Documentation & Architecture Guide
└── .gitignore          # Git Ignore Rules
```

---

## 🚀 Local Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/himanshu-2l/neural-studio-x.git
   cd neural-studio-x
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser!

---

## 🐳 Docker Deployment

Run Neural Studio X in a containerized environment:

```bash
# Build Docker image
docker build -t neural-studio-x .

# Run container
docker run -p 8501:8501 neural-studio-x
```

---

## 👤 Author & Maintainer

Developed & maintained by **Himanshu** ([@himanshu-2l](https://github.com/himanshu-2l)).
