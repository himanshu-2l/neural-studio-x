# 🧠 Neural Studio X — Kaggle AI & Data Science Web Suite

**Neural Studio X** is an interactive, full-stack Data Science & Machine Learning Web Application built with **Streamlit**, **PyTorch**, **Plotly**, **Scikit-Learn**, and **Pandas**. 

It provides automated Exploratory Data Analysis (EDA), real-time PyTorch Neural Network feature map inspection, interactive model training arenas, and 1-click Kaggle submission validation.

---

## 🌟 Key Features & Modules

### 📊 1. Automated EDA & 3D Analytics Engine
- **Universal Dataset Support**: Built-in profiler for Kaggle House Prices, Digit Recognizer, and custom uploaded CSV files.
- **Target Distribution Normalization**: Interactive comparison of raw target distributions vs log-transformed $\log(1 + x)$ (`np.log1p`).
- **Interactive 3D Scatter & Correlation Heatmaps**: Dynamic 3D Plotly visualizer mapping living area, basement square footage, overall quality, and sales price.

### ⚙️ 2. Intelligent Feature Engineering Workshop
- **Domain Feature Generation**: Interactive creation of combined square footage (`TotalSF`), total bathrooms (`TotalBath`), and property age (`HouseAge`).
- **Correlation Analytics**: Dynamic bar charts ranking engineered feature correlations with target metrics.

### 🧠 3. PyTorch Neural Lab
- **Interactive Grayscale Image Inspector**: View 28x28 grayscale handwritten digit samples.
- **Softmax Prediction Confidence Visualizer**: Live probability distribution bar charts across digit classes (0–9).
- **PyTorch Architecture Overview**: Layer-by-layer breakdown of `Conv2D` $\to$ `BatchNorm` $\to$ `MaxPool` $\to$ `Dropout` $\to$ `Linear` neural networks.

### 🏆 4. Model Training Arena & Hyperparameter Tuning
- **Model Comparison**: Evaluate Ridge Regression, Random Forest, Gradient Boosting, and PyTorch Neural Networks.
- **Interactive Tuning Sliders**: Adjust Learning Rates ($0.0001 - 0.05$), Number of Trees ($50 - 300$), and Dropout probabilities live.
- **Real-Time Training History**: Live Plotly loss & validation accuracy curves.

### 🚀 5. Automated Kaggle Submission Generator
- **Compliance & Assertion Engine**: Enforces exact row count compliance (`1459` test rows for House Prices, `1000` for Digit Recognizer) and non-null data assertions.
- **One-Click Browser Export**: Download formatted `submission.csv` files directly from the web app interface.

---

## 📁 Repository Structure

```
neural-studio-x/
├── app.py              # Main Multi-Module Streamlit Application
├── requirements.txt    # Application Dependencies
├── README.md           # Project Overview & Architecture Guide
└── .gitignore          # Git Ignore Configuration
```

---

## 🚀 Local Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/himanshu-2l/neural-studio-x.git
   cd neural-studio-x
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser!
