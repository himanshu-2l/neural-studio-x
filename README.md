# 🏠 House Prices - Advanced Regression Techniques (Kaggle)

Welcome to the Kaggle **House Prices: Advanced Regression Techniques** repository! This project contains a complete end-to-end Machine Learning pipeline for predicting residential property sales prices in Ames, Iowa.

---

## 📁 Repository Structure

```
.
├── house_prices_starter.ipynb   # Complete Kaggle Jupyter Notebook (EDA, ML, Visualizations)
├── generate_submission.py       # Standalone Python script to generate submission.csv
├── submission.csv               # Ready-to-upload formatted Kaggle prediction file
└── README.md                    # Project documentation & instructions
```

---

## 🎯 Machine Learning Workflow

1. **Target Normalization**: Raw `SalePrice` values are right-skewed. Taking $\log(1 + x)$ (`np.log1p`) normalizes variance to optimize Root Mean Squared Logarithmic Error (RMSLE).
2. **Feature Engineering**:
   - `TotalSF`: Sum of basement, 1st floor, and 2nd floor square footage.
   - `TotalBath`: Sum of full bathrooms + 0.5 * half bathrooms.
   - `HouseAge`: Age of property at sales time ($2026 - \text{YearBuilt}$).
3. **Preprocessing Pipeline**:
   - Numerical: Median Imputation + Standard Scaling.
   - Categorical: Constant Imputation + One-Hot Encoding (`handle_unknown='ignore'`).
4. **Model Comparison**: 5-Fold Cross Validation comparing **Ridge Regression**, **Random Forest**, and **Gradient Boosting / LightGBM**.

---

## 🚀 How to Use

### 1. Running on Kaggle
1. Go to the [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
2. Click **Code** $\to$ **New Notebook**.
3. Upload `house_prices_starter.ipynb` via **File $\to$ Upload Notebook**.
4. Click **Run All** $\to$ **Save Version** $\to$ Submit `submission.csv`!

### 2. Running Locally
Generate the submission file locally by running:
```bash
python generate_submission.py
```

### 3. Submitting via Kaggle CLI
```bash
kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "Baseline Gradient Boosting Model"
```

---

## 📊 Evaluation Metric
Submissions are evaluated on **Root Mean Squared Logarithmic Error (RMSLE)** between predicted and actual log house prices.
