import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def generate_synthetic_house_data(n_samples=600):
    np.random.seed(42)
    gr_liv_area = np.random.randint(800, 3500, size=n_samples)
    overall_qual = np.random.randint(1, 10, size=n_samples)
    total_bsmt_sf = np.random.randint(0, 2000, size=n_samples)
    year_built = np.random.randint(1950, 2021, size=n_samples)
    neighborhood = np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], size=n_samples)
    
    noise = np.random.normal(0, 15000, size=n_samples)
    sale_price = 30000 + (gr_liv_area * 60) + (overall_qual * 15000) + (total_bsmt_sf * 40) + ((year_built - 1950) * 500) + noise
    sale_price = np.maximum(sale_price, 50000)
    
    df = pd.DataFrame({
        'Id': np.arange(1, n_samples + 1),
        'MSSubClass': np.random.choice([20, 60, 70, 120], size=n_samples),
        'Neighborhood': neighborhood,
        'OverallQual': overall_qual,
        'YearBuilt': year_built,
        'TotalBsmtSF': total_bsmt_sf,
        'GrLivArea': gr_liv_area,
        'FullBath': np.random.randint(1, 4, size=n_samples),
        'HalfBath': np.random.randint(0, 2, size=n_samples),
        'GarageCars': np.random.randint(0, 4, size=n_samples),
        'SalePrice': sale_price
    })
    return df

print("1. Loading or generating data...")
train_path = 'train.csv'
test_path = 'test.csv'

if os.path.exists('../input/house-prices-advanced-regression-techniques/train.csv'):
    train_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
    test_df = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
    print("Loaded Kaggle dataset.")
elif os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Loaded local train.csv and test.csv.")
else:
    print("Generating demo synthetic data...")
    raw_df = generate_synthetic_house_data(600)
    train_df = raw_df.iloc[:400].copy()
    test_df = raw_df.iloc[400:].copy().drop(columns=['SalePrice'])

def engineer_features(df):
    df = df.copy()
    bsmt = df['TotalBsmtSF'] if 'TotalBsmtSF' in df.columns else 0
    gr_liv = df['GrLivArea'] if 'GrLivArea' in df.columns else 0
    df['TotalSF'] = bsmt + gr_liv
    full_bath = df['FullBath'] if 'FullBath' in df.columns else 0
    half_bath = df['HalfBath'] if 'HalfBath' in df.columns else 0
    df['TotalBath'] = full_bath + (0.5 * half_bath)
    if 'YearBuilt' in df.columns:
        df['HouseAge'] = 2026 - df['YearBuilt']
    return df

print("2. Feature engineering & pipeline setup...")
train_fe = engineer_features(train_df)
test_fe = engineer_features(test_df)

y_train_log = np.log1p(train_fe['SalePrice'])
X_train = train_fe.drop(columns=['Id', 'SalePrice'], errors='ignore')
X_test = test_fe.drop(columns=['Id', 'SalePrice'], errors='ignore')

num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

print("3. Training Gradient Boosting model...")
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
pipeline.fit(X_train, y_train_log)

print("4. Generating predictions & submission.csv...")
test_log_preds = pipeline.predict(X_test)
final_preds = np.expm1(test_log_preds)

sub_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': final_preds
})

sub_df.to_csv('submission.csv', index=False)
print("SUCCESS: submission.csv created with shape", sub_df.shape)
