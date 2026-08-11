# ─────────────────────────────────────────────────────────────
# Neural Studio X — ML Utilities (ml_utils.py)
# Pure ML logic with NO Streamlit dependencies.
# Imported by app.py, api.py, and all tests.
# ─────────────────────────────────────────────────────────────
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("neural_studio_x.ml")

try:
    import joblib
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_log_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Feature Engineering ────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Real feature engineering — no Streamlit dependency."""
    out = df.copy()
    if 'TotalBsmtSF' in out.columns and 'GrLivArea' in out.columns:
        out['TotalSF']   = out['TotalBsmtSF'] + out['GrLivArea']
    if 'FullBath' in out.columns:
        half = out['HalfBath'] if 'HalfBath' in out.columns else pd.Series(np.zeros(len(out)), index=out.index)
        out['TotalBath'] = out['FullBath'] + 0.5 * half
    if 'YearBuilt' in out.columns:
        out['HouseAge']  = 2026 - out['YearBuilt']
    return out


# ── Pipeline Builder ───────────────────────────────────────────
def get_pipeline(algo_name: str, random_state: int = 42) -> "Pipeline":
    """Return a real Scikit-Learn Pipeline for the given algorithm name."""
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is not installed.")

    if algo_name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08,
            max_depth=4, random_state=random_state
        )
    elif algo_name == "RandomForestRegressor":
        model = RandomForestRegressor(
            n_estimators=150, max_depth=14, random_state=random_state
        )
    else:
        model = Ridge(alpha=1.0)

    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model)
    ])


# ── K-Fold Cross-Validation ────────────────────────────────────
def run_kfold(df: pd.DataFrame, algo_name: str, n_splits: int = 5):
    """
    Real K-Fold CV on the house prices dataset.
    Returns (fold_scores, trained_pipeline, numeric_feature_names).
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is not installed.")

    fe_df    = build_features(df)
    num_cols = [c for c in fe_df.select_dtypes(include=[np.number]).columns if c != 'SalePrice']
    X        = fe_df[num_cols].values
    y        = np.log1p(fe_df['SalePrice'].values)

    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    model  = None

    for fold, (tr, val) in enumerate(kf.split(X)):
        pipe = get_pipeline(algo_name)
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[val])
        rmsle = float(np.sqrt(mean_squared_log_error(
            np.expm1(y[val]), np.expm1(preds)
        )))
        scores.append(rmsle)
        model = pipe
        logger.info(f"Fold {fold+1}/{n_splits} | {algo_name} | RMSLE={rmsle:.4f}")

    return scores, model, num_cols


# ── Synthetic Training Data ────────────────────────────────────
def get_house_data_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Generate reproducible synthetic house prices data."""
    np.random.seed(seed)
    gr_liv = np.random.randint(600, 4000, size=n)
    qual   = np.random.randint(1, 11, size=n)
    bsmt   = np.random.randint(0, 2500, size=n)
    year   = np.random.randint(1940, 2023, size=n)
    fbath  = np.random.randint(1, 4, size=n)
    hbath  = np.random.randint(0, 2, size=n)
    neigh  = np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], size=n)
    price  = np.maximum(
        30000 + gr_liv*65 + qual*16000 + bsmt*45 + (year-1940)*560
        + fbath*7500 + np.random.normal(0, 11000, n),
        50000
    )
    return pd.DataFrame({
        'GrLivArea': gr_liv, 'OverallQual': qual, 'TotalBsmtSF': bsmt,
        'YearBuilt': year,   'FullBath': fbath,   'HalfBath': hbath,
        'Neighborhood': neigh, 'SalePrice': price
    })
