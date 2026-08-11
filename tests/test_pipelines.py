import pytest
import numpy as np
import pandas as pd
import sys, os

# Make sure app module can be imported for utility functions
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_sample_house_df(n=100):
    np.random.seed(0)
    return pd.DataFrame({
        'GrLivArea':   np.random.randint(800, 3000, size=n),
        'OverallQual': np.random.randint(1, 10, size=n),
        'TotalBsmtSF': np.random.randint(0, 2000, size=n),
        'YearBuilt':   np.random.randint(1950, 2020, size=n),
        'FullBath':    np.random.randint(1, 3, size=n),
        'HalfBath':    np.random.randint(0, 2, size=n),
        'Neighborhood':np.random.choice(['A','B','C'], size=n),
        'SalePrice':   np.random.normal(180000, 30000, size=n)
    })


class TestBuildFeatures:
    def test_total_sf_created(self):
        from app import build_features
        df   = get_sample_house_df()
        out  = build_features(df)
        assert 'TotalSF' in out.columns

    def test_total_bath_created(self):
        from app import build_features
        df  = get_sample_house_df()
        out = build_features(df)
        assert 'TotalBath' in out.columns

    def test_house_age_created(self):
        from app import build_features
        df  = get_sample_house_df()
        out = build_features(df)
        assert 'HouseAge' in out.columns
        assert (out['HouseAge'] >= 0).all(), "HouseAge should not be negative"

    def test_total_sf_values(self):
        from app import build_features
        df  = get_sample_house_df()
        out = build_features(df)
        expected = df['TotalBsmtSF'] + df['GrLivArea']
        pd.testing.assert_series_equal(out['TotalSF'].reset_index(drop=True),
                                       expected.reset_index(drop=True), check_names=False)


class TestGetPipeline:
    def test_gradient_boosting_pipeline(self):
        from app import get_pipeline
        pipe = get_pipeline("GradientBoostingRegressor")
        assert pipe is not None
        assert 'model' in pipe.named_steps

    def test_random_forest_pipeline(self):
        from app import get_pipeline
        pipe = get_pipeline("RandomForestRegressor")
        assert pipe is not None

    def test_ridge_pipeline(self):
        from app import get_pipeline
        pipe = get_pipeline("Ridge")
        assert pipe is not None

    def test_pipeline_fit_predict(self):
        from app import get_pipeline, build_features
        df    = get_sample_house_df(50)
        fe_df = build_features(df)
        cols  = [c for c in fe_df.select_dtypes(include=[np.number]).columns if c != 'SalePrice']
        X = fe_df[cols].values
        y = np.log1p(fe_df['SalePrice'].values)
        pipe = get_pipeline("Ridge")
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert not np.any(np.isnan(preds))


class TestKFold:
    def test_kfold_returns_correct_fold_count(self):
        from app import run_kfold
        df     = get_sample_house_df(100)
        scores, model, feats = run_kfold(df, "Ridge", n_splits=3)
        assert len(scores) == 3

    def test_kfold_scores_are_positive(self):
        from app import run_kfold
        df     = get_sample_house_df(100)
        scores, _, _ = run_kfold(df, "Ridge", n_splits=3)
        assert all(s >= 0 for s in scores)

    def test_kfold_returns_trained_model(self):
        from app import run_kfold
        df = get_sample_house_df(100)
        _, model, _ = run_kfold(df, "Ridge", n_splits=3)
        assert model is not None
        assert hasattr(model, 'predict')
