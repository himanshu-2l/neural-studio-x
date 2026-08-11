"""
Tests for the FastAPI backend endpoints.
Run with: pytest tests/test_api.py -v
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

# Patch DB path before importing app
import database
from pathlib import Path
database.DB_PATH = Path(__file__).parent / "test_studio_x.db"
database.init_db()

from api import app, API_KEY

client = TestClient(app)
HEADERS = {"x-api-key": API_KEY}


class TestHealth:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "db_stats" in body


class TestAuth:
    def test_predict_without_api_key_returns_422_or_401(self):
        r = client.post("/predict", json={
            "GrLivArea": 1500, "OverallQual": 7, "TotalBsmtSF": 800,
            "YearBuilt": 2005, "FullBath": 2
        })
        assert r.status_code in (401, 422)

    def test_predict_with_wrong_key_returns_401(self):
        r = client.post("/predict",
            json={"GrLivArea": 1500, "OverallQual": 7, "TotalBsmtSF": 800,
                  "YearBuilt": 2005, "FullBath": 2},
            headers={"x-api-key": "wrong-key"}
        )
        assert r.status_code == 401


class TestTrain:
    def test_train_ridge_returns_200(self):
        r = client.post("/train",
            json={"algorithm": "Ridge", "cv_folds": 3, "username": "test"},
            headers=HEADERS
        )
        assert r.status_code == 200

    def test_train_response_schema(self):
        r = client.post("/train",
            json={"algorithm": "Ridge", "cv_folds": 3, "username": "test"},
            headers=HEADERS
        )
        body = r.json()
        assert "run_id"     in body
        assert "mean_rmsle" in body
        assert "std_rmsle"  in body
        assert body["mean_rmsle"] > 0

    def test_train_invalid_algo_returns_422(self):
        r = client.post("/train",
            json={"algorithm": "SVM", "cv_folds": 3},
            headers=HEADERS
        )
        assert r.status_code == 422


class TestPredict:
    def test_predict_after_training(self):
        # Train first
        client.post("/train",
            json={"algorithm": "Ridge", "cv_folds": 3, "username": "test"},
            headers=HEADERS
        )
        # Then predict
        r = client.post("/predict",
            json={"GrLivArea": 1850, "OverallQual": 7, "TotalBsmtSF": 1050,
                  "YearBuilt": 2005, "FullBath": 2, "HalfBath": 1,
                  "algorithm": "Ridge", "username": "test"},
            headers=HEADERS
        )
        assert r.status_code == 200
        body = r.json()
        assert body["predicted_price"] > 50000
        assert body["lower_bound"] < body["predicted_price"] < body["upper_bound"]

    def test_predict_validates_quality_range(self):
        r = client.post("/predict",
            json={"GrLivArea": 1500, "OverallQual": 99, "TotalBsmtSF": 800,
                  "YearBuilt": 2005, "FullBath": 2},
            headers=HEADERS
        )
        assert r.status_code == 422


class TestExperiments:
    def test_experiments_endpoint_returns_list(self):
        r = client.get("/experiments")
        assert r.status_code == 200
        body = r.json()
        assert "experiments" in body
        assert isinstance(body["experiments"], list)


class TestModels:
    def test_models_endpoint_returns_list(self):
        r = client.get("/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert isinstance(body["models"], list)
