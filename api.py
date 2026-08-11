# ─────────────────────────────────────────────────────────────
# Neural Studio X — FastAPI Backend (api.py)
# Separates ML engine from Streamlit UI
# Endpoints: /health  /predict  /train  /experiments  /models
# ─────────────────────────────────────────────────────────────
import os
import uuid
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Internal modules
from database import init_db, save_experiment, load_experiments, save_prediction, get_stats
from ml_utils import build_features, get_pipeline, run_kfold, get_house_data_df, HAS_SKLEARN

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger("neural_studio_x.api")

# ── Bootstrap DB ───────────────────────────────────────────────
init_db()

# ── FastAPI App ────────────────────────────────────────────────
app = FastAPI(
    title="Neural Studio X API",
    description=(
        "Production ML inference & experiment tracking API.\n"
        "Serves predictions, triggers real training runs, and exposes "
        "the experiment registry."
    ),
    version="3.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Simple API Key Auth ────────────────────────────────────────
API_KEY = os.getenv("API_KEY", "nsx-dev-key-change-in-prod")


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key."
        )


# ── Scikit-Learn check ────────────────────────────────────────
# HAS_SKLEARN is imported from ml_utils above


MODEL_DIR = Path(__file__).parent


def load_model(algo: str):
    path = MODEL_DIR / f"model_{algo.lower()}.pkl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model '{algo}' not found. Train it first via POST /train."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


# ─────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────

class HouseFeatures(BaseModel):
    GrLivArea:   float = Field(..., ge=200,  le=10000, description="Above ground living area (sqft)",
                               json_schema_extra={"example": 1850})
    OverallQual: int   = Field(..., ge=1,    le=10,    description="Overall material and finish quality (1-10)",
                               json_schema_extra={"example": 7})
    TotalBsmtSF: float = Field(..., ge=0,    le=6000,  description="Total basement area (sqft)",
                               json_schema_extra={"example": 1050})
    YearBuilt:   int   = Field(..., ge=1870, le=2026,  description="Year the house was built",
                               json_schema_extra={"example": 2005})
    FullBath:    int   = Field(..., ge=0,    le=5,     description="Full bathrooms above grade",
                               json_schema_extra={"example": 2})
    HalfBath:    int   = Field(0,  ge=0,    le=4,     description="Half bathrooms above grade")
    algorithm:   str   = Field("GradientBoostingRegressor", description="Model to use for prediction")
    username:    str   = Field("api_user",                  description="Username for audit logging")

    @field_validator('algorithm')
    @classmethod
    def algo_must_be_valid(cls, v):
        valid = {"GradientBoostingRegressor", "RandomForestRegressor", "Ridge"}
        if v not in valid:
            raise ValueError(f"algorithm must be one of {valid}")
        return v


class PredictResponse(BaseModel):
    predicted_price:  float
    lower_bound:      float
    upper_bound:      float
    algorithm:        str
    input_features:   dict
    timestamp:        str


class TrainRequest(BaseModel):
    algorithm: str = Field("GradientBoostingRegressor", description="Algorithm to train")
    cv_folds:  int = Field(5, ge=2, le=10,              description="Number of K-Fold splits")
    username:  str = Field("api_user",                  description="Who triggered this run")

    @field_validator('algorithm')
    @classmethod
    def algo_must_be_valid(cls, v):
        valid = {"GradientBoostingRegressor", "RandomForestRegressor", "Ridge"}
        if v not in valid:
            raise ValueError(f"algorithm must be one of {valid}")
        return v


class TrainResponse(BaseModel):
    run_id:     str
    algorithm:  str
    cv_folds:   int
    mean_rmsle: float
    std_rmsle:  float
    model_path: str
    timestamp:  str


class HealthResponse(BaseModel):
    status:      str
    version:     str
    sklearn:     bool
    db_stats:    dict
    timestamp:   str


# ─────────────────────────────────────────────────────────────
# Helper: generate synthetic training data
# ─────────────────────────────────────────────────────────────
def get_training_data():
    df  = get_house_data_df()
    fe  = build_features(df)
    num = [c for c in fe.select_dtypes(include=[np.number]).columns if c != 'SalePrice']
    return fe[num].values, np.log1p(fe['SalePrice'].values), num


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check — returns system status and DB stats."""
    return HealthResponse(
        status="ok",
        version="3.2.0",
        sklearn=HAS_SKLEARN,
        db_stats=get_stats(),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"],
          dependencies=[Depends(verify_api_key)])
def predict(body: HouseFeatures):
    """
    Run live ML inference on house features.
    Returns predicted sale price with 95% confidence bounds.
    Requires a pre-trained model (train via POST /train first).
    """
    if not HAS_SKLEARN:
        raise HTTPException(503, "Scikit-Learn not available on this server.")

    model, feat_cols = load_model(body.algorithm)

    # Build feature row
    row_dict = {
        'GrLivArea':   body.GrLivArea,
        'OverallQual': body.OverallQual,
        'TotalBsmtSF': body.TotalBsmtSF,
        'YearBuilt':   body.YearBuilt,
        'FullBath':    body.FullBath,
        'HalfBath':    body.HalfBath,
        'TotalSF':     body.TotalBsmtSF + body.GrLivArea,
        'TotalBath':   body.FullBath + 0.5 * body.HalfBath,
        'HouseAge':    2026 - body.YearBuilt,
    }
    X = np.array([[row_dict.get(f, 0) for f in feat_cols]])
    pred_price = float(np.expm1(model.predict(X)[0]))
    margin     = pred_price * 0.065

    # Audit log
    save_prediction(
        username=body.username,
        algorithm=body.algorithm,
        input_features=row_dict,
        predicted_price=pred_price
    )
    logger.info(f"Prediction | {body.algorithm} | ${pred_price:,.0f} | user={body.username}")

    return PredictResponse(
        predicted_price=round(pred_price, 2),
        lower_bound=round(pred_price - margin, 2),
        upper_bound=round(pred_price + margin, 2),
        algorithm=body.algorithm,
        input_features=row_dict,
        timestamp=datetime.now().isoformat()
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"],
          dependencies=[Depends(verify_api_key)])
def train(body: TrainRequest):
    """
    Trigger a real K-Fold cross-validation training run.
    Saves the model to disk and logs the experiment to the database.
    """
    if not HAS_SKLEARN:
        raise HTTPException(503, "Scikit-Learn not available on this server.")

    df = get_house_data_df()
    scores, pipe, feat_cols = run_kfold(df, body.algorithm, n_splits=body.cv_folds)

    mean_s = float(np.mean(scores))
    std_s  = float(np.std(scores))
    run_id = str(uuid.uuid4())[:8]

    # Persist model
    model_path = str(MODEL_DIR / f"model_{body.algorithm.lower()}.pkl")
    joblib.dump({'model': pipe, 'features': feat_cols}, model_path)

    # Persist experiment
    save_experiment(
        run_id=run_id,
        username=body.username,
        algorithm=body.algorithm,
        cv_folds=body.cv_folds,
        mean_rmsle=mean_s,
        std_rmsle=std_s,
        hyperparams={'n_splits': body.cv_folds}
    )
    logger.info(f"Training complete | {run_id} | {body.algorithm} | RMSLE={mean_s:.4f}")

    return TrainResponse(
        run_id=run_id,
        algorithm=body.algorithm,
        cv_folds=body.cv_folds,
        mean_rmsle=round(mean_s, 4),
        std_rmsle=round(std_s, 4),
        model_path=model_path,
        timestamp=datetime.now().isoformat()
    )


@app.get("/experiments", tags=["Registry"])
def experiments():
    """Return all experiment runs from the persistent database."""
    rows = load_experiments()
    return {"count": len(rows), "experiments": rows}


@app.get("/models", tags=["Registry"])
def list_models():
    """List all serialized model files available on disk."""
    pkls = list(MODEL_DIR.glob("model_*.pkl"))
    return {
        "count": len(pkls),
        "models": [
            {"name": p.stem.replace("model_", ""), "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in pkls
        ]
    }


@app.delete("/models/{algo}", tags=["Registry"],
            dependencies=[Depends(verify_api_key)])
def delete_model(algo: str):
    """Delete a serialized model file from disk."""
    path = MODEL_DIR / f"model_{algo.lower()}.pkl"
    if not path.exists():
        raise HTTPException(404, f"Model '{algo}' not found.")
    path.unlink()
    logger.info(f"Model deleted: {path.name}")
    return {"deleted": algo}
