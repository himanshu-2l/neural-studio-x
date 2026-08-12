# ─────────────────────────────────────────────────────────────
# Neural Studio X — Database Layer (SQLite)
# Persistent experiment tracking, model registry, user sessions
# ─────────────────────────────────────────────────────────────
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("neural_studio_x.db")

DB_PATH = Path(__file__).parent / "studio_x.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur  = conn.cursor()

    # Experiment runs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id        TEXT    NOT NULL UNIQUE,
            username      TEXT    NOT NULL DEFAULT 'anonymous',
            algorithm     TEXT    NOT NULL,
            cv_folds      INTEGER NOT NULL,
            mean_rmsle    REAL    NOT NULL,
            std_rmsle     REAL    NOT NULL,
            hyperparams   TEXT,          -- JSON string
            is_champion   INTEGER DEFAULT 0,
            created_at    TEXT    NOT NULL
        )
    """)

    # Prediction logs
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL DEFAULT 'anonymous',
            algorithm     TEXT    NOT NULL,
            input_features TEXT   NOT NULL,  -- JSON string
            predicted_price REAL  NOT NULL,
            created_at    TEXT    NOT NULL
        )
    """)

    # Alter experiments table to add is_production if not exists
    try:
        cur.execute("ALTER TABLE experiments ADD COLUMN is_production INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    logger.info(f"Database initialised at {DB_PATH}")



def save_experiment(run_id: str, username: str, algorithm: str,
                    cv_folds: int, mean_rmsle: float, std_rmsle: float,
                    hyperparams: dict = None) -> None:
    """Persist a training run to the database."""
    conn = get_connection()
    cur  = conn.cursor()

    # Reset previous champion for this algorithm
    cur.execute("UPDATE experiments SET is_champion = 0 WHERE algorithm = ?", (algorithm,))

    cur.execute("""
        INSERT OR REPLACE INTO experiments
            (run_id, username, algorithm, cv_folds, mean_rmsle, std_rmsle, hyperparams, is_champion, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)
    """, (run_id, username, algorithm, cv_folds,
          round(mean_rmsle, 5), round(std_rmsle, 5),
          json.dumps(hyperparams or {}),
          datetime.now().isoformat()))

    conn.commit()
    conn.close()
    logger.info(f"Experiment saved | run_id={run_id} | {algorithm} | RMSLE={mean_rmsle:.4f}")


def load_experiments() -> list[dict]:
    """Load all experiment runs ordered by RMSLE ascending."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT * FROM experiments ORDER BY mean_rmsle ASC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def save_prediction(username: str, algorithm: str,
                    input_features: dict, predicted_price: float) -> None:
    """Log a prediction to the database."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (username, algorithm, input_features, predicted_price, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (username, algorithm, json.dumps(input_features),
          round(predicted_price, 2), datetime.now().isoformat()))
    conn.commit()
    conn.close()


def load_predictions(username: str = None) -> list[dict]:
    """Load prediction history, optionally filtered by user."""
    conn = get_connection()
    cur  = conn.cursor()
    if username:
        cur.execute("SELECT * FROM predictions WHERE username = ? ORDER BY id DESC LIMIT 50", (username,))
    else:
        cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 50")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    """Return high-level dashboard statistics."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) as total FROM experiments")
    total_exp  = cur.fetchone()['total']
    cur.execute("SELECT COUNT(*) as total FROM predictions")
    total_pred = cur.fetchone()['total']
    cur.execute("SELECT MIN(mean_rmsle) as best FROM experiments")
    best_rmsle = cur.fetchone()['best'] or 0.0
    conn.close()
    return {
        'total_experiments': total_exp,
        'total_predictions': total_pred,
        'best_rmsle': round(best_rmsle, 4)
    }


def promote_experiment(run_id: str) -> None:
    """Set is_production = 1 for the given run_id, and 0 for all others."""
    conn = get_connection()
    cur  = conn.cursor()
    # Reset all
    cur.execute("UPDATE experiments SET is_production = 0")
    # Set run_id to production
    cur.execute("UPDATE experiments SET is_production = 1 WHERE run_id = ?", (run_id,))
    conn.commit()
    conn.close()


def get_production_algorithm() -> str:
    """Return the algorithm name for the active production model run, defaulting to GradientBoostingRegressor if none."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("SELECT algorithm FROM experiments WHERE is_production = 1 LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if row:
        return row['algorithm']
    return "GradientBoostingRegressor"


