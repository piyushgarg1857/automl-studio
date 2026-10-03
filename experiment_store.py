"""Local SQLite persistence for AutoML experiment runs."""
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("automl_experiments.db")


def _connect():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            task TEXT NOT NULL,
            target TEXT NOT NULL,
            model_count INTEGER NOT NULL,
            score_metric TEXT NOT NULL,
            score_value REAL,
            payload BLOB NOT NULL
        )
    """)
    connection.commit()
    return connection


def save_experiment(name, task, target, leaderboard, artifacts, score_metric):
    successful = leaderboard[leaderboard[score_metric].notna()]
    score = float(successful.iloc[0][score_metric]) if not successful.empty else None
    payload = pickle.dumps({"leaderboard": leaderboard, "artifacts": artifacts})
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _connect() as connection:
        cursor = connection.execute(
            """INSERT INTO experiments
            (name, created_at, task, target, model_count, score_metric, score_value, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, created_at, task, str(target), len(leaderboard), score_metric, score, payload),
        )
        return cursor.lastrowid


def list_experiments():
    with _connect() as connection:
        rows = connection.execute(
            """SELECT id, name, created_at, task, target, model_count,
                      score_metric, score_value FROM experiments ORDER BY id DESC"""
        ).fetchall()
    return [dict(row) for row in rows]


def load_experiment(experiment_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT payload FROM experiments WHERE id = ?", (int(experiment_id),)
        ).fetchone()
    if row is None:
        raise ValueError("Experiment not found.")
    return pickle.loads(row["payload"])


def delete_experiment(experiment_id):
    with _connect() as connection:
        cursor = connection.execute(
            "DELETE FROM experiments WHERE id = ?", (int(experiment_id),)
        )
        return cursor.rowcount > 0
