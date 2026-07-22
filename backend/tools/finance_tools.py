"""Finance helpers for parsing local bank and subscription data."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class FinanceStore:
    """Persist financial data locally in SQLite."""

    def __init__(self, db_path: str = "./data/agent_suite.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service_name TEXT NOT NULL,
                    amount REAL NOT NULL,
                    billing_cycle TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    def track_subscription(self, service_name: str, amount: float, billing_cycle: str) -> Dict[str, Any]:
        connection = sqlite3.connect(self.db_path)
        try:
            cursor = connection.execute(
                "INSERT INTO subscriptions (service_name, amount, billing_cycle, created_at) VALUES (?, ?, ?, datetime('now'))",
                (service_name, amount, billing_cycle),
            )
            connection.commit()
            return {"ok": True, "id": cursor.lastrowid}
        finally:
            connection.close()

    def get_subscriptions(self) -> List[Dict[str, Any]]:
        connection = sqlite3.connect(self.db_path)
        try:
            rows = connection.execute("SELECT service_name, amount, billing_cycle FROM subscriptions ORDER BY id DESC").fetchall()
        finally:
            connection.close()
        return [{"service_name": service_name, "amount": amount, "billing_cycle": billing_cycle} for service_name, amount, billing_cycle in rows]


def parse_transaction_email(email_content: str) -> Dict[str, Any]:
    return {"amount": None, "merchant": None, "date": None, "message": "No finance parser configured yet."}


def check_budget(category: str, spent_this_month: float) -> Dict[str, Any]:
    return {"category": category, "spent_this_month": spent_this_month, "budget_ok": True, "message": "No budget limits configured yet."}


def generate_monthly_summary() -> Dict[str, Any]:
    return {"income": 0, "expenses": {}, "subscriptions": []}


def flag_unusual_transaction(amount: float, merchant: str, historical_avg: float) -> Dict[str, Any]:
    return {"ok": False, "message": "No anomaly detection configured yet."}
