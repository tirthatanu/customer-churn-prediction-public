
import sqlite3
from pathlib import Path
from contextlib import closing


# 1. Define database location
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_DIR / "data" / "predictions.db"


# 2. Create database connection
def get_connection():
    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=10
    )
    connection.row_factory = sqlite3.Row
    return connection


# 3. Initialize database
def init_db():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with closing(get_connection()) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    tenure INTEGER NOT NULL
                        CHECK (tenure >= 0),

                    monthly_charges REAL NOT NULL
                        CHECK (monthly_charges >= 0),

                    total_charges REAL NOT NULL
                        CHECK (total_charges >= 0),

                    contract TEXT NOT NULL,

                    internet_service TEXT NOT NULL,

                    prediction TEXT NOT NULL
                        CHECK (
                            prediction IN ('Churn', 'No Churn')
                        ),

                    probability REAL NOT NULL
                        CHECK (
                            probability >= 0
                            AND probability <= 1
                        ),

                    risk_level TEXT NOT NULL
                        CHECK (
                            risk_level IN ('Low', 'Medium', 'High')
                        ),

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)


# 4. Save prediction to SQLite
def save_prediction(
    tenure,
    monthly_charges,
    total_charges,
    contract,
    internet_service,
    prediction,
    probability,
    risk_level
):
    with closing(get_connection()) as connection:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_history (
                    tenure,
                    monthly_charges,
                    total_charges,
                    contract,
                    internet_service,
                    prediction,
                    probability,
                    risk_level
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenure,
                    monthly_charges,
                    total_charges,
                    contract,
                    internet_service,
                    prediction,
                    probability,
                    risk_level
                )
            )

            return cursor.lastrowid


# 5. Retrieve prediction history
def get_prediction_history(limit=20, offset=0):
    with closing(get_connection()) as connection:

        # Count all prediction records
        total = connection.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]

        # Retrieve newest predictions first
        rows = connection.execute(
            """
            SELECT
                id,
                tenure,
                monthly_charges,
                total_charges,
                contract,
                internet_service,
                prediction,
                probability,
                risk_level,
                created_at
            FROM prediction_history
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()

        predictions = [
            dict(row) for row in rows
        ]

        return total, predictions


# 6. Verify SQLite database
def verify_database():
    with closing(get_connection()) as connection:

        columns = connection.execute(
            "PRAGMA table_info(prediction_history)"
        ).fetchall()

        count = connection.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]

        print("\nDatabase Verification")
        print("-------------------------")
        print("Database:", DATABASE_PATH)
        print("Table: prediction_history")

        print("\nTable Columns:")
        for column in columns:
            print("-", column["name"])

        print("\nStored Predictions:", count)
        print("\nSQLite Database Ready!")


# 7. Initialize and verify database
if __name__ == "__main__":
    print("\nInitializing SQLite Database...")
    init_db()
    verify_database()
