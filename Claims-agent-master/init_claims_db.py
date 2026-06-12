import sqlite3
from pathlib import Path

DB_PATH = Path("claims_system.db")


def initialize_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS members (
            member_id TEXT PRIMARY KEY,
            name TEXT,
            policy_number TEXT,
            status TEXT,
            policy_tier TEXT
        )
        """
    )

    sample_members = [
        ("M001", "Alex", "POL-1001", "Active", "Gold"),
        ("M002", "Sarah", "POL-1002", "Lapsed", "Silver"),
    ]

    cursor.executemany(
        """
        INSERT OR REPLACE INTO members (
            member_id,
            name,
            policy_number,
            status,
            policy_tier
        ) VALUES (?, ?, ?, ?, ?)
        """,
        sample_members,
    )

    conn.commit()
    conn.close()


def print_verification():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    rows = cursor.execute(
        "SELECT member_id, name, policy_number, status, policy_tier FROM members ORDER BY member_id"
    ).fetchall()

    print("Verified members:")
    for row in rows:
        print(row)

    conn.close()


if __name__ == "__main__":
    initialize_database()
    print_verification()
