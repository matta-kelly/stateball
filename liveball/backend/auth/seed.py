"""Create auth schema, users table, and initial admin user.

Usage:
    uv run python -m backend.auth.seed --username admin --password <pw>
"""

import argparse

from backend.auth.db import get_pg
from backend.auth.passwords import hash_password


SCHEMA_SQL = "CREATE SCHEMA IF NOT EXISTS auth"

TABLE_SQL = """
CREATE TABLE IF NOT EXISTS auth.users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username        TEXT UNIQUE NOT NULL,
    password        TEXT NOT NULL DEFAULT '',
    role            TEXT NOT NULL DEFAULT 'user',
    status          TEXT NOT NULL DEFAULT 'active',
    invite_token    TEXT,
    invite_expires  TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

MIGRATE_SQL = [
    "ALTER TABLE auth.users ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'",
    "ALTER TABLE auth.users ADD COLUMN IF NOT EXISTS invite_token TEXT",
    "ALTER TABLE auth.users ADD COLUMN IF NOT EXISTS invite_expires TIMESTAMPTZ",
]

UPSERT_SQL = """
INSERT INTO auth.users (username, password, role, status)
VALUES (%s, %s, %s, 'active')
ON CONFLICT (username) DO UPDATE SET password = EXCLUDED.password, role = EXCLUDED.role, status = 'active'
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed auth database")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", default="admin", choices=["admin", "user"])
    args = parser.parse_args()

    with get_pg() as conn:
        conn.execute(SCHEMA_SQL)
        conn.execute(TABLE_SQL)
        for sql in MIGRATE_SQL:
            conn.execute(sql)
        conn.execute(UPSERT_SQL, (args.username, hash_password(args.password), args.role))
        conn.commit()

    print(f"User '{args.username}' seeded with role '{args.role}'")


if __name__ == "__main__":
    main()
