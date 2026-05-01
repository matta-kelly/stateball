import psycopg

from backend.config import settings


def get_pg() -> psycopg.Connection:
    return psycopg.connect(
        host=settings.ducklake_db_host,
        port=settings.ducklake_db_port,
        user=settings.ducklake_db_user,
        password=settings.ducklake_db_password,
        dbname=settings.ducklake_db_name,
        options="-c search_path=auth",
    )
