"""Shared fixtures, including a real-Postgres harness for integration tests.

The harness provisions a throwaway database so integration tests never touch the
corpus indexed in the development database.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from rag.config import settings

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "scripts" / "init_db.sql"
TEST_DB = "ragdb_integration_test"


def _admin_url() -> str:
    """URL for the maintenance database, used to create/drop the test database."""
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/postgres"
    )


def _server_available() -> bool:
    try:
        with psycopg.connect(_admin_url(), connect_timeout=2):
            return True
    except psycopg.Error:
        return False


@pytest.fixture(scope="session")
def integration_db() -> None:
    """Create a throwaway database with the project schema and point settings at it."""
    if not _server_available():
        pytest.skip("no Postgres reachable; start one with 'make db-up'")

    with psycopg.connect(_admin_url(), autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')
        cur.execute(f'CREATE DATABASE "{TEST_DB}"')

    patch = pytest.MonkeyPatch()
    patch.setattr(settings, "postgres_db", TEST_DB)

    try:
        with psycopg.connect(settings.database_url, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        yield
    finally:
        patch.undo()
        with psycopg.connect(_admin_url(), autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(f'DROP DATABASE IF EXISTS "{TEST_DB}" WITH (FORCE)')


@pytest.fixture
def empty_index(integration_db):
    """Truncate documents so each integration test starts from a known state."""
    from rag import db

    db.clear_documents()
    return db
