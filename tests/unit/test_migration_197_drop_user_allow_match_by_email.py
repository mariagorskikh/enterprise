"""Real-DB tests for migration 197 (drop ``allow_match_by_email`` from ``user``)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from sqlalchemy import text
from sqlalchemy.engine import Engine

from tests import postgres_testdb


def _user_columns(engine: Engine) -> set[str]:
    with engine.connect() as conn:
        return set(
            conn.execute(
                text(
                    'SELECT column_name FROM information_schema.columns '
                    "WHERE table_name = 'user'"
                )
            ).scalars()
        )


def test_upgrade_drops_column(
    engine: Engine,
    test_database: postgres_testdb.TestDatabase,
    create_org,
):
    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'downgrade', '196'
    )
    # Raw SQL: the ORM model has columns that schema 196 does not have.
    user = SimpleNamespace(id=uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                'INSERT INTO "user" (id, current_org_id, email, allow_match_by_email) '
                "VALUES (:id, :org_id, 'kept@example.com', true)"
            ),
            {'id': user.id, 'org_id': create_org().id},
        )

    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'upgrade', '197'
    )

    assert 'allow_match_by_email' not in _user_columns(engine)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text('SELECT email FROM "user" WHERE id = :id'), {'id': user.id}
            ).scalar_one()
            == 'kept@example.com'
        )


def test_downgrade_restores_column_as_false(
    engine: Engine, test_database: postgres_testdb.TestDatabase, create_user
):
    user = create_user(email='restored@example.com')

    postgres_testdb.run_alembic(
        test_database.server, test_database.name, 'downgrade', '196'
    )

    with engine.connect() as conn:
        assert (
            conn.execute(
                text('SELECT allow_match_by_email FROM "user" WHERE id = :id'),
                {'id': user.id},
            ).scalar_one()
            is False
        )
