"""Neo4j async driver wrapper."""

from __future__ import annotations

from typing import Any, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

from ..config import Settings


class GraphDriver:
    """Thin wrapper around the Neo4j async driver."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._driver: Optional[AsyncDriver] = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
                max_connection_lifetime=60,
            )
        return self._driver

    async def run(self, query: str, **params: Any) -> list[dict]:
        """Execute a query, return list of record dicts."""
        async with self.driver.session() as session:
            result = await session.run(query, params)
            return [record.data() async for record in result]

    async def run_one(self, query: str, **params: Any) -> Optional[dict]:
        rows = await self.run(query, **params)
        return rows[0] if rows else None

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def ping(self) -> bool:
        try:
            await self.run("RETURN 1 AS ok")
            return True
        except Exception:
            return False
