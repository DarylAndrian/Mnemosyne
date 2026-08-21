"""Embedding client for Ollama (nomic-embed-text)."""

from __future__ import annotations

import httpx

from ..config import Settings


class OllamaEmbedder:
    """Generates 768-dim embeddings via the local Ollama API."""

    def __init__(self, settings: Settings):
        self._url = settings.ollama_url.rstrip("/")
        self._model = settings.embed_model
        self._client = httpx.AsyncClient(timeout=60.0)

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
            f"{self._url}/api/embed",
            json={"model": self._model, "input": text},
        )
        resp.raise_for_status()
        data = resp.json()
        vectors = data.get("embeddings") or []
        if not vectors:
            raise RuntimeError(f"Ollama returned no embedding: {data}")
        return vectors[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post(
            f"{self._url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        return resp.json().get("embeddings") or []

    async def health(self) -> bool:
        try:
            resp = await self._client.get(f"{self._url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()
