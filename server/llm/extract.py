"""Entity / relationship / fact extraction via local Ollama LLM.

Tuned for qwen2.5:3b on CPU: tight JSON-only prompt, validation, and
retry-on-parse-failure (max 2 retries).
"""

from __future__ import annotations

import json
import logging
import re

import httpx

from ..config import Settings
from ..models import ExtractionResult

log = logging.getLogger("mnemosyne.extract")

EXTRACTION_PROMPT = """You are a precise information extraction system. Analyze the conversation and extract structured information.

Return ONLY a single valid JSON object with this exact shape:
{{
  "entities": [
    {{"name": "...", "type": "...", "description": "...", "aliases": []}}
  ],
  "relationships": [
    {{"from_name": "...", "to_name": "...", "type": "...", "properties": {{}}}}
  ],
  "facts": [
    {{"subject_name": "...", "content": "...", "confidence": 0.9}}
  ]
}}

Field rules:
- entities.type must be one of: Person, Project, Product, Board, Identifier, Service, API, Tool, Decision, Concept, Task
- Extract ONLY names that appear verbatim in the conversation. Never invent a person, product, board, or project.
- Do not extract generic words (user, agent, system, memory, task, note, conversation).
- Do NOT extract labels, headings or test scaffolding as entities: strings like "TEST NOTE", "TEST FACT", "SMOKE", "TEMPORARY", or a bare codeword value are framing, not things.
- Prefer specific products, boards, projects, people, and explicit ids over broad topics.

Relationships: from_name is the SUBJECT and to_name is the OBJECT, so the triple reads as a
plain sentence "SUBJECT VERB OBJECT". Use ACTIVE verb types only:
  OWNS, MANAGES, USES, WORKS_AT, LOCATED_IN, CREATED, DEPENDS_ON, PART_OF,
  HAS_PROPERTY, HAS_BEHAVIOR, DECIDED_ON, WORKING_ON, INTEGRATES_WITH, RELATED_TO
Never use passive labels (OWNED_BY, MANAGED_BY, CREATED_BY). If the sentence is passive,
flip the direction instead of changing the label.

Examples (literal braces are doubled below only because this template is .format()ed):
- "Kenny and Timothy own HOKI Reward"
    -> {{"from_name":"Kenny","to_name":"HOKI Reward","type":"OWNS"}}
    -> {{"from_name":"Timothy","to_name":"HOKI Reward","type":"OWNS"}}
- "Gusta is the PIC for Travel SIM"
    -> {{"from_name":"Gusta","to_name":"Travel SIM","type":"MANAGES"}}
- "Kenny works at HOKI"
    -> {{"from_name":"Kenny","to_name":"HOKI","type":"WORKS_AT"}}
- "the warehouse is located in Sendai"
    -> {{"from_name":"the warehouse","to_name":"Sendai","type":"LOCATED_IN"}}
- "Daryl deploys to the tunnel"
    -> {{"from_name":"Daryl","to_name":"the tunnel","type":"USES"}}
- Read-back check: "Kenny OWNS HOKI Reward" is a sentence; "Kenny OWNED_BY HOKI Reward" is not.

- facts.confidence: explicit statement = 0.95, clear inference = 0.8, guess = 0.6
- Output ONLY the JSON object. No markdown fences, no commentary, no trailing text.
- Do not invent entities not mentioned in the conversation.
- Use consistent names (the most common form used in the conversation).
- Include aliases when alternative names appear.
- If there is nothing to extract, return {{"entities": [], "relationships": [], "facts": []}}.

Conversation:
{content}

Context: Agent={agent_id}, Session={session_id}, Tags={tags}
"""

ALLOWED_ENTITY_TYPES = {
    "Person", "Project", "Product", "Board", "Identifier", "Service", "API",
    "Tool", "Decision", "Concept", "Task", "Topic",
}

ALLOWED_REL_TYPES = {
    # Active verbs: the prompt asks for these so a triple reads as a sentence.
    "OWNS", "MANAGES", "USES", "WORKS_AT", "LOCATED_IN", "CREATED",
    "DEPENDS_ON", "PART_OF", "INTEGRATES_WITH", "HAS_PROPERTY", "HAS_BEHAVIOR",
    "DECIDED_ON", "WORKING_ON", "CONTRADICTS", "SUPERSEDES", "MENTIONED_IN",
    "RELATED_TO",
    # Legacy passive labels kept so pre-migration output still parses instead
    # of silently degrading to RELATED_TO. The prompt forbids these.
    "OWNED_BY", "MANAGED_BY", "CREATED_BY",
}


class ExtractionLLM:
    def __init__(self, settings: Settings):
        self._url = settings.ollama_url.rstrip("/")
        self._model = settings.extract_model
        # Extraction prompts can run 5-20s on CPU with 3b models
        self._client = httpx.AsyncClient(timeout=180.0)

    async def extract(
        self,
        content: str,
        agent_id: str,
        session_id: str = "",
        tags: list[str] | None = None,
    ) -> ExtractionResult:
        prompt = EXTRACTION_PROMPT.format(
            content=content[:8000],  # cap input size for CPU inference
            agent_id=agent_id,
            session_id=session_id or "n/a",
            tags=", ".join(tags or []) or "none",
        )

        last_error = "no attempts made"
        for attempt in range(3):  # initial + 2 retries
            raw = await self._generate(prompt)
            parsed = self._parse_json(raw)
            if parsed is not None:
                return self._normalize(parsed)
            last_error = f"attempt {attempt + 1}: unparseable JSON from model"
            log.warning("Extraction %s (%s chars)", last_error, len(raw or ""))
            # Nudge the model harder on retry
            prompt = (
                "IMPORTANT: respond with ONLY the JSON object, nothing else.\n\n"
                + prompt
            )

        log.error("Extraction failed after 3 attempts: %s", last_error)
        return ExtractionResult()  # graceful degradation: raw episode still stored

    async def _generate(self, prompt: str) -> str:
        resp = await self._client.post(
            f"{self._url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                # Reasoning models (Gemma 4, Qwen3, ...) burn their whole
                # output budget thinking before answering; with format="json"
                # that ran 144s per episode instead of 12s. Non-reasoning
                # models ignore this field (verified against qwen2.5:3b).
                "think": False,
                "options": {"temperature": 0.1, "num_ctx": 8192},
                "format": "json",
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Robust JSON extraction: direct, fenced-block, and first-brace scan."""
        text = text.strip()
        if not text:
            return None
        # Direct parse
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass
        # Markdown fence
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        # First { ... last } scan
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _normalize(data: dict) -> ExtractionResult:
        """Coerce model output into typed models, dropping invalid rows."""
        entities = []
        relationships = []
        facts = []
        try:
            for e in data.get("entities", []) or []:
                if not isinstance(e, dict) or not e.get("name"):
                    continue
                from ..models import EntityOut
                raw_type = str(e.get("type", "Concept")).strip()[:50] or "Concept"
                if raw_type not in ALLOWED_ENTITY_TYPES:
                    raw_type = "Concept"
                raw_aliases = e.get("aliases") or []
                if isinstance(raw_aliases, str):
                    raw_aliases = [raw_aliases]
                entities.append(EntityOut(
                    name=str(e["name"]).strip()[:200],
                    type=raw_type,
                    description=str(e.get("description", ""))[:1000],
                    aliases=[str(a).strip()[:200] for a in raw_aliases if str(a).strip()][:10],
                ))
            for r in data.get("relationships", []) or []:
                if not isinstance(r, dict):
                    continue
                f, t = str(r.get("from_name", "")).strip(), str(r.get("to_name", "")).strip()
                if not f or not t:
                    continue
                rel_type = str(r.get("type", "RELATED_TO")).strip().upper()
                if rel_type not in ALLOWED_REL_TYPES:
                    rel_type = "RELATED_TO"
                from ..models import RelationshipOut
                relationships.append(RelationshipOut(
                    from_name=f,
                    to_name=t,
                    type=rel_type,
                    properties=r.get("properties") if isinstance(r.get("properties"), dict) else {},
                ))
            for fct in data.get("facts", []) or []:
                if not isinstance(fct, dict) or not fct.get("content"):
                    continue
                try:
                    conf = float(fct.get("confidence", 0.8))
                except (TypeError, ValueError):
                    conf = 0.8
                from ..models import FactOut
                facts.append(FactOut(
                    subject_name=str(fct.get("subject_name", "")).strip()[:200],
                    content=str(fct["content"]).strip()[:2000],
                    confidence=max(0.0, min(1.0, conf)),
                ))
            return ExtractionResult(entities=entities, relationships=relationships, facts=facts)
        except Exception as exc:  # noqa: BLE001
            log.error("Normalization failed: %s", exc)
            return ExtractionResult()

    async def close(self) -> None:
        await self._client.aclose()
