"""Guard the extraction prompt template.

The prompt is a Python `.format()` template, so every literal JSON brace in it
must be doubled. A single `{` makes `extract()` raise `KeyError` on the first
real call, which is only visible as episodes stuck at `failed` -- and the
earlier direction-repair edits introduced exactly that. This test formats the
template the same way `ExtractionLLM.extract()` does.
"""
from __future__ import annotations

from server.llm.extract import ALLOWED_REL_TYPES, EXTRACTION_PROMPT


def _render(content: str = "Kenny owns HOKI Reward.") -> str:
    return EXTRACTION_PROMPT.format(
        content=content,
        agent_id="test",
        session_id="test",
        tags="[]",
    )


def test_template_formats_without_keyerror():
    prompt = _render()
    assert "Kenny owns HOKI Reward." in prompt
    assert "{content}" not in prompt


def test_template_placeholders_are_the_four_expected_ones():
    # .format() must not be carrying stray fields from unescaped braces.
    for field in ("content", "agent_id", "session_id", "tags"):
        assert "{" + field + "}" in EXTRACTION_PROMPT
    # Formatting must not leave any unreplaced {name} behind.
    prompt = _render()
    assert "{agent_id}" not in prompt
    assert "{session_id}" not in prompt


def test_prompt_forbids_passive_relation_labels():
    prompt = _render()
    assert "OWNED_BY" in prompt            # named as forbidden
    assert "Never use passive labels" in prompt
    assert "WORKS_AT" in prompt
    assert "LOCATED_IN" in prompt


def test_prompt_tells_the_model_to_skip_labels_and_test_scaffolding():
    prompt = _render()
    assert "TEST NOTE" in prompt
    assert "codeword" in prompt.lower()


def test_every_allowed_relation_type_has_a_static_query_branch():
    """The graph writer is a hardcoded FOREACH chain (APOC is not installed),
    so its type list must cover ALLOWED_REL_TYPES exactly.

    It previously lagged behind: the prompt and extract.py accepted OWNS /
    WORKS_AT / LOCATED_IN, but the Cypher only knew the old labels and fell
    through to MERGE (a)-[:RELATED_TO]->(b). Everything came out RELATED_TO
    while every Python-side check passed. Keep the two lists in lock-step.
    """
    from server.graph import queries as Q

    missing = [t for t in ALLOWED_REL_TYPES if f"'{t}'" not in Q.CREATE_RELATIONSHIP_STATIC]
    assert not missing, (
        f"types accepted by extract.py but not creatable in the graph: {sorted(missing)}"
    )

    # The fallback must also know about them, otherwise the case falls
    # through to RELATED_TO instead of being rejected.
    fallback = Q.CREATE_RELATIONSHIP_STATIC.splitlines()[-2]
    for t in ALLOWED_REL_TYPES:
        if t == "RELATED_TO":
            continue
        assert f"'{t}'" in fallback, f"{t} missing from the RELATED_TO fallback guard"
