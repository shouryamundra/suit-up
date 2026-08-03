"""JSON extraction and the validate-and-retry loop.

These are the parts that make a weakly-schema-compliant provider (Ollama) usable, so
they are tested against deliberately badly-behaved responses rather than happy paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from suitup.config import ProviderConfig
from suitup.llm.provider import StructuredLLM, StructuredOutputError, _extract_json


class Answer(BaseModel):
    name: str
    count: int


def make_llm(responses: list[str], tier="json_object") -> tuple[StructuredLLM, MagicMock]:
    """A StructuredLLM whose transport returns `responses` in order."""
    provider = ProviderConfig(name="fake", base_url="http://x/v1", api_key_env=None, tier=tier)
    llm = StructuredLLM(role="tester", provider=provider, model="fake-model")

    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=r))]) for r in responses
    ]
    llm._client = client
    return llm, client


class TestExtractJson:
    def test_plain_object_passes_through(self):
        assert _extract_json('{"a": 1}') == '{"a": 1}'

    def test_strips_markdown_fence(self):
        assert _extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_strips_bare_fence(self):
        assert _extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_ignores_surrounding_prose(self):
        # Small local models routinely preface JSON with an explanation.
        assert _extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == '{"a": 1}'

    def test_handles_nested_objects(self):
        payload = '{"a": {"b": {"c": 1}}}'
        assert _extract_json(f"noise {payload} trailing") == payload

    def test_braces_inside_strings_do_not_confuse_the_scan(self):
        payload = '{"text": "a } brace { inside"}'
        assert _extract_json(payload) == payload

    def test_escaped_quote_inside_string(self):
        payload = '{"text": "she said \\"hi\\" loudly"}'
        assert _extract_json(payload) == payload

    def test_no_object_returns_input_for_the_validator_to_reject(self):
        assert _extract_json("not json at all") == "not json at all"


class TestRetryLoop:
    def test_valid_first_response_returns_immediately(self):
        llm, client = make_llm(['{"name": "a", "count": 1}'])
        assert llm.complete("sys", "user", Answer) == Answer(name="a", count=1)
        assert client.chat.completions.create.call_count == 1

    def test_recovers_on_second_attempt(self):
        llm, client = make_llm(['{"name": "a"}', '{"name": "a", "count": 2}'])
        assert llm.complete("sys", "user", Answer).count == 2
        assert client.chat.completions.create.call_count == 2

    def test_validation_error_is_fed_back_to_the_model(self):
        llm, client = make_llm(['{"name": "a"}', '{"name": "a", "count": 2}'])
        llm.complete("sys", "user", Answer)

        # The corrective turn must carry the actual validation error, not a generic
        # "try again" — that feedback is why the retry succeeds more often than not.
        second_call_messages = client.chat.completions.create.call_args_list[1].kwargs["messages"]
        correction = second_call_messages[-1]["content"]
        assert "failed validation" in correction
        assert "count" in correction

    def test_gives_up_after_three_attempts(self):
        llm, client = make_llm(['{"name": "a"}'] * 3)
        with pytest.raises(StructuredOutputError) as exc:
            llm.complete("sys", "user", Answer)
        assert client.chat.completions.create.call_count == 3
        assert exc.value.role == "tester"
        assert exc.value.schema == "Answer"
        assert exc.value.attempts == 3

    def test_error_carries_the_last_raw_response_for_debugging(self):
        llm, _ = make_llm(["garbage"] * 3)
        with pytest.raises(StructuredOutputError) as exc:
            llm.complete("sys", "user", Answer)
        assert exc.value.last_raw == "garbage"


class TestResponseFormat:
    def test_schema_tier_requests_json_schema(self):
        llm, client = make_llm(['{"name": "a", "count": 1}'], tier="schema")
        llm.complete("sys", "user", Answer)
        fmt = client.chat.completions.create.call_args.kwargs["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["name"] == "Answer"

    def test_json_object_tier_requests_json_object(self):
        llm, client = make_llm(['{"name": "a", "count": 1}'], tier="json_object")
        llm.complete("sys", "user", Answer)
        assert client.chat.completions.create.call_args.kwargs["response_format"] == {
            "type": "json_object"
        }

    def test_schema_forbids_extra_properties(self):
        # Providers otherwise add commentary keys that pass their check and fail ours.
        llm, client = make_llm(['{"name": "a", "count": 1}'], tier="schema")
        llm.complete("sys", "user", Answer)
        schema = client.chat.completions.create.call_args.kwargs["response_format"]["json_schema"][
            "schema"
        ]
        assert schema["additionalProperties"] is False

    def test_schema_is_described_in_the_system_prompt(self):
        # Ollama gets no native enforcement, so the prompt is its only schema signal.
        llm, client = make_llm(['{"name": "a", "count": 1}'])
        llm.complete("do the thing", "user", Answer)
        system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "do the thing" in system
        assert '"count"' in system
