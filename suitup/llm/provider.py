"""One structured-output client for every provider.

Mistral, Gemini, and Ollama all speak OpenAI-compatible HTTP, so provider swapping is a
matter of base URL, key, and model — not three separate SDK integrations. `StructuredLLM`
is the only thing in the codebase that talks to a model, and agents receive one already
bound to their role.

Reliability comes from validate-and-retry rather than trust. Providers differ in how well
they honour a JSON schema (Ollama's OpenAI shim offers no schema enforcement at all), so
every response is parsed into a Pydantic model and, on failure, retried with the
validation error fed back. Three strikes raises `StructuredOutputError`, which the caller
turns into a labelled escalation rather than a traceback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, ValidationError

from suitup.config import ProviderConfig, SchemaTier

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_ATTEMPTS = 3


class StructuredOutputError(RuntimeError):
    """A model could not produce output matching the requested schema."""

    def __init__(self, role: str, schema: str, attempts: int, last_error: str, last_raw: str):
        self.role = role
        self.schema = schema
        self.attempts = attempts
        self.last_error = last_error
        self.last_raw = last_raw
        super().__init__(
            f"{role}: {schema} not produced after {attempts} attempts. "
            f"Last validation error: {last_error}"
        )


def _tighten(node: Any) -> Any:
    """Recursively forbid extra properties.

    Providers otherwise happily add commentary keys alongside the real fields, which
    passes their own schema check and then fails ours.
    """
    if isinstance(node, dict):
        out = {k: _tighten(v) for k, v in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_tighten(v) for v in node]
    return node


def _schema_of(model: type[BaseModel]) -> dict[str, Any]:
    return _tighten(model.model_json_schema())


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a response that may be wrapped in prose or fences.

    Smaller local models routinely ignore "return only JSON". Rather than burn a retry on
    something recoverable, take the outermost balanced object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(stripped[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : i + 1]
    return stripped[start:]


class StructuredLLM:
    """A model bound to one pipeline role, returning validated Pydantic objects."""

    def __init__(
        self,
        role: str,
        provider: ProviderConfig,
        model: str,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float = 120.0,
    ):
        self.role = role
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._tier: SchemaTier = provider.tier
        self._client = OpenAI(
            base_url=provider.base_url,
            api_key=provider.api_key(),
            timeout=timeout,
            max_retries=2,
        )

    def __repr__(self) -> str:
        return f"<StructuredLLM {self.role} -> {self.provider.name}:{self.model}>"

    # -- response_format construction -------------------------------------------------

    def _response_format(self, schema_model: type[BaseModel]) -> dict[str, Any]:
        if self._tier == "schema":
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_model.__name__,
                    "schema": _schema_of(schema_model),
                },
            }
        return {"type": "json_object"}

    def _system_prompt(self, system: str, schema_model: type[BaseModel]) -> str:
        """Always describe the schema in the prompt.

        Native schema enforcement is a guardrail, not an instruction — the model still
        writes better output when it is told what the fields mean, and this is the only
        schema signal Ollama gets at all.
        """
        schema = json.dumps(_schema_of(schema_model), indent=2)
        return (
            f"{system}\n\n"
            f"Respond with a single JSON object matching this schema exactly. "
            f"No prose, no markdown fences, no extra keys.\n\n"
            f"{schema}"
        )

    # -- the one public method --------------------------------------------------------

    def complete(self, system: str, user: str, schema: type[T]) -> T:
        """Call the model and return a validated instance of `schema`.

        Raises `StructuredOutputError` if the model cannot produce valid output within
        `MAX_ATTEMPTS`.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt(system, schema)},
            {"role": "user", "content": user},
        ]

        last_error = "no attempt made"
        last_raw = ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            raw = self._call(messages, schema)
            last_raw = raw
            try:
                return schema.model_validate_json(_extract_json(raw))
            except ValidationError as exc:
                last_error = str(exc)
                log.warning(
                    "%s: attempt %d/%d failed schema validation for %s",
                    self.role,
                    attempt,
                    MAX_ATTEMPTS,
                    schema.__name__,
                )
                if attempt < MAX_ATTEMPTS:
                    # Feed the error back. The model corrects far more often than it
                    # repeats itself, which is why this beats a blind retry.
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"That response failed validation:\n\n{last_error}\n\n"
                                f"Return corrected JSON matching the schema. Output only "
                                f"the JSON object."
                            ),
                        }
                    )

        raise StructuredOutputError(
            role=self.role,
            schema=schema.__name__,
            attempts=MAX_ATTEMPTS,
            last_error=last_error,
            last_raw=last_raw,
        )

    def _call(self, messages: list[dict[str, str]], schema: type[BaseModel]) -> str:
        """Issue the request, degrading to json_object if the provider rejects a schema."""
        extra = {"max_tokens": self.max_tokens} if self.max_tokens else {}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                response_format=self._response_format(schema),  # type: ignore[arg-type]
                **extra,
            )
        except BadRequestError:
            if self._tier != "schema":
                raise
            # Provider advertised schema support but refused this one. Drop to json_object
            # for the rest of this client's life rather than failing the run — the
            # validate-and-retry loop covers the difference.
            log.warning(
                "%s: %s rejected json_schema, falling back to json_object",
                self.role,
                self.provider.name,
            )
            self._tier = "json_object"
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.temperature,
                response_format={"type": "json_object"},
                **extra,
            )

        return response.choices[0].message.content or ""

    # -- diagnostics ------------------------------------------------------------------

    def ping(self) -> tuple[bool, str]:
        """Check reachability and whether this model is offered. Used by `suitup models --check`."""
        try:
            available = {m.id for m in self._client.models.list().data}
        except Exception as exc:  # noqa: BLE001 - surfacing any failure verbatim is the point
            return False, f"{type(exc).__name__}: {exc}"

        if not available:
            return True, "reachable (model list empty)"
        if self.model in available:
            return True, "reachable, model available"
        near = sorted(m for m in available if m.split(":")[0] in self.model or self.model in m)
        hint = f" Did you mean: {', '.join(near[:3])}?" if near else ""
        return False, f"reachable, but {self.model!r} not offered.{hint}"
