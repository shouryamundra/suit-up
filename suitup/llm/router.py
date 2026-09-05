"""Turns the role -> provider:model table into bound clients.

The indirection exists so agents depend on a role name, never a vendor. Swapping the
drafter from Gemini to Ollama is a config edit; no agent module changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from suitup.config import Config
from suitup.llm.provider import StructuredLLM


@dataclass(frozen=True)
class RoleRow:
    """A resolved routing decision, ready to display."""

    role: str
    provider: str
    model: str
    temperature: float
    has_key: bool


# Fallback sampling per role, used when models.yaml does not set one explicitly.
# Nodes that write prose want a little room; nodes that emit structure want none.
ROLE_TEMPERATURE = {
    "jd_analyst": 0.0,
    "listing_selector": 0.1,
    "bullet_selector": 0.1,
    "drafter": 0.4,
    "critic": 0.0,
    "tagger": 0.0,
}


class Router:
    """Lazily constructs and caches one `StructuredLLM` per role."""

    def __init__(self, config: Config):
        self.config = config
        self._cache: dict[str, StructuredLLM] = {}

    def for_role(self, role: str) -> StructuredLLM:
        if role not in self._cache:
            binding = self.config.roles[role]
            temperature = (
                binding.temperature
                if binding.temperature is not None
                else ROLE_TEMPERATURE.get(role, 0.2)
            )
            self._cache[role] = StructuredLLM(
                role=role,
                provider=self.config.providers[binding.provider],
                model=binding.model,
                temperature=temperature,
                max_tokens=binding.max_tokens,
            )
        return self._cache[role]

    def describe(self) -> list[RoleRow]:
        """One row per configured role, for `suitup models`."""
        rows = []
        for role in sorted(self.config.roles):
            binding = self.config.roles[role]
            provider = self.config.providers[binding.provider]
            temperature = (
                binding.temperature
                if binding.temperature is not None
                else ROLE_TEMPERATURE.get(role, 0.2)
            )
            rows.append(
                RoleRow(
                    role=role,
                    provider=binding.provider,
                    model=binding.model,
                    temperature=temperature,
                    has_key=provider.has_key(),
                )
            )
        return rows
