"""Configuration: filesystem paths, selection caps, and role -> model routing.

`models.yaml` is loaded here and nowhere else. Agents receive a bound client from the
router; they never read this module's provider tables directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

SchemaTier = Literal["schema", "json_object"]

# Roles the pipeline knows how to route. models.yaml must cover exactly these.
KNOWN_ROLES = frozenset(
    {
        "jd_analyst",
        "listing_selector",
        "bullet_selector",
        "drafter",
        "critic",
        "tagger",
    }
)


class ConfigError(RuntimeError):
    """Raised for a malformed or internally inconsistent configuration."""


class ProviderConfig(BaseModel):
    name: str
    base_url: str
    api_key_env: str | None = None
    tier: SchemaTier = "json_object"

    def api_key(self) -> str:
        """Resolve the key from the environment.

        Providers needing no auth still get a placeholder — the OpenAI client requires
        a non-empty key even when the server ignores it.
        """
        if self.api_key_env is None:
            return "not-needed"
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ConfigError(
                f"provider {self.name!r} needs {self.api_key_env} but it is unset. "
                f"Add it to .env (see .env.example)."
            )
        return key

    def has_key(self) -> bool:
        return self.api_key_env is None or bool(os.environ.get(self.api_key_env))


class RoleBinding(BaseModel):
    """Which model runs one pipeline role, and how it is sampled.

    Any role may point at any provider — nothing in the pipeline assumes a vendor.
    `temperature` and `max_tokens` are optional per-role overrides; when unset the router
    falls back to a sensible default for that role's job.
    """

    provider: str
    model: str
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class Caps(BaseModel):
    """One-page fit heuristics.

    These are count-based on purpose: the PRD rejects character-length estimation for the
    MVP because the compiler is the only honest source of truth on page count. Real values
    are derived from the user's existing one-page resumes during library bootstrap.
    """

    max_experience: int = 3
    max_projects: int = 2
    max_bullets_per_listing: int = 4
    # A challenger listing must beat a template's incumbent by at least this much to
    # displace it. Stops marginal score noise from churning a human-approved template.
    delta_margin: float = 0.15


class Thresholds(BaseModel):
    # Below this fraction of high-weight requirements covered, the Critic raises a
    # warning. A first guess, to be tuned once there are real runs — see PRD open questions.
    low_coverage: float = 0.6
    # Scoring weights for the Retriever's two match kinds.
    exact_phrase_weight: float = 1.0
    token_subset_weight: float = 0.4


class Paths(BaseModel):
    root: Path
    library: Path
    listings: Path
    templates: Path
    configurations: Path
    pending_variants: Path
    index: Path
    latex_template: Path
    runs: Path

    @classmethod
    def from_root(cls, root: Path) -> Paths:
        library = root / "library"
        return cls(
            root=root,
            library=library,
            listings=library / "listings",
            templates=library / "templates",
            configurations=library / "configurations.json",
            pending_variants=library / "pending_variants.yaml",
            index=library / "index.json",
            latex_template=root / "templates" / "latex" / "resume.tex.j2",
            runs=root / "runs",
        )


class Config(BaseModel):
    paths: Paths
    providers: dict[str, ProviderConfig]
    roles: dict[str, RoleBinding]
    caps: Caps = Field(default_factory=Caps)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    # The PRD wants the Critic blind to the Drafter, which same-model routing undermines.
    # Enforced by default, but this is a quality guardrail rather than a correctness one,
    # so it stays overridable — the abstraction should not refuse a routing the user wants.
    require_blind_critique: bool = True

    @model_validator(mode="after")
    def _validate_routing(self) -> Config:
        missing = KNOWN_ROLES - self.roles.keys()
        if missing:
            raise ConfigError(f"models.yaml is missing roles: {sorted(missing)}")

        unknown = self.roles.keys() - KNOWN_ROLES
        if unknown:
            raise ConfigError(f"models.yaml declares unknown roles: {sorted(unknown)}")

        for role, binding in self.roles.items():
            if binding.provider not in self.providers:
                raise ConfigError(
                    f"role {role!r} routes to undefined provider {binding.provider!r}; "
                    f"known providers: {sorted(self.providers)}"
                )

        # The PRD requires the Critic to evaluate blind to the Drafter's reasoning. Same
        # model on both ends makes that requirement nominal, so it is enforced, not assumed.
        if self.require_blind_critique and self.roles["drafter"] == self.roles["critic"]:
            raise ConfigError(
                "drafter and critic resolve to the same model, so the critique would not "
                f"be blind (both are {self.roles['drafter']}). Change one in models.yaml, "
                f"or set `require_blind_critique: false` if that is deliberate."
            )
        return self

    def provider_for(self, role: str) -> ProviderConfig:
        return self.providers[self.roles[role].provider]


def _find_root(start: Path | None = None) -> Path:
    """Walk upward for the directory holding models.yaml."""
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "models.yaml").is_file():
            return candidate
    raise ConfigError(
        f"no models.yaml found in {here} or any parent. Run suitup from the project root."
    )


def parse_override(spec: str) -> tuple[str, RoleBinding]:
    """Parse a `--model role=provider:model` override.

    Kept separate from CLI plumbing so the parsing rules are unit-testable on their own.
    """
    if "=" not in spec:
        raise ConfigError(f"bad --model override {spec!r}; expected role=provider:model")
    role, _, target = spec.partition("=")
    role = role.strip()
    if role not in KNOWN_ROLES:
        raise ConfigError(f"unknown role {role!r}; known roles: {sorted(KNOWN_ROLES)}")
    if ":" not in target:
        raise ConfigError(f"bad --model override {spec!r}; expected role=provider:model")
    provider, _, model = target.partition(":")
    provider, model = provider.strip(), model.strip()
    if not provider or not model:
        raise ConfigError(f"bad --model override {spec!r}; provider and model must be non-empty")
    return role, RoleBinding(provider=provider, model=model)


def load_config(
    root: Path | None = None,
    overrides: list[str] | None = None,
    profile: str | None = None,
) -> Config:
    """Load models.yaml and resolve the routing.

    Three layers, each beating the one before it:

    1. `roles:` — the defaults.
    2. `profiles.<name>:` — a named set of role overrides, e.g. flipping everything local.
    3. `--model role=provider:model` — explicit per-role overrides for one invocation.
    """
    root = root or _find_root()
    load_dotenv(root / ".env")

    raw = yaml.safe_load((root / "models.yaml").read_text()) or {}

    providers = {
        name: ProviderConfig(name=name, **spec)
        for name, spec in (raw.get("providers") or {}).items()
    }
    roles = {name: RoleBinding(**spec) for name, spec in (raw.get("roles") or {}).items()}

    profiles = raw.get("profiles") or {}
    if profile is not None:
        if profile not in profiles:
            raise ConfigError(
                f"unknown profile {profile!r}; defined profiles: {sorted(profiles) or 'none'}"
            )
        for role, spec in (profiles[profile] or {}).items():
            if role not in KNOWN_ROLES:
                raise ConfigError(f"profile {profile!r} sets unknown role {role!r}")
            roles[role] = RoleBinding(**spec)

    for spec in overrides or []:
        role, binding = parse_override(spec)
        roles[role] = binding

    settings = raw.get("settings") or {}

    return Config(
        paths=Paths.from_root(root),
        providers=providers,
        roles=roles,
        caps=Caps(**(raw.get("caps") or {})),
        thresholds=Thresholds(**(raw.get("thresholds") or {})),
        require_blind_critique=settings.get("require_blind_critique", True),
    )
