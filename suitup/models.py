"""Core data model for Suit-Up.

Three families of types live here:

1. **Library types** (`Listing`, `Bullet`, `Variant`, `Template`) mirror the on-disk YAML
   in `library/`. They are the human-authored source of truth and are read-only at runtime.
2. **Pipeline types** (`Requirement`, `Candidate`, `Selection`, `Draft`, `Critique`) are
   passed between agents. Every LLM node validates its output against one of these.
3. **Outcome types** (`CompileResult`, `RunResult`) describe how a run ended.

Nothing in this module imports an agent, a provider, or the graph — it is the shared
vocabulary those layers speak, and it must stay free of their dependencies.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Library types
# ---------------------------------------------------------------------------


class BulletRole(str, Enum):
    """Whether a bullet may be swapped or dropped.

    `essential` bullets set scope, team size, timeframe, or headline context. They are
    never scored, never swapped, and never omitted — only lightly polished for tense and
    flow. `flexible` bullets carry variants and are the only content the selector moves.
    """

    ESSENTIAL = "essential"
    FLEXIBLE = "flexible"


class ListingType(str, Enum):
    EXPERIENCE = "experience"
    PROJECT = "project"


class Variant(BaseModel):
    """One pre-approved phrasing of a flexible bullet, tagged for lexical matching."""

    variant_id: str
    keywords: list[str] = Field(default_factory=list)
    text: str

    @model_validator(mode="after")
    def _strip_text(self) -> Variant:
        # YAML block scalars carry trailing newlines and folded line breaks. Normalizing
        # here means every downstream consumer — scorer, renderer, hasher — sees one form.
        object.__setattr__(self, "text", " ".join(self.text.split()))
        return self


class Bullet(BaseModel):
    """A slot within a listing.

    An essential bullet carries `text` directly. A flexible bullet carries `variants` and
    no text of its own — the selector picks which variant fills the slot.
    """

    slot: int
    role: BulletRole
    text: str | None = None
    variants: list[Variant] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_role_shape(self) -> Bullet:
        if self.role is BulletRole.ESSENTIAL:
            if not self.text:
                raise ValueError(f"slot {self.slot}: essential bullet requires `text`")
            if self.variants:
                raise ValueError(f"slot {self.slot}: essential bullet cannot have `variants`")
            object.__setattr__(self, "text", " ".join(self.text.split()))
        else:
            if not self.variants:
                raise ValueError(f"slot {self.slot}: flexible bullet requires at least one variant")
            if self.text:
                raise ValueError(f"slot {self.slot}: flexible bullet cannot have bare `text`")
            ids = [v.variant_id for v in self.variants]
            if len(ids) != len(set(ids)):
                raise ValueError(f"slot {self.slot}: duplicate variant_id among {ids}")
        return self


class ListingContext(BaseModel):
    org: str
    dates: str
    location: str | None = None


class ListingMetadata(BaseModel):
    approved: bool = False
    last_used: date | None = None
    last_reviewed: date | None = None
    # Which source resume(s) this listing was bootstrapped from. Bootstrap provenance,
    # useful when auditing an auto-grouped variant that looks wrong.
    source: list[str] = Field(default_factory=list)


class Listing(BaseModel):
    """One experience or project entry: a heading plus its ordered bullet slots."""

    id: str
    type: ListingType
    title: str
    context: ListingContext
    tags: list[str] = Field(default_factory=list)
    bullets: list[Bullet]
    metadata: ListingMetadata = Field(default_factory=ListingMetadata)

    @model_validator(mode="after")
    def _check_slots(self) -> Listing:
        slots = [b.slot for b in self.bullets]
        if len(slots) != len(set(slots)):
            raise ValueError(f"{self.id}: duplicate slot numbers {slots}")
        if not self.bullets:
            raise ValueError(f"{self.id}: listing has no bullets")
        return self

    def bullet(self, slot: int) -> Bullet:
        for b in self.bullets:
            if b.slot == slot:
                return b
        raise KeyError(f"{self.id}: no slot {slot}")

    @property
    def flexible_slots(self) -> list[Bullet]:
        return [b for b in self.bullets if b.role is BulletRole.FLEXIBLE]


class Template(BaseModel):
    """A broad, user-authored resume for one career direction.

    The listing set here is a *starting point*, not a constraint — the Listing Selector
    decides deltas from it. `default_variants` records which variant each flexible slot
    used in the human-approved version, so an untouched template run reproduces a resume
    the user already blessed.
    """

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    listing_ids: list[str]
    # listing_id -> {slot -> variant_id}
    default_variants: dict[str, dict[int, str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline types
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    """One weighted requirement extracted from a job posting by the JD Analyst."""

    text: str
    weight: float = Field(ge=0.0, le=1.0)
    keywords: list[str] = Field(default_factory=list)


class Requirements(BaseModel):
    role_title: str
    requirements: list[Requirement]

    @property
    def high_weight(self) -> list[Requirement]:
        return [r for r in self.requirements if r.weight >= 0.7]


class KeywordHit(BaseModel):
    """Why a variant scored what it did — the audit trail behind every selection."""

    requirement: str
    keyword: str
    weight: float
    kind: Literal["exact_phrase", "token_subset"]
    contribution: float


class Candidate(BaseModel):
    """A scored variant for one flexible slot. Produced by the Retriever, pure code."""

    listing_id: str
    slot: int
    variant_id: str
    score: float
    hits: list[KeywordHit] = Field(default_factory=list)


class GenerationMode(str, Enum):
    """How a generated bullet came to exist.

    `synthesis` recombines phrasing already present in that slot's approved variants and
    is the preferred path. `novel` writes content not grounded in any of them and is a
    fallback that carries real hallucination risk — tracked separately so it can be
    audited and measured.
    """

    SYNTHESIS = "synthesis"
    NOVEL = "novel"


class BulletSource(str, Enum):
    ESSENTIAL = "essential"
    SELECTED = "selected"
    GENERATED = "generated"


class SlotSelection(BaseModel):
    """The chosen filling for one slot, or a flagged gap."""

    listing_id: str
    slot: int
    variant_id: str | None = None
    is_gap: bool = False
    reason: str = ""

    @model_validator(mode="after")
    def _gap_xor_variant(self) -> SlotSelection:
        if self.is_gap and self.variant_id:
            raise ValueError(f"{self.listing_id} slot {self.slot}: gap cannot name a variant")
        if not self.is_gap and not self.variant_id:
            raise ValueError(f"{self.listing_id} slot {self.slot}: non-gap needs a variant_id")
        return self


class Selections(BaseModel):
    listing_ids: list[str]
    selections: list[SlotSelection]

    @property
    def gaps(self) -> list[SlotSelection]:
        return [s for s in self.selections if s.is_gap]


class DraftBullet(BaseModel):
    """One bullet in the assembled draft, with provenance intact."""

    slot: int
    source: BulletSource
    text: str
    variant_id: str | None = None
    generation_mode: GenerationMode | None = None
    matched_keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _provenance_consistent(self) -> DraftBullet:
        if self.source is BulletSource.GENERATED and self.generation_mode is None:
            raise ValueError(f"slot {self.slot}: generated bullet must declare generation_mode")
        if self.source is BulletSource.SELECTED and not self.variant_id:
            raise ValueError(f"slot {self.slot}: selected bullet must name its variant_id")
        return self


class DraftListing(BaseModel):
    listing_id: str
    bullets: list[DraftBullet]


class Draft(BaseModel):
    """The assembled resume, ordered. Output of the Drafter, input to Critic and Formatter."""

    listings: list[DraftListing]
    unaddressed_requirements: list[str] = Field(default_factory=list)


class Severity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"


class CritiqueIssue(BaseModel):
    category: Literal[
        "keyword_coverage",
        "groundedness",
        "repetition",
        "redundant_emphasis",
        "ordering",
    ]
    severity: Severity
    detail: str
    listing_id: str | None = None
    slot: int | None = None


class Critique(BaseModel):
    """The Critic's verdict. `passed` gates whether the draft proceeds to rendering."""

    passed: bool
    issues: list[CritiqueIssue] = Field(default_factory=list)
    low_coverage_warning: bool = False
    coverage_estimate: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def severity_score(self) -> int:
        """Total severity, used by the Supervisor to require strict improvement per round.

        A round that fixes a major issue but introduces two minor ones has not improved,
        and this ranking says so.
        """
        weights = {Severity.NONE: 0, Severity.MINOR: 1, Severity.MAJOR: 4}
        return sum(weights[i.severity] for i in self.issues)


# ---------------------------------------------------------------------------
# Outcome types
# ---------------------------------------------------------------------------


class RunStatus(str, Enum):
    OK = "ok"
    NEEDS_REVIEW = "needs_review"
    CONTENT_FAIL = "content_fail"
    COMPILE_ERROR = "compile_error"
    OVERFLOW = "overflow"


class CompileResult(BaseModel):
    """Ground truth on page count. The compiler is the only thing that decides this."""

    ok: bool
    page_count: int | None = None
    tex_path: str
    pdf_path: str | None = None
    error_log: str | None = None


class RunResult(BaseModel):
    status: RunStatus
    run_dir: str
    page_count: int | None = None
    unaddressed_requirements: list[str] = Field(default_factory=list)
    low_coverage_warning: bool = False
    message: str = ""
