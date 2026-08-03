"""Score flexible bullet variants against posting requirements. Pure code, no LLM.

Every number this produces is written to the run folder, because the PRD's auditability
requirement means a selection must be explicable without asking a model why. That also
makes this the most heavily tested module in the project: a scoring bug does not crash,
it quietly produces a worse resume.

Scoring is deliberately lexical and unclever. For each requirement, each of its keywords
is matched against the variant:

* **exact phrase** — the keyword appears verbatim in the variant's normalised text or
  its declared keyword list. Full credit.
* **token subset** — every word of the keyword appears in the variant, but not adjacently.
  Partial credit, because "inference latency" matching a bullet that says "latency" and
  "inference" separately is real but weaker evidence.

A requirement contributes at most once, at its best match, so a requirement with eight
keywords cannot outweigh a more important one with two.
"""

from __future__ import annotations

import re

from suitup.config import Thresholds
from suitup.models import Bullet, BulletRole, Candidate, KeywordHit, Listing, Requirement, Variant

_PUNCTUATION = re.compile(r"[^a-z0-9+#.\s]")
# A period survives only between digits, so "9.0ms" and "mAP@0.5" keep their decimals
# while sentence-ending periods go. Without this, "latency." would not token-match
# "latency".
_NON_DECIMAL_PERIOD = re.compile(r"(?<!\d)\.|\.(?!\d)")
_WHITESPACE = re.compile(r"\s+")
# LaTeX the library legitimately contains: \texttimes{}, \(\pm\), $\to$, \textbf{...}.
# Stripped before matching so markup never counts as content.
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+\s*(\{[^{}]*\})?")
_LATEX_MATH = re.compile(r"\$[^$]*\$")


def normalize(text: str) -> str:
    """Lowercase, strip LaTeX and punctuation, collapse whitespace.

    Both sides of every comparison go through this, so `TensorRT FP16` in a posting and
    `a TensorRT FP16 engine.` in a variant compare equal on the phrase.
    """
    text = _LATEX_MATH.sub(" ", text)
    text = _LATEX_COMMAND.sub(" ", text)
    text = text.lower()
    text = _PUNCTUATION.sub(" ", text)
    text = _NON_DECIMAL_PERIOD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def _haystack(variant: Variant) -> tuple[str, set[str]]:
    """Normalised text of a variant plus its declared keywords, and its token set.

    Declared keywords are folded into the searchable text so a variant can be found by a
    term its prose implies but does not literally contain.
    """
    combined = " ".join([variant.text, *variant.keywords])
    normalised = normalize(combined)
    return normalised, set(normalised.split())


def score_variant(
    variant: Variant,
    requirements: list[Requirement],
    thresholds: Thresholds,
) -> tuple[float, list[KeywordHit]]:
    """Return the variant's score and the hits that produced it."""
    text, tokens = _haystack(variant)
    total = 0.0
    hits: list[KeywordHit] = []

    for requirement in requirements:
        best: KeywordHit | None = None

        for keyword in requirement.keywords:
            normalised = normalize(keyword)
            if not normalised:
                continue

            if normalised in text:
                kind, weight = "exact_phrase", thresholds.exact_phrase_weight
            elif set(normalised.split()) <= tokens:
                kind, weight = "token_subset", thresholds.token_subset_weight
            else:
                continue

            contribution = requirement.weight * weight
            if best is None or contribution > best.contribution:
                best = KeywordHit(
                    requirement=requirement.text,
                    keyword=keyword,
                    weight=requirement.weight,
                    kind=kind,  # type: ignore[arg-type]
                    contribution=contribution,
                )

        # One contribution per requirement, at its strongest match. Without this a
        # requirement listing many near-synonyms would dominate a more important one.
        if best is not None:
            total += best.contribution
            hits.append(best)

    hits.sort(key=lambda h: -h.contribution)
    return round(total, 6), hits


def score_slot(
    listing: Listing,
    bullet: Bullet,
    requirements: list[Requirement],
    thresholds: Thresholds,
) -> list[Candidate]:
    """Score every variant of one flexible slot, best first."""
    if bullet.role is BulletRole.ESSENTIAL:
        raise ValueError(
            f"{listing.id} slot {bullet.slot} is essential; essential bullets are always "
            f"included and are never scored"
        )

    candidates = []
    for variant in bullet.variants:
        score, hits = score_variant(variant, requirements, thresholds)
        candidates.append(
            Candidate(
                listing_id=listing.id,
                slot=bullet.slot,
                variant_id=variant.variant_id,
                score=score,
                hits=hits,
            )
        )
    # Ties broken by variant_id so the output is deterministic run to run — an unstable
    # order would make the run archives diff noisily for no reason.
    candidates.sort(key=lambda c: (-c.score, c.variant_id))
    return candidates


def retrieve(
    listings: list[Listing],
    requirements: list[Requirement],
    thresholds: Thresholds,
) -> dict[tuple[str, int], list[Candidate]]:
    """Score every flexible slot of every listing. Essential bullets are skipped."""
    out: dict[tuple[str, int], list[Candidate]] = {}
    for listing in listings:
        for bullet in listing.flexible_slots:
            out[(listing.id, bullet.slot)] = score_slot(listing, bullet, requirements, thresholds)
    return out


def score_listing(
    listing: Listing,
    requirements: list[Requirement],
    thresholds: Thresholds,
) -> float:
    """A listing's relevance: the sum of its best score per flexible slot.

    Used to rank whole listings when choosing which appear. Summing the best variant per
    slot rather than every variant stops a listing with many alternatives for one slot
    from outranking a listing that covers more distinct ground.
    """
    total = 0.0
    for bullet in listing.flexible_slots:
        scores = [score_variant(v, requirements, thresholds)[0] for v in bullet.variants]
        if scores:
            total += max(scores)
    return round(total, 6)


def score_template_tags(
    tags: list[str],
    requirements: list[Requirement],
    thresholds: Thresholds,
) -> float:
    """Lexical fit between a template's tags and the posting.

    This is the template matching the PRD gave to an LLM node. Comparing two bags of
    keywords does not need a model, and doing it in code keeps the choice inspectable.
    """
    pseudo = Variant(variant_id="_tags", keywords=tags, text=" ".join(tags))
    return score_variant(pseudo, requirements, thresholds)[0]
