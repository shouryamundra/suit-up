"""Detect text overlap between bullets that could appear together.

Two bullets in the same listing may be rendered side by side, so shared phrasing between
them shows up as visible repetition in the PDF. Variants *within* one slot are alternatives
and never co-occur, so overlap there is expected and ignored.

A condensed variant that deliberately absorbs another slot declares it via
`Variant.supersedes`, which suppresses the warning for that pair. Everything else is an
authoring bug.

Pure code, no LLM — this runs in the library integrity tests and is also what the Critic's
repetition check is measured against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from suitup.models import BulletRole, Listing

# Words too common to signal repetition. Kept deliberately small: an aggressive stop list
# hides real duplication like "localization stack".
STOPWORDS = frozenset(
    """a an the and or of to in for with by on at from as is are was were be been being
    that this it its into via while then so than not no using used use across over under
    after before during through each other more most both all any""".split()
)

TRIGRAM_THRESHOLD = 0.12
CONTENT_THRESHOLD = 0.45


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9.+#]+", text.lower()) if w not in STOPWORDS]


def _shingles(text: str, n: int = 3) -> set[str]:
    words = _words(text)
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _content(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 2}


@dataclass(frozen=True)
class Overlap:
    listing_id: str
    left: str
    right: str
    trigram: float
    content: float
    shared: tuple[str, ...]
    involves_essential: bool

    def __str__(self) -> str:
        tag = " [ESSENTIAL]" if self.involves_essential else ""
        shared = ", ".join(self.shared[:3]) or "-"
        return (
            f"{self.listing_id}: {self.left} vs {self.right}{tag} "
            f"(trigram {self.trigram:.2f}, content {self.content:.2f}; shared: {shared})"
        )


def _units(
    listing: Listing, chosen: dict[int, str] | None = None
) -> list[tuple[int, str, str, frozenset[int]]]:
    """(slot, id, text, supersedes) for every bullet, or only those a selection renders."""
    units: list[tuple[int, str, str, frozenset[int]]] = []
    for bullet in sorted(listing.bullets, key=lambda b: b.slot):
        if bullet.role is BulletRole.ESSENTIAL:
            units.append((bullet.slot, "essential", bullet.text or "", frozenset()))
            continue
        for variant in bullet.variants:
            if chosen is not None and chosen.get(bullet.slot) != variant.variant_id:
                continue
            units.append(
                (bullet.slot, variant.variant_id, variant.text, frozenset(variant.supersedes))
            )
    return units


def _compare(
    listing_id: str,
    units: list[tuple[int, str, str, frozenset[int]]],
    trigram_threshold: float,
    content_threshold: float,
) -> list[Overlap]:
    found = []
    for (slot_a, id_a, text_a, sup_a), (slot_b, id_b, text_b, sup_b) in combinations(units, 2):
        if slot_a == slot_b:
            continue  # alternatives for one slot never co-occur
        if slot_b in sup_a or slot_a in sup_b:
            continue  # a declared merge; selecting one drops the other

        sh_a, sh_b = _shingles(text_a), _shingles(text_b)
        co_a, co_b = _content(text_a), _content(text_b)
        if not sh_a or not sh_b or not co_a or not co_b:
            continue

        trigram = len(sh_a & sh_b) / min(len(sh_a), len(sh_b))
        content = len(co_a & co_b) / min(len(co_a), len(co_b))
        if trigram >= trigram_threshold or content >= content_threshold:
            found.append(
                Overlap(
                    listing_id=listing_id,
                    left=f"slot{slot_a}:{id_a}",
                    right=f"slot{slot_b}:{id_b}",
                    trigram=trigram,
                    content=content,
                    shared=tuple(sorted(sh_a & sh_b)),
                    involves_essential=(id_a == "essential" or id_b == "essential"),
                )
            )
    return found


def find_overlaps(
    listing: Listing,
    trigram_threshold: float = TRIGRAM_THRESHOLD,
    content_threshold: float = CONTENT_THRESHOLD,
) -> list[Overlap]:
    """Every undeclared overlap in the listing, across all variants.

    Advisory: a flagged pair is only a real defect if some selection renders both. Use
    `overlaps_against_essential` and `overlaps_in_render` for the checks that must pass.
    """
    return _compare(listing.id, _units(listing), trigram_threshold, content_threshold)


def overlaps_against_essential(
    listing: Listing,
    trigram_threshold: float = TRIGRAM_THRESHOLD,
    content_threshold: float = CONTENT_THRESHOLD,
) -> list[Overlap]:
    """Overlap between an essential bullet and any variant.

    Strict, because an essential bullet renders in *every* tailoring — so overlapping one
    is repetition the user will see on every single resume, regardless of selection. This
    is the check that catches an essential bullet being silently restated as a flexible
    variant elsewhere in the listing.
    """
    return [
        o
        for o in _compare(listing.id, _units(listing), trigram_threshold, content_threshold)
        if o.involves_essential
    ]


def overlaps_in_render(
    listing: Listing,
    chosen: dict[int, str],
    trigram_threshold: float = TRIGRAM_THRESHOLD,
    content_threshold: float = CONTENT_THRESHOLD,
) -> list[Overlap]:
    """Overlap among the bullets one concrete selection actually renders."""
    return _compare(listing.id, _units(listing, chosen), trigram_threshold, content_threshold)
