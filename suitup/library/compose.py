"""Turn a listing selection into a `Draft`.

Sits between the library and the renderer. The Drafter will normally produce a Draft with
polished text, but a template's default selection is already a complete, human-approved
composition — so this builds one directly, with no model involved.

That path matters twice: it is how a cache hit re-renders a resume the user already
approved, and it is what the golden end-to-end test exercises to prove the deterministic
core works without any LLM.
"""

from __future__ import annotations

from suitup.library.configs import listing_key, resume_key
from suitup.library.loader import Library
from suitup.models import (
    BulletRole,
    BulletSource,
    Draft,
    DraftBullet,
    DraftListing,
    Template,
)


class CompositionError(RuntimeError):
    """A selection cannot be realised against the library."""


def resolve_selection(
    library: Library,
    listing_id: str,
    chosen: dict[int, str],
) -> DraftListing:
    """Build one listing's draft content from a variant selection.

    Essential bullets are always included. Flexible slots appear only if selected, and a
    slot superseded by a chosen variant is dropped — selecting a condensed bullet that
    already absorbs another slot must not also emit the slot it absorbed.
    """
    listing = library.listing(listing_id)

    superseded: set[int] = set()
    for slot, variant_id in chosen.items():
        try:
            bullet = listing.bullet(slot)
        except KeyError:
            raise CompositionError(f"{listing_id}: no slot {slot}") from None
        for variant in bullet.variants:
            if variant.variant_id == variant_id:
                superseded |= set(variant.supersedes)

    clashes = superseded & set(chosen)
    if clashes:
        raise CompositionError(
            f"{listing_id}: selection includes slot(s) {sorted(clashes)} that a chosen "
            f"variant already absorbs; the same content would appear twice"
        )

    bullets: list[DraftBullet] = []
    for bullet in sorted(listing.bullets, key=lambda b: b.slot):
        if bullet.role is BulletRole.ESSENTIAL:
            bullets.append(
                DraftBullet(slot=bullet.slot, source=BulletSource.ESSENTIAL, text=bullet.text or "")
            )
            continue

        if bullet.slot not in chosen:
            continue

        variant_id = chosen[bullet.slot]
        variant = next((v for v in bullet.variants if v.variant_id == variant_id), None)
        if variant is None:
            raise CompositionError(
                f"{listing_id} slot {bullet.slot}: no variant {variant_id!r}; "
                f"available: {sorted(v.variant_id for v in bullet.variants)}"
            )
        bullets.append(
            DraftBullet(
                slot=bullet.slot,
                source=BulletSource.SELECTED,
                text=variant.text,
                variant_id=variant.variant_id,
                matched_keywords=list(variant.keywords),
            )
        )

    return DraftListing(listing_id=listing_id, bullets=bullets)


def draft_from_selection(
    library: Library,
    listing_ids: list[str],
    selections: dict[str, dict[int, str]],
    unaddressed: list[str] | None = None,
) -> Draft:
    """Assemble a full draft. `listing_ids` order is the resume's display order."""
    return Draft(
        listings=[resolve_selection(library, lid, selections.get(lid, {})) for lid in listing_ids],
        unaddressed_requirements=unaddressed or [],
    )


def draft_from_template(library: Library, template: Template | str) -> Draft:
    """Realise a template's default composition exactly as authored."""
    if isinstance(template, str):
        template = library.template(template)
    return draft_from_selection(library, template.listing_ids, dict(template.default_variants))


def key_for_draft(library: Library, draft: Draft) -> str:
    """The composition key for a draft, for looking it up in the approval cache."""
    keys = []
    for draft_listing in draft.listings:
        listing = library.listing(draft_listing.listing_id)
        chosen = {
            b.slot: b.variant_id
            for b in draft_listing.bullets
            if b.source is BulletSource.SELECTED and b.variant_id
        }
        keys.append(listing_key(listing, chosen))
    return resume_key(keys)
