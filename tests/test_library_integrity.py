"""Integrity checks on the real library in `library/`.

The library is hand-edited YAML by design, so these run against the actual files rather
than fixtures. They are the safety net that catches a typo'd variant id or a template
pointing at a slot that no longer exists — failures a schema check alone would miss,
because each file is individually valid and only the references between them are broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from suitup.models import BulletRole, Listing, ListingType, Template

LIBRARY = Path(__file__).resolve().parent.parent / "library"


def load_listings() -> dict[str, Listing]:
    return {
        (listing := Listing(**yaml.safe_load(p.read_text()))).id: listing
        for p in sorted((LIBRARY / "listings").glob("*.yaml"))
    }


def load_templates() -> dict[str, Template]:
    return {
        (template := Template(**yaml.safe_load(p.read_text()))).id: template
        for p in sorted((LIBRARY / "templates").glob("*.yaml"))
    }


@pytest.fixture(scope="module")
def listings():
    return load_listings()


@pytest.fixture(scope="module")
def templates():
    return load_templates()


class TestListings:
    def test_every_listing_file_parses(self, listings):
        assert listings, "no listings found"

    def test_filename_matches_listing_id(self):
        for path in sorted((LIBRARY / "listings").glob("*.yaml")):
            listing = Listing(**yaml.safe_load(path.read_text()))
            assert path.stem == listing.id, f"{path.name} declares id {listing.id!r}"

    def test_variant_ids_unique_within_a_listing(self, listings):
        # Not enforced by the schema, which only checks uniqueness per slot. A duplicate
        # across slots would make composition keys ambiguous.
        for listing in listings.values():
            seen = [v.variant_id for b in listing.bullets for v in b.variants]
            dupes = {v for v in seen if seen.count(v) > 1}
            assert not dupes, f"{listing.id}: variant ids reused across slots: {dupes}"

    def test_every_listing_has_exactly_one_essential_opener(self, listings):
        # Slot 1 sets scope and is what survives when everything else is cut.
        for listing in listings.values():
            first = min(listing.bullets, key=lambda b: b.slot)
            assert first.role is BulletRole.ESSENTIAL, (
                f"{listing.id}: slot {first.slot} is flexible; the opening bullet should "
                f"be essential so the listing always says what it was"
            )

    def test_flexible_variants_carry_keywords(self, listings):
        # The Retriever scores lexically. A variant with no keywords can never win a slot.
        for listing in listings.values():
            for bullet in listing.flexible_slots:
                for variant in bullet.variants:
                    assert variant.keywords, (
                        f"{listing.id} slot {bullet.slot} variant {variant.variant_id} "
                        f"has no keywords and would never be selected"
                    )

    def test_skills_variants_declare_a_label(self, listings):
        # The skills section renders as "label: text" rows, so a missing label would emit
        # a bare list with no category heading.
        for listing in listings.values():
            if listing.type is not ListingType.SKILLS:
                continue
            for bullet in listing.bullets:
                for variant in bullet.variants:
                    assert variant.label, (
                        f"{listing.id} slot {bullet.slot} variant {variant.variant_id} "
                        f"needs a label"
                    )


class TestTemplates:
    def test_every_template_file_parses(self, templates):
        assert templates, "no templates found"

    def test_template_listings_exist(self, templates, listings):
        for template in templates.values():
            for listing_id in template.listing_ids:
                assert listing_id in listings, f"{template.id} references unknown {listing_id}"

    def test_default_variants_resolve(self, templates, listings):
        for template in templates.values():
            for listing_id, chosen in template.default_variants.items():
                assert (
                    listing_id in template.listing_ids
                ), f"{template.id} picks variants for {listing_id}, which it does not include"
                listing = listings[listing_id]
                for slot, variant_id in chosen.items():
                    bullet = listing.bullet(slot)  # raises KeyError if the slot is gone
                    assert bullet.role is BulletRole.FLEXIBLE, (
                        f"{template.id}/{listing_id}: slot {slot} is essential and cannot "
                        f"have a variant chosen"
                    )
                    available = {v.variant_id for v in bullet.variants}
                    assert variant_id in available, (
                        f"{template.id}/{listing_id} slot {slot}: no variant "
                        f"{variant_id!r} (have {sorted(available)})"
                    )

    def test_templates_respect_the_configured_caps(self, templates, listings):
        # Each template reconstructs a resume that already fit on one page, so a template
        # exceeding the caps means the caps are wrong, not the template.
        from suitup.config import load_config

        caps = load_config(root=LIBRARY.parent).caps
        for template in templates.values():
            by_type: dict[ListingType, int] = {}
            for listing_id in template.listing_ids:
                listing_type = listings[listing_id].type
                by_type[listing_type] = by_type.get(listing_type, 0) + 1
            assert by_type.get(ListingType.EXPERIENCE, 0) <= caps.max_experience
            assert by_type.get(ListingType.PROJECT, 0) <= caps.max_projects

    def test_no_listing_exceeds_the_bullet_cap(self, templates, listings):
        from suitup.config import load_config

        cap = load_config(root=LIBRARY.parent).caps.max_bullets_per_listing
        for template in templates.values():
            for listing_id in template.listing_ids:
                listing = listings[listing_id]
                if not listing.type.competes_for_space:
                    continue
                essential = sum(1 for b in listing.bullets if b.role is BulletRole.ESSENTIAL)
                total = essential + len(template.default_variants.get(listing_id, {}))
                assert total <= cap, f"{template.id}/{listing_id} renders {total} bullets"


class TestSeedConfigurations:
    def test_configurations_file_is_valid_json(self):
        import json

        configs = json.loads((LIBRARY / "configurations.json").read_text())
        assert configs, "no seed configurations"
        for key, entry in configs.items():
            assert entry["approved"] is True
            assert entry["listing_keys"], f"{key} has no listing keys"
            assert key == "||".join(entry["listing_keys"]), (
                "the composition key must be derivable from its listing keys, or a cache "
                "lookup will never match what the pipeline computes"
            )

    def test_one_seed_per_template(self, templates):
        import json

        configs = json.loads((LIBRARY / "configurations.json").read_text())
        seeded = {entry["template_id"] for entry in configs.values()}
        assert seeded == set(templates), f"seeded {seeded}, templates {set(templates)}"
