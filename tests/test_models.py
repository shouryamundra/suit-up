"""Schema invariants for the library and pipeline types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from suitup.models import (
    Bullet,
    BulletRole,
    BulletSource,
    Critique,
    CritiqueIssue,
    DraftBullet,
    GenerationMode,
    Listing,
    ListingContext,
    ListingType,
    Severity,
    SlotSelection,
    Variant,
)


def make_listing(**overrides):
    base = dict(
        id="aruw_lead_swe_2025",
        type=ListingType.EXPERIENCE,
        title="Lead Software Engineer",
        context=ListingContext(org="ARUW", dates="Oct 2025 - Present"),
        bullets=[
            Bullet(slot=1, role=BulletRole.ESSENTIAL, text="Led a 15-engineer team."),
            Bullet(
                slot=2,
                role=BulletRole.FLEXIBLE,
                variants=[Variant(variant_id="v1", keywords=["yolo"], text="Improved mAP.")],
            ),
        ],
    )
    return Listing(**{**base, **overrides})


class TestBulletRoleShape:
    def test_essential_requires_text(self):
        with pytest.raises(ValidationError, match="requires `text`"):
            Bullet(slot=1, role=BulletRole.ESSENTIAL)

    def test_essential_rejects_variants(self):
        with pytest.raises(ValidationError, match="cannot have `variants`"):
            Bullet(
                slot=1,
                role=BulletRole.ESSENTIAL,
                text="Led a team.",
                variants=[Variant(variant_id="v1", text="x")],
            )

    def test_flexible_requires_variants(self):
        with pytest.raises(ValidationError, match="requires at least one variant"):
            Bullet(slot=2, role=BulletRole.FLEXIBLE)

    def test_flexible_rejects_bare_text(self):
        with pytest.raises(ValidationError, match="cannot have bare `text`"):
            Bullet(
                slot=2,
                role=BulletRole.FLEXIBLE,
                text="oops",
                variants=[Variant(variant_id="v1", text="x")],
            )

    def test_duplicate_variant_ids_rejected(self):
        with pytest.raises(ValidationError, match="duplicate variant_id"):
            Bullet(
                slot=2,
                role=BulletRole.FLEXIBLE,
                variants=[Variant(variant_id="v1", text="a"), Variant(variant_id="v1", text="b")],
            )


class TestTextNormalization:
    def test_variant_text_folds_whitespace(self):
        # YAML block scalars arrive with embedded newlines; scoring and hashing both
        # depend on a single canonical form.
        variant = Variant(variant_id="v1", text="Improved\n  mAP  from\n82% to 89%.\n")
        assert variant.text == "Improved mAP from 82% to 89%."

    def test_essential_text_folds_whitespace(self):
        bullet = Bullet(slot=1, role=BulletRole.ESSENTIAL, text="Led a\n  15-engineer team.\n")
        assert bullet.text == "Led a 15-engineer team."


class TestListing:
    def test_duplicate_slots_rejected(self):
        with pytest.raises(ValidationError, match="duplicate slot"):
            make_listing(
                bullets=[
                    Bullet(slot=1, role=BulletRole.ESSENTIAL, text="a"),
                    Bullet(slot=1, role=BulletRole.ESSENTIAL, text="b"),
                ]
            )

    def test_empty_bullets_rejected(self):
        with pytest.raises(ValidationError, match="no bullets"):
            make_listing(bullets=[])

    def test_flexible_slots_excludes_essential(self):
        assert [b.slot for b in make_listing().flexible_slots] == [2]

    def test_bullet_lookup_by_slot(self):
        assert make_listing().bullet(1).role is BulletRole.ESSENTIAL
        with pytest.raises(KeyError):
            make_listing().bullet(99)


class TestSlotSelection:
    def test_gap_cannot_name_a_variant(self):
        with pytest.raises(ValidationError, match="gap cannot name a variant"):
            SlotSelection(listing_id="x", slot=2, variant_id="v1", is_gap=True)

    def test_non_gap_requires_a_variant(self):
        with pytest.raises(ValidationError, match="needs a variant_id"):
            SlotSelection(listing_id="x", slot=2)


class TestDraftBulletProvenance:
    def test_generated_must_declare_mode(self):
        with pytest.raises(ValidationError, match="must declare generation_mode"):
            DraftBullet(slot=2, source=BulletSource.GENERATED, text="new bullet")

    def test_selected_must_name_variant(self):
        with pytest.raises(ValidationError, match="must name its variant_id"):
            DraftBullet(slot=2, source=BulletSource.SELECTED, text="reused bullet")

    def test_valid_generated_bullet(self):
        bullet = DraftBullet(
            slot=2,
            source=BulletSource.GENERATED,
            text="new bullet",
            generation_mode=GenerationMode.SYNTHESIS,
        )
        assert bullet.generation_mode is GenerationMode.SYNTHESIS


class TestListingTypes:
    def test_only_experience_and_projects_compete_for_space(self):
        # The count caps exist to fit one page. Structural sections are always present,
        # so counting them against the cap would silently squeeze out real content.
        assert ListingType.EXPERIENCE.competes_for_space
        assert ListingType.PROJECT.competes_for_space
        assert not ListingType.SKILLS.competes_for_space
        assert not ListingType.EDUCATION.competes_for_space
        assert not ListingType.ACHIEVEMENTS.competes_for_space

    def test_dated_listings_require_context(self):
        with pytest.raises(ValidationError, match="requires `context`"):
            Listing(
                id="x",
                type=ListingType.EXPERIENCE,
                title="t",
                bullets=[Bullet(slot=1, role=BulletRole.ESSENTIAL, text="a")],
            )

    def test_structural_listings_render_without_context(self):
        listing = Listing(
            id="technical_skills",
            type=ListingType.SKILLS,
            title="Technical Skills",
            bullets=[Bullet(slot=1, role=BulletRole.ESSENTIAL, label="Languages", text="C++")],
        )
        assert listing.context is None

    def test_variant_label_overrides_slot_label(self):
        # A skills slot is not one category with different contents — across tailorings it
        # is a different category heading entirely.
        bullet = Bullet(
            slot=2,
            role=BulletRole.FLEXIBLE,
            label="Fallback",
            variants=[
                Variant(variant_id="v1", label="Systems and Concurrency", text="Mutexes"),
                Variant(variant_id="v2", label="ML and Computer Vision", text="PyTorch"),
            ],
        )
        assert [v.label for v in bullet.variants] == [
            "Systems and Concurrency",
            "ML and Computer Vision",
        ]
        assert bullet.label == "Fallback"


class TestCritiqueSeverity:
    def _issue(self, severity):
        return CritiqueIssue(category="repetition", severity=severity, detail="d")

    def test_major_outweighs_multiple_minors(self):
        # The supervisor requires strict improvement. Trading one major for two minors
        # must register as progress, not a wash.
        major = Critique(passed=False, issues=[self._issue(Severity.MAJOR)])
        two_minor = Critique(passed=False, issues=[self._issue(Severity.MINOR)] * 2)
        assert major.severity_score > two_minor.severity_score

    def test_clean_critique_scores_zero(self):
        assert Critique(passed=True).severity_score == 0
