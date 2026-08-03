"""Lexical scoring.

Everything downstream depends on these numbers being right, and a scoring bug does not
crash — it quietly produces a worse resume. So this is the heaviest suite in the project.
"""

from __future__ import annotations

import pytest

from suitup.agents.retriever import (
    normalize,
    retrieve,
    score_listing,
    score_slot,
    score_template_tags,
    score_variant,
)
from suitup.models import Requirement, Variant


class TestNormalize:
    def test_lowercases_and_collapses_whitespace(self):
        assert normalize("  TensorRT   FP16\n engine ") == "tensorrt fp16 engine"

    def test_strips_punctuation(self):
        assert normalize("latency, 30%: down!") == "latency 30 down"

    def test_keeps_plus_and_hash(self):
        # C++ and C# are real skills; stripping these would make them unmatchable.
        assert normalize("C++ and C#") == "c++ and c#"

    def test_strips_latex_commands(self):
        # Library text is authored LaTeX; markup must never count as content.
        assert normalize(r"Scaled 3\texttimes{} to 30 streams") == "scaled 3 to 30 streams"

    def test_strips_latex_math(self):
        assert normalize(r"latency (9.0ms $\to$ 6.3ms)") == "latency 9.0ms 6.3ms"

    def test_strips_textbf_but_keeps_its_content_out(self):
        # \textbf{...} is markup; the braces content is dropped with the command since
        # keeping it would let a label leak into scoring.
        assert "textbf" not in normalize(r"\textbf{Relevant Coursework:} Algorithms")


class TestScoreVariant:
    def test_exact_phrase_scores_full_weight(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text="Cut inference latency 30%.")
        req = [Requirement(text="r", weight=1.0, keywords=["inference latency"])]
        score, hits = score_variant(variant, req, thresholds)
        assert score == pytest.approx(1.0)
        assert hits[0].kind == "exact_phrase"

    def test_token_subset_scores_partial_weight(self, thresholds):
        # Words present but not adjacent: real evidence, weaker than a phrase match.
        variant = Variant(variant_id="v", keywords=[], text="Latency fell after inference tuning.")
        req = [Requirement(text="r", weight=1.0, keywords=["inference latency"])]
        score, hits = score_variant(variant, req, thresholds)
        assert score == pytest.approx(thresholds.token_subset_weight)
        assert hits[0].kind == "token_subset"

    def test_no_match_scores_zero(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text="Wrote documentation.")
        req = [Requirement(text="r", weight=1.0, keywords=["kubernetes"])]
        assert score_variant(variant, req, thresholds)[0] == 0.0

    def test_declared_keywords_are_searchable(self, thresholds):
        # A variant can be found by a term its prose implies but never says.
        variant = Variant(variant_id="v", keywords=["TensorRT"], text="Quantised the detector.")
        req = [Requirement(text="r", weight=1.0, keywords=["TensorRT"])]
        assert score_variant(variant, req, thresholds)[0] == pytest.approx(1.0)

    def test_requirement_weight_scales_contribution(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text="Cut inference latency 30%.")
        low = [Requirement(text="r", weight=0.25, keywords=["inference latency"])]
        assert score_variant(variant, low, thresholds)[0] == pytest.approx(0.25)

    def test_requirement_contributes_once_at_its_best_match(self, thresholds):
        # Eight near-synonyms for one requirement must not outweigh a more important
        # requirement that happens to list two.
        variant = Variant(
            variant_id="v",
            keywords=[],
            text="Cut inference latency with TensorRT on embedded Jetson hardware.",
        )
        req = [
            Requirement(
                text="r",
                weight=1.0,
                keywords=["inference latency", "TensorRT", "embedded", "Jetson"],
            )
        ]
        score, hits = score_variant(variant, req, thresholds)
        assert score == pytest.approx(1.0), "four keyword hits must still contribute once"
        assert len(hits) == 1

    def test_best_match_wins_within_a_requirement(self, thresholds):
        # A phrase hit and a subset hit on the same requirement: the phrase should win.
        variant = Variant(
            variant_id="v", keywords=[], text="Reduced inference latency; embedded work too."
        )
        req = [
            Requirement(text="r", weight=1.0, keywords=["embedded systems", "inference latency"])
        ]
        _, hits = score_variant(variant, req, thresholds)
        assert hits[0].kind == "exact_phrase"
        assert hits[0].keyword == "inference latency"

    def test_multiple_requirements_accumulate(self, thresholds):
        variant = Variant(
            variant_id="v", keywords=[], text="Cut inference latency and raised mAP to 89%."
        )
        req = [
            Requirement(text="a", weight=1.0, keywords=["inference latency"]),
            Requirement(text="b", weight=0.5, keywords=["mAP"]),
        ]
        assert score_variant(variant, req, thresholds)[0] == pytest.approx(1.5)

    def test_hits_are_ordered_by_contribution(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text="inference latency and mAP")
        req = [
            Requirement(text="small", weight=0.2, keywords=["mAP"]),
            Requirement(text="big", weight=1.0, keywords=["inference latency"]),
        ]
        _, hits = score_variant(variant, req, thresholds)
        assert [h.requirement for h in hits] == ["big", "small"]

    def test_empty_keyword_is_ignored(self, thresholds):
        # An empty string is a substring of everything and would match unconditionally.
        variant = Variant(variant_id="v", keywords=[], text="Anything at all.")
        req = [Requirement(text="r", weight=1.0, keywords=["", "   "])]
        assert score_variant(variant, req, thresholds)[0] == 0.0

    def test_matching_ignores_case_and_punctuation(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text="a TENSORRT, FP16 engine.")
        req = [Requirement(text="r", weight=1.0, keywords=["tensorrt fp16"])]
        assert score_variant(variant, req, thresholds)[0] == pytest.approx(1.0)

    def test_latex_markup_does_not_block_a_match(self, thresholds):
        variant = Variant(variant_id="v", keywords=[], text=r"Scaled 3\texttimes{} to 30 streams.")
        req = [Requirement(text="r", weight=1.0, keywords=["30 streams"])]
        assert score_variant(variant, req, thresholds)[0] == pytest.approx(1.0)


class TestScoreSlot:
    def test_ranks_variants_best_first(self, sample_listing, requirements, thresholds):
        bullet = sample_listing.bullet(2)
        candidates = score_slot(sample_listing, bullet, requirements, thresholds)
        assert [c.variant_id for c in candidates] == ["v1_latency", "v2_accuracy"]
        assert candidates[0].score > candidates[1].score

    def test_ties_break_deterministically(self, sample_listing, thresholds):
        # An unstable order would make run archives diff noisily for no reason.
        bullet = sample_listing.bullet(2)
        nothing = [Requirement(text="r", weight=1.0, keywords=["unrelated"])]
        first = [c.variant_id for c in score_slot(sample_listing, bullet, nothing, thresholds)]
        second = [c.variant_id for c in score_slot(sample_listing, bullet, nothing, thresholds)]
        assert first == second == sorted(first)

    def test_scoring_an_essential_slot_is_an_error(self, sample_listing, requirements, thresholds):
        with pytest.raises(ValueError, match="essential"):
            score_slot(sample_listing, sample_listing.bullet(1), requirements, thresholds)

    def test_candidates_carry_their_audit_trail(self, sample_listing, requirements, thresholds):
        candidates = score_slot(sample_listing, sample_listing.bullet(2), requirements, thresholds)
        best = candidates[0]
        assert best.hits, "a scoring variant must record why it scored"
        assert best.listing_id == "demo_role"
        assert best.slot == 2


class TestRetrieve:
    def test_covers_every_flexible_slot_and_no_essential_one(
        self, sample_listing, requirements, thresholds
    ):
        result = retrieve([sample_listing], requirements, thresholds)
        assert set(result) == {("demo_role", 2), ("demo_role", 3), ("demo_role", 4)}


class TestScoreListing:
    def test_sums_the_best_variant_per_slot(self, sample_listing, requirements, thresholds):
        # Summing best-per-slot, not all variants: a listing with many alternatives for one
        # slot must not outrank one that covers more distinct ground.
        total = score_listing(sample_listing, requirements, thresholds)
        slot2_best = max(
            score_variant(v, requirements, thresholds)[0] for v in sample_listing.bullet(2).variants
        )
        assert total >= slot2_best
        assert total < slot2_best * len(sample_listing.bullet(2).variants) + 1

    def test_irrelevant_listing_scores_zero(self, sample_listing, thresholds):
        unrelated = [Requirement(text="r", weight=1.0, keywords=["actuarial modelling"])]
        assert score_listing(sample_listing, unrelated, thresholds) == 0.0


class TestScoreTemplateTags:
    def test_matching_tags_score(self, thresholds, requirements):
        assert score_template_tags(["TensorRT", "embedded"], requirements, thresholds) > 0

    def test_unrelated_tags_score_zero(self, thresholds, requirements):
        assert score_template_tags(["marketing", "copywriting"], requirements, thresholds) == 0.0
