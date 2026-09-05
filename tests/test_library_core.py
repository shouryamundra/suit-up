"""Loader, composition keys, the approval cache, and draft composition."""

from __future__ import annotations

import json

import pytest

from tests.conftest import all_template_ids
import yaml

from suitup.config import Paths
from suitup.library.compose import (
    CompositionError,
    draft_from_selection,
    draft_from_template,
    key_for_draft,
    resolve_selection,
)
from suitup.library.configs import (
    ConfigurationCache,
    listing_key,
    resume_key,
    split_resume_key,
)
from suitup.library.loader import Library, LibraryError, load_library
from suitup.models import BulletSource


def write_library(
    tmp_path, listings: dict, templates: dict | None = None, profile: dict | None = None
):
    paths = Paths.from_root(tmp_path)
    paths.listings.mkdir(parents=True)
    paths.templates.mkdir(parents=True)
    for name, data in listings.items():
        (paths.listings / f"{name}.yaml").write_text(yaml.safe_dump(data))
    for name, data in (templates or {}).items():
        (paths.templates / f"{name}.yaml").write_text(yaml.safe_dump(data))
    paths.profile.write_text(
        yaml.safe_dump(profile or {"name": "Test", "links": [{"label": "e", "url": "mailto:e"}]})
    )
    return paths


MINIMAL = {
    "id": "demo",
    "type": "experience",
    "title": "Engineer",
    "context": {"org": "Corp", "dates": "2025"},
    "bullets": [
        {"slot": 1, "role": "essential", "text": "Led a team."},
        {
            "slot": 2,
            "role": "flexible",
            "variants": [{"variant_id": "v1", "keywords": ["a"], "text": "Did a thing."}],
        },
    ],
}


class TestLoader:
    def test_loads_the_real_library(self, library):
        assert library.listings and library.templates
        assert library.profile.name

    def test_selectable_excludes_structural_sections(self, library):
        ids = {listing.id for listing in library.selectable}
        assert "skills" not in ids
        assert "education_state_u" not in ids
        assert "backend_intern" in ids

    def test_unknown_listing_names_what_exists(self, library):
        with pytest.raises(LibraryError, match="unknown listing"):
            library.listing("nope")

    def test_filename_must_match_id(self, tmp_path):
        paths = write_library(tmp_path, {"wrong_name": MINIMAL})
        with pytest.raises(LibraryError, match="filename and id must match"):
            load_library(paths)

    def test_malformed_yaml_names_the_file(self, tmp_path):
        paths = write_library(tmp_path, {"demo": MINIMAL})
        (paths.listings / "broken.yaml").write_text("id: [unclosed\n")
        with pytest.raises(LibraryError, match="broken.yaml"):
            load_library(paths)

    def test_schema_violation_names_the_file(self, tmp_path):
        bad = {**MINIMAL, "id": "bad", "bullets": [{"slot": 1, "role": "essential"}]}
        paths = write_library(tmp_path, {"bad": bad})
        with pytest.raises(LibraryError, match="bad.yaml"):
            load_library(paths)

    def test_empty_library_is_an_error(self, tmp_path):
        paths = write_library(tmp_path, {})
        with pytest.raises(LibraryError, match="no listing files"):
            load_library(paths)

    def test_missing_profile_is_an_error(self, tmp_path):
        paths = write_library(tmp_path, {"demo": MINIMAL})
        paths.profile.unlink()
        with pytest.raises(LibraryError, match="no profile"):
            load_library(paths)

    def test_template_referencing_a_missing_listing_fails(self, tmp_path):
        template = {"id": "t", "name": "T", "listing_ids": ["ghost"]}
        paths = write_library(tmp_path, {"demo": MINIMAL}, {"t": template})
        with pytest.raises(LibraryError, match="unknown listing 'ghost'"):
            load_library(paths)

    def test_template_referencing_a_missing_variant_fails(self, tmp_path):
        template = {
            "id": "t",
            "name": "T",
            "listing_ids": ["demo"],
            "default_variants": {"demo": {2: "v_missing"}},
        }
        paths = write_library(tmp_path, {"demo": MINIMAL}, {"t": template})
        with pytest.raises(LibraryError, match="no variant 'v_missing'"):
            load_library(paths)

    def test_template_choosing_a_variant_for_an_unincluded_listing_fails(self, tmp_path):
        other = {**MINIMAL, "id": "other"}
        template = {
            "id": "t",
            "name": "T",
            "listing_ids": ["demo"],
            "default_variants": {"other": {2: "v1"}},
        }
        paths = write_library(tmp_path, {"demo": MINIMAL, "other": other}, {"t": template})
        with pytest.raises(LibraryError, match="does not include"):
            load_library(paths)


class TestCompositionKeys:
    def test_includes_essential_and_selected_slots(self, sample_listing):
        key = listing_key(sample_listing, {2: "v1_latency"})
        assert key == "demo_role|slot1:essential|slot2:v1_latency"

    def test_omitted_slots_change_the_key(self, sample_listing):
        # Dropping a bullet changes the composition, so it must change the key or the
        # cache would treat two different resumes as the same.
        with_slot = listing_key(sample_listing, {2: "v1_latency", 4: "v1_tooling"})
        without = listing_key(sample_listing, {2: "v1_latency"})
        assert with_slot != without

    def test_slots_are_ordered_numerically_not_by_insertion(self, sample_listing):
        # dict ordering must not leak into the key, or a cache hit would depend on the
        # order the selector happened to fill slots in.
        forward = listing_key(sample_listing, {2: "v1_latency", 4: "v1_tooling"})
        reverse = listing_key(sample_listing, {4: "v1_tooling", 2: "v1_latency"})
        assert forward == reverse

    def test_resume_key_preserves_listing_order(self):
        # Two resumes with the same content in a different order are different resumes.
        assert resume_key(["a", "b"]) != resume_key(["b", "a"])

    def test_resume_key_round_trips(self):
        keys = ["demo|slot1:essential", "other|slot1:essential|slot2:v1"]
        assert split_resume_key(resume_key(keys)) == keys

    def test_matches_the_seeded_format(self, paths):
        # The bootstrap seeds were written by a throwaway script. If this module ever
        # disagrees with them, no cache lookup will hit and nothing will say why.
        seeds = json.loads(paths.configurations.read_text())
        for key, entry in seeds.items():
            assert key == resume_key(entry["listing_keys"])

    def test_seeded_keys_are_reproducible_from_the_library(self, library, paths):
        seeds = json.loads(paths.configurations.read_text())
        for entry in seeds.values():
            template = library.template(entry["template_id"])
            rebuilt = key_for_draft(library, draft_from_template(library, template))
            assert rebuilt == resume_key(
                entry["listing_keys"]
            ), f"{entry['template_id']}: composition key drifted from its seed"


class TestConfigurationCache:
    def test_missing_file_loads_empty(self, tmp_path):
        cache = ConfigurationCache.load(tmp_path / "nope.json")
        assert len(cache) == 0

    def test_reads_the_real_seeds(self, paths, library):
        # Count is derived, not hardcoded: one approved seed per template, so adding a
        # career direction should not require editing this test.
        cache = ConfigurationCache.load(paths.configurations)
        assert len(cache) == len(library.templates)
        assert all(cache.is_approved(key) for key in json.loads(paths.configurations.read_text()))

    def test_unknown_key_is_not_approved(self, paths):
        cache = ConfigurationCache.load(paths.configurations)
        assert not cache.is_approved("something|slot1:essential")

    def test_record_and_round_trip(self, tmp_path):
        path = tmp_path / "configurations.json"
        cache = ConfigurationCache.load(path)
        cache.record("demo|slot1:essential", approved=True, template_id="t")
        cache.save()

        reloaded = ConfigurationCache.load(path)
        assert reloaded.is_approved("demo|slot1:essential")
        assert reloaded.get("demo|slot1:essential").template_id == "t"

    def test_unapproved_entry_is_recorded_but_not_approved(self, tmp_path):
        cache = ConfigurationCache(tmp_path / "c.json")
        cache.record("k", approved=False)
        assert "k" in cache
        assert not cache.is_approved("k")

    def test_invalid_json_is_reported(self, tmp_path):
        path = tmp_path / "configurations.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match="invalid JSON"):
            ConfigurationCache.load(path)


class TestCompose:
    def test_essential_bullets_are_always_included(self, sample_listing):
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        drafted = resolve_selection(lib, "demo_role", {})
        assert [b.slot for b in drafted.bullets] == [1]
        assert drafted.bullets[0].source is BulletSource.ESSENTIAL

    def test_selected_variants_carry_provenance(self, sample_listing):
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        drafted = resolve_selection(lib, "demo_role", {2: "v1_latency"})
        chosen = next(b for b in drafted.bullets if b.slot == 2)
        assert chosen.source is BulletSource.SELECTED
        assert chosen.variant_id == "v1_latency"
        assert chosen.matched_keywords

    def test_selecting_a_variant_alongside_the_slot_it_supersedes_is_rejected(self, sample_listing):
        # The whole point of `supersedes` is that the condensed bullet already says it.
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        with pytest.raises(CompositionError, match="already absorbs"):
            resolve_selection(lib, "demo_role", {3: "v1_merged", 4: "v1_tooling"})

    def test_superseding_variant_alone_is_fine(self, sample_listing):
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        drafted = resolve_selection(lib, "demo_role", {3: "v1_merged"})
        assert {b.slot for b in drafted.bullets} == {1, 3}

    def test_unknown_variant_lists_the_alternatives(self, sample_listing):
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        with pytest.raises(CompositionError, match="v1_latency"):
            resolve_selection(lib, "demo_role", {2: "ghost"})

    def test_unknown_slot_is_rejected(self, sample_listing):
        lib = Library(listings={"demo_role": sample_listing}, templates={}, profile=None)
        with pytest.raises(CompositionError, match="no slot 99"):
            resolve_selection(lib, "demo_role", {99: "v1"})

    def test_draft_preserves_requested_listing_order(self, library):
        ids = ["awards", "education_state_u"]
        draft = draft_from_selection(library, ids, {})
        assert [dl.listing_id for dl in draft.listings] == ids

    @pytest.mark.parametrize("template_id", all_template_ids())
    def test_every_template_composes(self, library, template_id):
        draft = draft_from_template(library, template_id)
        assert draft.listings
        assert all(dl.bullets for dl in draft.listings)
