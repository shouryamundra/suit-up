"""Read the on-disk library into validated objects.

The library is hand-edited YAML, so the loader's real job is failing usefully: a typo in
one file should name that file, not surface three layers later as a KeyError during
rendering.

Everything here is read-only. Nothing in the pipeline writes back to `library/listings/`
or `library/templates/` — the approval cache and pending variants are the only mutable
state, and they live in their own modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from suitup.config import Paths
from suitup.models import Listing, ListingType, Template


class LibraryError(RuntimeError):
    """A library file is missing, malformed, or inconsistent with its neighbours."""


class ProfileLink(BaseModel):
    label: str
    url: str


class Profile(BaseModel):
    """The resume header. Static across tailorings; nothing scores or selects it."""

    name: str
    links: list[ProfileLink]


def _load_yaml(path: Path) -> dict:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise LibraryError(f"{path.name}: invalid YAML — {exc}") from exc
    if not isinstance(data, dict):
        raise LibraryError(f"{path.name}: expected a mapping at the top level")
    return data


def _parse(model: type, path: Path, what: str):
    try:
        return model(**_load_yaml(path))
    except ValidationError as exc:
        # Pydantic's own message is good; the filename is what it lacks.
        raise LibraryError(f"{path.name}: invalid {what}\n{exc}") from exc


@dataclass(frozen=True)
class Library:
    """Everything loaded from `library/`, indexed for lookup."""

    listings: dict[str, Listing]
    templates: dict[str, Template]
    profile: Profile

    def listing(self, listing_id: str) -> Listing:
        try:
            return self.listings[listing_id]
        except KeyError:
            raise LibraryError(
                f"unknown listing {listing_id!r}; known: {sorted(self.listings)}"
            ) from None

    def template(self, template_id: str) -> Template:
        try:
            return self.templates[template_id]
        except KeyError:
            raise LibraryError(
                f"unknown template {template_id!r}; known: {sorted(self.templates)}"
            ) from None

    def by_type(self, listing_type: ListingType) -> list[Listing]:
        return [listing for listing in self.listings.values() if listing.type is listing_type]

    @property
    def selectable(self) -> list[Listing]:
        """Listings that compete for space, i.e. what the Listing Selector chooses among."""
        return [listing for listing in self.listings.values() if listing.type.competes_for_space]


def load_library(paths: Paths) -> Library:
    """Load and cross-validate the whole library.

    Cross-file checks live here rather than in the schema because no single file can see
    them: a template referencing a deleted listing is only detectable once both are loaded.
    """
    if not paths.listings.is_dir():
        raise LibraryError(f"no listings directory at {paths.listings}")

    listings: dict[str, Listing] = {}
    for path in sorted(paths.listings.glob("*.yaml")):
        listing = _parse(Listing, path, "listing")
        if listing.id != path.stem:
            raise LibraryError(
                f"{path.name}: declares id {listing.id!r}; filename and id must match so a "
                f"listing can be found from either"
            )
        if listing.id in listings:
            raise LibraryError(f"duplicate listing id {listing.id!r}")
        listings[listing.id] = listing

    if not listings:
        raise LibraryError(f"no listing files found in {paths.listings}")

    templates: dict[str, Template] = {}
    if paths.templates.is_dir():
        for path in sorted(paths.templates.glob("*.yaml")):
            template = _parse(Template, path, "template")
            if template.id in templates:
                raise LibraryError(f"duplicate template id {template.id!r}")
            _validate_template(template, listings, path.name)
            templates[template.id] = template

    if not paths.profile.is_file():
        raise LibraryError(f"no profile at {paths.profile}; the resume header comes from it")
    profile = _parse(Profile, paths.profile, "profile")

    return Library(listings=listings, templates=templates, profile=profile)


def _validate_template(template: Template, listings: dict[str, Listing], filename: str) -> None:
    for listing_id in template.listing_ids:
        if listing_id not in listings:
            raise LibraryError(f"{filename}: references unknown listing {listing_id!r}")

    for listing_id, chosen in template.default_variants.items():
        if listing_id not in template.listing_ids:
            raise LibraryError(
                f"{filename}: picks variants for {listing_id!r}, which it does not include"
            )
        listing = listings[listing_id]
        for slot, variant_id in chosen.items():
            try:
                bullet = listing.bullet(slot)
            except KeyError:
                raise LibraryError(f"{filename}: {listing_id} has no slot {slot}") from None
            available = {v.variant_id for v in bullet.variants}
            if variant_id not in available:
                raise LibraryError(
                    f"{filename}: {listing_id} slot {slot} has no variant {variant_id!r}; "
                    f"available: {sorted(available) or 'none (slot is essential)'}"
                )
