"""Composition keys and the approval cache.

A composition key names exactly which content a resume is made of. Its whole value is
exactness: the cache exists so a composition the user already reviewed is not re-reviewed,
and a key that varies with dict ordering or whitespace would never hit.

Key format, fixed by the seeds written during bootstrap:

    listing key   aruw_lead_swe_2025|slot1:essential|slot2:v1_ros2_pubsub
    resume key    <listing key>||<listing key>||...

Slots ascend numerically within a listing. Essential slots are always present and always
render as `essential`. Listing keys appear in the resume's own display order, because two
resumes with the same content in a different order are genuinely different resumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from suitup.models import BulletRole, Listing

SLOT_SEPARATOR = "|"
LISTING_SEPARATOR = "||"


def listing_key(listing: Listing, chosen: dict[int, str]) -> str:
    """Key for one listing under a given variant selection.

    Slots absent from `chosen` are omitted — a dropped flexible bullet changes the
    composition, so it must change the key.
    """
    parts = [listing.id]
    for bullet in sorted(listing.bullets, key=lambda b: b.slot):
        if bullet.role is BulletRole.ESSENTIAL:
            parts.append(f"slot{bullet.slot}:essential")
        elif bullet.slot in chosen:
            parts.append(f"slot{bullet.slot}:{chosen[bullet.slot]}")
    return SLOT_SEPARATOR.join(parts)


def resume_key(listing_keys: list[str]) -> str:
    """Key for a whole resume. Order is significant and deliberately not normalised."""
    return LISTING_SEPARATOR.join(listing_keys)


def split_resume_key(key: str) -> list[str]:
    return key.split(LISTING_SEPARATOR) if key else []


@dataclass
class ConfigurationEntry:
    approved: bool
    reviewed_at: str | None = None
    template_id: str | None = None
    listing_keys: list[str] | None = None
    source: str | None = None
    company: str | None = None
    role: str | None = None
    note: str | None = None

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class ConfigurationCache:
    """`configurations.json` — which compositions a human has already approved.

    Loaded and saved whole. The file is small by construction (one entry per reviewed
    composition) and stays human-readable and git-diffable, which is the point.
    """

    def __init__(self, path: Path, entries: dict[str, ConfigurationEntry] | None = None):
        self.path = path
        self._entries: dict[str, ConfigurationEntry] = entries or {}

    @classmethod
    def load(cls, path: Path) -> ConfigurationCache:
        if not path.is_file():
            # A fresh library has no approvals yet; that is not an error.
            return cls(path)
        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON — {exc}") from exc

        entries = {}
        for key, value in raw.items():
            known = {f: value.get(f) for f in ConfigurationEntry.__annotations__}
            entries[key] = ConfigurationEntry(**known)
        return cls(path, entries)

    def save(self) -> None:
        payload = {key: entry.to_json() for key, entry in sorted(self._entries.items())}
        self.path.write_text(json.dumps(payload, indent=2) + "\n")

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> ConfigurationEntry | None:
        return self._entries.get(key)

    def is_approved(self, key: str) -> bool:
        entry = self._entries.get(key)
        return entry is not None and entry.approved

    def record(
        self,
        key: str,
        approved: bool,
        listing_keys: list[str] | None = None,
        template_id: str | None = None,
        note: str | None = None,
    ) -> ConfigurationEntry:
        entry = ConfigurationEntry(
            approved=approved,
            reviewed_at=date.today().isoformat(),
            template_id=template_id,
            listing_keys=listing_keys if listing_keys is not None else split_resume_key(key),
            note=note,
        )
        self._entries[key] = entry
        return entry
