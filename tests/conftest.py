"""Shared fixtures.

Tests run against the fictional library in `examples/library`, never the user's real one.
The real library holds personal data and is untracked, so a fresh clone would have nothing
to test against — and assertions on real bullets would break every time one was reworded.
Pure-logic tests use the even smaller hand-built fixtures below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from suitup.config import Paths, Thresholds, load_config
from suitup.library.loader import load_library
from suitup.models import (
    Bullet,
    BulletRole,
    Listing,
    ListingContext,
    ListingType,
    Requirement,
    Variant,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tests run against the example library, never the user's real one. The real library holds
# personal data and is untracked, so a fresh clone would have nothing to test against; and
# tests that asserted on real bullets would break every time one was reworded.
EXAMPLE_LIBRARY = PROJECT_ROOT / "examples" / "library"

# Canonical ids from the example library, so tests never name real-library content.
PRIMARY_TEMPLATE = "backend"
PRIMARY_EXPERIENCE = "backend_intern"


@pytest.fixture(scope="session")
def paths() -> Paths:
    return Paths.from_root(PROJECT_ROOT, library_dir=EXAMPLE_LIBRARY)


@pytest.fixture(scope="session")
def config():
    return load_config(root=PROJECT_ROOT)


@pytest.fixture(scope="session")
def library(paths):
    """The real library. Used by integration-flavoured tests."""
    return load_library(paths)


def all_template_ids() -> list[str]:
    """Every template on disk, for parametrising at collection time.

    Derived rather than hardcoded so adding a career direction is automatically covered by
    the compile and composition tests instead of silently skipping them.
    """
    return sorted(
        load_library(Paths.from_root(PROJECT_ROOT, library_dir=EXAMPLE_LIBRARY)).templates
    )


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds()


@pytest.fixture
def sample_listing() -> Listing:
    """A small, stable listing for scoring and composition tests."""
    return Listing(
        id="demo_role",
        type=ListingType.EXPERIENCE,
        title="Engineer",
        context=ListingContext(org="Demo Corp", dates="2025 -- Present", location="Seattle, WA"),
        tags=["systems", "cuda"],
        bullets=[
            Bullet(slot=1, role=BulletRole.ESSENTIAL, text="Led a team of four engineers."),
            Bullet(
                slot=2,
                role=BulletRole.FLEXIBLE,
                variants=[
                    Variant(
                        variant_id="v1_latency",
                        keywords=["inference latency", "TensorRT"],
                        text="Cut inference latency 30% with a TensorRT FP16 engine.",
                    ),
                    Variant(
                        variant_id="v2_accuracy",
                        keywords=["object detection", "mAP"],
                        text="Raised object detection mAP from 82% to 89%.",
                    ),
                ],
            ),
            Bullet(
                slot=3,
                role=BulletRole.FLEXIBLE,
                variants=[
                    Variant(
                        variant_id="v1_merged",
                        keywords=["profiling"],
                        supersedes=[4],
                        text="Cut latency and built the profiling tooling that found it.",
                    ),
                ],
            ),
            Bullet(
                slot=4,
                role=BulletRole.FLEXIBLE,
                variants=[
                    Variant(
                        variant_id="v1_tooling",
                        keywords=["profiling", "telemetry"],
                        text="Built profiling tooling for live GPU telemetry.",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def requirements() -> list[Requirement]:
    return [
        Requirement(
            text="Optimise inference latency on embedded hardware",
            weight=1.0,
            keywords=["inference latency", "TensorRT", "embedded"],
        ),
        Requirement(
            text="Computer vision model accuracy",
            weight=0.5,
            keywords=["object detection", "mAP"],
        ),
    ]
