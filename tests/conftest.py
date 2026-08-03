"""Shared fixtures.

Tests run against the real library where they are checking the real library, and against
small hand-built fixtures where they are checking logic — a scoring test that depends on
the user's actual bullets would break every time a variant is reworded.
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


@pytest.fixture(scope="session")
def paths() -> Paths:
    return Paths.from_root(PROJECT_ROOT)


@pytest.fixture(scope="session")
def config():
    return load_config(root=PROJECT_ROOT)


@pytest.fixture(scope="session")
def library(paths):
    """The real library. Used by integration-flavoured tests."""
    return load_library(paths)


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
