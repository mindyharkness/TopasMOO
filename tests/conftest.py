"""Fixtures shared across the test suite.

pytest collects this file automatically, so tests use these fixtures by naming
them as arguments -- no import needed. Keeping ``opt_dir`` here in particular
means the path to the bundled example is written down once instead of in every
module that needs a real ``GenerateTopasScripts`` / ``TopasObjectiveFunction``.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create and cleanup temporary directory"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def opt_dir():
    """Path to DevelopmentExample with GenerateTopasScripts/TopasObjectiveFunction."""
    return Path(__file__).parent.parent / "examples" / "DevelopmentExample"
