"""Integration checks for the DIALS-enabled container image."""

import os
import sys

# Load the native DIALS/cctbx stack during collection, before pytest imports
# tests that use Pandas or Reciprocalspaceship.  This mirrors the production
# entry point and protects the container regression suite from the observed
# Boost.Python import-order crash.
if os.environ.get("CARELESS_CONTAINER_TESTS") == "1":
    from cctbx import sgtbx  # noqa: F401
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

from importlib.metadata import version
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("CARELESS_CONTAINER_TESTS") != "1",
    reason="DIALS is a container-only dependency",
)


def test_dials_native_modules_share_the_careless_python():
    """DIALS and Careless must import from the image's NVIDIA Python."""
    import careless
    import dials
    import numpy as np
    import torch
    from careless.distributions import TruncatedNormal
    from careless.utils.distributions import FoldedNormal, Rice
    table = flex.reflection_table()
    table["id"] = flex.int([0, 1])

    assert careless.__version__
    assert dials.__path__
    assert version("dials") == "3.30"
    assert version("ray") == "2.54.0"
    assert version("rs-distributions") == "0.0.4"
    assert version("safetensors") == "0.7.0"
    assert sys.executable == "/usr/bin/python"
    assert np.__version__ == "2.1.1"
    assert ".nv26.01." in torch.__version__
    assert torch.version.cuda == "13.1"
    assert Path(torch.__file__).is_relative_to(
        "/usr/local/lib/python3.12/dist-packages/torch"
    )
    assert list(table["id"]) == [0, 1]
    assert ExperimentListFactory is not None
    assert TruncatedNormal is not None
    assert FoldedNormal is not None
    assert Rice is not None


def test_dials_imports_in_a_ray_worker():
    """Ray must be able to import native DIALS extensions in a fresh worker."""
    import ray

    from careless.io.dials_ray import DialsReader

    ray.init(num_cpus=1, include_dashboard=False)
    try:
        actor = DialsReader.remote(0, [])
        assert ray.get(actor.__ray_ready__.remote()) is True
    finally:
        ray.shutdown()


def test_max_files_selects_complete_pairs(tmp_path):
    from careless.io.dials_ray import discover_pairs

    for index in (1, 2, 3):
        (tmp_path / f"image_{index:03d}.expt").touch()
    for index in (1, 3, 4):
        (tmp_path / f"image_{index:03d}.refl").touch()

    manifest, report = discover_pairs(tmp_path, "*.expt", max_files=3)
    assert [item["relative_expt"] for item in manifest] == [
        "image_001.expt",
        "image_003.expt",
    ]
    assert report["complete_pairs"] == 2
    assert report["selected_pairs"] == 2
    assert report["missing_reflection_files"] == ["image_002.refl"]
    assert report["missing_experiment_files"] == ["image_004.expt"]
