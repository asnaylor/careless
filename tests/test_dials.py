"""Integration checks for the DIALS-enabled container image."""

import os
import sys
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
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory

    table = flex.reflection_table()
    table["id"] = flex.int([0, 1])

    assert careless.__version__
    assert dials.__path__
    assert version("dials") == "3.30"
    assert version("rs-distributions") == "0.0.4"
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
