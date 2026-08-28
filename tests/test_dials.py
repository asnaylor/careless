"""Integration checks for the DIALS-enabled container image."""

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest


if os.environ.get("CARELESS_CONTAINER_TESTS") == "1":
    from cctbx import sgtbx  # noqa: F401
    from dials.array_family import flex
    from dxtbx.model import BeamFactory, Crystal
    from dxtbx.model.experiment_list import Experiment, ExperimentList, ExperimentListFactory


pytestmark = pytest.mark.skipif(
    os.environ.get("CARELESS_CONTAINER_TESTS") != "1",
    reason="DIALS is a container-only dependency",
)


def test_dials_native_modules_share_the_careless_python():
    import careless
    import dials
    import numpy as np
    import torch

    table = flex.reflection_table()
    table["id"] = flex.int([0, 1])
    assert careless.__version__
    assert dials.__path__
    assert version("dials") == "3.30"
    assert version("ray") == "2.54.0"
    assert version("safetensors") == "0.7.0"
    assert sys.executable == "/usr/bin/python"
    assert np.__version__ == "2.1.1"
    assert ".nv26.01." in torch.__version__
    assert torch.version.cuda == "13.1"
    assert ExperimentListFactory is not None


def test_dials_imports_in_a_ray_worker():
    import ray

    from careless.io.dials_ray import _inventory_partition

    ray.init(num_cpus=1, include_dashboard=False)
    try:
        result = ray.get(_inventory_partition.remote([]))
        assert result["reports"] == []
    finally:
        ray.shutdown()


def test_discovery_requires_complete_pairs(tmp_path):
    from careless.io.dials_ray import discover_pairs

    for index in (1, 2, 3):
        (tmp_path / f"image_{index:03d}.expt").touch()
    for index in (1, 3, 4):
        (tmp_path / f"image_{index:03d}.refl").touch()

    with pytest.raises(FileNotFoundError, match="unmatched DIALS input files"):
        discover_pairs(tmp_path, "*.expt", max_files=3)


def test_partition_count_is_bounded_for_twenty_thousand_pairs():
    from careless.io.dials_ray import contiguous_weighted_partitions

    items = [{"index": index, "refl_bytes": 1 + index % 17} for index in range(20_000)]
    partitions = contiguous_weighted_partitions(items, 16)

    assert len(partitions) == 16
    assert [item["index"] for part in partitions for item in part] == list(range(20_000))
    assert all(partitions)


def _write_dials_pair(tmp_path, stem="image"):
    beam = BeamFactory.simple(1.3)
    crystal = Crystal(
        (40.0, 0.0, 0.0),
        (0.0, 50.0, 0.0),
        (0.0, 0.0, 60.0),
        "P 1",
    )
    experiments = ExperimentList([Experiment(beam=beam, crystal=crystal)])
    expt = tmp_path / f"{stem}.expt"
    refl = tmp_path / f"{stem}.refl"
    experiments.as_file(expt)

    table = flex.reflection_table()
    table["id"] = flex.int([0, 0])
    table["miller_index"] = flex.miller_index([(1, 0, 0), (0, 1, 0)])
    s0 = beam.get_s0()
    a = crystal.get_A()
    predicted = flex.mat3_double([a, a]) * table["miller_index"].as_vec3_double()
    table["s1"] = predicted + flex.vec3_double([s0, s0])
    table["xyzcal.px"] = flex.vec3_double([(10.0, 20.0, 0.0), (30.0, 40.0, 0.0)])
    table["xyzobs.px.value"] = flex.vec3_double([(11.0, 22.0, 0.0), (33.0, 44.0, 0.0)])
    table["xyzobs.px.variance"] = flex.vec3_double([(4.0, 9.0, 1.0), (16.0, 25.0, 1.0)])
    table["intensity.sum.value"] = flex.double([100.0, 200.0])
    table["intensity.sum.variance"] = flex.double([25.0, 100.0])
    table.as_file(refl)
    return expt, refl


def test_converter_emits_complete_reference_schema(tmp_path):
    import numpy as np

    from careless.io.dials import convert_pair
    from careless.io.prepared import DATASET_COLUMNS, NUMPY_DTYPES

    expt, refl = _write_dials_pair(tmp_path)
    columns, report = convert_pair(str(expt), str(refl), batch_offset=7)

    assert tuple(columns) == DATASET_COLUMNS
    assert report["rows"] == 2
    np.testing.assert_array_equal(columns["BATCH"], [7, 7])
    np.testing.assert_array_equal(columns["H"], [1, 0])
    np.testing.assert_array_equal(columns["K"], [0, 1])
    np.testing.assert_array_equal(columns["L"], [0, 0])
    np.testing.assert_array_equal(columns["I"], [100.0, 200.0])
    np.testing.assert_array_equal(columns["SigI"], [5.0, 10.0])
    np.testing.assert_array_equal(columns["xcal"], [10.0, 30.0])
    np.testing.assert_array_equal(columns["ycal"], [20.0, 40.0])
    np.testing.assert_array_equal(columns["xobs"], [11.0, 33.0])
    np.testing.assert_array_equal(columns["yobs"], [22.0, 44.0])
    np.testing.assert_array_equal(columns["sigxobs"], [2.0, 4.0])
    np.testing.assert_array_equal(columns["sigyobs"], [3.0, 5.0])
    h = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a = np.asarray(Crystal(
        (40.0, 0.0, 0.0), (0.0, 50.0, 0.0), (0.0, 0.0, 60.0), "P 1"
    ).get_A()).reshape(3, 3)
    s0 = np.asarray(BeamFactory.simple(1.3).get_s0())
    expected_offset = np.linalg.norm((a @ h.T).T + s0, axis=1) - 1.0 / 1.3
    np.testing.assert_allclose(columns["ewald_offset"], expected_offset, atol=1e-7)
    np.testing.assert_allclose(
        np.column_stack([
            columns["cartesian_fixed_obs_x"],
            columns["cartesian_fixed_obs_y"],
            columns["cartesian_fixed_obs_z"],
        ]),
        np.column_stack([
            columns["cartesian_fixed_x"],
            columns["cartesian_fixed_y"],
            columns["cartesian_fixed_z"],
        ]),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        np.column_stack([
            columns["cartesian_delta_x"],
            columns["cartesian_delta_y"],
            columns["cartesian_delta_z"],
        ]),
        0.0,
        atol=1e-7,
    )
    for name, values in columns.items():
        assert values.dtype == NUMPY_DTYPES[name]
        assert values.shape == (2,)


def test_direct_ray_ingestion_and_optional_cache_are_equivalent(tmp_path):
    import numpy as np

    from careless.io.dials_ray import read_dials_dataset
    from careless.io.prepared import DATASET_COLUMNS, read_prepared_dataset

    _write_dials_pair(tmp_path, "a")
    _write_dials_pair(tmp_path, "b")
    cache = tmp_path / "cache"
    direct = read_dials_dataset(
        tmp_path,
        workers_per_node=2,
        block_mib=1,
        save_tensors=cache,
        tensor_shards=2,
    )
    restored = read_prepared_dataset(cache)

    direct_frame = direct.reset_index()
    restored_frame = restored.reset_index()
    assert tuple(direct_frame.columns) == DATASET_COLUMNS
    np.testing.assert_array_equal(direct_frame["BATCH"], [0, 0, 1, 1])
    for name in DATASET_COLUMNS:
        np.testing.assert_array_equal(direct_frame[name], restored_frame[name])


def test_raw_dials_runs_through_the_existing_mono_cli(tmp_path):
    _write_dials_pair(tmp_path)
    output = tmp_path / "merge" / "run"
    result = subprocess.run(
        [
            sys.executable, "-m", "careless.careless", "mono",
            "--disable-gpu", "--disable-progress-bar", "--iterations", "1",
            "dHKL,cartesian_fixed_x", str(tmp_path), str(output),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
        timeout=180,
    )
    assert result.returncode == 0, result.stdout
    assert output.with_name("run_0.mtz").is_file()
    assert output.with_name("run_history.csv").is_file()
