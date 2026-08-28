import json

import numpy as np
import pytest
import reciprocalspaceship as rs

from careless.io.formatter import MonoFormatter
from careless.io.prepared import (
    DATASET_COLUMNS,
    MTZ_TYPES,
    is_prepared_dataset,
    read_prepared_dataset,
    validate_prepared_dataset,
    write_prepared_dataset,
)


def prepared_source_dataset():
    rows = 4
    values = {
        "H": np.asarray([1, 0, 1, 2], dtype=np.int32),
        "K": np.asarray([0, 1, 1, 0], dtype=np.int32),
        "L": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "BATCH": np.asarray([0, 0, 1, 1], dtype=np.int32),
        "I": np.asarray([10.0, 12.0, 8.0, 15.0], dtype=np.float32),
        "SigI": np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32),
    }
    for index, name in enumerate(DATASET_COLUMNS):
        if name not in values:
            values[name] = np.arange(rows, dtype=np.float32) + index / 10.0
    dataset = rs.DataSet(
        {name: values[name] for name in DATASET_COLUMNS},
        cell=[40.0, 50.0, 60.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
    )
    for name in dataset.columns:
        dataset[name] = dataset[name].astype(MTZ_TYPES.get(name, "R"))
    dataset.set_index(["H", "K", "L"], inplace=True)
    return dataset


def test_prepared_dataset_round_trip(tmp_path):
    source = prepared_source_dataset()
    destination = tmp_path / "prepared"
    manifest = write_prepared_dataset(
        source,
        destination,
        shards=3,
        source={"files": [{"expt": "one.expt", "refl": "one.refl"}]},
        wavelength=1.31,
    )

    assert is_prepared_dataset(destination)
    assert validate_prepared_dataset(destination, verify_files=True) == manifest
    restored = read_prepared_dataset(destination, read_workers=2)
    source_frame = source.reset_index()
    restored_frame = restored.reset_index()
    assert tuple(restored_frame.columns) == DATASET_COLUMNS
    for name in DATASET_COLUMNS:
        assert str(restored_frame[name].dtype) == str(source_frame[name].dtype)
        np.testing.assert_array_equal(
            restored_frame[name].to_numpy(), source_frame[name].to_numpy()
        )


def test_prepared_reader_loads_selected_columns(tmp_path):
    destination = tmp_path / "prepared"
    write_prepared_dataset(prepared_source_dataset(), destination, shards=2)

    restored = read_prepared_dataset(
        destination, columns={"H", "K", "L", "I", "SigI", "BATCH", "xobs"}
    )

    assert tuple(restored.reset_index().columns) == (
        "H", "K", "L", "BATCH", "I", "SigI", "xobs"
    )


def test_prepared_dataset_has_identical_monoformatter_inputs(tmp_path):
    source = prepared_source_dataset()
    destination = tmp_path / "prepared"
    write_prepared_dataset(source, destination, shards=2)

    def format_dataset(dataset):
        formatter = MonoFormatter(
            intensity_key="I",
            uncertainty_key="SigI",
            image_key="BATCH",
            metadata_keys=["dHKL", "xobs", "yobs", "ewald_offset"],
            separate_outputs=False,
            anomalous=False,
            standardize=True,
        )
        return formatter((dataset,))[0]

    expected = format_dataset(source.copy())
    formatter = MonoFormatter(
        intensity_key="I",
        uncertainty_key="SigI",
        image_key="BATCH",
        metadata_keys=["dHKL", "xobs", "yobs", "ewald_offset"],
        separate_outputs=False,
        anomalous=False,
        standardize=True,
    )
    observed = formatter.format_files([destination])[0]
    assert len(expected) == len(observed)
    for expected_tensor, observed_tensor in zip(expected, observed):
        np.testing.assert_array_equal(observed_tensor, expected_tensor)


def test_careless_parser_accepts_a_prepared_dataset(tmp_path):
    destination = tmp_path / "prepared"
    write_prepared_dataset(prepared_source_dataset(), destination)

    from careless.parser import parser

    parsed = parser.parse_args([
        "mono", "--disable-gpu", "dHKL,xobs,yobs,ewald_offset",
        str(destination), str(tmp_path / "output"),
    ])
    assert parsed.reflection_files == [str(destination)]
    assert parsed.input_kind == "prepared"

    with pytest.raises(SystemExit):
        parser.parse_args([
            "poly", "dHKL,xobs,yobs,ewald_offset",
            str(destination), str(tmp_path / "poly-output"),
        ])


def test_careless_parser_classifies_raw_dials_only_for_mono(tmp_path):
    (tmp_path / "image.expt").touch()
    (tmp_path / "image.refl").touch()

    from careless.parser import parser

    parsed = parser.parse_args([
        "mono", "--ray-workers-per-node", "2", "dHKL,ewald_offset",
        str(tmp_path), str(tmp_path / "output"),
    ])
    assert parsed.input_kind == "dials"
    assert parsed.ray_workers_per_node == 2

    with pytest.raises(SystemExit):
        parser.parse_args([
            "poly", "dHKL,ewald_offset", str(tmp_path), str(tmp_path / "poly-output")
        ])


def test_save_tensors_is_rejected_for_mtz(off_file, tmp_path):
    from careless.parser import parser

    with pytest.raises(SystemExit):
        parser.parse_args([
            "mono", "--save-tensors", str(tmp_path / "cache"), "dHKL,image_id",
            off_file, str(tmp_path / "output"),
        ])


def test_old_prototype_artifact_is_rejected(tmp_path):
    destination = tmp_path / "old"
    destination.mkdir()
    manifest = {"artifact_type": "careless-pre-monoformatter", "schema_version": 1}
    payload = json.dumps(manifest).encode()
    (destination / "manifest.json").write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    (destination / "COMPLETE").write_text(f"manifest_sha256={digest}\n")

    with pytest.raises(RuntimeError, match="old prototype artifacts are not supported"):
        validate_prepared_dataset(destination)


def test_existing_mtz_loader_does_not_import_dials_or_ray(off_file, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "ray" or name.startswith(("dials", "dxtbx", "cctbx")):
            raise AssertionError(f"ordinary MTZ loading unexpectedly imported {name}")
        return original_import(name, *args, **kwargs)

    formatter = MonoFormatter(
        intensity_key="I",
        uncertainty_key="SigI",
        image_key="BATCH",
        metadata_keys=["dHKL", "image_id"],
        separate_outputs=False,
        anomalous=False,
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    inputs, _ = formatter.format_files([off_file])
    assert len(inputs[0]) > 0
