import numpy as np
import reciprocalspaceship as rs

from careless.io.formatter import MonoFormatter
from careless.io.prepared import (
    DATASET_COLUMNS,
    is_prepared_dataset,
    read_prepared_dataset,
    validate_prepared_dataset,
    write_prepared_dataset,
)


def prepared_source_dataset():
    dataset = rs.DataSet(
        {
            "H": [1, 0, 1, 2],
            "K": [0, 1, 1, 0],
            "L": [0, 0, 1, 1],
            "I": [10.0, 12.0, 8.0, 15.0],
            "SigI": [1.0, 1.5, 2.0, 2.5],
            "BATCH": [0, 0, 1, 1],
            "xobs": [1.0, 2.0, 4.0, 8.0],
            "yobs": [3.0, 5.0, 7.0, 11.0],
            "ewald_offset": [-0.02, -0.01, 0.01, 0.03],
        },
        cell=[40.0, 50.0, 60.0, 90.0, 90.0, 90.0],
        spacegroup="P 1",
    )
    mtz_types = {"H": "H", "K": "H", "L": "H", "I": "J", "SigI": "Q", "BATCH": "B"}
    for name in dataset.columns:
        dataset[name] = dataset[name].astype(mtz_types.get(name, "R"))
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
    )

    assert is_prepared_dataset(destination)
    assert validate_prepared_dataset(destination) == manifest
    restored = read_prepared_dataset(destination, read_workers=2)
    source_frame = source.reset_index()
    restored_frame = restored.reset_index()
    assert tuple(restored_frame.columns) == DATASET_COLUMNS
    for name in DATASET_COLUMNS:
        assert str(restored_frame[name].dtype) == str(source_frame[name].dtype)
        np.testing.assert_array_equal(
            restored_frame[name].to_numpy(), source_frame[name].to_numpy()
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

    parsed = parser.parse_args(
        [
            "mono",
            "--disable-gpu",
            "dHKL,xobs,yobs,ewald_offset",
            str(destination),
            str(tmp_path / "output"),
        ]
    )
    assert parsed.reflection_files == [str(destination)]
