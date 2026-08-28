"""Validated storage for exporter-complete DIALS SafeTensors datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ARTIFACT_TYPE = "careless-dials-safetensors"
SCHEMA_VERSION = 1
DATASET_COLUMNS = (
    "H", "K", "L", "BATCH",
    "cartesian_fixed_obs_x", "cartesian_fixed_obs_y", "cartesian_fixed_obs_z",
    "cartesian_fixed_x", "cartesian_fixed_y", "cartesian_fixed_z",
    "ewald_offset", "I", "SigI", "xcal", "ycal", "xobs", "yobs",
    "sigxobs", "sigyobs",
    "cartesian_delta_x", "cartesian_delta_y", "cartesian_delta_z",
)
MTZ_TYPES = {"H": "H", "K": "H", "L": "H", "I": "J", "SigI": "Q", "BATCH": "B"}
PANDAS_DTYPES = {
    name: {"H": "HKL", "K": "HKL", "L": "HKL", "I": "Intensity",
           "SigI": "Stddev", "BATCH": "Batch"}.get(name, "MTZReal")
    for name in DATASET_COLUMNS
}
NUMPY_DTYPES = {
    name: np.dtype(np.int32) if name in ("H", "K", "L", "BATCH") else np.dtype(np.float32)
    for name in DATASET_COLUMNS
}
TORCH_DTYPES = {"int32": torch.int32, "float32": torch.float32}


def is_prepared_dataset(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "manifest.json").is_file() and (path / "COMPLETE").is_file()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_path(path: str | Path) -> None:
    path = Path(path)
    flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_staging_directory(destination: str | Path) -> tuple[Path, str]:
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    preparation_id = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.tmp-{preparation_id}"
    staging.mkdir()
    return staging, preparation_id


def shard_metadata(preparation_id: str, shard_index: int, rows: int) -> dict[str, str]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": str(SCHEMA_VERSION),
        "preparation_id": preparation_id,
        "shard_index": str(shard_index),
        "rows": str(rows),
    }


def _validate_manifest_shape(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(
            "unsupported prepared dataset type: "
            f"{manifest.get('artifact_type')!r}; old prototype artifacts are not supported"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported prepared schema version: {manifest.get('schema_version')!r}")
    total_rows = manifest.get("total_rows")
    if not isinstance(total_rows, int) or total_rows < 1:
        raise RuntimeError(f"invalid prepared row count: {total_rows!r}")
    if manifest.get("column_order") != list(DATASET_COLUMNS):
        raise RuntimeError("prepared column order is invalid")
    columns = manifest.get("columns")
    if not isinstance(columns, dict) or set(columns) != set(DATASET_COLUMNS):
        raise RuntimeError("prepared column schema is invalid")
    for name in DATASET_COLUMNS:
        details = columns[name]
        if details.get("shape") != [total_rows]:
            raise RuntimeError(f"prepared shape is invalid for {name}: {details.get('shape')}")
        if details.get("dtype") != NUMPY_DTYPES[name].name:
            raise RuntimeError(f"prepared dtype is invalid for {name}: {details.get('dtype')}")
        if details.get("mtz_type") != MTZ_TYPES.get(name, "R"):
            raise RuntimeError(f"prepared MTZ type is invalid for {name}")
    cell = manifest.get("cell")
    if not isinstance(cell, list) or len(cell) != 6 or not all(np.isfinite(cell)):
        raise RuntimeError(f"prepared unit cell is invalid: {cell!r}")
    if not isinstance(manifest.get("space_group"), str) or not manifest["space_group"]:
        raise RuntimeError("prepared space group is invalid")
    wavelength = manifest.get("wavelength")
    if not isinstance(wavelength, (int, float)) or not np.isfinite(wavelength) or wavelength <= 0:
        raise RuntimeError(f"prepared wavelength is invalid: {wavelength!r}")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("prepared source metadata is invalid")
    if manifest.get("source_sha256") != hashlib.sha256(_canonical_json(source)).hexdigest():
        raise RuntimeError("prepared source metadata digest is invalid")


def validate_prepared_dataset(path: str | Path, verify_files: bool = False) -> dict[str, Any]:
    """Validate a completed artifact and optionally hash every shard."""
    from safetensors import safe_open

    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"prepared dataset directory does not exist: {path}")
    manifest_path = path / "manifest.json"
    complete_path = path / "COMPLETE"
    if not manifest_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"prepared dataset is incomplete: {path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if complete_path.read_text() != f"manifest_sha256={manifest_sha256}\n":
        raise RuntimeError(f"prepared dataset completion marker is invalid: {path}")
    manifest = json.loads(manifest_bytes)
    _validate_manifest_shape(manifest)

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("prepared shard list is invalid")
    cursor = 0
    names = set()
    preparation_id = manifest.get("preparation_id")
    if not isinstance(preparation_id, str) or not preparation_id:
        raise RuntimeError("prepared dataset has no preparation id")
    for index, shard in enumerate(shards):
        name = shard.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise RuntimeError(f"invalid prepared shard name: {name!r}")
        if name in names:
            raise RuntimeError(f"duplicate prepared shard name: {name}")
        names.add(name)
        if shard.get("index") != index or shard.get("row_start") != cursor:
            raise RuntimeError(f"prepared shard coverage is invalid before {name}")
        rows = shard.get("rows")
        row_stop = shard.get("row_stop")
        if not isinstance(rows, int) or rows < 1 or row_stop != cursor + rows:
            raise RuntimeError(f"prepared shard range is invalid for {name}")
        shard_path = path / name
        if not shard_path.is_file():
            raise FileNotFoundError(f"prepared shard is missing: {shard_path}")
        if shard_path.stat().st_size != shard.get("bytes"):
            raise RuntimeError(f"prepared shard size changed: {shard_path}")
        digest = shard.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"prepared shard digest is invalid for {name}")
        if verify_files and sha256_file(shard_path) != digest:
            raise RuntimeError(f"prepared shard digest changed: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            if set(source.keys()) != set(DATASET_COLUMNS):
                raise RuntimeError(f"prepared shard schema is invalid: {shard_path}")
            if source.metadata() != shard_metadata(preparation_id, index, rows):
                raise RuntimeError(f"prepared shard metadata is invalid: {shard_path}")
            for column in DATASET_COLUMNS:
                if source.get_slice(column).get_shape() != [rows]:
                    raise RuntimeError(f"prepared tensor shape changed for {column}: {shard_path}")
        cursor = row_stop
    if cursor != manifest["total_rows"]:
        raise RuntimeError(f"prepared shard coverage ends at {cursor}, expected {manifest['total_rows']}")
    observed_names = {item.name for item in path.glob("*.safetensors")}
    if observed_names != names:
        raise RuntimeError(
            f"prepared shard files differ from manifest: {sorted(observed_names)} != {sorted(names)}"
        )
    return manifest


def finalize_staging_dataset(
    staging: str | Path,
    destination: str | Path,
    preparation_id: str,
    shard_reports: list[dict[str, Any]],
    cell: Iterable[float],
    space_group: str,
    wavelength: float,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Write the manifest and atomically publish already-written shards."""
    staging = Path(staging).absolute()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {destination}")
    reports = sorted(shard_reports, key=lambda item: item["index"])
    if [item["index"] for item in reports] != list(range(len(reports))):
        raise RuntimeError("prepared shard indexes are not contiguous")
    cursor = 0
    shards = []
    for report in reports:
        rows = int(report["rows"])
        shards.append({
            "index": report["index"], "file": report["file"],
            "row_start": cursor, "row_stop": cursor + rows, "rows": rows,
            "bytes": report["bytes"], "sha256": report["sha256"],
        })
        cursor += rows
    if cursor < 1:
        raise ValueError("cannot prepare an empty dataset")
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": SCHEMA_VERSION,
        "preparation_id": preparation_id,
        "total_rows": cursor,
        "cell": [float(value) for value in cell],
        "space_group": str(space_group),
        "wavelength": float(wavelength),
        "column_order": list(DATASET_COLUMNS),
        "columns": {
            name: {"shape": [cursor], "dtype": NUMPY_DTYPES[name].name,
                   "mtz_type": MTZ_TYPES.get(name, "R"),
                   "pandas_dtype": PANDAS_DTYPES[name]}
            for name in DATASET_COLUMNS
        },
        "source": source,
        "source_sha256": hashlib.sha256(_canonical_json(source)).hexdigest(),
        "shards": shards,
    }
    manifest_path = staging / "manifest.json"
    manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    fsync_path(manifest_path)
    complete_path = staging / "COMPLETE"
    complete_path.write_text(f"manifest_sha256={sha256_file(manifest_path)}\n")
    fsync_path(complete_path)
    fsync_path(staging)
    validate_prepared_dataset(staging, verify_files=True)
    os.replace(staging, destination)
    fsync_path(destination.parent)
    return manifest


def _dataset_arrays(dataset: Any) -> dict[str, np.ndarray]:
    frame = dataset.reset_index()
    if set(frame.columns) != set(DATASET_COLUMNS):
        raise RuntimeError(f"prepared columns changed: {list(frame.columns)} != {list(DATASET_COLUMNS)}")
    return {
        name: np.ascontiguousarray(frame[name].to_numpy(), dtype=NUMPY_DTYPES[name])
        for name in DATASET_COLUMNS
    }


def dataset_from_arrays(
    arrays: dict[str, np.ndarray],
    cell: Iterable[float],
    space_group: str,
) -> Any:
    """Construct the single exporter-complete DataSet used by MonoFormatter."""
    import gemmi
    import pandas as pd
    import reciprocalspaceship as rs

    if tuple(arrays) != DATASET_COLUMNS:
        raise RuntimeError(f"DIALS columns changed: {tuple(arrays)}")
    rows = len(arrays["H"])
    if rows < 1:
        raise ValueError("cannot construct an empty DIALS dataset")
    for name, values in arrays.items():
        if values.dtype != NUMPY_DTYPES[name] or values.shape != (rows,):
            raise RuntimeError(f"DIALS array {name} has an invalid schema")
    dataset = rs.DataSet(
        pd.DataFrame(arrays, copy=False),
        cell=gemmi.UnitCell(*[float(value) for value in cell]),
        spacegroup=gemmi.SpaceGroup(str(space_group)),
    )
    for name in dataset.columns:
        dataset[name] = dataset[name].astype(MTZ_TYPES.get(name, "R"))
    dataset.set_index(["H", "K", "L"], inplace=True)
    return dataset


def write_prepared_dataset(
    dataset: Any,
    destination: str | Path,
    shards: int = 8,
    source: dict[str, Any] | None = None,
    wavelength: float = 1.0,
) -> dict[str, Any]:
    """Write a complete in-memory dataset, primarily for tests and utilities."""
    from safetensors.numpy import save_file

    if shards < 1:
        raise ValueError("prepared shard count must be positive")
    destination = Path(destination).absolute()
    staging, preparation_id = create_staging_directory(destination)
    try:
        arrays = _dataset_arrays(dataset)
        total_rows = len(dataset)
        if total_rows < 1:
            raise ValueError("cannot prepare an empty dataset")
        shard_count = min(shards, total_rows)
        reports = []
        for index in range(shard_count):
            start = index * total_rows // shard_count
            stop = (index + 1) * total_rows // shard_count
            name = f"part-{index:05d}-of-{shard_count:05d}.safetensors"
            path = staging / name
            save_file(
                {column: values[start:stop] for column, values in arrays.items()},
                path,
                metadata=shard_metadata(preparation_id, index, stop - start),
            )
            fsync_path(path)
            reports.append({"index": index, "file": name, "rows": stop - start,
                            "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        return finalize_staging_dataset(
            staging, destination, preparation_id, reports,
            dataset.cell.parameters, dataset.spacegroup.xhm(), wavelength,
            {} if source is None else source,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def read_prepared_dataset(
    path: str | Path,
    read_workers: int | None = None,
    columns: Iterable[str] | None = None,
) -> Any:
    """Validate shards and reconstruct one logical ``rs.DataSet``."""
    from safetensors import safe_open

    import gemmi
    import pandas as pd
    import reciprocalspaceship as rs

    path = Path(path).resolve()
    manifest = validate_prepared_dataset(path)
    if read_workers is None:
        read_workers = int(os.environ.get("CARELESS_PREPARED_READ_WORKERS", "8"))
    if read_workers < 1:
        raise ValueError("prepared read worker count must be positive")
    requested = set(DATASET_COLUMNS if columns is None else columns)
    requested.update(("H", "K", "L"))
    unknown = requested - set(DATASET_COLUMNS)
    if unknown:
        raise KeyError(f"prepared columns are unavailable: {sorted(unknown)}")
    selected = tuple(name for name in DATASET_COLUMNS if name in requested)
    destination = {
        name: torch.empty(manifest["total_rows"], dtype=TORCH_DTYPES[manifest["columns"][name]["dtype"]])
        for name in selected
    }

    def load_shard(shard: dict[str, Any]) -> None:
        shard_path = path / shard["file"]
        if sha256_file(shard_path) != shard["sha256"]:
            raise RuntimeError(f"prepared shard digest changed: {shard_path}")
        start, stop = shard["row_start"], shard["row_stop"]
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            for name in selected:
                tensor = source.get_tensor(name)
                expected = destination[name][start:stop]
                if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                    raise RuntimeError(f"prepared tensor schema changed for {name}: {shard_path}")
                expected.copy_(tensor)

    workers = min(read_workers, len(manifest["shards"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_shard, shard) for shard in manifest["shards"]]
        for future in as_completed(futures):
            future.result()
    frame = pd.DataFrame({name: destination[name].numpy() for name in selected})
    dataset = rs.DataSet(
        frame,
        cell=gemmi.UnitCell(*manifest["cell"]),
        spacegroup=gemmi.SpaceGroup(manifest["space_group"]),
    )
    for name in dataset.columns:
        dataset[name] = dataset[name].astype(MTZ_TYPES.get(name, "R"))
    dataset.set_index(["H", "K", "L"], inplace=True)
    return dataset
