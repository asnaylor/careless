"""Validated SafeTensors storage for complete pre-formatter Careless datasets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch


ARTIFACT_TYPE = "careless-pre-monoformatter"
SCHEMA_VERSION = 1
DATASET_COLUMNS = (
    "H",
    "K",
    "L",
    "I",
    "SigI",
    "BATCH",
    "xobs",
    "yobs",
    "ewald_offset",
)
PANDAS_DTYPES = {
    "H": "HKL",
    "K": "HKL",
    "L": "HKL",
    "I": "Intensity",
    "SigI": "Stddev",
    "BATCH": "Batch",
    "xobs": "MTZReal",
    "yobs": "MTZReal",
    "ewald_offset": "MTZReal",
}
MTZ_TYPES = {
    "H": "H",
    "K": "H",
    "L": "H",
    "I": "J",
    "SigI": "Q",
    "BATCH": "B",
}
TORCH_DTYPES = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
    "float64": torch.float64,
}


def is_prepared_dataset(path: str | Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "manifest.json").is_file() and (path / "COMPLETE").is_file()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _fsync(path: Path) -> None:
    flags = os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _dataset_tensors(dataset: Any) -> dict[str, torch.Tensor]:
    frame = dataset.reset_index()
    if tuple(frame.columns) != DATASET_COLUMNS:
        raise RuntimeError(
            f"prepared columns changed: {list(frame.columns)} != {list(DATASET_COLUMNS)}"
        )
    tensors = {}
    for name in DATASET_COLUMNS:
        observed_dtype = str(frame[name].dtype)
        if observed_dtype != PANDAS_DTYPES[name]:
            raise RuntimeError(
                f"prepared dtype changed for {name}: "
                f"{observed_dtype} != {PANDAS_DTYPES[name]}"
            )
        array = np.ascontiguousarray(frame[name].to_numpy())
        if array.dtype.kind not in "ifu" or array.dtype.hasobject:
            raise TypeError(f"prepared column {name} has unsupported dtype {array.dtype}")
        tensors[name] = torch.from_numpy(array)
    return tensors


def _shard_metadata(manifest: dict[str, Any], shard: dict[str, Any]) -> dict[str, str]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "schema_version": str(SCHEMA_VERSION),
        "source_sha256": manifest["source_sha256"],
        "row_start": str(shard["row_start"]),
        "row_stop": str(shard["row_stop"]),
        "total_rows": str(manifest["total_rows"]),
    }


def validate_prepared_dataset(path: str | Path) -> dict[str, Any]:
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
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise RuntimeError(f"unsupported prepared dataset type: {manifest.get('artifact_type')!r}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported prepared schema version: {manifest.get('schema_version')!r}"
        )
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
        if details.get("dtype") not in TORCH_DTYPES:
            raise RuntimeError(f"prepared dtype is invalid for {name}: {details.get('dtype')}")
        if details.get("pandas_dtype") != PANDAS_DTYPES[name]:
            raise RuntimeError(f"prepared pandas dtype is invalid for {name}")
        digest = details.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"prepared digest is invalid for {name}")

    cell = manifest.get("cell")
    if not isinstance(cell, list) or len(cell) != 6:
        raise RuntimeError(f"prepared unit cell is invalid: {cell!r}")
    if not isinstance(manifest.get("space_group"), str) or not manifest["space_group"]:
        raise RuntimeError("prepared space group is invalid")
    if not isinstance(manifest.get("source"), dict):
        raise RuntimeError("prepared source metadata is invalid")
    expected_source_hash = hashlib.sha256(_canonical_json(manifest["source"])).hexdigest()
    if manifest.get("source_sha256") != expected_source_hash:
        raise RuntimeError("prepared source metadata digest is invalid")

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise RuntimeError("prepared shard list is invalid")
    cursor = 0
    names = set()
    for shard in shards:
        name = shard.get("file")
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise RuntimeError(f"invalid prepared shard name: {name!r}")
        if name in names:
            raise RuntimeError(f"duplicate prepared shard name: {name}")
        names.add(name)
        if shard.get("row_start") != cursor:
            raise RuntimeError(f"prepared shard coverage has a gap or overlap before {name}")
        row_stop = shard.get("row_stop")
        if not isinstance(row_stop, int) or row_stop <= cursor or row_stop > total_rows:
            raise RuntimeError(f"prepared shard range is invalid for {name}")
        if not isinstance(shard.get("bytes"), int) or shard["bytes"] < 1:
            raise RuntimeError(f"prepared shard byte count is invalid for {name}")
        if not isinstance(shard.get("sha256"), str) or len(shard["sha256"]) != 64:
            raise RuntimeError(f"prepared shard digest is invalid for {name}")
        cursor = row_stop
    if cursor != total_rows:
        raise RuntimeError(f"prepared shard coverage ends at {cursor}, expected {total_rows}")
    observed_names = {item.name for item in path.glob("*.safetensors")}
    if observed_names != names:
        raise RuntimeError(
            f"prepared shard files differ from manifest: {sorted(observed_names)} != {sorted(names)}"
        )
    return manifest


def write_prepared_dataset(
    dataset: Any,
    destination: str | Path,
    shards: int = 8,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write one logical dataset as deterministic row shards."""
    from safetensors.torch import save_file

    if shards < 1:
        raise ValueError("prepared shard count must be positive")
    destination = Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        tensors = _dataset_tensors(dataset)
        total_rows = len(dataset)
        if total_rows < 1:
            raise ValueError("cannot prepare an empty dataset")
        shard_count = min(shards, total_rows)
        source = {} if source is None else source
        source_sha256 = hashlib.sha256(_canonical_json(source)).hexdigest()
        shard_records = []
        for index in range(shard_count):
            row_start = index * total_rows // shard_count
            row_stop = (index + 1) * total_rows // shard_count
            name = f"part-{index:05d}-of-{shard_count:05d}.safetensors"
            shard_path = temporary / name
            block = {
                column: tensor[row_start:row_stop].clone()
                for column, tensor in tensors.items()
            }
            save_file(
                block,
                shard_path,
                metadata={
                    "artifact_type": ARTIFACT_TYPE,
                    "schema_version": str(SCHEMA_VERSION),
                    "source_sha256": source_sha256,
                    "row_start": str(row_start),
                    "row_stop": str(row_stop),
                    "total_rows": str(total_rows),
                },
            )
            _fsync(shard_path)
            shard_records.append(
                {
                    "file": name,
                    "row_start": row_start,
                    "row_stop": row_stop,
                    "bytes": shard_path.stat().st_size,
                    "sha256": _sha256_file(shard_path),
                }
            )

        manifest = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "total_rows": total_rows,
            "cell": list(dataset.cell.parameters),
            "space_group": dataset.spacegroup.xhm(),
            "column_order": list(DATASET_COLUMNS),
            "columns": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "pandas_dtype": PANDAS_DTYPES[name],
                    "sha256": _tensor_sha256(tensor),
                }
                for name, tensor in tensors.items()
            },
            "source": source,
            "source_sha256": source_sha256,
            "shards": shard_records,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        _fsync(manifest_path)
        complete_path = temporary / "COMPLETE"
        complete_path.write_text(f"manifest_sha256={_sha256_file(manifest_path)}\n")
        _fsync(complete_path)
        _fsync(temporary)
        validate_prepared_dataset(temporary)
        os.replace(temporary, destination)
        _fsync(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def read_prepared_dataset(
    path: str | Path, read_workers: int | None = None
) -> Any:
    """Validate all shards and reconstruct one complete ``rs.DataSet``."""
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
    total_rows = manifest["total_rows"]
    destination = {
        name: torch.empty(total_rows, dtype=TORCH_DTYPES[manifest["columns"][name]["dtype"]])
        for name in DATASET_COLUMNS
    }

    def load_shard(shard: dict[str, Any]) -> None:
        shard_path = path / shard["file"]
        if not shard_path.is_file():
            raise FileNotFoundError(f"prepared shard is missing: {shard_path}")
        if shard_path.stat().st_size != shard["bytes"]:
            raise RuntimeError(f"prepared shard size changed: {shard_path}")
        if _sha256_file(shard_path) != shard["sha256"]:
            raise RuntimeError(f"prepared shard digest changed: {shard_path}")
        row_start, row_stop = shard["row_start"], shard["row_stop"]
        with safe_open(shard_path, framework="pt", device="cpu") as source:
            if set(source.keys()) != set(DATASET_COLUMNS):
                raise RuntimeError(f"prepared shard schema is invalid: {shard_path}")
            if source.metadata() != _shard_metadata(manifest, shard):
                raise RuntimeError(f"prepared shard metadata is invalid: {shard_path}")
            for name in DATASET_COLUMNS:
                tensor = source.get_tensor(name)
                expected = destination[name][row_start:row_stop]
                if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
                    raise RuntimeError(f"prepared tensor schema changed for {name}: {shard_path}")
                expected.copy_(tensor)

    workers = min(read_workers, len(manifest["shards"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(load_shard, shard) for shard in manifest["shards"]]
        for future in as_completed(futures):
            future.result()
    for name, tensor in destination.items():
        if _tensor_sha256(tensor) != manifest["columns"][name]["sha256"]:
            raise RuntimeError(f"prepared column digest changed: {name}")

    frame = pd.DataFrame({name: tensor.numpy() for name, tensor in destination.items()})
    dataset = rs.DataSet(
        frame,
        cell=gemmi.UnitCell(*manifest["cell"]),
        spacegroup=gemmi.SpaceGroup(manifest["space_group"]),
    )
    for name in dataset.columns:
        dataset[name] = dataset[name].astype(MTZ_TYPES.get(name, "R"))
    dataset.set_index(["H", "K", "L"], inplace=True)
    return dataset
