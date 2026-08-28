"""Direct, bounded Ray ingestion of paired DIALS experiments and reflections."""

from __future__ import annotations

import gc
import json
import os
import socket
from collections import deque
from pathlib import Path
from typing import Any

# Import native DIALS/cctbx modules before NumPy, Ray, Pandas, Gemmi, or
# Reciprocalspaceship in a fresh Careless CLI or Ray worker process.
from careless.io.dials import convert_pair, inventory_pair  # noqa: E402

import numpy as np  # noqa: E402
import ray  # noqa: E402
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: E402
from tqdm import tqdm  # noqa: E402

from careless.io.prepared import (  # noqa: E402
    DATASET_COLUMNS,
    NUMPY_DTYPES,
    dataset_from_arrays,
    write_prepared_dataset,
)


BYTES_PER_ROW = sum(dtype.itemsize for dtype in NUMPY_DTYPES.values())


def discover_pairs(
    input_dir: str | Path,
    expt_glob: str = "**/*.expt",
    max_files: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return complete same-stem pairs in deterministic relative-path order."""
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"DIALS input directory does not exist: {input_dir}")
    if not expt_glob.endswith(".expt"):
        raise ValueError("--dials-expt-glob must end in '.expt'")
    refl_glob = f"{expt_glob[:-len('.expt')]}.refl"
    expts = {
        path.relative_to(input_dir).with_suffix("").as_posix(): path
        for path in input_dir.glob(expt_glob)
        if path.is_file()
    }
    refls = {
        path.relative_to(input_dir).with_suffix("").as_posix(): path
        for path in input_dir.glob(refl_glob)
        if path.is_file()
    }
    if not expts:
        raise FileNotFoundError(f"no experiment files matched {expt_glob!r}")
    missing_refl = sorted(set(expts) - set(refls))
    missing_expt = sorted(set(refls) - set(expts))
    if missing_refl or missing_expt:
        details = {
            "missing_reflection_files": [
                expts[stem].with_suffix(".refl").relative_to(input_dir).as_posix()
                for stem in missing_refl
            ],
            "missing_experiment_files": [
                refls[stem].with_suffix(".expt").relative_to(input_dir).as_posix()
                for stem in missing_expt
            ],
        }
        raise FileNotFoundError(
            f"unmatched DIALS input files: {json.dumps(details, sort_keys=True)}"
        )
    stems = sorted(expts)
    selected = stems if max_files is None else stems[:max_files]
    if not selected:
        raise FileNotFoundError("no complete .expt/.refl pairs were selected")
    manifest = [
        {
            "index": index,
            "expt": str(expts[stem].resolve()),
            "refl": str(refls[stem].resolve()),
            "relative_expt": expts[stem].relative_to(input_dir).as_posix(),
            "relative_refl": refls[stem].relative_to(input_dir).as_posix(),
            "refl_bytes": refls[stem].stat().st_size,
        }
        for index, stem in enumerate(selected)
    ]
    return manifest, {
        "experiment_files": len(expts),
        "reflection_files": len(refls),
        "selected_pairs": len(selected),
        "max_files": max_files,
    }


def contiguous_weighted_partitions(
    items: list[dict[str, Any]], count: int
) -> list[list[dict[str, Any]]]:
    """Make nonempty contiguous partitions balanced by reflection-file bytes."""
    if count < 1:
        raise ValueError("partition count must be positive")
    count = min(count, len(items))
    if count == 1:
        return [items]
    partitions: list[list[dict[str, Any]]] = []
    start = 0
    remaining_weight = sum(max(1, int(item["refl_bytes"])) for item in items)
    for partition_index in range(count - 1):
        remaining_parts = count - partition_index
        target = remaining_weight / remaining_parts
        maximum_stop = len(items) - (remaining_parts - 1)
        stop = start
        weight = 0
        while stop < maximum_stop:
            next_weight = max(1, int(items[stop]["refl_bytes"]))
            if stop > start and weight + next_weight > target:
                break
            weight += next_weight
            stop += 1
        if stop == start:
            stop += 1
            weight = max(1, int(items[start]["refl_bytes"]))
        partitions.append(items[start:stop])
        start = stop
        remaining_weight -= weight
    partitions.append(items[start:])
    return partitions


def _live_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    nodes.sort(key=lambda node: (str(node.get("NodeManagerAddress")), str(node["NodeID"])))
    if not nodes:
        raise RuntimeError("Ray reports no live nodes")
    return nodes


@ray.remote(num_cpus=1, max_retries=2)
def _inventory_partition(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "reports": [inventory_pair(item) for item in items],
    }


def _global_metadata(
    manifest: list[dict[str, Any]], reports: list[dict[str, Any]]
) -> tuple[list[float], str, float, dict[int, int]]:
    reports.sort(key=lambda item: item["index"])
    if [item["index"] for item in reports] != list(range(len(manifest))):
        raise RuntimeError("inventory reports changed the input order")
    groups = {item["space_group"] for item in reports}
    if len(groups) != 1:
        provenance = {item["relative_expt"]: item["space_group"] for item in reports}
        raise ValueError(f"incompatible space groups: {json.dumps(provenance, sort_keys=True)}")
    crystal_count = sum(int(item["crystal_models"]) for item in reports)
    beam_count = sum(int(item["beam_models"]) for item in reports)
    if crystal_count < 1 or beam_count < 1:
        raise ValueError("input contains no crystal or beam models")
    cell = sum(
        np.asarray(item["cell_mean"], dtype=np.float64) * int(item["crystal_models"])
        for item in reports
    ) / crystal_count
    wavelength = sum(float(item["wavelength_sum"]) for item in reports) / beam_count
    offsets: dict[int, int] = {}
    cursor = 0
    for item in reports:
        offsets[item["index"]] = cursor
        cursor += int(item["experiments"])
    if cursor - 1 > np.iinfo(np.int32).max:
        raise OverflowError("global experiment count exceeds int32 BATCH capacity")
    return cell.tolist(), groups.pop(), wavelength, offsets


@ray.remote(num_cpus=1, max_restarts=0)
class DialsReader:
    """Stateful actor that holds one converted partition and serves bounded blocks."""

    def __init__(self, index: int, items: list[dict[str, Any]]) -> None:
        self.index = index
        self.items = items
        self.parts: list[dict[str, np.ndarray]] | None = None
        self.part_ranges: list[tuple[int, int]] = []
        self.rows = 0
        self.file_reports: list[dict[str, Any]] = []

    def load(self) -> dict[str, Any]:
        if self.parts is not None:
            raise RuntimeError("DIALS reader load may only be called once")
        self.parts = []
        cursor = 0
        for item in self.items:
            columns, report = convert_pair(
                item["expt"], item["refl"], item["batch_offset"]
            )
            report.update({
                "index": item["index"],
                "relative_expt": item["relative_expt"],
                "relative_refl": item["relative_refl"],
                "local_row_start": cursor,
                "local_row_stop": cursor + report["rows"],
            })
            self.file_reports.append(report)
            self.part_ranges.append((cursor, cursor + report["rows"]))
            cursor += report["rows"]
            self.parts.append(columns)
        if not self.parts:
            raise ValueError("a DIALS reader received an empty partition")
        self.rows = cursor
        return {
            "reader_index": self.index,
            "hostname": socket.gethostname(),
            "files": len(self.items),
            "rows": cursor,
            "file_reports": self.file_reports,
        }

    def get_block(self, start: int, stop: int) -> dict[str, np.ndarray]:
        if self.parts is None:
            raise RuntimeError("load the DIALS reader before requesting blocks")
        if start < 0 or stop <= start or stop > self.rows:
            raise IndexError(
                f"invalid DIALS block [{start}, {stop}) for {self.rows} rows"
            )
        block = {
            name: np.empty(stop - start, dtype=NUMPY_DTYPES[name])
            for name in DATASET_COLUMNS
        }
        copied = 0
        for part, (part_start, part_stop) in zip(self.parts, self.part_ranges):
            overlap_start = max(start, part_start)
            overlap_stop = min(stop, part_stop)
            if overlap_start >= overlap_stop:
                continue
            source_start = overlap_start - part_start
            source_stop = overlap_stop - part_start
            destination_start = overlap_start - start
            destination_stop = overlap_stop - start
            for name in DATASET_COLUMNS:
                block[name][destination_start:destination_stop] = part[name][
                    source_start:source_stop
                ]
            copied += overlap_stop - overlap_start
        if copied != stop - start:
            raise RuntimeError(f"DIALS actor copied {copied} block rows, expected {stop - start}")
        return block

    def release_all(self) -> int:
        rows = 0 if self.parts is None else self.rows
        self.parts = None
        self.part_ranges = []
        self.items = []
        self.file_reports = []
        gc.collect()
        return rows


def _on_node(remote: Any, node: dict[str, Any]) -> Any:
    return remote.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=str(node["NodeID"]), soft=False
        )
    )


def _inventory(
    partitions: list[list[dict[str, Any]]], nodes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    futures = [
        _on_node(_inventory_partition, nodes[index % len(nodes)]).remote(partition)
        for index, partition in enumerate(partitions)
    ]
    reports: list[dict[str, Any]] = []
    with tqdm(total=sum(map(len, partitions)), unit="file", desc="DIALS inventory") as progress:
        pending = list(futures)
        try:
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                result = ray.get(ready[0])
                reports.extend(result["reports"])
                progress.update(len(result["reports"]))
        except BaseException:
            for reference in pending:
                ray.cancel(reference, force=True)
            raise
    return reports


def _make_readers(
    partitions: list[list[dict[str, Any]]], nodes: list[dict[str, Any]]
) -> list[Any]:
    return [
        _on_node(DialsReader, nodes[index % len(nodes)]).remote(index, partition)
        for index, partition in enumerate(partitions)
    ]


def _load_readers(readers: list[Any], file_count: int) -> list[dict[str, Any]]:
    pending = [reader.load.remote() for reader in readers]
    reports = []
    with tqdm(total=file_count, unit="file", desc="DIALS conversion") as progress:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            report = ray.get(ready[0])
            reports.append(report)
            progress.update(report["files"])
    reports.sort(key=lambda item: item["reader_index"])
    return reports


def _transfer_blocks(
    readers: list[Any], reports: list[dict[str, Any]], block_mib: int
) -> dict[str, np.ndarray]:
    total_rows = sum(int(report["rows"]) for report in reports)
    if total_rows < 1:
        raise ValueError("DIALS input contains no reflections")
    destination = {
        name: np.empty(total_rows, dtype=NUMPY_DTYPES[name])
        for name in DATASET_COLUMNS
    }
    rows_per_block = max(1, block_mib * 1024 * 1024 // BYTES_PER_ROW)
    queues: list[deque[tuple[int, int, int, int]]] = []
    cursor = 0
    for report in reports:
        rows = int(report["rows"])
        queues.append(deque(
            (start, min(start + rows_per_block, rows), cursor + start,
             cursor + min(start + rows_per_block, rows))
            for start in range(0, rows, rows_per_block)
        ))
        cursor += rows
    if cursor != total_rows:
        raise RuntimeError("DIALS reader row accounting changed")

    pending: dict[Any, tuple[int, int, int]] = {}

    def schedule(reader_index: int) -> None:
        if not queues[reader_index]:
            return
        local_start, local_stop, global_start, global_stop = queues[reader_index].popleft()
        reference = readers[reader_index].get_block.remote(local_start, local_stop)
        pending[reference] = (reader_index, global_start, global_stop)

    for reader_index in range(len(readers)):
        schedule(reader_index)
    copied_rows = 0
    with tqdm(total=total_rows, unit="refl", desc="Ray transfer") as progress:
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            reader_index, start, stop = pending.pop(reference)
            block = ray.get(reference)
            expected_rows = stop - start
            if tuple(block) != DATASET_COLUMNS:
                raise RuntimeError(f"DIALS block schema changed: {tuple(block)}")
            for name in DATASET_COLUMNS:
                values = block[name]
                if values.dtype != NUMPY_DTYPES[name] or values.shape != (expected_rows,):
                    raise RuntimeError(f"DIALS block column {name} has an invalid schema")
                destination[name][start:stop] = values
            copied_rows += expected_rows
            progress.update(expected_rows)
            del block
            del reference
            schedule(reader_index)
    if copied_rows != total_rows:
        raise RuntimeError(f"copied {copied_rows} DIALS rows, expected {total_rows}")
    return destination


def _source_metadata(
    pairing: dict[str, Any], reports: list[dict[str, Any]]
) -> dict[str, Any]:
    files = []
    cursor = 0
    for reader in reports:
        for report in reader["file_reports"]:
            rows = int(report["rows"])
            files.append({
                "index": report["index"],
                "expt": report["relative_expt"],
                "refl": report["relative_refl"],
                "experiments": report["experiments"],
                "batch_offset": report["batch_offset"],
                "row_start": cursor,
                "row_stop": cursor + rows,
            })
            cursor += rows
    files.sort(key=lambda item: item["index"])
    return {
        "conversion": "cctbx-export-careless-compatible",
        "global_order": "sorted-relative-expt-path",
        "wavelength_filter": None,
        "pairing": pairing,
        "files": files,
    }


def read_dials_dataset(
    input_dir: str | Path,
    *,
    expt_glob: str = "**/*.expt",
    max_files: int | None = None,
    workers_per_node: int = 4,
    block_mib: int = 256,
    save_tensors: str | Path | None = None,
    tensor_shards: int | None = None,
) -> Any:
    """Build one in-memory DataSet from raw DIALS pairs, optionally caching it."""
    if workers_per_node < 1 or block_mib < 1:
        raise ValueError("Ray worker and block sizes must be positive")
    if max_files is not None and max_files < 1:
        raise ValueError("DIALS max-files must be positive")
    if tensor_shards is not None and tensor_shards < 1:
        raise ValueError("tensor shard count must be positive")
    if tensor_shards is not None and save_tensors is None:
        raise ValueError("tensor_shards requires save_tensors")
    if save_tensors is not None:
        destination = Path(save_tensors).absolute()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"refusing to overwrite prepared dataset: {destination}")
    manifest, pairing = discover_pairs(input_dir, expt_glob, max_files)
    initialized_here = not ray.is_initialized()
    if initialized_here:
        ray.init(
            address=os.environ.get("RAY_ADDRESS") or None,
            include_dashboard=False,
        )
    readers: list[Any] = []
    try:
        nodes = _live_nodes()
        worker_count = min(len(manifest), len(nodes) * workers_per_node)
        partitions = contiguous_weighted_partitions(manifest, worker_count)
        inventory = _inventory(partitions, nodes)
        cell, space_group, wavelength, batch_offsets = _global_metadata(manifest, inventory)
        for item in manifest:
            item["batch_offset"] = batch_offsets[item["index"]]
        readers = _make_readers(partitions, nodes)
        reports = _load_readers(readers, len(manifest))
        arrays = _transfer_blocks(readers, reports, block_mib)
        released = sum(ray.get([reader.release_all.remote() for reader in readers]))
        if released != len(arrays["H"]):
            raise RuntimeError(f"Ray readers released {released} rows, expected {len(arrays['H'])}")
        source = _source_metadata(pairing, reports)
        shard_default = len(readers)
    finally:
        for reader in readers:
            try:
                ray.kill(reader, no_restart=True)
            except Exception:
                pass
        if initialized_here:
            ray.shutdown()

    dataset = dataset_from_arrays(arrays, cell, space_group)
    if save_tensors is not None:
        write_prepared_dataset(
            dataset,
            save_tensors,
            shards=tensor_shards or shard_default,
            source=source,
            wavelength=wavelength,
        )
    return dataset
