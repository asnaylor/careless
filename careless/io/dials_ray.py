"""Prepare one complete Careless dataset from paired DIALS files using Ray."""

from __future__ import annotations

import argparse
import faulthandler
import json
import os
import socket
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


faulthandler.enable(all_threads=True)

# Import native DIALS/cctbx extensions before NumPy, Ray, Pandas, Gemmi, or
# Reciprocalspaceship. Ray workers import this module before constructing an
# actor, which avoids the observed Boost.Python import-order crashes.
from cctbx import sgtbx, uctbx  # noqa: E402
from dials.array_family import flex  # noqa: E402
from dxtbx.model.experiment_list import ExperimentListFactory  # noqa: E402

import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy  # noqa: E402
from tqdm import tqdm  # noqa: E402

from careless.io.prepared import MTZ_TYPES, write_prepared_dataset  # noqa: E402


REQUIRED_COLUMNS = (
    "id",
    "intensity.sum.variance",
    "intensity.sum.value",
    "miller_index",
    "xyzobs.px.value",
)
FLOAT_COLUMNS = ("I", "SigI", "xobs", "yobs", "ewald_offset")
PACKED_FLOATS = "float64_columns"
TRANSPORT_COLUMNS = ("hkl", "batch", PACKED_FLOATS)
BYTES_PER_ROW = 3 * 4 + 8 + len(FLOAT_COLUMNS) * 8


def _contiguous(array: np.ndarray, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _space_group(experiment: Any) -> str:
    info = experiment.crystal.get_space_group().info()
    symbols = sgtbx.space_group_symbols(info.symbol_and_number().split("(")[0])
    return symbols.universal_hermann_mauguin()


def _validate_reflections(reflections: Any, experiments: Any, path: str) -> np.ndarray:
    missing = sorted(set(REQUIRED_COLUMNS) - set(reflections.keys()))
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")
    if reflections.size() == 0:
        raise ValueError(f"{path}: reflection table is empty")
    if len(experiments) == 0:
        raise ValueError(f"{path}: experiment list is empty")
    experiment_ids = _contiguous(
        reflections["id"].as_numpy_array(), np.dtype(np.int64)
    )
    invalid = np.flatnonzero((experiment_ids < 0) | (experiment_ids >= len(experiments)))
    if invalid.size:
        row = int(invalid[0])
        raise ValueError(
            f"{path}: invalid experiment id at row {row}: "
            f"{int(experiment_ids[row])}; reflections are never filtered"
        )
    variances = reflections["intensity.sum.variance"].as_numpy_array()
    invalid = np.flatnonzero(~np.isfinite(variances) | (variances <= 0))
    if invalid.size:
        row = int(invalid[0])
        raise ValueError(
            f"{path}: invalid intensity.sum.variance at row {row}: "
            f"{variances[row]!r}; reflections are never filtered"
        )
    return experiment_ids


def convert_pair(expt_path: str, refl_path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Apply the validated serial DIALS mapping without filtering any rows."""
    experiments = ExperimentListFactory.from_json_file(expt_path, check_format=False)
    reflections = flex.reflection_table.from_file(refl_path)
    row_count = reflections.size()
    experiment_ids = _validate_reflections(reflections, experiments, refl_path)

    miller_indices = reflections["miller_index"].as_vec3_double()
    hkl = _contiguous(miller_indices.as_numpy_array(), np.dtype(np.int32))
    xyzobs = _contiguous(reflections["xyzobs.px.value"].as_numpy_array())
    intensities = _contiguous(reflections["intensity.sum.value"].as_numpy_array())
    variances = _contiguous(reflections["intensity.sum.variance"].as_numpy_array())
    uncertainties = _contiguous(np.sqrt(variances))
    if hkl.shape != (row_count, 3) or xyzobs.shape != (row_count, 3):
        raise ValueError(
            f"{refl_path}: unexpected hkl/xyz shapes: {hkl.shape}, {xyzobs.shape}"
        )
    if not (
        intensities.dtype
        == uncertainties.dtype
        == xyzobs.dtype
        == np.dtype(np.float64)
    ):
        raise TypeError(f"{refl_path}: DIALS floating-point columns are not float64")

    selector = flex.size_t(experiment_ids.astype(np.uint64, copy=False))
    matrices = flex.mat3_double(
        [experiment.crystal.get_A() for experiment in experiments]
    ).select(selector)
    incident_beams = flex.vec3_double(
        [experiment.beam.get_s0() for experiment in experiments]
    ).select(selector)
    wavelengths = flex.double(
        [experiment.beam.get_wavelength() for experiment in experiments]
    ).select(selector)
    predicted_s1 = matrices * miller_indices + incident_beams
    ewald_offset = _contiguous(
        (predicted_s1.norms() - (1.0 / wavelengths)).as_numpy_array()
    )

    used_ids = np.unique(experiment_ids)
    cells = []
    space_groups = []
    for experiment_id in used_ids:
        experiment = experiments[int(experiment_id)]
        cells.append(tuple(experiment.crystal.get_unit_cell().parameters()))
        space_groups.append(_space_group(experiment))
    if len(set(space_groups)) != 1:
        raise ValueError(
            f"{expt_path}: used experiments have incompatible space groups: "
            f"{sorted(set(space_groups))}"
        )

    columns = {
        "hkl": torch.from_numpy(hkl),
        "batch": torch.from_numpy(experiment_ids),
        "I": torch.from_numpy(intensities),
        "SigI": torch.from_numpy(uncertainties),
        "xobs": torch.from_numpy(_contiguous(xyzobs[:, 0])),
        "yobs": torch.from_numpy(_contiguous(xyzobs[:, 1])),
        "ewald_offset": torch.from_numpy(ewald_offset),
    }
    for name, tensor in columns.items():
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise RuntimeError(f"{refl_path}: {name} is not a contiguous CPU tensor")
        if tensor.shape[0] != row_count:
            raise RuntimeError(f"{refl_path}: {name} row count changed")
    return columns, {
        "rows": row_count,
        "experiments": len(experiments),
        "cells": cells,
        "space_group": space_groups[0],
    }


def _concatenate(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    names = tuple(parts[0])
    if any(tuple(part) != names for part in parts):
        raise RuntimeError("converted tensor schemas differ between files")
    return {name: torch.cat([part[name] for part in parts], dim=0) for name in names}


def _pack_block(
    columns: dict[str, torch.Tensor], start: int, stop: int
) -> dict[str, torch.Tensor]:
    rows = stop - start
    block = {
        "hkl": columns["hkl"][start:stop].clone(),
        "batch": columns["batch"][start:stop].clone(),
        PACKED_FLOATS: torch.empty((rows, len(FLOAT_COLUMNS)), dtype=torch.float64),
    }
    for index, name in enumerate(FLOAT_COLUMNS):
        block[PACKED_FLOATS][:, index].copy_(columns[name][start:stop])
    return block


@ray.remote(num_cpus=1)
class DialsReader:
    def __init__(self, index: int, manifest: list[dict[str, Any]]) -> None:
        self._native_modules = (flex, sgtbx, uctbx, ExperimentListFactory)
        self.index = index
        self.manifest = manifest
        self.columns: dict[str, torch.Tensor] | None = None
        self.reports = []
        self.file_slices = []

    def load(self) -> dict[str, Any]:
        if self.columns is not None:
            raise RuntimeError("reader load may only be called once")
        parts = []
        cursor = 0
        for item in self.manifest:
            columns, report = convert_pair(item["expt"], item["refl"])
            stop = cursor + report["rows"]
            self.file_slices.append((item["index"], cursor, stop))
            report["index"] = item["index"]
            report["relative_expt"] = item["relative_expt"]
            self.reports.append(report)
            parts.append(columns)
            cursor = stop
        self.columns = _concatenate(parts)
        return {
            "reader_index": self.index,
            "hostname": socket.gethostname(),
            "files": len(self.manifest),
            "rows": cursor,
            "reports": self.reports,
        }

    def assign_image_ids(self, offsets: dict[int, int]) -> None:
        if self.columns is None:
            raise RuntimeError("load before assigning image ids")
        for file_index, start, stop in self.file_slices:
            self.columns["batch"][start:stop].add_(int(offsets[file_index]))

    def block_descriptors(self, rows_per_block: int, global_start: int) -> list[dict[str, int]]:
        if self.columns is None:
            raise RuntimeError("load before requesting blocks")
        rows = self.columns["batch"].numel()
        return [
            {
                "local_start": start,
                "local_stop": min(start + rows_per_block, rows),
                "global_start": global_start + start,
                "global_stop": global_start + min(start + rows_per_block, rows),
            }
            for start in range(0, rows, rows_per_block)
        ]

    def get_block(self, start: int, stop: int) -> dict[str, torch.Tensor]:
        if self.columns is None:
            raise RuntimeError("load before requesting blocks")
        return _pack_block(self.columns, start, stop)

    def release_all(self) -> int:
        rows = self.columns["batch"].numel() if self.columns is not None else 0
        self.columns = None
        return rows


@ray.remote(num_cpus=1)
class DatasetAssembler:
    def __init__(self, total_rows: int) -> None:
        self._native_modules = (flex, sgtbx, uctbx, ExperimentListFactory)
        self.total_rows = total_rows
        self.columns = {
            "hkl": torch.empty((total_rows, 3), dtype=torch.int32),
            "batch": torch.empty(total_rows, dtype=torch.int64),
            **{
                name: torch.empty(total_rows, dtype=torch.float64)
                for name in FLOAT_COLUMNS
            },
        }
        self.intervals: dict[str, tuple[int, int]] = {}

    def receive(self, token: str, start: int, stop: int, block: dict[str, torch.Tensor]) -> str:
        if token in self.intervals:
            raise RuntimeError(f"duplicate block token: {token}")
        if start < 0 or stop <= start or stop > self.total_rows:
            raise RuntimeError(f"invalid block range for {token}: [{start}, {stop})")
        for other, (other_start, other_stop) in self.intervals.items():
            if start < other_stop and other_start < stop:
                raise RuntimeError(f"block {token} overlaps {other}")
        if tuple(block) != TRANSPORT_COLUMNS:
            raise RuntimeError(f"invalid packed block schema: {tuple(block)}")
        rows = stop - start
        expected = {
            "hkl": ((rows, 3), torch.int32),
            "batch": ((rows,), torch.int64),
            PACKED_FLOATS: ((rows, len(FLOAT_COLUMNS)), torch.float64),
        }
        for name, tensor in block.items():
            shape, dtype = expected[name]
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise RuntimeError(f"block {token} tensor {name} is not contiguous CPU")
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise RuntimeError(f"block {token} tensor {name} has the wrong schema")
        self.columns["hkl"][start:stop].copy_(block["hkl"])
        self.columns["batch"][start:stop].copy_(block["batch"])
        for index, name in enumerate(FLOAT_COLUMNS):
            self.columns[name][start:stop].copy_(block[PACKED_FLOATS][:, index])
        self.intervals[token] = (start, stop)
        return token

    def write(
        self,
        destination: str,
        shards: int,
        cell: list[float],
        space_group: str,
        source: dict[str, Any],
    ) -> dict[str, Any]:
        cursor = 0
        for token, (start, stop) in sorted(self.intervals.items(), key=lambda item: item[1]):
            if start != cursor:
                raise RuntimeError(f"assembled rows have a gap before {token}: {cursor} != {start}")
            cursor = stop
        if cursor != self.total_rows:
            raise RuntimeError(f"assembled rows end at {cursor}, expected {self.total_rows}")

        import gemmi
        import pandas as pd
        import reciprocalspaceship as rs

        frame = pd.DataFrame(
            {
                "H": self.columns["hkl"][:, 0].numpy(),
                "K": self.columns["hkl"][:, 1].numpy(),
                "L": self.columns["hkl"][:, 2].numpy(),
                "I": self.columns["I"].numpy(),
                "SigI": self.columns["SigI"].numpy(),
                "BATCH": self.columns["batch"].numpy(),
                "xobs": self.columns["xobs"].numpy(),
                "yobs": self.columns["yobs"].numpy(),
                "ewald_offset": self.columns["ewald_offset"].numpy(),
            }
        )
        dataset = rs.DataSet(
            frame,
            cell=gemmi.UnitCell(*cell),
            spacegroup=gemmi.SpaceGroup(space_group),
        )
        for name in dataset.columns:
            dataset[name] = dataset[name].astype(MTZ_TYPES.get(name, "R"))
        dataset.set_index(["H", "K", "L"], inplace=True)
        self.columns = {}
        manifest = write_prepared_dataset(dataset, destination, shards=shards, source=source)
        return {
            "path": str(Path(destination).absolute()),
            "rows": len(dataset),
            "shards": len(manifest["shards"]),
            "space_group": manifest["space_group"],
            "cell": manifest["cell"],
        }


def discover_pairs(
    input_dir: Path, expt_glob: str, max_files: int | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")
    if not expt_glob.endswith(".expt"):
        raise ValueError("--expt-glob must end in '.expt'")
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
    complete = sorted(set(expts) & set(refls))
    selected = complete if max_files is None else complete[:max_files]
    report = {
        "experiment_files": len(expts),
        "reflection_files": len(refls),
        "complete_pairs": len(complete),
        "selected_pairs": len(selected),
        "max_files": max_files,
        "missing_reflection_files": [
            expts[stem].with_suffix(".refl").relative_to(input_dir).as_posix()
            for stem in sorted(set(expts) - set(refls))
        ],
        "missing_experiment_files": [
            refls[stem].with_suffix(".expt").relative_to(input_dir).as_posix()
            for stem in sorted(set(refls) - set(expts))
        ],
    }
    print(
        "PAIRING "
        f"experiments={len(expts)} reflections={len(refls)} "
        f"complete={len(complete)} selected={len(selected)} "
        f"missing_reflections={len(report['missing_reflection_files'])} "
        f"missing_experiments={len(report['missing_experiment_files'])}",
        flush=True,
    )
    if report["missing_reflection_files"]:
        print(
            "MISSING_REFLECTION_SAMPLE="
            + json.dumps(report["missing_reflection_files"][:10]),
            flush=True,
        )
    if report["missing_experiment_files"]:
        print(
            "MISSING_EXPERIMENT_SAMPLE="
            + json.dumps(report["missing_experiment_files"][:10]),
            flush=True,
        )
    if not selected:
        raise FileNotFoundError("no complete .expt/.refl pairs were found")
    manifest = [
        {
            "index": index,
            "expt": str(expts[stem].resolve()),
            "refl": str(refls[stem].resolve()),
            "relative_expt": expts[stem].relative_to(input_dir).as_posix(),
            "relative_refl": refls[stem].relative_to(input_dir).as_posix(),
        }
        for index, stem in enumerate(selected)
    ]
    return manifest, report


def _partition(manifest: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    count = min(count, len(manifest))
    return [
        manifest[index * len(manifest) // count : (index + 1) * len(manifest) // count]
        for index in range(count)
    ]


def _live_nodes() -> list[dict[str, Any]]:
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    nodes.sort(key=lambda node: (str(node.get("NodeManagerAddress")), str(node["NodeID"])))
    if not nodes:
        raise RuntimeError("Ray reports no live nodes")
    return nodes


def _global_metadata(
    manifest: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    length_tolerance: float,
    angle_tolerance: float,
) -> tuple[list[float], str, dict[int, int], dict[int, int]]:
    reports.sort(key=lambda item: item["index"])
    if [item["relative_expt"] for item in reports] != [
        item["relative_expt"] for item in manifest
    ]:
        raise RuntimeError("reader manifest order changed")
    groups = {item["space_group"] for item in reports}
    if len(groups) != 1:
        raise ValueError(f"incompatible space groups: {sorted(groups)}")
    cells = [tuple(cell) for item in reports for cell in item["cells"]]
    reference = uctbx.unit_cell(cells[0])
    for cell in cells[1:]:
        if not reference.is_similar_to(
            uctbx.unit_cell(cell),
            relative_length_tolerance=length_tolerance,
            absolute_angle_tolerance=angle_tolerance,
        ):
            raise ValueError(f"incompatible unit cells: {cells[0]} and {cell}")
    row_offsets = {}
    image_offsets = {}
    row_cursor = 0
    image_cursor = 0
    for item in reports:
        row_offsets[item["index"]] = row_cursor
        image_offsets[item["index"]] = image_cursor
        row_cursor += item["rows"]
        image_cursor += item["experiments"]
    return (
        np.mean(np.asarray(cells, dtype=np.float64), axis=0).tolist(),
        groups.pop(),
        row_offsets,
        image_offsets,
    )


def _source_metadata(
    manifest: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    row_offsets: dict[int, int],
    image_offsets: dict[int, int],
    pairing: dict[str, Any],
) -> dict[str, Any]:
    reports_by_index = {item["index"]: item for item in reports}
    files = []
    for item in manifest:
        index = item["index"]
        report = reports_by_index[index]
        files.append(
            {
                "index": index,
                "expt": item["relative_expt"],
                "refl": item["relative_refl"],
                "row_start": row_offsets[index],
                "row_stop": row_offsets[index] + report["rows"],
                "image_start": image_offsets[index],
                "image_stop": image_offsets[index] + report["experiments"],
            }
        )
    return {
        "conversion": "validated-dials-stills-mapping",
        "global_order": "sorted-relative-pair-path",
        "global_image_ids": "cumulative-experiment-count-in-global-order",
        "pairing": pairing,
        "files": files,
    }


def _make_readers(
    partitions: list[list[dict[str, Any]]], nodes: list[dict[str, Any]]
) -> list[Any]:
    readers = []
    for index, partition in enumerate(partitions):
        node = nodes[index % len(nodes)]
        readers.append(
            DialsReader.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=str(node["NodeID"]), soft=False
                )
            ).remote(index, partition)
        )
    return readers


def _transfer(
    readers: list[Any],
    reader_reports: list[dict[str, Any]],
    assembler: Any,
    total_rows: int,
    block_mib: int,
    max_in_flight: int,
) -> int:
    rows_per_block = max(1, block_mib * 1024 * 1024 // BYTES_PER_ROW)
    rows_by_reader = {item["reader_index"]: item["rows"] for item in reader_reports}
    queues = {}
    global_start = 0
    for index, reader in enumerate(readers):
        queues[index] = deque(
            ray.get(reader.block_descriptors.remote(rows_per_block, global_start))
        )
        global_start += rows_by_reader[index]
    if global_start != total_rows:
        raise RuntimeError(f"reader rows sum to {global_start}, expected {total_rows}")

    in_flight = {}
    active_by_reader = defaultdict(int)
    block_index = 0
    next_reader = 0

    def schedule(reader_index: int) -> bool:
        nonlocal block_index
        if not queues[reader_index] or active_by_reader[reader_index]:
            return False
        descriptor = queues[reader_index].popleft()
        token = f"object-{reader_index}-{block_index}"
        block_index += 1
        block = readers[reader_index].get_block.remote(
            descriptor["local_start"], descriptor["local_stop"]
        )
        acknowledgement = assembler.receive.remote(
            token,
            descriptor["global_start"],
            descriptor["global_stop"],
            block,
        )
        # Hold the object reference until the assembler confirms its copy.
        in_flight[acknowledgement] = (reader_index, token, block)
        active_by_reader[reader_index] = 1
        return True

    def fill() -> None:
        nonlocal next_reader
        checked = 0
        while len(in_flight) < max_in_flight and checked < len(readers):
            reader_index = next_reader
            next_reader = (next_reader + 1) % len(readers)
            checked = 0 if schedule(reader_index) else checked + 1

    fill()
    while in_flight:
        ready, _ = ray.wait(list(in_flight), num_returns=1)
        acknowledgement = ready[0]
        reader_index, token, _block = in_flight.pop(acknowledgement)
        if ray.get(acknowledgement) != token:
            raise RuntimeError(f"block acknowledgement changed for {token}")
        active_by_reader[reader_index] = 0
        fill()
    return block_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--prepared-out", type=Path, required=True)
    parser.add_argument("--expt-glob", default="**/*.expt")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--readers-per-node", type=int, default=4)
    parser.add_argument("--block-mib", type=int, default=256)
    parser.add_argument("--max-in-flight", type=int, choices=(1, 2), default=2)
    parser.add_argument("--prepared-shards", type=int, default=8)
    parser.add_argument("--cell-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--cell-angle-tolerance", type=float, default=1.0)
    args = parser.parse_args()
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if args.readers_per_node < 1 or args.block_mib < 1 or args.prepared_shards < 1:
        parser.error("reader, block, and shard counts must be positive")
    if args.cell_relative_tolerance <= 0 or args.cell_angle_tolerance <= 0:
        parser.error("cell tolerances must be positive")
    return args


def main() -> None:
    args = parse_args()
    manifest, pairing = discover_pairs(
        args.input_dir.resolve(), args.expt_glob, args.max_files
    )
    ray.init(
        address="auto" if os.environ.get("RAY_ADDRESS") else None,
        include_dashboard=False,
    )
    try:
        nodes = _live_nodes()
        partitions = _partition(manifest, len(nodes) * args.readers_per_node)
        readers = _make_readers(partitions, nodes)
        pending = {reader.load.remote(): reader for reader in readers}
        reader_reports = []
        with tqdm(total=len(manifest), unit="file", desc="DIALS ingestion") as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                report = ray.get(ready[0])
                pending.pop(ready[0])
                reader_reports.append(report)
                progress.update(report["files"])
        reader_reports.sort(key=lambda item: item["reader_index"])
        file_reports = [item for reader in reader_reports for item in reader["reports"]]
        cell, space_group, row_offsets, image_offsets = _global_metadata(
            manifest,
            file_reports,
            args.cell_relative_tolerance,
            args.cell_angle_tolerance,
        )
        ray.get([reader.assign_image_ids.remote(image_offsets) for reader in readers])
        total_rows = sum(item["rows"] for item in file_reports)
        assembler_node = nodes[-1]
        assembler = DatasetAssembler.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=str(assembler_node["NodeID"]), soft=False
            )
        ).remote(total_rows)
        blocks = _transfer(
            readers,
            reader_reports,
            assembler,
            total_rows,
            args.block_mib,
            args.max_in_flight,
        )
        released_rows = sum(ray.get([reader.release_all.remote() for reader in readers]))
        if released_rows != total_rows:
            raise RuntimeError(f"readers released {released_rows} rows, expected {total_rows}")
        source = _source_metadata(
            manifest, file_reports, row_offsets, image_offsets, pairing
        )
        artifact = ray.get(
            assembler.write.remote(
                str(args.prepared_out.resolve()),
                args.prepared_shards,
                cell,
                space_group,
                source,
            )
        )
        result = {
            "status": "pass",
            "files": len(manifest),
            "rows": total_rows,
            "blocks": blocks,
            "readers": len(readers),
            "nodes": [str(node["NodeManagerAddress"]) for node in nodes],
            "pairing": {
                "experiment_files": pairing["experiment_files"],
                "reflection_files": pairing["reflection_files"],
                "complete_pairs": pairing["complete_pairs"],
                "selected_pairs": pairing["selected_pairs"],
                "max_files": pairing["max_files"],
                "missing_reflections": len(pairing["missing_reflection_files"]),
                "missing_experiments": len(pairing["missing_experiment_files"]),
            },
            "prepared": artifact,
        }
        print(f"CARELESS_PREPARE_RESULT={json.dumps(result, sort_keys=True)}", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
