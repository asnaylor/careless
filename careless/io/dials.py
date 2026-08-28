"""Auditable DIALS mapping used by the Careless SafeTensors exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Native extensions must be loaded before NumPy/Pandas/Ray in worker processes.
from cctbx import sgtbx
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory
from scitbx import matrix

import numpy as np

from careless.io.prepared import DATASET_COLUMNS, NUMPY_DTYPES


REQUIRED_COLUMNS = (
    "id",
    "intensity.sum.value",
    "intensity.sum.variance",
    "miller_index",
    "s1",
    "xyzcal.px",
    "xyzobs.px.value",
    "xyzobs.px.variance",
)
INT32_MAX = np.iinfo(np.int32).max


def _space_group(crystal: Any) -> str:
    info = crystal.get_space_group().info()
    symbols = sgtbx.space_group_symbols(info.symbol_and_number().split("(")[0])
    return symbols.universal_hermann_mauguin()


def _load_experiments(path: str) -> Any:
    experiments = ExperimentListFactory.from_json_file(path, check_format=False)
    if len(experiments) == 0:
        raise ValueError(f"{path}: experiment list is empty")
    if len(experiments.crystals()) == 0 or len(experiments.beams()) == 0:
        raise ValueError(f"{path}: experiments have no crystal or beam models")
    if any(experiment.crystal is None or experiment.beam is None for experiment in experiments):
        raise ValueError(f"{path}: every experiment must have a crystal and beam model")
    return experiments


def inventory_pair(item: dict[str, Any]) -> dict[str, Any]:
    """Read small experiment metadata needed before distributed conversion."""
    experiments = _load_experiments(item["expt"])
    crystals = experiments.crystals()
    beams = experiments.beams()
    groups = {_space_group(crystal) for crystal in crystals}
    if len(groups) != 1:
        raise ValueError(f"{item['relative_expt']}: crystals have incompatible space groups: {sorted(groups)}")
    cells = np.asarray(
        [crystal.get_unit_cell().parameters() for crystal in crystals],
        dtype=np.float64,
    )
    wavelengths = np.asarray([beam.get_wavelength() for beam in beams], dtype=np.float64)
    if not np.all(np.isfinite(cells)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError(f"{item['relative_expt']}: non-finite crystal or beam metadata")
    return {
        "index": item["index"],
        "relative_expt": item["relative_expt"],
        "experiments": len(experiments),
        "crystal_models": len(crystals),
        "beam_models": len(beams),
        "cell_mean": cells.mean(axis=0).tolist(),
        "cell_min": cells.min(axis=0).tolist(),
        "cell_max": cells.max(axis=0).tolist(),
        "wavelength_sum": float(wavelengths.sum()),
        "wavelength_min": float(wavelengths.min()),
        "wavelength_max": float(wavelengths.max()),
        "space_group": groups.pop(),
    }


def _array(values: Any, dtype: np.dtype[Any]) -> np.ndarray:
    return np.ascontiguousarray(values.as_numpy_array(), dtype=dtype)


def _a_inverse(crystal: Any) -> Any:
    """Use the exporter API when present, with an exact base-Crystal fallback."""
    method = getattr(crystal, "get_A_inverse_as_sqr", None)
    if method is not None:
        return method()
    return matrix.sqr(crystal.get_A()).inverse()


def convert_pair(
    expt_path: str,
    refl_path: str,
    batch_offset: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reproduce the observable column mapping of the cctbx MTZ exporter."""
    experiments = _load_experiments(expt_path)
    reflections = flex.reflection_table.from_file(refl_path)
    rows = reflections.size()
    if rows == 0:
        raise ValueError(f"{refl_path}: reflection table is empty")
    missing = sorted(set(REQUIRED_COLUMNS) - set(reflections.keys()))
    if missing:
        raise ValueError(f"{refl_path}: missing required columns: {missing}")

    experiment_ids = np.ascontiguousarray(
        reflections["id"].as_numpy_array(), dtype=np.int64
    )
    invalid = np.flatnonzero(
        (experiment_ids < 0) | (experiment_ids >= len(experiments))
    )
    if invalid.size:
        row = int(invalid[0])
        raise ValueError(
            f"{refl_path}: invalid experiment id at row {row}: "
            f"{int(experiment_ids[row])}"
        )
    global_ids = experiment_ids + int(batch_offset)
    if global_ids.min() < 0 or global_ids.max() > INT32_MAX:
        raise OverflowError(f"{refl_path}: global BATCH values exceed int32")

    selector = flex.size_t(experiment_ids.astype(np.uint64, copy=False))
    matrices_a = flex.mat3_double(
        [crystal.get_A() for crystal in experiments.crystals()]
    ).select(selector)
    matrices_ainv = flex.mat3_double(
        [_a_inverse(crystal) for crystal in experiments.crystals()]
    ).select(selector)
    matrices_b = flex.mat3_double(
        [crystal.get_B() for crystal in experiments.crystals()]
    ).select(selector)
    incident = flex.vec3_double(
        [experiment.beam.get_s0() for experiment in experiments]
    ).select(selector)
    wavelengths = flex.double(
        [experiment.beam.get_wavelength() for experiment in experiments]
    ).select(selector)
    h = reflections["miller_index"].as_vec3_double()
    predicted_s1 = matrices_a * h + incident
    ewald_offset = predicted_s1.norms() - (1.0 / wavelengths)
    miller_index_obs = matrices_ainv * (reflections["s1"] - incident)
    cartesian_fixed_obs = matrices_b * miller_index_obs
    cartesian_fixed = matrices_b * h

    hkl = _array(h, np.dtype(np.float64))
    observed_fixed = _array(cartesian_fixed_obs, np.dtype(np.float64))
    predicted_fixed = _array(cartesian_fixed, np.dtype(np.float64))
    xyzcal = _array(reflections["xyzcal.px"], np.dtype(np.float64))
    xyzobs = _array(reflections["xyzobs.px.value"], np.dtype(np.float64))
    xyzvariance = _array(reflections["xyzobs.px.variance"], np.dtype(np.float64))
    intensity = _array(reflections["intensity.sum.value"], np.dtype(np.float64))
    variance = _array(reflections["intensity.sum.variance"], np.dtype(np.float64))
    offset = _array(ewald_offset, np.dtype(np.float64))
    if any(array.shape[0] != rows for array in (
        hkl, observed_fixed, predicted_fixed, xyzcal, xyzobs, xyzvariance,
        intensity, variance, offset,
    )):
        raise RuntimeError(f"{refl_path}: converted column row counts differ")

    values: dict[str, Any] = {
        "H": hkl[:, 0], "K": hkl[:, 1], "L": hkl[:, 2],
        "BATCH": global_ids,
        "cartesian_fixed_obs_x": observed_fixed[:, 0],
        "cartesian_fixed_obs_y": observed_fixed[:, 1],
        "cartesian_fixed_obs_z": observed_fixed[:, 2],
        "cartesian_fixed_x": predicted_fixed[:, 0],
        "cartesian_fixed_y": predicted_fixed[:, 1],
        "cartesian_fixed_z": predicted_fixed[:, 2],
        "ewald_offset": offset,
        "I": intensity,
        "SigI": np.sqrt(variance),
        "xcal": xyzcal[:, 0], "ycal": xyzcal[:, 1],
        "xobs": xyzobs[:, 0], "yobs": xyzobs[:, 1],
        "sigxobs": np.sqrt(xyzvariance[:, 0]),
        "sigyobs": np.sqrt(xyzvariance[:, 1]),
        "cartesian_delta_x": observed_fixed[:, 0] - predicted_fixed[:, 0],
        "cartesian_delta_y": observed_fixed[:, 1] - predicted_fixed[:, 1],
        "cartesian_delta_z": observed_fixed[:, 2] - predicted_fixed[:, 2],
    }
    columns = {
        name: np.ascontiguousarray(values[name], dtype=NUMPY_DTYPES[name])
        for name in DATASET_COLUMNS
    }
    return columns, {
        "rows": rows,
        "experiments": len(experiments),
        "batch_offset": int(batch_offset),
        "expt": str(Path(expt_path)),
        "refl": str(Path(refl_path)),
    }
