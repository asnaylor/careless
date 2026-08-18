"""Summarize Careless training ranges from an Nsight Systems SQLite export."""

import argparse
import sqlite3
from pathlib import Path


RANGE_DURATIONS_QUERY = """
    SELECT events.end - events.start AS duration_ns
    FROM NVTX_EVENTS AS events
    LEFT JOIN StringIds AS strings ON strings.id = events.textId
    WHERE COALESCE(events.text, strings.value) = ?
      AND events.end IS NOT NULL
    ORDER BY events.start
"""


def range_durations(connection, range_name):
    return [
        row[0]
        for row in connection.execute(RANGE_DURATIONS_QUERY, (range_name,))
    ]


def summarize(sqlite_file, skip_first=0):
    sqlite_file = Path(sqlite_file)
    if not sqlite_file.is_file():
        raise ValueError(f"Profile does not exist: {sqlite_file}")
    if skip_first < 0:
        raise ValueError("--skip-first must be non-negative")

    connection = sqlite3.connect(sqlite_file)
    try:
        run_durations = range_durations(connection, "train.run")
        step_durations = range_durations(connection, "train.step")
    except sqlite3.Error as error:
        raise ValueError(
            f"Could not read NVTX ranges from {sqlite_file}: {error}"
        ) from error
    finally:
        connection.close()

    if not run_durations:
        raise ValueError("No 'train.run' NVTX ranges found")
    if not step_durations:
        raise ValueError("No 'train.step' NVTX ranges found")
    if skip_first >= len(step_durations):
        raise ValueError(
            f"Cannot skip {skip_first} steps; profile contains {len(step_durations)}"
        )

    measured_steps = step_durations[skip_first:]
    return {
        "runs": len(run_durations),
        "total_training_seconds": sum(run_durations) / 1e9,
        "steps": len(step_durations),
        "average_step_milliseconds": sum(measured_steps) / len(measured_steps) / 1e6,
        "skipped_steps": skip_first,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Report total training time and average step time from a Careless "
            "Nsight Systems SQLite export."
        )
    )
    parser.add_argument("sqlite_profile", type=Path)
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Exclude this many warm-up steps from the average (default: 0).",
    )
    args = parser.parse_args()

    try:
        summary = summarize(args.sqlite_profile, args.skip_first)
    except ValueError as error:
        parser.error(str(error))

    print(f"Training runs: {summary['runs']}")
    print(f"Total training time: {summary['total_training_seconds']:.6f} s")
    print(f"Training steps: {summary['steps']}")
    qualifier = ""
    if summary["skipped_steps"]:
        qualifier = f" (excluding first {summary['skipped_steps']})"
    print(
        "Average step time"
        f"{qualifier}: {summary['average_step_milliseconds']:.3f} ms"
    )


if __name__ == "__main__":
    main()
