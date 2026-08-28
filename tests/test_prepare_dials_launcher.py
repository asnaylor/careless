import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY / "scripts" / "run_prepare_dials_perlmutter"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _launcher_environment(
    tmp_path: Path, *, cpus: int = 8
) -> Tuple[Dict[str, str], Path]:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "scontrol",
        "#!/usr/bin/env bash\nprintf 'nid000001\\nnid000002\\n'\n",
    )
    _write_executable(fake_bin / "podman-hpc", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "srun",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$FAKE_SRUN_ARGS"
prepared_out=""
for ((index = 1; index <= $#; index++)); do
  argument="${!index}"
  case "$argument" in
    --prepared-out)
      next=$((index + 1))
      prepared_out="${!next}"
      ;;
    --prepared-out=*)
      prepared_out="${argument#*=}"
      ;;
  esac
done
[[ -n "$prepared_out" ]]
if [[ "${FAKE_PREPARE_MODE:-success}" == "success" ]]; then
  mkdir -p "$prepared_out"
  printf '{"total_rows": 1, "shards": [{"file": "part-00000-of-00001.safetensors"}]}\n' \
    > "$prepared_out/manifest.json"
  printf 'manifest_sha256=fake\n' > "$prepared_out/COMPLETE"
  printf 'fake shard\n' > "$prepared_out/part-00000-of-00001.safetensors"
fi
printf 'CARELESS_PREPARE_RESULT={"prepared": {"path": "%s"}, "status": "pass"}\n' \
  "$prepared_out"
sleep 30
""",
    )
    arguments_path = tmp_path / "srun-arguments.txt"
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "IMAGE": "example.invalid/careless:test",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NODELIST": "nid[000001-000002]",
            "SLURM_CPUS_PER_TASK": str(cpus),
            "SCRATCH": str(tmp_path / "scratch"),
            "CARELESS_PREPARE_LOG": str(tmp_path / "prepare.log"),
            "FAKE_SRUN_ARGS": str(arguments_path),
        }
    )
    return environment, arguments_path


def _run_launcher(arguments: List[str], environment: Dict[str, str]):
    return subprocess.run(
        [str(LAUNCHER), *arguments],
        cwd=REPOSITORY,
        env=environment,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=15,
        check=False,
    )


def test_launcher_mounts_input_and_output_and_preserves_arguments(tmp_path):
    input_dir = tmp_path / "input data"
    input_dir.mkdir()
    output = tmp_path / "prepared outputs" / "smoke test"
    environment, arguments_path = _launcher_environment(tmp_path)

    result = _run_launcher(
        [
            "--input-dir",
            str(input_dir),
            "--expt-glob",
            "*_reintegrated_*.expt",
            "--max-files",
            "10",
            "--prepared-out",
            str(output),
            "--readers-per-node",
            "4",
            "--block-mib",
            "256",
            "--max-in-flight",
            "2",
            "--prepared-shards",
            "16",
        ],
        environment,
    )

    assert result.returncode == 0, result.stdout
    assert "passed and persisted" in result.stdout
    assert (output / "manifest.json").is_file()
    assert (output / "COMPLETE").is_file()
    forwarded = arguments_path.read_text().splitlines()
    resolved_input = str(input_dir.resolve())
    resolved_parent = str(output.parent.resolve())
    assert f"{resolved_input}:{resolved_input}:ro" in forwarded
    assert f"{resolved_parent}:{resolved_parent}:rw" in forwarded
    assert "*_reintegrated_*.expt" in forwarded
    input_index = forwarded.index("--input-dir")
    output_index = forwarded.index("--prepared-out")
    assert forwarded[input_index + 1] == resolved_input
    assert forwarded[output_index + 1] == str(output.resolve())


def test_launcher_rejects_container_only_false_success(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output = tmp_path / "output" / "prepared"
    environment, _ = _launcher_environment(tmp_path)
    environment["FAKE_PREPARE_MODE"] = "no-artifact"

    result = _run_launcher(
        ["--input-dir", str(input_dir), "--prepared-out", str(output)],
        environment,
    )

    assert result.returncode == 2
    assert "prepared output was not persisted on the host" in result.stdout


def test_launcher_rejects_existing_output_before_starting_srun(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output = tmp_path / "prepared"
    output.mkdir()
    environment, arguments_path = _launcher_environment(tmp_path)

    result = _run_launcher(
        ["--input-dir", str(input_dir), "--prepared-out", str(output)],
        environment,
    )

    assert result.returncode == 2
    assert "prepared output already exists" in result.stdout
    assert not arguments_path.exists()


def test_launcher_reserves_one_cpu_for_assembly(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output = tmp_path / "prepared"
    environment, arguments_path = _launcher_environment(tmp_path, cpus=3)

    result = _run_launcher(
        [
            f"--input-dir={input_dir}",
            f"--prepared-out={output}",
            "--readers-per-node=3",
        ],
        environment,
    )

    assert result.returncode == 2
    assert "must be lower than CPUs per task" in result.stdout
    assert not arguments_path.exists()


def test_launcher_help_does_not_require_slurm():
    result = subprocess.run(
        [str(LAUNCHER), "--help"],
        cwd=REPOSITORY,
        env={"PATH": os.environ["PATH"]},
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=5,
        check=False,
    )

    assert result.returncode == 0
    assert "--input-dir PATH" in result.stdout
    assert "--prepared-out PATH" in result.stdout
