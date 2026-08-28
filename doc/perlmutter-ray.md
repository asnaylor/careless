# Ray DIALS input on Perlmutter

Careless can read same-stem DIALS `.expt`/`.refl` pairs directly for a
monochromatic merge. Ray distributes discovery and conversion across the
allocated nodes, then the head process runs the ordinary Careless formatter,
training, and output stages. Saving a tensor cache is optional.

## Before a full run

Start with `--dials-max-files 20` and a small `--iterations` value. Remove
`--dials-max-files` for the full dataset; omitting it means every complete pair
matching `--dials-expt-glob` is processed. The input directory is treated as a
single logical dataset and must be the only reflection input. A matching
`.expt` or `.refl` without its same-stem partner is reported as an error before
conversion begins.

The cache destination supplied to `--save-tensors` must not exist. This avoids
silently mixing or overwriting datasets. Removing `--save-tensors` and
`--tensor-shards` keeps the converted dataset in memory and writes no cache.

## Run mounted source without installing it

When the repository is mounted at `/workdir/careless`, put it first on
`PYTHONPATH` inside the container:

```bash
export PYTHONPATH="/workdir/careless${PYTHONPATH:+:$PYTHONPATH}"
/usr/bin/python -c 'import careless; print(careless.__file__)'
```

This skips an editable `pip install` and avoids concurrent installation from
each Slurm task. It does not replace the container's dependencies; it only
makes Python import the mounted Careless source before an installed Careless
package. The printed path should begin with `/workdir/careless`.

## Two-node Slurm launch

Use one Slurm task per node. The following is the essential pattern from the
tested CPU-node run; substitute the account, image, and host paths.

```bash
#!/bin/bash
#SBATCH --account=<account>
#SBATCH --constraint=cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

set -euo pipefail

IMAGE=ghcr.io/asnaylor/careless:pytorch-ray
CARELESS_DIR=/path/to/careless
INPUT_DIR=/path/to/dials
OUTPUT_DIR="$SCRATCH/careless-ray-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

HEAD_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
HEAD_IP="$(srun -N 1 -n 1 -w "$HEAD_NODE" hostname -I | awk '{print $1}')"
RAY_PORT=$((20000 + SLURM_JOB_ID % 20000))

export RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"
export RAY_NODE_COUNT="$SLURM_NNODES"
export RAY_CPUS_PER_NODE="$SLURM_CPUS_PER_TASK"

srun \
  --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  podman-hpc run --rm \
    --network=host \
    --group-add keep-groups \
    --shm-size=64g \
    -v "$INPUT_DIR":/input:ro \
    -v "$OUTPUT_DIR":/output \
    -v "$CARELESS_DIR":/workdir/careless:ro \
    -e RAY_ADDRESS \
    -e RAY_NODE_COUNT \
    -e RAY_CPUS_PER_NODE \
    "$IMAGE" \
    /bin/bash -lc '
      export PYTHONPATH="/workdir/careless${PYTHONPATH:+:$PYTHONPATH}"
      exec ray symmetric-run \
        --address "$RAY_ADDRESS" \
        --min-nodes "$RAY_NODE_COUNT" \
        --num-cpus "$RAY_CPUS_PER_NODE" \
        --num-gpus 0 \
        -- \
        /usr/bin/python -m careless.careless mono \
          --disable-gpu \
          --iterations 1 \
          --dials-expt-glob "**/*.expt" \
          --ray-workers-per-node 2 \
          --ray-block-mib 64 \
          --save-tensors /output/run_cache \
          --tensor-shards 8 \
          dHKL,cartesian_fixed_x,cartesian_fixed_y,cartesian_fixed_z,ewald_offset \
          /input /output/run
    '
```

Every node must see the input, output, and mounted source at the same container
paths. `ray symmetric-run` is entered by every Slurm task, but executes the
Careless command only on the Ray head. `--ray-workers-per-node 2` creates two
stateful reader actors on each live Ray node. `--ray-block-mib` bounds each
block transferred from an actor; it is not a limit on the final in-memory
dataset.

The tested multi-node path used CPU nodes, `--network=host`, and
`--disable-gpu`. A useful workflow is to create the cache in that job and then
run training from the cache in a later GPU job.

The same script can be launched inside an interactive Slurm allocation. Slurm
environment variables such as `SLURM_JOB_NODELIST` and `SLURM_NNODES` are
available there, so running `./test_dials_ray_slurm.sh` uses the allocated
nodes. Do not start it on a login node without an allocation.

## Check the result

A completed cache contains `manifest.json`, `COMPLETE`, and exactly the number
of `.safetensors` files requested by `--tensor-shards` (or the reader-actor
count when that option is omitted):

```bash
test -f "$OUTPUT_DIR/run_cache/COMPLETE"
find "$OUTPUT_DIR/run_cache" -name '*.safetensors' -type f | wc -l
ls -lh "$OUTPUT_DIR"
```

The same command also runs Careless training and writes its normal outputs,
including merged and prediction MTZ files, history, scale, and structure-factor
artifacts. The cache does not write MTZ files itself.

To skip DIALS and Ray ingestion on a later run, use the completed cache as the
single reflection input:

```bash
/usr/bin/python -m careless.careless mono \
  --iterations 100 \
  dHKL,cartesian_fixed_x,cartesian_fixed_y,cartesian_fixed_z,ewald_offset \
  /output/run_cache /output/run_from_cache
```

Do not pass `--save-tensors`, `--tensor-shards`, or DIALS/Ray input options when
loading a completed cache. If a direct-input run reports `refusing to overwrite
prepared dataset`, choose a new cache path or intentionally remove the old
cache before starting the job.
