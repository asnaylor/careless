name = "DIALS / Ray input"
description = (
    "Options used only when the single reflection input is a directory of "
    "same-stem DIALS .expt/.refl pairs. These options are available for "
    "monochromatic merging only."
)

args_and_kwargs = (
    (("--dials-expt-glob",), {
        "default": "**/*.expt", "metavar": "PATTERN",
        "help": "Glob relative to the input directory used to discover .expt files. "
                "Each corresponding .refl file must have the same stem. Default: **/*.expt.",
    }),
    (("--dials-max-files",), {
        "type": int, "metavar": "N",
        "help": "Use only the first N sorted, complete DIALS pairs.",
    }),
    (("--ray-workers-per-node",), {
        "type": int, "default": 4, "metavar": "N",
        "help": "Number of stateful DIALS reader actors per Ray node. Default: 4.",
    }),
    (("--ray-block-mib",), {
        "type": int, "default": 256, "metavar": "MIB",
        "help": "Maximum converted block size transferred from each Ray actor. Default: 256 MiB.",
    }),
    (("--save-tensors",), {
        "metavar": "PATH",
        "help": "Atomically save the complete 22-column pre-formatter dataset as "
                "SafeTensors shards for reuse by a later Careless run.",
    }),
    (("--tensor-shards",), {
        "type": int, "metavar": "N",
        "help": "Number of SafeTensors shards to save. Default: the number of "
                "DIALS reader actors, never the number of input pairs.",
    }),
)
