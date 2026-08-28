#!/usr/bin/env python


def main():
    from . import __version__
    print(f"Careless version {__version__}")
    from careless.parser import parser
    parser = parser.parse_args()
    run_careless(parser)


def run_careless(parser):
    dials_dataset = None
    if getattr(parser, 'input_kind', 'files') == 'dials':
        # Keep this before the scientific Python stack: DIALS loads native
        # cctbx extensions with import-order constraints.
        from careless.io.dials_ray import read_dials_dataset
        dials_dataset = read_dials_dataset(
            parser.reflection_files[0],
            expt_glob=parser.dials_expt_glob,
            max_files=parser.dials_max_files,
            workers_per_node=parser.ray_workers_per_node,
            block_mib=parser.ray_block_mib,
            save_tensors=parser.save_tensors,
            tensor_shards=parser.tensor_shards,
        )

    import numpy as np
    import torch
    import reciprocalspaceship as rs
    from careless.io.manager import DataManager
    from careless.io.formatter import MonoFormatter, LaueFormatter
    from careless.models.base import BaseModel

    np.random.seed(parser.seed)
    torch.manual_seed(parser.seed)

    if parser.type == 'poly':
        df = LaueFormatter.from_parser(parser)
    elif parser.type == 'mono':
        df = MonoFormatter.from_parser(parser)
    elif parser.type == 'devices':
        print("###############################################")
        print("# PyTorch can access the following devices    #")
        print("###############################################")
        print(f" - CPU")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f" - CUDA:{i}: {torch.cuda.get_device_name(i)}")
        from sys import exit
        exit()

    # Avoid per-call GPU reductions and host synchronizations from distribution
    # argument/support checks in the training loop.
    torch.distributions.Distribution.set_default_validate_args(False)

    # Allow float32 matrix multiplications to use TF32 tensor cores on supported
    # NVIDIA GPUs while keeping model parameters and outputs in float32.
    torch.set_float32_matmul_precision("high")

    if dials_dataset is None:
        inputs, rac = df.format_files(parser.reflection_files)
    else:
        inputs, rac = df((dials_dataset,))
    dm = DataManager(inputs, rac, parser=parser)

    if parser.test_fraction is not None:
        train, test = dm.split_data_by_refl(parser.test_fraction)
    else:
        train, test = dm.inputs, None

    model = dm.build_model()

    # Select compute device
    if not parser.disable_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{parser.gpu_id}')
    else:
        device = torch.device('cpu')
    model.to(device)

    # Initialize any LazyLinear layers before loading weights or freezing parameters
    with torch.no_grad():
        from careless.models.base import reset_losses_and_metrics
        reset_losses_and_metrics()
        _init_inputs = tuple(
            torch.as_tensor(d, dtype=torch.float32).to(device) if d.dtype in (np.float64,) else torch.as_tensor(d).to(device)
            for d in train
        )
        model(_init_inputs)

        # Apply TF v0.5.4-style identity initialization to the scaling model.
        # All nn.LazyLinear layers are now materialized after the forward pass above.
        # Weights: identity matrix in the top-left min(out, in) block, zeros elsewhere.
        # Biases: zeros.  This matches Keras kernel_initializer='identity' + default
        # bias_initializer='zeros', giving scale_raw ≈ 0 → scale ≈ 1 at init for
        # standardized metadata inputs.
        for m in model.scaling_model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.weight)
                k = min(m.weight.shape)
                m.weight.data[:k, :k] = torch.eye(k, device=m.weight.device)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    if parser.scale_file is not None:
        model.scaling_model.load_state_dict(torch.load(parser.scale_file, weights_only=True))
    if parser.freeze_scales:
        for p in model.scaling_model.parameters():
            p.requires_grad_(False)

    if parser.structure_factor_file is not None:
        model.surrogate_posterior.load_state_dict(
            torch.load(parser.structure_factor_file, weights_only=True)
        )
    if parser.freeze_structure_factors:
        for p in model.surrogate_posterior.parameters():
            p.requires_grad_(False)

    validation_frequency = parser.validation_frequency
    progress = not parser.disable_progress_bar

    history = model.train_model(
        train,
        parser.iterations,
        message="Training",
        validation_data=test,
        validation_frequency=validation_frequency,
        progress=progress,
    )

    import os
    out_dir = os.path.dirname(parser.output_base)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i, ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=train)):
        filename = parser.output_base + f'_{i}.mtz'
        ds.write_mtz(filename)

    filename = parser.output_base + '_history.csv'
    rs.DataSet(history).to_csv(filename, index_label='step')

    torch.save(
        model.surrogate_posterior.state_dict(),
        parser.output_base + '_structure_factor'
    )
    torch.save(
        model.scaling_model.state_dict(),
        parser.output_base + '_scale'
    )

    if parser.save_data_manager:
        import pickle
        with open(parser.output_base + "_data_manager.pickle", "wb") as out:
            pickle.dump(dm, out)

    if test is not None:
        for file_id, (ds_train, ds_test) in enumerate(zip(
            dm.get_predictions(model, train, test_value=0),
            dm.get_predictions(model, test, test_value=1),
        )):
            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            rs.concat((ds_train, ds_test)).write_mtz(filename)
    else:
        for file_id, ds_train in enumerate(dm.get_predictions(model, train, test_value=0)):
            filename = parser.output_base + f'_predictions_{file_id}.mtz'
            ds_train.write_mtz(filename)

    if parser.merge_half_datasets:
        scaling_model = model.scaling_model
        for p in scaling_model.parameters():
            p.requires_grad_(False)

        xval_data = [None] * len(dm.asu_collection)
        for repeat in range(parser.half_dataset_repeats):
            for half_id, half in enumerate(dm.split_data_by_image()):
                model = dm.build_model(scaling_model=scaling_model)
                model.to(device)
                history = model.train_model(
                    half,
                    parser.iterations,
                    message=f"Merging repeat {repeat + 1} half {half_id + 1}",
                    progress=progress,
                )

                for file_id, ds in enumerate(dm.get_results(model.surrogate_posterior, inputs=half)):
                    ds['repeat'] = rs.DataSeries(repeat, index=ds.index, dtype='I')
                    ds['half'] = rs.DataSeries(half_id, index=ds.index, dtype='I')
                    if xval_data[file_id] is None:
                        xval_data[file_id] = ds
                    else:
                        xval_data[file_id] = rs.concat((xval_data[file_id], ds))

        for file_id, ds in enumerate(xval_data):
            filename = parser.output_base + f'_xval_{file_id}.mtz'
            ds.write_mtz(filename)

    if parser.embed:
        from IPython import embed
        embed(colors='Linux')


if __name__ == "__main__":
    main()
