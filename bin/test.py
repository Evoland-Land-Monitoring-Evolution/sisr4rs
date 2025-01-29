#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""

import argparse
import glob
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from pytorch_lightning.loggers.csv_logs import CSVLogger

from torchsisr import dataset, double_sisr_model, training
from torchsisr.bin_utils import write_image
from torchsisr.custom_types import BatchData, NetworkInput
from torchsisr.patches import crop


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Use CARN network to for test step"
    )

    parser.add_argument(
        "--checkpoint", "-cp", type=str, help="Path to model checkpoint", required=True
    )
    parser.add_argument(
        "--testing_sites",
        nargs="+",
        required=False,
        default=None,
        help="Override the testing sites",
    )

    parser.add_argument(
        "--config_overrides",
        "-cfg",
        type=str,
        nargs="*",
        help="overrides of the default config",
        required=False,
        default=None,
    )

    parser.add_argument("--disable_testing", action="store_true")
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        help="Optional path to metrics config, in case we want more metrics in testing",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--bicubic", action="store_true", help="Evaluate bicubic default model"
    )

    parser.add_argument(
        "--load_best_checkpoint",
        type=str,
        required=False,
        default=None,
        help="Load best checkpoint according to given metric",
    )
    parser.add_argument(
        "--nb_workers",
        required=False,
        default=4,
        type=int,
        help="Override number of workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size to use for inference",
        default=64,
        required=False,
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sample patches selection"
    )

    parser.add_argument(
        "--crop", type=int, default=32, help="Crop for images generation"
    )

    parser.add_argument("--cache", action="store_true", help="Cache testing dataset")
    return parser


def main():
    """
    Main method
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Working directory: %s", Path.cwd())

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    # Parser arguments
    args = get_parser().parse_args()
    pl.seed_everything(args.seed)
    # Find on which device to run
    dev = "cpu"
    if torch.cuda.is_available():
        dev = "cuda"
    logging.info("Processing will happen on device %s", dev)

    # We instantiate the checkpoint configuration
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(Path(__file__).parent.resolve(), "../hydra/"),
    ):
        config = compose(
            config_name="main.yaml",
            overrides=args.config_overrides,
        )
        if config.get("mat_mul_precision"):
            torch.set_float32_matmul_precision(config.get("mat_mul_precision"))
        logging.info(
            "Instantiating datamodule <%s>", str(config.datamodule.data_module)
        )
        datamodule: dataset.WorldStratDataModule | dataset.Sen2VnsDataModule
        worldstrat = isinstance(
            hydra.utils.instantiate(config.datamodule.config),
            dataset.WorldStratDataModule,
        )
        scale_1: float = 2.0
        scale_2: float | None = None
        if not worldstrat:
            if args.testing_sites is not None:
                datamodule_cfg = hydra.utils.instantiate(
                    config.datamodule.config,
                    testing_validation_batch_size=args.batch_size,
                    num_workers=args.nb_workers,
                    cache_testing_dataset=args.cache,
                    testing_sites=args.testing_sites,
                )
            else:
                datamodule_cfg = hydra.utils.instantiate(
                    config.datamodule.config,
                    testing_validation_batch_size=args.batch_size,
                    num_workers=args.nb_workers,
                    cache_testing_dataset=args.cache,
                )
            if datamodule_cfg.single_site_config.load_10m_data:
                scale_1 = 2.0
                scale_2 = None
                if datamodule_cfg.single_site_config.load_20m_data:
                    scale_2 = 2.0
            else:
                scale_1 = 4.0

            datamodule = dataset.Sen2VnsDataModule(datamodule_cfg)
        else:
            scale_1 = config.model.generator.upsampling_factor
            scale_2 = None
            datamodule = dataset.WorldStratDataModule(
                config.datamodule.data_module.dataset_index_path,
                testing_validation_batch_size=args.batch_size,
                num_workers=args.nb_workers,
                min_correlation=0.2,
            )

        name = config.name
        version = config.label

        additional_metrics: torch.nn.ModuleList | None = None
        if args.metrics is not None:
            additional_metrics = torch.nn.ModuleList(
                hydra.utils.instantiate(OmegaConf.load(args.metrics))
            )

        nb_testing_batches = len(datamodule.test_dataloader())
        logging.info(f"{nb_testing_batches=}")

        bicubic_model = double_sisr_model.DoubleSuperResolutionModel(
            sisr_model=double_sisr_model.BicubicInterpolation(
                upsampling_factor=scale_1
            ),
            lr_to_hr_model=(
                None
                if scale_2 is None
                else double_sisr_model.BicubicInterpolation(upsampling_factor=scale_2)
            ),
        )

        if Path(args.checkpoint).is_file():
            checkpoints = [args.checkpoint]
        else:
            checkpoints = glob.glob(str(Path(args.checkpoint, "*_*.ckpt")))

        patch_examples_dataset = datamodule.testing_dataset
        patch_examples_dataloader = torch.utils.data.DataLoader(
            patch_examples_dataset,
            batch_size=32,
            drop_last=True,
            shuffle=True,
            collate_fn=dataset.batch_data_collate_fn,
            num_workers=args.nb_workers,
            prefetch_factor=1,
            pin_memory=True,
        )
        batch = next(iter(patch_examples_dataloader))

        batch = BatchData(
            NetworkInput(
                batch.network_input.hr_tensor.to(device=dev),
                batch.network_input.hr_bands,
                (
                    batch.network_input.lr_tensor.to(device=dev)
                    if batch.network_input.lr_tensor is not None
                    else None
                ),
                batch.network_input.lr_bands,
            ),
            target=batch.target.to(device=dev),
            target_bands=batch.target_bands,
        )

        if not args.bicubic:
            model = hydra.utils.instantiate(config.training_module.training_module)

            if additional_metrics is not None:
                model.test_metrics.extend(additional_metrics)

            for checkpoint in checkpoints:
                step = torch.load(checkpoint, weights_only=True)["global_step"]
                # Init test logger
                save_dir = os.path.join(config.original_work_dir, "test_logs")
                version_string = f"{version}_step_{step}"
                logging.info(
                    f"Results will be stored in {os.path.join(save_dir, name, version_string)}"
                )
                test_logger = CSVLogger(
                    save_dir=save_dir,
                    name=name,
                    version=version_string,
                )
                if not args.disable_testing:
                    trainer = hydra.utils.instantiate(
                        config.trainer,
                        logger=test_logger,
                        accelerator=dev,
                        precision=32,
                    )
                else:
                    trainer = hydra.utils.instantiate(
                        config.trainer,
                        logger=test_logger,
                        accelerator=dev,
                        limit_test_batches=1,
                    )
                # Log hyperparameters
                trainer.logger.log_hyperparams({"ckpt_path": checkpoint})

                model.config.testval_geometric_registration = False
                model.config.testval_radiometric_registration = False

                trainer.test(
                    model=model,
                    dataloaders=datamodule.test_dataloader(),
                    ckpt_path=checkpoint,
                )
                logging.info("Logging sample images")

                with torch.no_grad():
                    model = model.to(device=dev)
                    bicubic_model = bicubic_model.to(device=dev)

                    # Perform inference
                    prediction = model.predict(batch)
                    # batch_std = model.standardize_batch(batch)
                    # batch_std = model.register_target(
                    #     batch_std, radiometric_registration=False
                    # )
                    # batch.target = patches.unstandardize(
                    #     batch_std.target,
                    #     model.mean[:8],
                    #     model.std[:8],
                    # )
                    prediction_margin = args.crop

                    predicted = (10000 * prediction.prediction).cpu()
                    predicted_sim = (
                        10000 * model.predict(batch, simulate=True).prediction
                    ).cpu()
                    predicted_bicubic = (
                        10000
                        * bicubic_model(
                            dataset.batch_to_millirefl(
                                batch, dtype=torch.float32
                            ).network_input
                        ).prediction.cpu()
                    )
                    predicted_sim_bicubic = (
                        10000
                        * bicubic_model(
                            dataset.simulate_batch(
                                dataset.batch_to_millirefl(batch, dtype=torch.float32),
                                mtf=model.config.batch_simulation.mtf_min,
                                noise_std=model.noise_std,
                            ).network_input
                        ).prediction.cpu()
                    )
                    target_cropped = crop(batch.target.cpu(), prediction_margin)
                    predicted = crop(predicted, prediction_margin)
                    predicted_sim = crop(predicted_sim, prediction_margin)
                    predicted_bicubic = crop(predicted_bicubic, prediction_margin)
                    predicted_sim_bicubic = crop(
                        predicted_sim_bicubic, prediction_margin
                    )

                    out_img_dir = os.path.join(test_logger.log_dir, "images")
                    Path(out_img_dir).mkdir(parents=True, exist_ok=True)
                    # Write all patches
                    write_image(
                        target_cropped,
                        os.path.join(out_img_dir, "target.tif"),
                        nrows=8,
                        ncols=4,
                    )
                    write_image(
                        predicted,
                        os.path.join(out_img_dir, "pred_real.tif"),
                        nrows=8,
                        ncols=4,
                    )
                    write_image(
                        predicted_sim,
                        os.path.join(out_img_dir, "pred_sim.tif"),
                        nrows=8,
                        ncols=4,
                    )
                    write_image(
                        predicted_bicubic,
                        os.path.join(out_img_dir, "input_real.tif"),
                        nrows=8,
                        ncols=4,
                    )
                    write_image(
                        predicted_sim_bicubic,
                        os.path.join(out_img_dir, "input_sim.tif"),
                        nrows=8,
                        ncols=4,
                    )

        else:
            logging.info("Evaluating bicubic reference model")
            save_dir = os.path.join(config.original_work_dir, "test_logs")
            name = "bicubic"
            version_string = "reference"
            logging.info(
                f"Results will be stored in {os.path.join(save_dir, name, version_string)}"
            )
            test_logger = CSVLogger(
                save_dir=save_dir,
                name=name,
                version=version_string,
            )
            if not args.disable_testing:
                trainer = hydra.utils.instantiate(
                    config.trainer, logger=test_logger, accelerator=dev
                )
            else:
                trainer = hydra.utils.instantiate(
                    config.trainer,
                    logger=test_logger,
                    accelerator=dev,
                    limit_test_batches=1,
                )

            training_module_config = hydra.utils.instantiate(
                config.training_module.config, model=bicubic_model
            )
            model = training.DoubleSISRTrainingModule(training_module_config)

            if additional_metrics is not None:
                model.test_metrics.extend(additional_metrics)

            trainer.test(model=model, dataloaders=datamodule.test_dataloader())


if __name__ == "__main__":
    main()
