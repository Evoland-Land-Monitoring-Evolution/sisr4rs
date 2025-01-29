#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""

import argparse
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from pytorch_lightning.loggers.csv_logs import CSVLogger

from torchsisr import dataset
from torchsisr.bin_utils import write_image
from torchsisr.custom_types import BatchData, NetworkInput
from torchsisr.dataset import batch_to_millirefl
from torchsisr.registration import simulate_warp, warp


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
        "--config_overrides",
        "-cfg",
        type=str,
        nargs="*",
        help="overrides of the default config",
        required=False,
        default=None,
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
        "--worldstrat",
        action="store_true",
        help="For evaluation on worldstrat dataset",
        required=False,
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for sample patches selection"
    )

    parser.add_argument("--out", type=str, help="Output directory")

    parser.add_argument("--cache", action="store_true", help="Cache testing dataset")
    return parser


def main():
    """
    Main method
    """
    logging.info("Working directory: %s", Path.cwd())

    # configure logging at the root level of Lightning
    # Define logger (from https://github.com/ashleve/lightning-hydra-template/blob
    # /a4b5299c26468e98cd264d3b57932adac491618a/src/testing_pipeline.py)
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.INFO,
        datefmt="%y-%m-%d %H:%M:%S",
        format="%(asctime)s :: %(levelname)s :: %(message)s",
    )

    # Parser arguments
    args = get_parser().parse_args()

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
        pl.seed_everything(args.seed)
        if config.get("mat_mul_precision"):
            torch.set_float32_matmul_precision(config.get("mat_mul_precision"))
        logging.info(
            "Instantiating datamodule <%s>", str(config.datamodule.data_module)
        )
        worldstrat = isinstance(
            hydra.utils.instantiate(config.datamodule.config),
            dataset.WorldStratDataModule,
        )

        datamodule: dataset.Sen2VnsDataModule | dataset.WorldStratDataModule
        if not worldstrat:
            datamodule_cfg = hydra.utils.instantiate(
                config.datamodule.config,
                testing_validation_batch_size=args.batch_size,
                num_workers=args.nb_workers,
                cache_testing_dataset=args.cache,
            )

            datamodule = dataset.Sen2VnsDataModule(datamodule_cfg)
        else:
            datamodule = dataset.WorldStratDataModule(
                config.datamodule.data_module.dataset_index_path,
                testing_validation_batch_size=args.batch_size,
                num_workers=args.nb_workers,
                min_correlation=0.2,
            )

        nb_testing_batches = len(datamodule.test_dataloader())
        print(f"{nb_testing_batches=}")

        test_logger = CSVLogger(
            save_dir=args.out,
        )
        trainer = hydra.utils.instantiate(
            config.trainer,
            logger=test_logger,
        )
        print(f"{config.training_module.training_module.config.pretrain_registration=}")
        model = hydra.utils.instantiate(config.training_module.training_module)
        trainer.test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            ckpt_path=args.checkpoint,
        )

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

        model = hydra.utils.instantiate(config.training_module.training_module)

        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))

        parameters = {
            k.split(".", maxsplit=1)[1]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("registration_module")
        }

        model.registration_module.load_state_dict(parameters)

        with torch.no_grad():
            model = model.to(device=dev)
            batch = batch_to_millirefl(batch)
            # batch = align_min_max_batch(batch)
            batch_std = model.standardize_batch(batch)
            sim_batch = model.simulate_batch(batch_std)

            target = sim_batch.network_input.hr_tensor
            target_band = target[
                :, model.config.registration.registration_channel, :, :
            ]

            target_flow = simulate_warp(
                target,
                max_range=2,
                max_width=10,
            )
            source = warp(sim_batch.network_input.hr_tensor, target_flow)
            source_band = source[
                :, model.config.registration.registration_channel, :, :
            ]

            flow = model.registration_module(source_band, target_band)

            print(f"{torch.nn.functional.smooth_l1_loss(target_flow, flow)=}")
            warped = warp(source, flow)

            out_img_dir = os.path.join(args.out, "simulated")
            Path(out_img_dir).mkdir(parents=True, exist_ok=True)
            # Write all patches
            write_image(
                flow,
                os.path.join(out_img_dir, "flow.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                target_flow,
                os.path.join(out_img_dir, "target_flow.tif"),
                nrows=8,
                ncols=4,
            )

            write_image(
                target,
                os.path.join(out_img_dir, "target.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                source,
                os.path.join(out_img_dir, "source.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                warped,
                os.path.join(out_img_dir, "warped.tif"),
                nrows=8,
                ncols=4,
            )

            target_band = batch_std.network_input.hr_tensor[
                :, model.config.registration.registration_channel, :, :
            ]

            source_band = sim_batch.network_input.hr_tensor[
                :, model.config.registration.registration_channel, :, :
            ]

            psf_kernel = torch.tensor(
                dataset.generate_psf_kernel(1.0, 1.0, 0.000001, 7),
                device=flow.device,
                dtype=flow.dtype,
            )

            flow = model.registration_module(source_band, target_band)

            # pylint: disable=not-callable
            flow = torch.nn.functional.conv2d(
                flow,
                psf_kernel[None, None, :, :].expand(flow.shape[1], -1, -1, -1),
                groups=flow.shape[1],
                padding="same",
            )

            warped_10m = warp(sim_batch.network_input.hr_tensor, flow)

            warped_20m: torch.Tensor | None = None
            if sim_batch.network_input.lr_tensor is not None:
                flow_sub = 0.5 * torch.nn.functional.interpolate(
                    flow, mode="bicubic", scale_factor=0.5, align_corners=False
                )

                warped_20m = warp(sim_batch.network_input.lr_tensor, flow_sub)

            psf_kernel = torch.tensor(
                dataset.generate_psf_kernel(1.0, 1.0, 0.00001, 7),
                device=flow.device,
                dtype=flow.dtype,
            )
            hr_scale_factor = (
                batch_std.target.shape[-1] / batch_std.network_input.hr_tensor.shape[-1]
            )

            radio_corr_10m = torch.nn.functional.interpolate(
                # pylint: disable=not-callable
                torch.nn.functional.conv2d(
                    batch_std.network_input.hr_tensor - warped_10m,
                    psf_kernel[None, None, :, :].expand(
                        warped_10m.shape[1], -1, -1, -1
                    ),
                    groups=warped_10m.shape[1],
                    padding="same",
                ),
                mode="bicubic",
                scale_factor=hr_scale_factor,
                align_corners=False,
            )
            radio_corr_20m: torch.Tensor | None = None
            if batch_std.network_input.lr_tensor is not None:
                assert warped_20m
                radio_corr_20m = torch.nn.functional.interpolate(
                    torch.nn.functional.conv2d(
                        batch_std.network_input.lr_tensor[:, :4, ...]
                        - warped_20m[:, :4, ...],
                        psf_kernel[None, None, :, :].expand(
                            radio_corr_10m.shape[1], -1, -1, -1
                        ),
                        groups=radio_corr_10m.shape[1],
                        padding="same",
                    ),
                    mode="bicubic",
                    scale_factor=2 * hr_scale_factor,
                    align_corners=False,
                )

            sim_batch = model.simulate_batch(batch_std)
            source = sim_batch.network_input.hr_tensor
            target = batch_std.network_input.hr_tensor
            radiometric_correction = radio_corr_10m
            if sim_batch.network_input.lr_tensor is not None:
                source = torch.cat(
                    (
                        sim_batch.network_input.hr_tensor,
                        torch.nn.functional.interpolate(
                            sim_batch.network_input.lr_tensor,
                            mode="bicubic",
                            scale_factor=2.0,
                            align_corners=False,
                        ),
                    ),
                    dim=1,
                )

                target = torch.cat(
                    (
                        batch_std.network_input.hr_tensor,
                        torch.nn.functional.interpolate(
                            batch_std.network_input.lr_tensor,
                            mode="bicubic",
                            scale_factor=2.0,
                            align_corners=False,
                        ),
                    ),
                    dim=1,
                )
                assert radio_corr_20m
                radiometric_correction = torch.cat(
                    (radio_corr_10m, radio_corr_20m), dim=1
                )
            flow_x2 = hr_scale_factor * torch.nn.functional.interpolate(
                flow, mode="bicubic", scale_factor=hr_scale_factor, align_corners=False
            )
            warped_reference = warp(batch_std.target, flow_x2)

            warped_corrected_reference = warped_reference + radiometric_correction

            out_img_dir = os.path.join(args.out, "real")
            Path(out_img_dir).mkdir(parents=True, exist_ok=True)
            # Write all patches
            write_image(
                flow,
                os.path.join(out_img_dir, "flow.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                target,
                os.path.join(out_img_dir, "target.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                source,
                os.path.join(out_img_dir, "source.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                warped_10m,
                os.path.join(out_img_dir, "warped.tif"),
                nrows=8,
                ncols=4,
            )
            write_image(
                batch_std.target,
                os.path.join(out_img_dir, "reference.tif"),
                nrows=8,
                ncols=4,
                res=1 / hr_scale_factor,
            )

            write_image(
                warped_reference,
                os.path.join(out_img_dir, "warped_reference.tif"),
                nrows=8,
                ncols=4,
                res=1 / hr_scale_factor,
            )

            write_image(
                radiometric_correction,
                os.path.join(out_img_dir, "reference_radiometric_correction.tif"),
                nrows=8,
                ncols=4,
                res=1 / hr_scale_factor,
            )

            write_image(
                warped_corrected_reference,
                os.path.join(out_img_dir, "warped_corrected_reference.tif"),
                nrows=8,
                ncols=4,
                res=1 / hr_scale_factor,
            )
            warped_corrected_reference2 = model.register_target(
                batch_std, radiometric_registration=True
            ).target

            write_image(
                warped_corrected_reference2,
                os.path.join(out_img_dir, "warped_corrected_reference2.tif"),
                nrows=8,
                ncols=4,
                res=1 / hr_scale_factor,
            )


if __name__ == "__main__":
    main()
