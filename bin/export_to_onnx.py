#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to export checkpoint to onnx format
"""
import argparse
import logging
import os
from pathlib import Path

import hydra
import onnx  # type: ignore
import torch
import yaml
from hydra import compose, initialize_config_dir

from torchsisr import dataset


class PreprocessingModule(torch.nn.Module):
    """
    Preprocessing module in charge of data normalization for export
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Handles standardization and conversion from millirefl
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        data = data / 10000
        data = (data - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return data


class PostProcessingModule(torch.nn.Module):
    """
    Post-processing module in charge of data de-normalization for export
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Handles unstardardization and conversion to millirefl
        """
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward
        """
        data = self.mean[None, :, None, None] + (data * self.std[None, :, None, None])
        return 10000 * data


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Export trained model to onnx"
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

    parser.add_argument("--output", type=str, help="Ouptut file path", required=False)
    return parser


def main():
    """
    Main method
    """
    # Parser arguments
    args = get_parser().parse_args()
    logging.getLogger().setLevel(logging.INFO)
    checkpoint = torch.load(
        args.checkpoint, map_location=torch.device("cpu"), weights_only=True
    )
    for k in checkpoint["state_dict"].keys():
        model_checkpoint = {
            k.split(".", maxsplit=2)[2]: v
            for k, v in checkpoint["state_dict"].items()
            if not (
                k.startswith("discriminator")
                or k.startswith("mean")
                or k.startswith("std")
                or k.startswith("wald")
                or k.startswith("noise")
                or k.startswith("loss")
            )
        }

    export_file = args.output
    if export_file is None:
        export_file = args.checkpoint[:-5] + ".onnx"

    # We instantiate the checkpoint configuration
    with initialize_config_dir(
        version_base=None,
        config_dir=os.path.join(Path(__file__).parent.resolve(), "../hydra/"),
    ):
        config = compose(
            config_name="main.yaml",
            overrides=args.config_overrides,
        )
        srnet = hydra.utils.instantiate(config.model.generator)
        srnet.load_state_dict(model_checkpoint, strict=False)

        mean = torch.tensor(config.training_module.standardization_parameters.mean)
        std = torch.tensor(config.training_module.standardization_parameters.std)
        preprocessing = PreprocessingModule(
            mean,
            std,
        )

        postprocessing = PostProcessingModule(mean, std)
        logging.info("Mean: %s", config.training_module.standardization_parameters.mean)
        logging.info("Std: %s", config.training_module.standardization_parameters.std)
        logging.info("Margin: %s", srnet.get_prediction_margin())
        net = torch.nn.Sequential(preprocessing, srnet, postprocessing)

        fake_input = torch.randint(
            0,
            2000,
            (1, len(config.training_module.standardization_parameters.mean), 32, 32),
            dtype=torch.float32,
        )

        input_names = ["input"]
        output_names = ["output"]
        logging.info("Exporting model ...")
        torch.onnx.export(
            net,
            (fake_input,),
            export_file,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=False,
            dynamic_axes={"input": [0, 2, 3], "output": [0, 2, 3]},
        )

        # # Now export to onnx
        # export_output = torch.onnx.dynamo_export(net, fake_input)

        # # Save the onnx file
        # export_output.save(args.ouptut)

        logging.info("Export done, loading back")
        # Load it back
        onnx_model = onnx.load(export_file)

        # Sanity check
        logging.info("Checking exported model ...")
        onnx.checker.check_model(onnx_model)

        logging.info("Writing yaml metadata ...")

        yaml_file = export_file[:-5] + ".yaml"

    bands: list[str] = []

    datamodule = hydra.utils.instantiate(config.datamodule.data_module)
    print(datamodule)
    if isinstance(
        datamodule,
        dataset.WorldStratDataModule,
    ):
        bands = ["B2", "B3", "B4", "B8"]
    else:
        if isinstance(datamodule, dataset.Sen2VnsDataModule):
            if config.datamodule.single_site_config.load_10m_data:
                bands.extend(["B2", "B3", "B4", "B8"])
            if config.datamodule.single_site_config.load_20m_data:
                bands.extend(["B5", "B6", "B7", "B8A"])
            if config.datamodule.single_site_config.load_b11b12:
                bands.extend(["B11", "B12"])

    with open(yaml_file, "w") as f:
        yaml.dump(
            {
                "model": os.path.basename(export_file),
                "bands": bands,
                "margin": srnet.get_prediction_margin(),
                "factor": config.model.generator.upsampling_factor,
            },
            f,
        )


if __name__ == "__main__":
    main()
