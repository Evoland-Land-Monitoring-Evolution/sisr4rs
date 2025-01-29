#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from torchsisr import double_sisr_model, hydra_utils, training


@hydra.main(version_base=None, config_path="../hydra/", config_name="main.yaml")
def main(config: DictConfig):
    """
    Hydra main function
    """
    # Call extras
    hydra_utils.extras(config)

    # Change current working directory to checkpoints dir, to enable
    # auto restart checkpointing
    checkpoints_dir = os.path.join(config.original_work_dir, "checkpoints", config.name)
    Path(checkpoints_dir).mkdir(parents=True, exist_ok=True)
    os.chdir(checkpoints_dir)

    logging.info("Working directory: %s", Path.cwd())

    if config.get("loglevel"):
        # Configure logging
        numeric_level = getattr(logging, config.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {config.loglevel}")

        logging.basicConfig(
            level=numeric_level,
            datefmt="%y-%m-%d %H:%M:%S",
            format="%(asctime)s :: %(levelname)s :: %(message)s",
        )

    # configure logging at the root level of Lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    # Apply seed if needed
    if config.get("seed"):
        pl.seed_everything(config.get("seed"), workers=True)

    if config.get("mat_mul_precision"):
        torch.set_float32_matmul_precision(config.get("mat_mul_precision"))

    # load samples
    datamodule = hydra.utils.instantiate(config.datamodule.data_module)

    training_module_config = hydra.utils.instantiate(config.training_module.config)

    if config.start_from_checkpoint is not None:
        logging.info("Restoring generator state from %s", config.start_from_checkpoint)
        checkpoint = torch.load(
            config.start_from_checkpoint, map_location=torch.device("cpu")
        )
        for k in checkpoint["state_dict"].keys():
            model_checkpoint = {
                k.split(".", maxsplit=2)[2]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("generator")
            }
        srnet = hydra.utils.instantiate(config.model.generator)
        srnet.load_state_dict(model_checkpoint)
        model = double_sisr_model.DoubleSuperResolutionModel(sisr_model=srnet)
        training_module_config.model = model

    training_module = training.DoubleSISRTrainingModule(training_module_config)

    if config.load_registration_checkpoint is not None:
        logging.info(
            "Restoring registration_module from %s", config.load_registration_checkpoint
        )
        checkpoint = torch.load(
            config.load_registration_checkpoint, map_location=torch.device("cpu")
        )
        registration_module_parameters = {
            k.split(".", maxsplit=1)[1]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("registration_module")
        }
        if registration_module_parameters:
            training_module.registration_module.load_state_dict(
                registration_module_parameters
            )

    # Define callbacks
    # (from https://github.com/ashleve/lightning-hydra-template/blob/
    # a4b5299c26468e98cd264d3b57932adac491618a/src/training_pipeline.py#L50)
    callbacks: list[pl.Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                # pylint: disable=protected-access
                logging.info("Instantiating callback <%s>", cb_conf._target_)
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    loggers: list[pl.loggers.logger.Logger] = []
    if "loggers" in config:
        for _, lg_conf in config.loggers.items():
            if "_target_" in lg_conf:
                # pylint: disable=protected-access
                logging.info("Instantiating logger <%s>", lg_conf._target_)
                loggers.append(hydra.utils.instantiate(lg_conf))

    nb_training_batches = len(datamodule.train_dataloader())
    nb_validation_batches = len(datamodule.val_dataloader())

    logging.info(
        "nb_training_batches=%s, nb_validation_batches=%s",
        str(nb_training_batches),
        str(nb_validation_batches),
    )

    if config.get("track_carbon"):
        # If carbon emission tracker is activated, the model is trained for 1 epoch
        config.trainer.max_epochs = 1

    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    if config.resume_from_checkpoint is None:
        # Fit the network
        trainer.fit(training_module, datamodule)
    else:
        trainer.fit(
            training_module, datamodule, ckpt_path=config.resume_from_checkpoint
        )


if __name__ == "__main__":
    # We disable this pylint error since value for parameter config is provided
    # by hydra decorator
    # pylint: disable=no-value-for-parameter
    main()
