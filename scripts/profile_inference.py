#!/usr/bin/env python

# Copyright: (c) 2021 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to generate samples for training of super-resolution algorithms
"""

import time

import numpy as np
import torch
from tqdm import tqdm

from torchsisr.double_carn import SuperResolutionModel


def generate_fake_data(width_20m: int = 256, batch_size: int = 32):
    data_20m = torch.rand((32, 4, width_20m, width_20m)).to(
        dtype=torch.float32, device="cpu"
    )
    data_10m = torch.rand((32, 4, 2 * width_20m, 2 * width_20m)).to(
        dtype=torch.float32, device="cpu"
    )

    return {"source_std_10": data_10m, "source_std_20": data_20m}


def main():
    # Find on which device to run
    cpu_dev = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.init()
        gpu_dev = torch.device("cuda")

    # Simu param
    nb_tries = 20
    width = 128
    batch_size = 32

    # model = SuperResolutionModel(nb_bands=8,
    #                              nb_bands_20=4,
    #                              groups=2,
    #                              shared_weights=True,
    #                              nb_features_per_factor=64,
    #                              nb_features_per_factor_20=32,
    #                              upsampling_factor=2.,
    #                              nb_blocks=3,
    #                              nb_blocks_20=2,
    #                              kernel_size=3,
    #                              double_carn=True,
    #                              bicubic_upsampling_20m=True)

    model = SuperResolutionModel(
        nb_bands=8,
        nb_bands_20=4,
        groups=2,
        shared_weights=False,
        nb_features_per_factor=32,
        nb_features_per_factor_20=8,
        upsampling_factor=2.0,
        nb_blocks=2,
        nb_blocks_20=2,
        kernel_size=3,
        double_carn=True,
        bicubic_upsampling_20m=True,
    )

    model = model.eval()
    model = model.to(device=gpu_dev)

    nb_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Number of parameters: {nb_params}")

    measured_times = []
    for _ in tqdm(range(nb_tries), total=nb_tries, desc="Profiling GPU inference time"):
        fake_data = generate_fake_data(width, batch_size)
        start = time.perf_counter()
        with torch.no_grad():
            fake_data["source_std_10"] = fake_data["source_std_10"].to(device=gpu_dev)
            fake_data["source_std_20"] = fake_data["source_std_20"].to(device=gpu_dev)
            resp = model(fake_data)
            resp = resp.cpu()
            torch.cuda.synchronize()
        stop = time.perf_counter()
        measured_times.append(stop - start)

        mean_gpu_time = np.mean(measured_times)
        std_gpu_time = np.std(measured_times)
    print(
        f"GPU time {batch_size}x8x{width}x{width}: {mean_gpu_time:.6f}+/-{std_gpu_time:.6f}s"
    )
    model = model.to(device=cpu_dev)

    measured_times = []
    for _ in tqdm(range(nb_tries), total=nb_tries, desc="Profiling CPU inference time"):
        fake_data = generate_fake_data(width, batch_size)
        start = time.perf_counter()
        with torch.no_grad():
            model(fake_data)
        stop = time.perf_counter()
        measured_times.append(stop - start)
        mean_cpu_time = np.mean(measured_times)
        std_cpu_time = np.mean(measured_times)
    print(
        f"CPU time {batch_size}x8x{width}x{width}: {mean_cpu_time:.6f}+/-{std_cpu_time:.6f}s"
    )


if __name__ == "__main__":
    main()
