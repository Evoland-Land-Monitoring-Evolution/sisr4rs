#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales

"""
A tool to find best checkpoint for a given metric
"""


import argparse
import glob
import os

import numpy as np


def get_parser() -> argparse.ArgumentParser:
    """
    Generate argument parser for cli
    """
    parser = argparse.ArgumentParser(
        os.path.basename(__file__), description="Analyse results from several testing"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory of checkpoints",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="val_total_loss",
        help="Name of metric to get checkpoints for",
    )

    parser.add_argument("--verbose", action="store_true", help="Print more logs")

    return parser


def main():
    """
    Main method
    """
    # Parser arguments
    parser = get_parser()
    args = parser.parse_args()

    all_checkpoints = glob.glob(
        "**/" + "*" + args.metric + "*" + ".ckpt", root_dir=args.input, recursive=True
    )

    # Sort according to date
    all_checkpoints = sorted(all_checkpoints)

    if args.verbose:
        print("Available checkpoints:")
        for p in all_checkpoints:
            print(p)

    loss_values = [
        float(p.rsplit("=", maxsplit=1)[1].replace(".ckpt", ""))
        for p in all_checkpoints
    ]

    best_loss_idx = np.argmin(loss_values)

    if args.verbose:
        print(f"Best checkpoint: {loss_values[best_loss_idx]}")
    print(os.path.abspath(os.path.join(args.input, all_checkpoints[best_loss_idx])))


if __name__ == "__main__":
    main()
