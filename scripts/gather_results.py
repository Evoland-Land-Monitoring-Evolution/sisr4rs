#!/usr/bin/env python

# Copyright: (c) 2023 CESBIO / Centre National d'Etudes Spatiales
"""
A tool to gather several testing results and make some plots
"""
import argparse
import glob
import os
from itertools import compress
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tabulate import tabulate


def list_results(path: str) -> dict[str, str]:
    """
    List results in a dictionary
    """
    out: dict[str, str] = {}

    csv_list = glob.glob(os.path.join(path, "*/*/metrics.csv"))

    for csv in csv_list:
        print(csv)
        run_id = csv.split("/")[-2]
        method = csv.split("/")[-3]

        print(run_id, method)
        out[method + "/" + run_id] = csv
    return out


def merge_csvs_dict(csvs_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge csvs dict into a single dataframe
    """
    csvs_dict = {k: pd.read_csv(v) for k, v in csvs_dict.items()}

    # Will store transformed dict
    models_dict: dict[str, pd.DataFrame] = {}

    for model_name, model_df in csvs_dict.items():
        if "step" in model_df.columns:
            model_df = model_df.drop("step", axis=1)
        models_dict[model_name] = model_df.iloc[-1:]

    final_df = pd.concat([m for m in models_dict.values()]).copy()
    final_df["method"] = [k for k in models_dict.keys()]
    final_df = final_df.set_index("method", drop=False).sort_index()
    return final_df


DEFAULT_BANDS = ["B2", "B3", "B4", "B8"]


def plot_perf(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    out_path: str,
    suffix: str = "",
    bands=DEFAULT_BANDS,
):
    """
    Plot performances
    """
    df = df.sort_index(ascending=True)
    # df = df.sort_values(by=metric + "_" + bands[0], ascending=False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    xpos = np.arange(2.5, 2.5 * (len(bands) + 1), 2.5)
    width = 1.8 / len(df)
    shifts = 0.5 * width + np.arange(-0.9, 0.9, width)
    metrics = [metric + "_" + b for b in bands]
    metrics_mask = [m in df.columns.values for m in metrics]

    metrics = list(compress(metrics, metrics_mask))
    bands = list(compress(bands, metrics_mask))
    xpos = list(compress(xpos, metrics_mask))

    for method, shift in zip(df.method, shifts):
        try:
            ax.bar(
                xpos + shift,
                height=df.loc[method][metrics].values,
                width=width,
                label=method,
            )
            ax.set_ylabel(metric_label)
        except KeyError as e:
            print(e)
        ax.grid(True)

        ax.set_xticks(xpos, bands)
        ax.set_xlabel("Sentinel-2 spectral band")
        ax.legend(bbox_to_anchor=(0.25, -0.6), loc="lower left")
    out_pdf = os.path.join(out_path, f"perf_{metric.replace('/','_')}{suffix}.pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf


def print_table(
    df: pd.DataFrame,
    metric: str,
    metric_name: str | None = None,
    bands: list[str] = DEFAULT_BANDS,
    floatfmt: str = ".2f",
):
    """
    Print results table
    """
    if metric_name is None:
        metric_name = metric
    metrics = [metric + "_" + b for b in bands]
    metric_labels = ["*" + b + "*" for b in bands]
    df = df.filter(items=metrics, axis=1)
    df.index = df.index.str.split("/", expand=True).get_level_values(0)
    df.index = df.index.str.replace("_prod", "")
    df.index = "~" + df.index + "~"
    table = tabulate(
        df,
        tablefmt="orgtbl",
        headers=["*method*"] + metric_labels,
        floatfmt=floatfmt,
    )
    print(table)


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
        help="Directory of testing logs",
    )

    parser.add_argument(
        "--output", type=str, help="Where to store the results", required=True
    )
    return parser


def main():
    """
    Main method
    """
    # Parser arguments
    args = get_parser().parse_args()

    # Ensure output dir exists
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Gather all results
    results_csvs = list_results(args.input)

    print(results_csvs)
    # Merge all dataframes
    results_df = merge_csvs_dict(results_csvs)

    # Store csvs to disk
    Path(os.path.join(args.output, "data")).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(os.path.join(args.output, "data", "real_metrics.csv"), sep="\t")
    # Generate figures
    Path(os.path.join(args.output, "figures")).mkdir(parents=True, exist_ok=True)
    for metric, label in zip(
        [
            "hr_rmse",
            "psnr",
            "lr_rmse",
            "high_grad_strata_rmse",
            "ssim",
            "fft_hf_power_variation_bicubic",
            "fft_hf_power_variation_target",
            "brisque",
            "brisque_variation_bicubic",
            "brisque_variation_target",
            "tv_variation_bicubic",
            "tv_variation_target",
        ],
        [
            "High Spatial Frequencies RMSE wrt. Venµs",
            "PSNR wrt. Venµs",
            "RMSE wrt. Sentinel-2",
            "RMSE of high gradient pixels wrt. Venµs",
            "Structural Similarity Image Metric",
            "FFT HF power variation wrt. bicubic",
            "FFT HF power variation wrt. reference",
            "BRISQUE",
            "BRISQUE variation wrt. bicubic",
            "BRISQUE variation wrt. target",
            "TV variation wrt. bicubic",
            "TV variation wrt. target",
        ],
    ):
        plot_perf(
            results_df,
            metric="test_real_metrics_per_band/" + metric,
            metric_label=label,
            suffix="_real",
            bands=["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"],
            out_path=os.path.join(args.output, "figures"),
        )
        plot_perf(
            results_df,
            metric="test_sim_metrics_per_band/" + metric,
            metric_label=label,
            suffix="_sim",
            bands=["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A"],
            out_path=os.path.join(args.output, "figures"),
        )

        print("test_real_metrics_per_band/" + metric)
        print_table(
            results_df,
            metric="test_real_metrics_per_band/" + metric,
            metric_name="",
            bands=["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"],
        )

        print("test_sim_metrics_per_band/" + metric)
        print_table(
            results_df,
            metric="test_sim_metrics_per_band/" + metric,
            metric_name="",
            bands=["B2", "B3", "B4", "B8", "B5", "B6", "B7", "B8A", "B11", "B12"],
        )


if __name__ == "__main__":
    main()
