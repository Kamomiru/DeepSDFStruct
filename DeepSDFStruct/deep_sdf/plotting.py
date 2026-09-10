"""
Training Visualization and Logging
==================================

This module provides utilities for visualizing DeepSDF training progress
and generating plots of training metrics.

Functions
---------

plot_logs
    Plot training loss curves with smoothing and optional learning rate overlay.
    Useful for monitoring training progress and diagnosing convergence issues.

plot_reconstruction_loss
    Visualize loss during shape reconstruction, showing how well the model
    fits to target geometries.

extract_paths
    Helper for extracting file paths from nested data structures.

running_mean
    Compute running average for smoothing noisy loss curves.

The module integrates with the workspace utilities to load training logs
and generate publication-quality plots for analysis and presentation.
"""

import numpy as np
import os
import logging
import torch
import matplotlib.pyplot as plt

import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.SDF import SDFfromDeepSDF
from DeepSDFStruct.deep_sdf.workspace import load_trained_model

logger = logging.getLogger(__name__)


def extract_paths(data, current_path=""):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{current_path}/{key}" if current_path else key
            paths.extend(extract_paths(value, new_path))

    elif isinstance(data, list):
        for item in data:
            paths.extend(extract_paths(item, current_path))

    else:
        paths.append(f"{current_path}/{data}")

    return paths


def running_mean(x, N):

    if isinstance(x, list):
        x = [
            item.detach().cpu().item() if torch.is_tensor(item) else item
            for item in x
        ]

    elif torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    x = np.asarray(x)

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_logs(experiment_directory, show_lr=False, ax=None, filename=None, GAN = False, snapshot_epochs = []):

    logs = torch.load(os.path.join(experiment_directory, ws.logs_filename))

    if GAN == False:

        num_iters = len(logs["loss"])
        iters_per_epoch = num_iters / logs["epoch"]

        smoothed_loss_41 = running_mean(logs["loss"], 41)

        show_plt = False

        if show_lr:
            if ax is None:
                fig, ax = plt.subplots(2, 1)
                fig.tight_layout()
                show_plt = True
        else:
            if ax is None:
                fig, ax = plt.subplots()
                show_plt = True
            ax = [ax]

        ax[0].plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss"],
            "#82c6eb",
            np.arange(20, num_iters - 20) / iters_per_epoch,
            smoothed_loss_41,
            "#2a9edd",
        )
        ax[0].set_yscale("log")

        ax[0].set(xlabel="Epoch", ylabel="Loss")
        ax[0].legend(["Loss", "Loss (Running Mean)", "Loss (Running Mean 41)"])

        if show_lr:
            combined_lrs = np.array(logs["learning_rate"])
            ax[1].plot(
                np.arange(combined_lrs.shape[0]),
                combined_lrs[:, 0],
                np.arange(combined_lrs.shape[0]),
                combined_lrs[:, 1],
            )
            ax[1].set(xlabel="Epoch", ylabel="Learning Rate")
            ax[1].legend(["Decoder", "Latent Vector"])

        for axis in ax:
            axis.grid()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        elif show_plt:
            plt.show()

    if GAN:

        num_iters = len(logs["loss_D"])
        iters_per_epoch = num_iters / logs["epoch"]

        smoothed_loss_D = running_mean(logs["loss_D"], 41)
        smoothed_loss_G = running_mean(logs["loss_G"], 41)

        show_plt = False

        if ax is None:
            if "loss_G_reg" in logs:
                fig, ax = plt.subplots(
                    3,
                    2,
                    figsize=(14, 8)
                )
            else:
                fig, ax = plt.subplots(
                    2,
                    2,
                    figsize=(14, 8)
                )
            show_plt = True

        # Flatten for easier indexing
        ax = ax.flatten() #type: ignore


        # --------------------
        # Losses (top left)
        # --------------------

        # Discriminator
        ax[0].plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss_D"],
            color="#e74c3c",
            alpha=0.6,
            linewidth=1,
            label="_nolegend_",
        )

        smoothed_loss_D = running_mean(logs["loss_D"], 41)

        ax[0].plot(
            np.arange(20, num_iters - 20) / iters_per_epoch,
            smoothed_loss_D,
            color="#c0392b",
            linewidth=2,
            label="D",
        )


        # Generator total
        ax[0].plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss_G"],
            color="#1F4E79",
            alpha=0.6,
            linewidth=1,
            label="_nolegend_",
        )

        smoothed_loss_G = running_mean(logs["loss_G"], 41)

        ax[0].plot(
            np.arange(20, num_iters - 20) / iters_per_epoch,
            smoothed_loss_G,
            color="#1F4E79",
            linewidth=2,
            label="G total",
        )


        # Generator GAN component
        if "loss_G_GAN" in logs:

            ax[0].plot(
                np.arange(num_iters) / iters_per_epoch,
                logs["loss_G_GAN"],
                color="#2E75B6",
                alpha=0.6,
                linewidth=1,
                label="_nolegend_",
            )

            smoothed_loss_G_GAN = running_mean(logs["loss_G_GAN"], 41)

            ax[0].plot(
                np.arange(20, num_iters - 20) / iters_per_epoch,
                smoothed_loss_G_GAN,
                color="#2E75B6",
                linewidth=1.8,
                linestyle="--",
                label="G GAN",
            )


        # Generator regressor component
        if "loss_G_reg" in logs:

            ax[0].plot(
                np.arange(num_iters) / iters_per_epoch,
                logs["loss_G_reg"],
                color="#6FA8DC",
                alpha=0.6,
                linewidth=1,
                label="_nolegend_",
            )

            smoothed_loss_G_reg = running_mean(logs["loss_G_reg"], 41)

            ax[0].plot(
                np.arange(20, num_iters - 20) / iters_per_epoch,
                smoothed_loss_G_reg,
                color="#6FA8DC",
                linewidth=1.8,
                linestyle="--",
                label="G Reg",
            )

        ax[0].set_yscale("log")

        ax[0].set(
            xlabel="Epoch",
            ylabel="Loss",
            title="GAN Losses"
        )

        ax[0].legend(
            loc="upper right",
            fontsize=8,
            ncol=2,
            framealpha=0.8,
        )


        # --------------------
        # Learning Rates (top right)
        # --------------------
        ax[1].plot(
            logs["lr_log_D"],
            label="Discriminator LR"
        )

        ax[1].plot(
            logs["lr_log_G"],
            label="Generator LR"
        )

        if "lr_log_R" in logs:
            ax[1].plot(
                logs["lr_log_R"],
                label="Regressor LR",
                color = "#6c3483"
            )

        ax[1].set(
            xlabel="Epoch",
            ylabel="Learning Rate",
            title="Learning Rates"
        )
        ax[1].legend()


        # --------------------
        # Discriminator Predictions (middle left)
        # --------------------
        ax[2].plot(
            logs["avg_real_pred"],
            label="Real Prediction"
        )

        ax[2].plot(
            logs["avg_fake_pred"],
            label="Fake Prediction"
        )

        ax[2].axhline(
            0,
            color="#4B4B4B",
            alpha = 0.5,
            label="Decision Boundary",
            linestyle="dashdot")

        ax[2].set(
            xlabel="Epoch",
            ylabel="Avg. Logit Score",
            title="Discriminator Predictions"
        )
        ax[2].legend()



        # --------------------
        # Discriminator Accuracy (middle right)
        # --------------------
        ax[3].plot(
            logs["pred_accuracy"],
            label="Accuracy"
        )

        ax[3].set(
            xlabel="Epoch",
            ylabel="Accuracy (%)",
            title="Discriminator Accuracy"
        )

        ax[3].set_ylim(-2, 102)
        ax[3].legend()


        # --------------------
        # Regressor RMSE (bottom left)
        # --------------------
        if "RMSE_error_log_R" in logs:
            ax[4].plot(
                logs["RMSE_error_log_R"],
                label="Regressor RMSE Error"
            )

            ax[4].set(
                        xlabel="Epoch",
                        ylabel="RMSE Error",
                        title="Regressor RMSE Error"
                    )

            ax[4].legend()


        # --------------------
        # Regressor Loss (bottom right)
        # --------------------
        if "loss_G_reg" in logs:

            ax[5].plot(
                np.arange(num_iters) / iters_per_epoch,
                logs["loss_G_reg_base"],
                color="#9b59b6",
                alpha=0.6,
                linewidth=1,
                label="Base Regressor Loss",
            )

            smoothed_loss_G_reg_detail = running_mean(logs["loss_G_reg_base"], 41)

            ax[5].plot(
                np.arange(20, num_iters - 20) / iters_per_epoch,
                smoothed_loss_G_reg_detail,
                color="#6c3483",
                linewidth=2,
                label="Base Regressor Loss (Mean)",
            )
            ax[5].set(
                xlabel="Epoch",
                ylabel="Regressor Loss",
                title="Base Regressor Loss",
                
            )
            ax[5].legend()

        #Plot Snapshots as Vertical Lines
        for axis in ax:
            for snapshot in snapshot_epochs:
                axis.axvline(
                    x=snapshot,
                    color="grey",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                )
        plt.plot([], [], linestyle="--", color="gray", label=f"Snapshots: {snapshot_epochs}") #ads legend for snapshots

        axis.grid() #type: ignore

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
            plt.close()
        elif show_plt:
            plt.show()





def plot_reconstruction_loss(loss_history, iters_per_epoch, filename=None):

    losses = np.array(loss_history)
    num_iters = len(losses)

    latest_epoch = num_iters / iters_per_epoch
    logger.info("latest epoch is {}".format(latest_epoch))
    logger.info("{} iters per epoch".format(iters_per_epoch))

    smoothed_loss_41 = running_mean(losses, 41)

    fig, ax = plt.subplots()

    ax.plot(
        np.arange(num_iters) / iters_per_epoch,
        losses,
        "#82c6eb",
        np.arange(20, num_iters - 20) / iters_per_epoch,
        smoothed_loss_41,
        "#2a9edd",
    )
    ax.set_yscale("log")
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend(["Loss", "Loss (Running Mean 41)"])
    ax.grid()
    plt.savefig(filename, bbox_inches="tight")

def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _get_control_code_bounds(specs):
    """
    Read ControlCodeBounds from specs.json.
    """
    lo, hi = specs.get("ControlCodeBounds", [0.0, 1.0])
    if lo >= hi:
        raise ValueError(f"ControlCodeBounds must have min < max, got [{lo}, {hi}]")
    return lo, hi


def _build_plot_latent(z_vec, code_val, code_dim, device, vary_dim=0, fixed_code=None):
    """
    Build a latent vector from a fixed z-part plus a control-code part for all plotting purposes.

    If `fixed_code` is None (default), the latent_vec is simply  [z_vec, code_val]

    If `fixed_code` (!= lenght code_dim) is given, every control-code entry is
    taken from `fixed_code` except `vary_dim`, which is set to `code_val`.
    """
    if fixed_code is None:
        code_vec = torch.full((code_dim,), float(code_val), device=device)
    else:
        code_vec = torch.as_tensor(fixed_code, dtype=torch.float32, device=device).clone()
        if code_vec.numel() != code_dim:
            raise ValueError(
                f"fixed_code has length {code_vec.numel()}, expected code_dim={code_dim}"
            )
        code_vec[vary_dim] = float(code_val)
    return torch.cat([z_vec, code_vec])


def plot_decoder_set(
    decoder,
    ax,
    origin=(0,0,0),
    normal=(0,1,0),
    code_vals=[0.1,0.5,0.9],
    code_dim=1,
    latent_size=None,
    z_vec=None,
    device="cpu",
    vary_dim=0,
    fixed_code=None,
    var_label="c",
):

    decoder.eval()

    if z_vec is None:
        if latent_size is None:
            raise ValueError(
                "plot_decoder_set needs either z_vec or latent_size (the decoder's "
                "total latent width, i.e. specs['CodeLength']) to size z."
            )
        z_dim = latent_size - code_dim
        z_vec = torch.zeros(z_dim, device=device)
    else:
        z_vec = z_vec.to(device)

    latent0 = _build_plot_latent(
        z_vec, code_vals[0], code_dim, device, vary_dim=vary_dim, fixed_code=fixed_code
    )
    sdf = SDFfromDeepSDF(
        DeepSDFModel(decoder, latent0.unsqueeze(0), device)
    )

    for i, code_val in enumerate(code_vals):
        latent = _build_plot_latent(
            z_vec, code_val, code_dim, device, vary_dim=vary_dim, fixed_code=fixed_code
        )
        sdf.set_latent_vec(latent)
        sdf.plot_slice(origin, normal, ax=ax[i])
        ax[i].set_title(f"{var_label} = {code_val:.3g}")


def plot_decoder_latent_effect_2d(
    decoder,
    axes,
    origin=(0,0,0),
    normal=(0,1,0),
    code_vals_0=[0.1,0.5,0.9],
    code_vals_1=[0.1,0.5,0.9],
    latent_size=None,
    z_vec=None,
    device="cpu",
):
    """
    Plot a decoder over a 2D grid for a ControlCodeDim=2 control code:
    control-code dim 0 varies down the rows of `axes`, dim 1 varies across
    the columns. Mirrors plot_decoder_set, but for two independently
    varying control-code dimensions instead of one.

    `axes` must be indexable as axes[i, j] (e.g. the array returned by
    plt.subplots(n, n)).
    """
    decoder.eval()
    code_dim = 2

    if z_vec is None:
        if latent_size is None:
            raise ValueError(
                "plot_decoder_grid2d needs either z_vec or latent_size (the decoder's "
                "total latent width, i.e. specs['CodeLength']) to size z."
            )
        z_dim = latent_size - code_dim
        z_vec = torch.zeros(z_dim, device=device)
    else:
        z_vec = z_vec.to(device)

    latent0 = torch.cat([
        z_vec,
        torch.tensor([code_vals_0[0], code_vals_1[0]], dtype=z_vec.dtype, device=device),
    ])
    sdf = SDFfromDeepSDF(
        DeepSDFModel(decoder, latent0.unsqueeze(0), device)
    )

    for i, c0 in enumerate(code_vals_0):
        for j, c1 in enumerate(code_vals_1):
            code_vec = torch.tensor([c0, c1], dtype=z_vec.dtype, device=device)
            latent = torch.cat([z_vec, code_vec])
            sdf.set_latent_vec(latent)
            ax = axes[i, j]
            sdf.plot_slice(origin, normal, ax=ax)
            ax.grid(True)
            if i == 0:
                ax.set_title(f"c₁ = {c1:.3g}")
            if j == 0:
                ax.set_ylabel(f"c₀ = {c0:.3g}")


def plot_decoder_evolution(
    experiment_directory,
    snapshot_epochs,
    origin=(0.0, 0.0, 0.0),
    normal=(0.0, 1.0, 0.0),
    code_vals=None,
    fixed_code_vals=None,
    device="cpu"
):
    """
    Plot decoder output across training snapshots, sweeping control-code
    dimension 0 across `code_vals` (default: 3 values evenly spaced across
    ControlCodeBounds). If ControlCodeDim > 1, the remaining control-code
    dimensions are held fixed at `fixed_code_vals` (default: the mdasdsadwads specs = ws.load_experiment_specifications(experiment_directory)
    """
    specs = ws.load_experiment_specifications(experiment_directory)
    code_dim = specs.get("ControlCodeDim", 1)
    latent_size = specs["CodeLength"]
    lo, hi = _get_control_code_bounds(specs)

    if code_vals is None:
        code_vals = np.linspace(lo, hi, 3).tolist()

    if code_dim > 1:
        if fixed_code_vals is None:
            mid = (lo + hi) / 2.0
            fixed_code_vals = [mid] * code_dim
        elif len(fixed_code_vals) != code_dim:
            raise ValueError(
                f"fixed_code_vals must have length code_dim={code_dim}, "
                f"got {len(fixed_code_vals)}"
            )
    else:
        fixed_code_vals = None

    var_label = "c" if code_dim == 1 else "c₀"

    fig, ax = plt.subplots(
        len(snapshot_epochs),
        len(code_vals),
        squeeze=False,
        figsize=(4 * len(code_vals), 4 * len(snapshot_epochs))
    )

    # General title
    fig.suptitle("Decoder Evolution", fontsize=16)

    for i, snapshot in enumerate(snapshot_epochs):
        decoder = load_trained_model(
            experiment_directory,
            "SnapshotE-" + str(snapshot),
            device
        )

        # z is re-derived as zeros inside plot_decoder_set every call, so
        # it's identical across snapshot rows automatically -- any
        # differences you see row-to-row are the decoder's evolving
        # response to c, not noise from a different z draw. When
        # code_dim > 1, only dim 0 is swept; the other dims are held at
        # fixed_code_vals for every row/snapshot.
        plot_decoder_set(
            decoder,
            ax=ax[i],
            origin=origin,
            normal=normal,
            code_vals=code_vals,
            code_dim=code_dim,
            latent_size=latent_size,
            device=device,
            vary_dim=0,
            fixed_code=fixed_code_vals,
            var_label=var_label,
        )

        # Optional: label each row by epoch
        ax[i, 0].set_ylabel(f"Epoch {snapshot}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for subtitle #type: ignore
    plt.savefig(str(experiment_directory) + "/DecoderTrainingPlot.png")
    plt.close(fig)


def plot_decoder_latent_effect(
    experiment_directory,
    epoch,
    code_vals=None,
    device=torch.device("cpu")
):
    """
    Visualize how the decoder responds across the full ControlCodeBounds
    range.

    - ControlCodeDim == 1: same 3x3 (9-subplot) layout as before, sweeping
      the single control-code dimension across `code_vals` (default: 9
      values evenly spaced across ControlCodeBounds).
    - ControlCodeDim == 2: a 5x5 grid where each axis is one control-code
      dimension, both swept across `code_vals` (default: 5 values evenly
      spaced across ControlCodeBounds; the same range is used for both
      axes since ControlCodeBounds is a single shared [min, max] pair).
    """
    specs = ws.load_experiment_specifications(experiment_directory)
    code_dim = specs.get("ControlCodeDim", 1)
    latent_size = specs["CodeLength"]
    lo, hi = _get_control_code_bounds(specs)

    decoder = load_trained_model(experiment_directory, "latest", device)
    decoder.eval()

    if code_dim == 1:
        if code_vals is None:
            code_vals = np.linspace(lo, hi, 9).tolist()
        elif len(code_vals) != 9:
            raise ValueError(
                f"plot_decoder_latent_effect (ControlCodeDim=1) expects exactly 9 "
                f"code_vals for the 3x3 grid, got {len(code_vals)}."
            )

        fig, axes = plt.subplots(3, 3, figsize=(12, 12), squeeze=False)
        axes = axes.flatten()

        plot_decoder_set(
            decoder,
            axes,
            code_vals=code_vals,
            code_dim=code_dim,
            latent_size=latent_size,
            device=device,
        )

        for ax in axes:
            ax.grid(True)

        fig.suptitle("Decoder Latent Effect", fontsize=16)
        plt.tight_layout()
        plt.savefig(str(experiment_directory) + f"/LatentEffect-E{epoch}.png")
        plt.close(fig)

    elif code_dim == 2:
        if code_vals is None:
            code_vals = np.linspace(lo, hi, 5).tolist()

        n = len(code_vals)
        fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n), squeeze=False)

        plot_decoder_latent_effect_2d(
            decoder,
            axes,
            code_vals_0=code_vals,
            code_vals_1=code_vals,
            latent_size=latent_size,
            device=device,
        )

        fig.suptitle("Decoder Latent Effect", fontsize=16)
        plt.tight_layout()
        plt.savefig(str(experiment_directory) + f"/LatentEffect-E{epoch}.png")
        plt.close(fig)

    else:
        raise NotImplementedError(
            f"plot_decoder_latent_effect currently supports ControlCodeDim 1 or 2, "
            f"got {code_dim}. For higher-dimensional control codes, use "
            f"plot_decoder_set / plot_decoder_grid2d directly on a chosen slice of "
            f"dimensions."
        )

def plot_decoder_scatter(
    decoder,
    experiment_directory,
    epoch,
    latent_vec=None,
    code_dim=1,
    code_value=0.5,
    z_vec=None,
    latent_size=None,
    ax=None,
    resolution=30,
    y_value=0.5,
):
    # Get device from decoder
    device = next(decoder.parameters()).device

    # Latent vector: pass `latent_vec` directly to override with a specific
    # full-width vector; otherwise one is built from a fixed (default-zero)
    # z and `code_value` broadcast across code_dim, same convention as
    # plot_decoder_set. `latent_size` sizes z when z_vec isn't given
    # explicitly either -- if not passed, it's read from specs.json (needs
    # experiment_directory for that).
    if latent_vec is None:
        if z_vec is None:
            if latent_size is None:
                if experiment_directory is None:
                    raise ValueError(
                        "plot_decoder_scatter needs latent_vec, z_vec, or latent_size "
                        "when experiment_directory isn't given -- there's no specs.json "
                        "to read CodeLength from."
                    )
                specs = ws.load_experiment_specifications(experiment_directory)
                latent_size = specs["CodeLength"]
            z_dim = latent_size - code_dim
            z_vec = torch.zeros(z_dim, device=device)
        else:
            z_vec = z_vec.to(device)
        latent_vec = _build_plot_latent(z_vec, code_value, code_dim, device).unsqueeze(0)
    else:
        latent_vec = latent_vec.to(device)

    # Create SDF
    sdf = SDFfromDeepSDF(DeepSDFModel(decoder,latent_vec,device))

    # -------- Scatter Plot --------
    x = torch.linspace(-1.0,1.0,resolution,device=device)
    z = torch.linspace(-1.0,1.0,resolution,device=device)
    X, Z = torch.meshgrid(x,z,indexing="ij")

    points = torch.stack([
        X.reshape(-1),
        torch.full((X.numel(),),y_value,device=device),
        Z.reshape(-1)
    ], dim=1)

    # Evaluate decoder
    with torch.no_grad():
        values = sdf.forward(points).squeeze()

    # Move data to CPU for matplotlib
    points_cpu = points.detach().cpu()
    values_cpu = values.detach().cpu()

    # Create figure
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(
            111,
            projection="3d"
        )
    else:
        fig = ax.figure

    # -------- Scatter --------

    sc = ax.scatter(
        points_cpu[:, 0].numpy(),
        values_cpu.numpy(),
        points_cpu[:, 2].numpy(),
        cmap="viridis",
        c=values_cpu.numpy(),
        s=40,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.set_zlabel("z")

    ax.set_title(
        f"Decoder Scatter Plot - Epoch {epoch}"
    )

    fig.colorbar(
        sc,
        ax=ax,
        label="SDF Value"
    )

    fig.tight_layout()

    # -------- Save --------

    if experiment_directory is not None:
        from pathlib import Path

        experiment_directory = Path(experiment_directory)
        experiment_directory.mkdir(parents=True,exist_ok=True)
        fig.savefig(experiment_directory /f"decoderScatterPlot-E{epoch}.png",dpi=200,bbox_inches="tight")

    return fig, ax