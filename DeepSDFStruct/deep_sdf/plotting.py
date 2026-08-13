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
            fig, ax = plt.subplots(
                2,
                2,
                figsize=(14, 8)
            )
            show_plt = True

        # Flatten for easier indexing
        ax = ax.flatten()


        # --------------------
        # GAN Losses (top left)
        # --------------------
        ax[0].plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss_D"],
            "#e74c3c",
            alpha=0.3,
            label="Discriminator Loss",
        )

        ax[0].plot(
            np.arange(20, num_iters-20) / iters_per_epoch,
            smoothed_loss_D,
            "#c0392b",
            label="Discriminator Loss (Mean)",
        )

        ax[0].plot(
            np.arange(num_iters) / iters_per_epoch,
            logs["loss_G"],
            "#3498db",
            alpha=0.3,
            label="Generator Loss",
        )

        ax[0].plot(
            np.arange(20, num_iters-20) / iters_per_epoch,
            smoothed_loss_G,
            "#21618c",
            label="Generator Loss (Mean)",
        )

        ax[0].set_yscale("log")
        ax[0].set(
            xlabel="Epoch",
            ylabel="Loss",
            title="GAN Losses"
        )
        ax[0].legend()


        # --------------------
        # Predictions (top right)
        # --------------------
        ax[1].plot(
            logs["avg_real_pred"],
            label="Real Prediction"
        )

        ax[1].plot(
            logs["avg_fake_pred"],
            label="Fake Prediction"
        )

        ax[1].set(
            xlabel="Epoch",
            ylabel="Avg. Logit Score",
            title="Discriminator Predictions"
        )
        ax[1].legend()


        # --------------------
        # Learning Rates (bottom left)
        # --------------------
        ax[2].plot(
            logs["lr_log_D"],
            label="Discriminator LR"
        )

        ax[2].plot(
            logs["lr_log_G"],
            label="Generator LR"
        )

        ax[2].set(
            xlabel="Epoch",
            ylabel="Learning Rate",
            title="Learning Rates"
        )
        ax[2].legend()


        # --------------------
        # Accuracy (bottom right)
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

def plot_decoder_set(
    experiment_directory,
    decoder,
    ax,
    origin=(0,0,0),
    normal=(0,1,0),
    lat_vec_set=[0.1,0.5,0.9],
    device="cpu"
):
    decoder.eval()

    sdf = SDFfromDeepSDF(
        DeepSDFModel(decoder, torch.tensor([[lat_vec_set[0]]]), device)
    )

    for i, lat_vec_val in enumerate(lat_vec_set):
        sdf.set_latent_vec(torch.tensor([lat_vec_val]))
        sdf.plot_slice(origin, normal, ax=ax[i])
        ax[i].set_title(f"Latent Vector: {lat_vec_val}")

def plot_decoder_evolution(
    experiment_directory,
    snapshot_epochs,
    origin=(0.0, 0.0, 0.0),
    normal=(0.0, 1.0, 0.0),
    lat_vec_set=[0.1, 0.5, 0.9],
    device="cpu"
):
    fig, ax = plt.subplots(
        len(snapshot_epochs),
        len(lat_vec_set),
        squeeze=False,
        figsize=(4 * len(lat_vec_set), 4 * len(snapshot_epochs))
    )

    # General title
    fig.suptitle("Decoder Evolution", fontsize=16)

    # Column titles
    for j, lat_vec in enumerate(lat_vec_set):
        ax[0, j].set_title(f"Latent Vector = {lat_vec}")

    for i, snapshot in enumerate(snapshot_epochs):
        decoder = load_trained_model(
            experiment_directory,
            "SnapshotE-" + str(snapshot),
            device
        )

        plot_decoder_set(
            experiment_directory,
            decoder,
            ax=ax[i],
            origin=origin,
            normal=normal,
            lat_vec_set=lat_vec_set,
            device=device
        )

        # Optional: label each row by epoch
        ax[i, 0].set_ylabel(f"Epoch {snapshot}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
    plt.savefig(experiment_directory + "/DecoderTrainingPlot.png")