"""
Experiment Workspace Management
===============================

This module provides utilities for managing DeepSDF experiment workspaces,
including directory structures, file naming conventions, and model loading/saving.

Constants
---------
The module defines standard subdirectory and file names for organizing
experiment artifacts:
- Model parameters and checkpoints
- Optimizer states
- Latent code vectors
- Training logs and plots
- Reconstructions and evaluations
- Dataset samples and normalization parameters

Architecture Registry
--------------------
ARCHITECTURES: dict
    Maps architecture names to decoder classes, enabling dynamic model
    instantiation from configuration files.

Functions
---------

load_experiment_specifications
    Load experiment configuration from specs.json file.

load_trained_model
    Load a trained decoder network from checkpoint.

load_latent_vectors
    Load learned latent codes from checkpoint.

create_experiment_directory
    Initialize directory structure for a new experiment.

The workspace utilities ensure consistent organization across experiments
and simplify model loading for inference and continued training.
"""

#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import pathlib
import torch
from typing import TypedDict

from .networks.analytic_round_cross import RoundCrossDecoder
from .networks.deep_sdf_decoder import DeepSDFDecoder
from .networks.hierarchical_deep_sdf_decoder import HierachicalDeepSDFDecoder
from .networks.resnet_positional_sdf_decoder import ResNetPositionalDeepSDFDecoder
from .networks.hierarchical_positional_sdf_decoder import (
    HierachicalPositionalDeepSDFDecoder,
)

screenshots_subdir = "Screenshots"
model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
latent_code_data_map_filename = "latent_code_data_map.json"
logs_filename = "Logs.pth"
logplot_filename = "Logs.png"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"
experiment_summary_name = "training_summary.json"

# Map architecture name to Decoder class
ARCHITECTURES = {
    "analytic_round_cross": RoundCrossDecoder,
    "deep_sdf_decoder": DeepSDFDecoder,
    "hierarchical_deep_sdf_decoder": HierachicalDeepSDFDecoder,
    "resnet_positional_deep_sdf_decoder": ResNetPositionalDeepSDFDecoder,
    "hierarchical_positional_deep_sdf_decoder": HierachicalPositionalDeepSDFDecoder,
}


def load_experiment_specifications(experiment_directory):

    filename = os.path.join(experiment_directory, specifications_filename)

    if not os.path.isfile(filename):
        raise Exception(
            f"The experiment directory ({experiment_directory}) does not include specifications file "
            + '"specs.json"'
        )
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_latent_vectors(experiment_directory, checkpoint, device):

    filename = os.path.join(
        experiment_directory, latent_codes_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(
            f"The experiment directory ({experiment_directory}) does not include a latent code file"
            + f" for checkpoint '{checkpoint}'"
        )

    data = torch.load(filename, map_location=device, weights_only=True)

    if isinstance(data["latent_codes"], torch.Tensor):

        num_vecs = data["latent_codes"].size()[0]

        lat_vecs = []
        for i in range(num_vecs):
            lat_vecs.append(data["latent_codes"][i])

        return lat_vecs

    else:

        num_embeddings, embedding_dim = data["latent_codes"]["weight"].shape

        lat_vecs = torch.nn.Embedding(num_embeddings, embedding_dim, device=device)

        lat_vecs.load_state_dict(data["latent_codes"])

        return lat_vecs.weight.data.detach()


def load_model_parameters(
    experiment_directory, checkpoint, decoder: torch.nn.Module, device
):

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location=device, weights_only=True)

    decoder.load_state_dict(data["model_state_dict"], strict=False)

    return data["epoch"]


def load_optimizer(
    experiment_directory, checkpoint, optimizer: torch.nn.Module, device
):

    filename = os.path.join(
        experiment_directory, optimizer_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception(f'optimizer state dict "{filename}" does not exist')

    data = torch.load(filename, map_location=device, weights_only=True)

    optimizer.load_state_dict(data["optimizer_state_dict"])

    return data["epoch"]


def get_data_source_map_filename(data_dir):
    return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir,
    epoch,
    dataset,
    class_name,
    instance_name,
    create_dir=True,
    filetype="ply",
):
    fname_raw = os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_meshes_subdir,
        dataset,
        class_name,
        instance_name + "." + filetype,
    )
    fname = pathlib.Path(fname_raw)
    if not os.path.isdir(fname.parent) and create_dir:
        os.makedirs(fname.parent)
    return fname


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

    return os.path.join(
        experiment_dir,
        reconstructions_subdir,
        str(epoch),
        reconstruction_codes_subdir,
        dataset,
        class_name,
        instance_name + ".pth",
    )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, model_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_screenshots_dir(experiment_dir, create_if_nonexistent=True):

    dir = os.path.join(experiment_dir, screenshots_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, optimizer_params_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

    dir = os.path.join(experiment_dir, latent_codes_subdir)

    if create_if_nonexistent and not os.path.isdir(dir):
        os.makedirs(dir)

    return dir


def get_latent_code_data_map_filename(experiment_dir):
    """Return absolute path for the latent-to-data mapping JSON file."""
    return os.path.join(
        get_latent_codes_dir(experiment_dir, create_if_nonexistent=True),
        latent_code_data_map_filename,
    )


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
    return os.path.join(
        data_dir,
        normalization_param_subdir,
        dataset_name,
        class_name,
        instance_name + ".npz",
    )


def init_decoder(experiment_specs, device, data_parallel):
    arch_name = experiment_specs["NetworkArch"]
    if arch_name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {arch_name}")

    latent_size = experiment_specs["CodeLength"]
    DecoderClass = ARCHITECTURES[arch_name]

    decoder = DecoderClass(latent_size, **experiment_specs["NetworkSpecs"]).to(device)
    if data_parallel:
        decoder = torch.nn.DataParallel(decoder)
    return decoder


def load_trained_model(
    experiment_directory: str, checkpoint: str, device=None, data_parallel=False
):
    specs_filename = os.path.join(experiment_directory, "specs.json")
    with open(specs_filename, "r", encoding="utf-8") as f:
        experiment_specs = json.load(f)
    if device is None:
        device = get_default_device()

    filename = os.path.join(
        experiment_directory, model_params_subdir, checkpoint + ".pth"
    )

    if not os.path.isfile(filename):
        raise Exception('model state dict "{}" does not exist'.format(filename))

    data = torch.load(filename, map_location=device)
    decoder = init_decoder(experiment_specs, device, data_parallel)
    try:
        decoder.load_state_dict(data["model_state_dict"], strict=False)
    except RuntimeError:
        state_dict = {}
        for k, v in data["model_state_dict"].items():
            new_key = k.replace("module.", "", 1) if k.startswith("module.") else k
            state_dict[new_key] = v
        decoder.load_state_dict(state_dict, strict=False)
    decoder = decoder.to(device)
    return decoder


def get_default_device():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def print_model_specifications(experiment_directory: str):
    specs = load_experiment_specifications(experiment_directory)
    print("Model Specifications:")
    for key in specs:
        print(f"  {key}: {specs[key]}")
    print("\n")


class ExperimentSummary(TypedDict):
    loss: float
    num_epochs: int
    timestamp: str
    host_name: str
    device: str
    training_duration: str
    data_dir: str
    version: str


def save_experiment_summary(experiment_directory: str, summary: ExperimentSummary):
    with open(os.path.join(experiment_directory, experiment_summary_name), "w") as f:
        json.dump(summary, f, indent=4)

def save_model(experiment_directory, filename, decoder, epoch):

    model_params_dir = get_model_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        os.path.join(model_params_dir, filename),
    )

def save_optimizer(experiment_directory, filename, optimizer, epoch):

    optimizer_params_dir = get_optimizer_params_dir(experiment_directory, True)

    torch.save(
        {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        os.path.join(optimizer_params_dir, filename),
    )

def save_latent_vectors(experiment_directory, filename, latent_vec, epoch):

    latent_codes_dir = get_latent_codes_dir(experiment_directory, True)

    all_latents = latent_vec.state_dict()

    torch.save(
        {"epoch": epoch, "latent_codes": all_latents},
        os.path.join(latent_codes_dir, filename),
    )

def save_latest(epoch, experiment_directory, decoder, optimizer_all, lat_vecs, GAN = False):

    if GAN == False:
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)
    if GAN == True:
        save_model(experiment_directory, "latest.pth", decoder, epoch)

def save_logs(
    experiment_directory,
    loss_log,
    lr_log,
    timing_log,
    lat_mag_log,
    param_mag_log,
    epoch
):

    torch.save(
        {
            "epoch": epoch,
            "loss": loss_log,
            "learning_rate": lr_log,
            "timing": timing_log,
            "latent_magnitude": lat_mag_log,
            "param_magnitude": param_mag_log,
        },
        os.path.join(experiment_directory, logs_filename),
    )

def save_logs_GAN(
        experiment_directory,
        loss_log_D,
        loss_log_G,
        lr_log_D,
        lr_log_G,
        avg_real_pred,
        avg_fake_pred,
        pred_accuracy,
        loss_log_C,
        RMSE_error_log_C,
        lr_log_C,
        loss_log_G_GAN,
        loss_log_G_cla,
        epoch
):

    logs = {
        "epoch": epoch,
        "loss_D": loss_log_D,
        "loss_G": loss_log_G,
        "lr_log_D": lr_log_D,
        "lr_log_G": lr_log_G,
        "avg_real_pred": avg_real_pred,
        "avg_fake_pred": avg_fake_pred,
        "pred_accuracy": pred_accuracy,
    }

    # Only add classifier-related logs if they contain data
    if len(loss_log_C) > 0:
        logs["loss_C"] = loss_log_C
        logs["RMSE_error_log_C"] = RMSE_error_log_C
        logs["lr_log_C"] = lr_log_C
        logs["loss_G_GAN"] = loss_log_G_GAN
        logs["loss_G_cla"] = loss_log_G_cla

    torch.save(
        logs,
        os.path.join(experiment_directory, logs_filename)
    )


def load_logs(experiment_directory):

    full_filename = os.path.join(experiment_directory, logs_filename)

    if not os.path.isfile(full_filename):
        raise Exception(f'log file "{full_filename}" does not exist')

    data = torch.load(full_filename)

    #GAN Logs
    if "loss_D" in data:

        # GAN + Classifier
        if "loss_C" in data:
            return (
                data["loss_D"],
                data["loss_G"],
                data["lr_log_D"],
                data["lr_log_G"],
                data["avg_real_pred"],
                data["avg_fake_pred"],
                data["pred_accuracy"],
                data["loss_C"],
                data["RMSE_error_log_C"],
                data["lr_log_C"],
                data["loss_G_GAN"],
                data["loss_G_cla"],
                data["epoch"],
            )

        # GAN without Classifier
        return (
            data["loss_D"],
            data["loss_G"],
            data["lr_log_D"],
            data["lr_log_G"],
            data["avg_real_pred"],
            data["avg_fake_pred"],
            data["pred_accuracy"],
            data["epoch"],
        )

    # Standard logs
    return (
        data["loss"],
        data["learning_rate"],
        data["timing"],
        data["latent_magnitude"],
        data["param_magnitude"],
        data["epoch"],
    )


