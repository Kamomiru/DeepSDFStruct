import torch
import logging
import socket
import math
import tqdm
import time


import DeepSDFStruct.deep_sdf.workspace as ws
from DeepSDFStruct.deep_sdf.GAN_helpers.gan_sampling import *
from DeepSDFStruct.sdf_primitives import CrossMsSDF
from DeepSDFStruct.deep_sdf.GAN_helpers.Hinge_GAN_loss import *
from DeepSDFStruct.deep_sdf.plotting import *
from DeepSDFStruct.deep_sdf.GAN_helpers.gan_sampling import ConvGAN_SDF_Sampler

def plot_sdf_slices(sdf, y=0.0, extent=None, cmap="RdBu_r"):
    """
    Plot the y=constant slice of an SDF.

    Args:
        sdf: Tensor of shape [B, 1, D, H, W]
        y: y-coordinate of the slice. Assumes voxel coordinates span
           [-1, 1] unless `extent` is provided.
        extent: (xmin, xmax, zmin, zmax), optional.
        cmap: matplotlib colormap.
    """
    assert sdf.ndim == 5, f"Expected [B, 1, D, H, W], got {sdf.shape}"
    assert sdf.shape[1] == 1, f"Expected 1 channel, got {sdf.shape[1]}"

    # [B, D, H, W]
    sdf = sdf[:, 0]

    B, D, H, W = sdf.shape

    # If coordinates are [-1, 1], convert y coordinate -> voxel index.
    if extent is None:
        y_idx = round((y + 1) / 2 * (H - 1))
    else:
        # extent = (xmin, xmax, zmin, zmax)
        # We still need the y coordinate bounds; assume [-1, 1].
        y_idx = round((y + 1) / 2 * (H - 1))

    y_idx = max(0, min(H - 1, y_idx))

    # Slice normal (0, 1, 0) => fix Y, leaving X/Z
    slices = sdf[:, :, y_idx, :].detach().cpu().numpy()

    # Symmetric color scale across all samples
    vmax = abs(slices).max()
    vmin = -vmax

    # Automatically choose grid size
    ncols = min(5, B)
    nrows = math.ceil(B / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    axes = axes.flatten()

    for i in range(B):
        ax = axes[i]

        im = ax.imshow(
            slices[i],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
        )

        ax.set_title(f"Sample {i}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")

        # Zero level set = SDF surface
        ax.contour(
            slices[i],
            levels=[0],
            colors="black",
            linewidths=1.0,
            origin="lower",
            extent=extent,
        )

    # Hide unused axes
    for i in range(B, len(axes)):
        axes[i].axis("off")

    fig.colorbar(im, ax=axes[:B], shrink=0.8, label="SDF")
    fig.suptitle(f"SDF slice: normal=(0,1,0), y={y}", fontsize=14)

    plt.tight_layout()
    plt.show()

device = torch.device("cpu")
experiment_directory = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiments/gan_test_experiment28 Chi"

#load experiment specs
specs = ws.load_experiment_specifications(experiment_directory)

decoder = ws.init_decoder(specs, device, data_parallel = False).to(device) 

#real_sdf = CrossMsSDF(0.0)

sampler = ConvGAN_SDF_Sampler(decoder, specs, device)

real_samples = sampler.fetch_real_batch()

real_samples = torch.stack(tuple(real_samples), dim=0)

print(real_samples.shape)

#print(sampler._update_random_params)

# torch.Size([10, 1, 32, 32, 32])

plot_sdf_slices(real_samples)