import torch
from matplotlib import pyplot as plt

from DeepSDFStruct.deep_sdf.workspace import *
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.deep_sdf.plotting import *


if __name__ == "__main__":

    # --------------- MODEL LOADING ---------------
    device = torch.device("cpu")

    path = (
        "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/"
        "DeepSDFStruct/trained_models/experiments/"
        "gan_experiment3"
    )

    decoder = load_trained_model(path, "latest", device)
    decoder.eval()

    lat_vec_set = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # --------------- PLOTTING ---------------
    fig, axes = plt.subplots(
        3, 3,
        figsize=(12, 12),
        squeeze=False
    )

    axes = axes.flatten()

    plot_decoder_set(
        path,
        decoder,
        axes,
        lat_vec_set=lat_vec_set
    )

    # Add latent value as title to each subplot
    for ax, latent_value in zip(axes, lat_vec_set):
        ax.set_title(f"Latent vector = {latent_value}")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(path + "/LatestLatentEffect.png")
    plt.show()