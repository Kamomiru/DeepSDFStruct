'''
Loads a Trained Model from given path and visualizes a slice through the xy plane
'''

import torch
from matplotlib import pyplot as plt

from DeepSDFStruct.deep_sdf.workspace import *
from DeepSDFStruct.pretrained_models import get_model, PretrainedModels
from DeepSDFStruct.deep_sdf.models import DeepSDFModel
from DeepSDFStruct.SDF import SDFfromDeepSDF

if __name__ == "__main__":

    #---------------MODEL LOADING---------------
    device = torch.device("cpu")
    path = "C:/Users/camil/Desktop/Bachelorarbeit/DeepSDFStruct/DeepSDFStruct/trained_models/test_experiment4"


    decoder = load_trained_model(path, "latest", device)
    latent_vectors = load_latent_vectors(path, "latest", device)
    
    decoder.eval()

    deep_sdf_model = DeepSDFModel(decoder, latent_vectors, device) # type: ignore

    fig, ax = plt.subplots(1,3)

    sdf = SDFfromDeepSDF(deep_sdf_model)
    sdf.set_latent_vec(torch.tensor([0.9]))
    sdf.plot_slice(origin=(0.5, 0.0, 0.0), normal=(0.0,1.0,0.0), ax = ax[0])
    sdf.set_latent_vec(torch.tensor([0.5]))
    sdf.plot_slice(origin=(0.0, 0.5, 0.0), normal=(0.0,1.0,0.0), ax = ax[1])
    sdf.set_latent_vec(torch.tensor([0.1]))
    sdf.plot_slice(origin=(0.0, 0.5, 0.0), normal=(0.0,1.0,0.0), ax = ax[2])

    #---------------PLOTTING---------------
    x = torch.linspace(-1.0, 1.0, 30)
    z = torch.linspace(-1.0, 1.0, 30)
    X, Z = torch.meshgrid(x, z, indexing="ij")

    # Flatten and stack into (N, 3)
    points = torch.stack([
        X.reshape(-1),
        torch.full((X.numel(),), 0.5),
        Z.reshape(-1)
    ], dim=1)


    
    values = sdf.forward(points)
    #print(values)
    

    values = values.squeeze()
    values = values.detach()

fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection="3d")



sc = ax2.scatter(
    points[:, 0].numpy(),                 # x # pyright: ignore[reportPossiblyUnboundVariable]
    values.numpy(),                       # height # pyright: ignore[reportArgumentType]
    points[:, 2].numpy(),                 # z # pyright: ignore[reportPossiblyUnboundVariable]
    c=values.detach().numpy(),              # pyright: ignore[reportPossiblyUnboundVariable]
    cmap="viridis",
    s=40
)

ax2.set_xlabel("x")
ax2.set_ylabel("value")
ax2.set_zlabel("z")

fig2.colorbar(sc, ax=ax2, label="Value")

plt.show()


    