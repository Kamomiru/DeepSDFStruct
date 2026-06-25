import numpy as np
from matplotlib import pyplot as plt

def visualize_sdf_samples(samples, max_sample_n = -1):
    """
    samples: numpy array of shape (N, 4)

    columns:
        0 -> x
        1 -> y
        2 -> z
        3 -> sdf distance
    """

    assert samples.ndim == 2
    assert samples.shape[1] == 4

    if max_sample_n != -1 and samples.shape[0] > max_sample_n:
        xyz = samples[:max_sample_n, :3]
        sdf = samples[:max_sample_n, 3]
    else :
        xyz = samples[:, :3]
        sdf = samples[:, 3]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=sdf,
        s=2,
        alpha=0.8,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("SDF Samples")

    cbar = plt.colorbar(scatter)
    cbar.set_label("Signed Distance")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file = "C:/Users/camil/Desktop/Chi3D_center_00040.npz"
    data = np.load(file)

    visualize_sdf_samples(data["neg"], 5000)