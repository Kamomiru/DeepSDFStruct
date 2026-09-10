"""
validate_chi3d_sdf.py
=====================

Sanity checks you should run before you feed anything to the GAN:

1.  mesh-SDF vs. analytic prism-SDF agreement (they are computed by completely
    different routes, so agreement means both are right),
2.  |SDF| == 0 on the tile surface,
3.  marching cubes on the SDF reproduces the spline tile,
4.  pictures: SDF slices + reconstructed unit cells.

Run: ``python validate_chi3d_sdf.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import trimesh
from skimage import measure

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chi3d_sdf import Chi3DMeshSDF, Chi3DPrismSDF, check_chi_parameters  # noqa: E402

warnings.filterwarnings("ignore")

REFERENCE = [-np.pi / 8, 0.10, 0.20, 0.10, 0.05]


def numeric_report(params=REFERENCE, n_points: int = 100_000, n_faces: int = 20):
    mesh_sdf = Chi3DMeshSDF(params, n_faces=n_faces)
    prism_sdf = Chi3DPrismSDF(params)

    g = torch.Generator().manual_seed(0)
    q = torch.rand(n_points, 3, generator=g) * 2 - 1
    d_mesh = mesh_sdf(q)
    d_prism = prism_sdf(q)

    err = (d_mesh - d_prism).abs()
    print(f"parameters            : {np.round(params, 4).tolist()}")
    print(f"volume fraction       : {check_chi_parameters(params).volume_fraction:.4f}")
    print(f"max |mesh - prism|    : {err.max().item():.2e}  (domain is [-1,1]^3)")
    print(f"mean |mesh - prism|   : {err.mean().item():.2e}")
    print(f"sign agreement        : {((d_mesh < 0) == (d_prism < 0)).float().mean():.5f}")
    print(f"inside fraction       : mesh {(d_mesh<0).float().mean():.4f} "
          f"| prism {(d_prism<0).float().mean():.4f}")

    surf, _ = trimesh.sample.sample_surface(mesh_sdf.mesh, 20_000)
    d_surf = prism_sdf(torch.tensor(np.asarray(surf), dtype=torch.float32))
    print(f"|prism SDF| on surface: max {d_surf.abs().max().item():.2e}, "
          f"mean {d_surf.abs().mean().item():.2e}")

    # marching cubes round trip: every reconstructed vertex must sit on the
    # zero level set of the *other* SDF.  (The reconstruction is open where the
    # struts leave the cell -- that is expected, the unit cell is not a closed
    # body; use CappedBorderSDF if you want it capped.)
    rec = marching_cubes(prism_sdf, resolution=96)
    d_rec = mesh_sdf(torch.tensor(rec.vertices, dtype=torch.float32))
    print(f"marching-cubes surface: max |mesh SDF| {d_rec.abs().max().item():.2e}, "
          f"mean {d_rec.abs().mean().item():.2e}")
    print(f"mesh volume (in [-1,1]^3): {abs(mesh_sdf.mesh.volume) / 8:.4f} of the cell "
          f"| Monte-Carlo inside fraction {(d_mesh < 0).float().mean():.4f}")
    return mesh_sdf, prism_sdf, rec


def marching_cubes(
    sdf, resolution: int = 96, batch: int = 200_000, close: bool = False
) -> trimesh.Trimesh:
    """Extract the zero level set of an SDF on [-1, 1]^3.

    ``close=True`` pads the sampled field with a positive shell so the struts
    are capped where they leave the unit cell -- only for nicer pictures, the
    real unit cell is open there.
    """
    lin = torch.linspace(-1, 1, resolution)
    grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    vals = torch.cat(
        [sdf(grid[i : i + batch]) for i in range(0, grid.shape[0], batch)]
    ).reshape(resolution, resolution, resolution).detach().numpy()
    spacing = 2.0 / (resolution - 1)
    offset = -1.0
    if close:
        vals = np.pad(vals, 1, mode="constant", constant_values=1.0)
        offset -= spacing
    verts, faces, _, _ = measure.marching_cubes(vals, level=0.0, spacing=(spacing,) * 3)
    verts += offset
    return trimesh.Trimesh(vertices=verts, faces=faces)


def figure_slices(mesh_sdf, prism_sdf, fname="chi3d_sdf_slices.png"):
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4))
    normals = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    titles = [
        "slice z = 0 (along the prism axis)",
        "slice y = 0 (the chi cross section)",
        "slice x = 0 (along the prism axis)",
    ]
    for col, (nrm, title) in enumerate(zip(normals, titles)):
        for row, (sdf, label) in enumerate(
            [(mesh_sdf, "mesh SDF (libigl)"), (prism_sdf, "prism SDF (analytic)")]
        ):
            ax = axes[row, col]
            sdf.plot_slice(origin=(0, 0, 0), normal=nrm, res=(220, 220), ax=ax)
            ax.set_title(f"{label}\n{title}", fontsize=9)
    fig.suptitle("Chi3D unit cell: signed distance field, black line = zero level set")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("wrote", fname)


def _render(ax, mesh, view=(0.45, 1.0, 0.30), up_ref=(0, 0, 1),
            color=(0.30, 0.47, 0.66)):
    """Tiny painter's-algorithm renderer (matplotlib 3D does not z-sort).

    The default view looks mostly down the prism axis (y), which is the only
    direction from which the chi cross section is readable.
    """
    from matplotlib.collections import PolyCollection

    fwd = np.asarray(view, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.asarray(up_ref, float), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)

    tri = mesh.vertices[mesh.faces]
    screen = np.stack([tri @ right, tri @ up], axis=-1)
    depth = (tri @ fwd).mean(axis=1)

    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([0.35, 0.40, 0.85])
    light /= np.linalg.norm(light)
    shade = np.abs(n @ light)
    facecolors = 0.30 + 0.70 * shade[:, None] * np.array(color)[None, :]

    order = np.argsort(depth)
    ax.add_collection(
        PolyCollection(screen[order], facecolors=facecolors[order],
                       edgecolors="none", linewidths=0)
    )
    lim = 1.5
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect(1); ax.set_axis_off()


def figure_gallery(param_list, fname="chi3d_gallery.png", resolution=90):
    n = len(param_list)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.3 * rows))
    for ax, params in zip(np.ravel(axes), param_list):
        rec = marching_cubes(Chi3DPrismSDF(params), resolution=resolution, close=True)
        _render(ax, rec)
        vf = check_chi_parameters(params).volume_fraction
        ax.set_title(
            "phi={:.2f}  t={:.2f}  x1={:.2f}\nx2={:.2f}  r={:.2f}   vf={:.2f}".format(
                *params, vf
            ),
            fontsize=8,
        )
    fig.suptitle(
        "marching cubes on the Chi3D SDF - this is what the network has to learn",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print("wrote", fname)


if __name__ == "__main__":
    mesh_sdf, prism_sdf, rec = numeric_report()
    figure_slices(mesh_sdf, prism_sdf)

    gallery = [
        [-np.pi / 8, 0.10, 0.20, 0.10, 0.05],
        [0.00, 0.06, 0.25, 0.25, 0.03],
        [-0.60, 0.16, 0.30, -0.15, 0.09],
        [-0.35, 0.20, 0.28, 0.05, 0.12],
        [0.25, 0.08, 0.18, -0.20, 0.04],
        [-0.50, 0.05, 0.14, 0.22, 0.05],
        [-0.20, 0.13, 0.27, -0.10, 0.06],
        [-0.66, 0.09, 0.25, 0.16, 0.11],
    ]
    figure_gallery(gallery)
