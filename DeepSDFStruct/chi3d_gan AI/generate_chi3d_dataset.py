"""
generate_chi3d_dataset.py
=========================

Build a DeepSDF-style training set of Chi3D unit cells with *multiple* shape
parameters, ready for your GAN.

This is the same pipeline your supervisor used (``DeepSDFStruct.sampling.
SDFSampler``, see the README of DeepSDFStruct), with three additions:

1. a **Sobol / LHS / grid design of experiments** over the 5 (or 9) tile
   parameters instead of a hand-written double for-loop,
2. **rejection of invalid parameter sets** (overlapping struts, control points
   leaving the unit cell, NaNs) -- roughly half of a naive box sample is
   geometrically broken and would otherwise end up in your training data with
   wrong signed distances,
3. a **``parameters.json`` / ``parameters.csv``** side-car that maps every
   instance name to its parameter vector, so the GAN can be conditioned on
   the parameters (the .npz itself only holds points + distances).

Usage
-----
    python generate_chi3d_dataset.py --n 200 --out ./training_data \
        --dataset-name chi3d_multiparam --n-samples 100000 --n-faces 20

Then train exactly as before::

    from DeepSDFStruct.deep_sdf.training import train_deep_sdf
    train_deep_sdf(exp_dir, "./training_data")
"""

from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import numpy as np
import splinepy

from chi3d_sdf import PARAM_NAMES, chi_multipatch, check_chi_parameters

# Ranges that keep ~50 % of a Sobol sample valid and cover volume fractions
# from ~0.10 to ~0.58.  Verified by sweeping each parameter around the
# reference cell [-pi/8, 0.1, 0.2, 0.1, 0.05]; see the sweep figure.
DEFAULT_RANGES = {
    "phi": (-0.70, 0.35),   # strut angle [rad]; < -0.9 or > 0.6 breaks the cell
    "t": (0.04, 0.22),      # nominal thickness; large t makes struts overlap
    "x1": (0.12, 0.32),     # arm position; > 0.33 pushes control points outside
    "x2": (-0.25, 0.30),    # arm offset; must not be 0 (0/0 -> NaN)
    "r": (0.02, 0.14),      # fillet radius; must be > 0, large r merges struts
}


def sample_parameters(
    n: int,
    ranges: dict | None = None,
    method: str = "sobol",
    seed: int = 0,
    max_tries: int = 50,
    verbose: bool = True,
) -> np.ndarray:
    """Draw ``n`` *valid* Chi3D parameter vectors.

    Parameters
    ----------
    method : {"sobol", "lhs", "random", "grid"}
        ``sobol``/``lhs`` give a space-filling design (recommended);
        ``grid`` builds a full factorial grid with ``round(n ** (1/d))``
        levels per parameter and then filters.

    Returns
    -------
    np.ndarray, shape (n, 5) -- rows are ``[phi, t, x1, x2, r]``.
    """
    from scipy.stats import qmc

    ranges = ranges or DEFAULT_RANGES
    lo = np.array([ranges[k][0] for k in PARAM_NAMES], dtype=float)
    hi = np.array([ranges[k][1] for k in PARAM_NAMES], dtype=float)
    d = len(PARAM_NAMES)

    if method == "grid":
        levels = max(2, int(round(n ** (1.0 / d))))
        axes = [np.linspace(lo[i], hi[i], levels) for i in range(d)]
        cand = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, d)
        rng = np.random.default_rng(seed)
        rng.shuffle(cand)
    else:
        engine = {
            "sobol": lambda: qmc.Sobol(d, scramble=True, seed=seed),
            "lhs": lambda: qmc.LatinHypercube(d, seed=seed),
            "random": lambda: qmc.Halton(d, seed=seed),
        }[method]()
        cand = lo + engine.random(n * 4) * (hi - lo)

    accepted, rejected = [], 0
    for row in cand:
        if len(accepted) >= n:
            break
        check = check_chi_parameters(row)
        if check.ok:
            accepted.append(row)
        else:
            rejected += 1

    if len(accepted) < n:
        raise RuntimeError(
            f"only {len(accepted)}/{n} valid parameter sets found "
            f"({rejected} rejected). Shrink the ranges or raise the candidate pool."
        )
    if verbose:
        rate = 100 * len(accepted) / (len(accepted) + rejected)
        print(f"design of experiments: {len(accepted)} accepted, {rejected} rejected "
              f"({rate:.0f} % acceptance)")
    return np.asarray(accepted)


def build_dataset(
    n: int = 200,
    outdir: str = "./training_data",
    dataset_name: str = "chi3d_multiparam",
    class_name: str = "Chi3D",
    n_samples: int = 100_000,
    n_faces: int = 20,
    method: str = "sobol",
    seed: int = 0,
    workers: int = 0,
    ranges: dict | None = None,
    params: np.ndarray | None = None,
    store_params_as_C: bool = False,
    overwrite: bool = True,
):
    """Generate the dataset. Returns the (n, 5) parameter table.

    Pass ``params`` explicitly (shape (n, 5)) to use your own design instead of
    the built-in DOE.
    """
    from DeepSDFStruct.sampling import SDFSampler

    outdir = pathlib.Path(outdir)
    splitdir = outdir / "splits"

    if params is None:
        params = sample_parameters(n, ranges=ranges, method=method, seed=seed)
    else:
        params = np.atleast_2d(np.asarray(params, dtype=float))

    tiles = [chi_multipatch(row) for row in params]

    sampler = SDFSampler(
        str(outdir), str(splitdir), dataset_name, overwrite_existing=overwrite
    )
    sampler.add_class(
        tiles,
        class_name=class_name,
        n_faces=n_faces,
        homogenized_c=[p.reshape(1, -1) for p in params] if store_params_as_C else None,
    )
    sampler.process_geometries(
        sampling_strategy="uniform",
        n_samples=n_samples,
        add_surface_samples=True,
        also_save_mesh=True,
        n_workers=workers,
    )
    sampler.write_json(f"{dataset_name}_train.json")

    # side-car: instance name -> parameter vector (what the GAN conditions on)
    table = {
        f"{class_name}_{i:05}": {k: float(v) for k, v in zip(PARAM_NAMES, row)}
        for i, row in enumerate(params)
    }
    param_path = outdir / dataset_name / "parameters.json"
    with open(param_path, "w") as f:
        json.dump(table, f, indent=2)
    with open(outdir / dataset_name / "parameters.csv", "w") as f:
        f.write("instance," + ",".join(PARAM_NAMES) + "\n")
        for name, row in table.items():
            f.write(name + "," + ",".join(f"{row[k]:.8f}" for k in PARAM_NAMES) + "\n")

    print(f"wrote {len(params)} instances to {outdir / dataset_name}")
    print(f"parameter table: {param_path}")
    return params


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="./training_data")
    ap.add_argument("--dataset-name", default="chi3d_multiparam")
    ap.add_argument("--class-name", default="Chi3D")
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--n-faces", type=int, default=20)
    ap.add_argument("--method", default="sobol", choices=["sobol", "lhs", "random", "grid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--store-params-as-C", action="store_true",
                    help="also write the parameter vector into the npz 'C' field")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    build_dataset(
        n=args.n,
        outdir=args.out,
        dataset_name=args.dataset_name,
        class_name=args.class_name,
        n_samples=args.n_samples,
        n_faces=args.n_faces,
        method=args.method,
        seed=args.seed,
        workers=args.workers,
        store_params_as_C=args.store_params_as_C,
    )


if __name__ == "__main__":
    main()
