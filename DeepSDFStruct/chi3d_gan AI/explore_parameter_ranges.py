"""
explore_parameter_ranges.py
===========================

Find out which Chi3D parameter values actually produce a usable unit cell,
before you spend hours generating a training set out of broken geometry.

Produces
--------
* a printed 1D sweep of every parameter around a reference cell, with the
  reason each rejected value was rejected,
* ``chi3d_param_sweep.png``: the cross section for every swept value, blue =
  accepted, red = rejected (look at the red ones -- the struts overlap and the
  boundary loops back into the material).

Run: ``python explore_parameter_ranges.py``
"""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chi3d_sdf import PARAM_NAMES, chi_boundary_loop, check_chi_parameters  # noqa: E402

warnings.filterwarnings("ignore")

REFERENCE = [-np.pi / 8, 0.10, 0.20, 0.10, 0.05]

SWEEPS = {
    "phi": np.linspace(-1.05, 1.05, 15),
    "t": np.linspace(0.02, 0.35, 15),
    "x1": np.linspace(0.05, 0.45, 15),
    "x2": np.concatenate([np.linspace(-0.35, -0.01, 7), np.linspace(0.01, 0.35, 8)]),
    "r": np.linspace(0.005, 0.26, 15),
}


def sweep_report(reference=REFERENCE, sweeps=SWEEPS):
    limits = {}
    for i, name in enumerate(PARAM_NAMES):
        print(f"\n=== {name} (others fixed at {np.round(reference, 3).tolist()}) ===")
        good = []
        for value in sweeps[name]:
            p = list(reference)
            p[i] = float(value)
            check = check_chi_parameters(p)
            tag = "ok " if check.ok else "BAD"
            why = "; ".join(r[:60] for r in check.reasons)
            print(f"  {name}={value: .4f}  {tag}  vf={check.volume_fraction:.3f}  {why}")
            if check.ok:
                good.append(value)
        if good:
            limits[name] = (min(good), max(good))
    print("\nusable 1D range around the reference cell:")
    for name, (lo, hi) in limits.items():
        print(f"  {name:4s}: [{lo:+.3f}, {hi:+.3f}]")
    print(
        "\nNote these are 1D slices -- the parameters interact (thick struts + "
        "small x1 + large r overlap sooner), so always keep the rejection test "
        "in the loop when you sample the full box."
    )
    return limits


def figure_sweep(reference=REFERENCE, sweeps=SWEEPS, n_cols=7,
                 fname="chi3d_param_sweep.png"):
    fig, axes = plt.subplots(len(PARAM_NAMES), n_cols,
                             figsize=(2.3 * n_cols, 2.45 * len(PARAM_NAMES)))
    for i, name in enumerate(PARAM_NAMES):
        values = np.asarray(sweeps[name])
        picks = values[np.linspace(0, len(values) - 1, n_cols).astype(int)]
        for j, value in enumerate(picks):
            ax = axes[i, j]
            p = list(reference)
            p[i] = float(value)
            check = check_chi_parameters(p)
            try:
                poly = chi_boundary_loop(p, simplify=0.0)
                ax.fill(poly[:, 0], poly[:, 1],
                        color="#4C78A8" if check.ok else "#D1495B", alpha=0.85)
                ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                        "k-", lw=0.5)
                ax.set_title(f"{name}={value:.3f}\nvf={check.volume_fraction:.2f}"
                             f"  {'ok' if check.ok else 'REJECTED'}", fontsize=7.5)
            except Exception:
                ax.set_title(f"{name}={value:.3f}\ntile creation failed",
                             fontsize=7.5, color="#D1495B")
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="gray",
                                       ls="--", lw=0.6))
            ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 1.12)
            ax.set_aspect(1); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Chi3D cross section vs. each parameter "
                 "(blue = accepted, red = rejected by check_chi_parameters)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=110)
    print("wrote", fname)


def figure_arm_thickness(reference=REFERENCE, fname="chi3d_arm_thickness.png"):
    """Show that the four arms can carry independent thickness (9-D design)."""
    from chi3d_sdf import chi_parameters

    cases = [
        ((0.10, 0.10, 0.10, 0.10), "uniform  t = 0.10"),
        ((0.04, 0.10, 0.16, 0.10), "tS = 0.04, tE = 0.16"),
        ((0.16, 0.04, 0.04, 0.16), "tS, tN thick"),
        ((0.05, 0.05, 0.16, 0.16), "tE, tN thick"),
    ]
    fig, axes = plt.subplots(1, len(cases), figsize=(3.0 * len(cases), 3.2))
    for ax, (arms, title) in zip(axes, cases):
        p = chi_parameters(reference[0], reference[1], 0.22, reference[3],
                           reference[4], arm_thickness=arms)
        poly = chi_boundary_loop(p, simplify=0.0)
        ax.fill(poly[:, 0], poly[:, 1], color="#4C78A8")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, ec="gray", ls="--"))
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        ax.set_aspect(1); ax.axis("off")
    fig.suptitle("per-arm thickness: rows 0/1/3/4 of the parameter array "
                 "(south / west / east / north)", fontsize=11)
    fig.tight_layout()
    fig.savefig(fname, dpi=115)
    print("wrote", fname)


if __name__ == "__main__":
    sweep_report()
    figure_sweep()
    figure_arm_thickness()
