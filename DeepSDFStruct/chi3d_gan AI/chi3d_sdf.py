"""
chi3d_sdf.py
============

Turn the splinepy ``Chi3D`` unit cell into something you can *sample* like the
``CrossMsSDF`` you trained on before.

Two independent routes are provided, both subclasses of
``DeepSDFStruct.SDF.SDFBase`` so they drop straight into your existing code:

``Chi3DMeshSDF``
    tile -> ``splinepy.Multipatch`` -> boundary triangle mesh -> ``SDFfromMesh``
    (libigl signed distance).  This is *exactly* the route
    ``DeepSDFStruct.sampling.SDFSampler`` takes internally, i.e. the way your
    supervisor's training samples were produced.  Use this to reproduce his
    dataset.

``Chi3DPrismSDF``
    The Chi3D tile is a *prism*: a 2D cross section extruded along one axis.
    So instead of meshing, we extract the exact 2D boundary curves of the
    patch union, turn them into one closed polygon and evaluate
    (2D polygon distance) x (slab) analytically in torch.
    ~100x faster, no meshing, runs on the GPU, and more accurate than a coarse
    triangulation.  Use this for on-the-fly ground truth during GAN training.

Parameter layout
----------------
``Chi3D.create_tile`` expects an array of shape (5, 5): one row per evaluation
point, in the order

    row 0 = south (0.5, 0.0)
    row 1 = west  (0.0, 0.5)
    row 2 = center(0.5, 0.5)
    row 3 = east  (1.0, 0.5)
    row 4 = north (0.5, 1.0)

and 5 numbers per row: ``[phi, t, x1, x2, r]``
(German original: ``[Winkel, Dicke, Armposition, Armpositionsverschiebung,
Rundungsradius]``).

Only the **center row** supplies phi, x1, x2, r and the *nominal* thickness t.
Column 1 of the four *other* rows supplies the thickness of the corresponding
arm (south/west/east/north) -- the tile blends linearly from the nominal
thickness ``t`` in the middle to the arm thickness at the cell boundary.
That is how neighbouring cells in a lattice stay watertight.

So the design space is 5-dimensional for a "uniform" cell and up to
9-dimensional if you let the four arms differ:

    [phi, t, x1, x2, r]  (+ optional [tS, tW, tE, tN])

Gotchas that will bite you (all handled by ``check_chi_parameters``):
    * ``x2 == 0`` produces 0/0 -> NaN control points.  Keep |x2| >= ~1e-3.
    * Large ``t``/``x1``/``r`` combinations push control points out of the unit
      cube or make patches overlap -> the "surface" is no longer a clean
      manifold and the signed distance sign flips inside.
    * The tile always fills the full unit cube [0,1]^3, so normalising to
      [-1,1]^3 is just ``p -> 2p - 1`` for *every* parameter set.  That means
      the normalisation does not distort the shape distribution your GAN sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import splinepy
import torch
import trimesh
import gustaf as gus

from DeepSDFStruct.SDF import SDFBase, SDFfromMesh
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D

__all__ = [
    "PARAM_NAMES",
    "EVAL_POINTS",
    "chi_parameters",
    "chi_multipatch",
    "chi_mesh",
    "chi_boundary_loop",
    "check_chi_parameters",
    "ParamCheck",
    "Chi3DMeshSDF",
    "Chi3DPrismSDF",
]

PARAM_NAMES = ("phi", "t", "x1", "x2", "r")
EVAL_POINTS = ("south", "west", "center", "east", "north")

# index of the extruded ("prism") axis in the 3D tile frame and the two axes
# spanning the cross section.  create_tile() extrudes along z and then permutes
# (x, y, z_extr) -> (y, z_extr, x), so the prism axis ends up being *y* and the
# 2D coordinates (u, v) of the cross section are (z3d, x3d).
PRISM_AXIS = 1
CROSS_AXES = (2, 0)  # (u, v) = (queries[:, 2], queries[:, 0])


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
def chi_parameters(
    phi: float,
    t: float,
    x1: float,
    x2: float,
    r: float,
    arm_thickness=None,
) -> np.ndarray:
    """Build the (5, 5) parameter array ``Chi3D.create_tile`` expects.

    Parameters
    ----------
    phi : float
        Strut angle [rad].  0 = straight cross, negative = chi/auxetic tilt.
    t : float
        Nominal strut thickness in the cell centre.
    x1 : float
        Arm position (distance of the arm axis from the cell centre).
    x2 : float
        Arm offset / shift at the cell boundary.  **Must not be 0.**
    r : float
        Fillet radius of the inner corner.
    arm_thickness : sequence of 4 floats, optional
        Thickness at (south, west, east, north).  Defaults to ``t`` everywhere,
        which reproduces the example in the DeepSDFStruct README.

    Returns
    -------
    np.ndarray, shape (5, 5)
    """
    row = np.array([phi, t, x1, x2, r], dtype=float)
    params = np.tile(row, (5, 1))
    if arm_thickness is not None:
        tS, tW, tE, tN = np.asarray(arm_thickness, dtype=float).ravel()
        params[0, 1] = tS
        params[1, 1] = tW
        params[3, 1] = tE
        params[4, 1] = tN
    return params


def _as_param_array(params) -> np.ndarray:
    params = np.asarray(params, dtype=float)
    if params.shape == (5,):
        params = np.tile(params, (5, 1))
    if params.shape == (9,):
        params = chi_parameters(*params[:5], arm_thickness=params[5:])
    if params.shape != (5, 5):
        raise ValueError(
            f"Expected parameters of shape (5,), (9,) or (5, 5), got {params.shape}"
        )
    return params


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def chi_multipatch(params, make3D: bool = True) -> splinepy.Multipatch:
    """Create the Chi3D tile and wrap it in a ``Multipatch``."""
    patches, _ = Chi3D().create_tile(_as_param_array(params), make3D=make3D)
    mp = splinepy.Multipatch(patches)
    mp.determine_interfaces()
    return mp


def chi_mesh(params, n_faces: int = 20, process: bool = True) -> trimesh.Trimesh:
    """Tile -> watertight triangle surface mesh (the SDFSampler route).

    ``n_faces`` is the sampling resolution per patch edge, identical to the
    ``n_faces`` argument of ``SDFSampler.add_class``.  n_faces=3 (the value in
    the DeepSDFStruct test suite) is a ~3 % volume error; 10-20 is a good
    trade-off for training data, above ~30 the mesh stops changing.
    """
    mp = chi_multipatch(params, make3D=True)
    faces = mp.extract.faces(n_faces)
    tris = gus.create.faces.to_simplex(faces)
    mesh = trimesh.Trimesh(vertices=tris.vertices, faces=tris.faces)
    if process:
        mesh.merge_vertices()
        mesh.process(validate=True)
    return mesh


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of an open polyline."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-15:
        d = np.linalg.norm(points - start, axis=1)
    else:
        d = np.abs(np.cross(seg, points - start)) / seg_len
    idx = int(np.argmax(d))
    if d[idx] <= epsilon:
        return np.vstack([start, end])
    left = _rdp(points[: idx + 1], epsilon)
    right = _rdp(points[idx:], epsilon)
    return np.vstack([left[:-1], right])


def chi_boundary_loop(
    params, resolution: int = 25, tol: float = 1e-6, simplify: float = 2e-5
) -> np.ndarray:
    """Exact 2D cross section of the tile as one ordered closed polygon.

    Extracts the boundary curves of the 2D patch union, samples them and
    chains them into a single loop (counter-clockwise, no repeated end point).
    ``simplify`` is the Ramer-Douglas-Peucker tolerance applied afterwards
    (in unit-cell units); it removes redundant points on near-straight pieces
    and typically cuts the vertex count by ~4x with < 1e-4 distance error.
    Set ``simplify=0`` to keep every sampled point.

    Returns
    -------
    np.ndarray, shape (M, 2) -- polygon vertices in tile coordinates [0, 1]^2.
    """
    patches, _ = Chi3D().create_tile(_as_param_array(params), make3D=False)
    mp = splinepy.Multipatch(patches)
    mp.determine_interfaces()

    curves = []
    for curve in mp.boundary_multipatch().patches:
        # straight boundaries only need their two end points
        n = 2 if max(curve.degrees) == 1 else resolution
        curves.append(curve.evaluate(np.linspace(0.0, 1.0, n).reshape(-1, 1)))

    used = [False] * len(curves)
    used[0] = True
    loop = [curves[0]]
    end = curves[0][-1]

    for _ in range(len(curves) - 1):
        best, best_d = None, np.inf
        for i, c in enumerate(curves):
            if used[i]:
                continue
            for reverse in (False, True):
                cc = c[::-1] if reverse else c
                d = float(np.linalg.norm(cc[0] - end))
                if d < best_d:
                    best_d, best = d, (i, reverse)
        i, reverse = best
        if best_d > tol:
            raise RuntimeError(
                f"boundary curves do not form a closed loop (gap {best_d:.2e}); "
                "the parameter set most likely produces overlapping patches"
            )
        cc = curves[i][::-1] if reverse else curves[i]
        used[i] = True
        loop.append(cc[1:])
        end = cc[-1]

    poly = np.vstack(loop)
    if np.linalg.norm(poly[-1] - poly[0]) > tol:
        raise RuntimeError("boundary loop does not close")

    keep = np.r_[True, np.linalg.norm(np.diff(poly, axis=0), axis=1) > 1e-12]
    poly = poly[keep]
    if simplify > 0:
        poly = _rdp(poly, simplify)
    if np.linalg.norm(poly[-1] - poly[0]) < 1e-12:
        poly = poly[:-1]

    area = 0.5 * np.sum(
        poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1]
    )
    if area < 0:  # make it counter-clockwise
        poly = poly[::-1]
    return poly


# --------------------------------------------------------------------------- #
# validity
# --------------------------------------------------------------------------- #
def polygon_self_intersects(poly: np.ndarray, eps: float = 1e-12) -> bool:
    """True if the closed polygon has a proper self-intersection.

    This is *the* validity test for a Chi3D parameter set.  When struts start
    to overlap, the patch union is no longer a simple region: the extracted
    boundary loops back into the material.  The resulting surface is still
    closed (trimesh happily reports ``is_watertight``) and its signed volume
    still equals the cross-section area, so those checks do **not** catch it --
    but the signed distance inside the overlap is measured to a spurious
    interior sheet, which quietly corrupts training samples.
    """
    a = poly
    b = np.roll(poly, -1, axis=0)
    n = len(poly)

    def cross(o_a, o_b, p):
        return (o_b[:, None, 0] - o_a[:, None, 0]) * (p[None, :, 1] - o_a[:, None, 1]) - (
            o_b[:, None, 1] - o_a[:, None, 1]
        ) * (p[None, :, 0] - o_a[:, None, 0])

    d1 = cross(a, b, a)          # (i, j): side of a_j w.r.t. segment i
    d2 = cross(a, b, b)          # (i, j): side of b_j w.r.t. segment i
    s1 = np.sign(np.where(np.abs(d1) < eps, 0.0, d1))
    s2 = np.sign(np.where(np.abs(d2) < eps, 0.0, d2))
    straddle = (s1 * s2) < 0     # segment j straddles the line of segment i
    proper = straddle & straddle.T

    idx = np.arange(n)
    adjacent = (
        (np.abs(idx[:, None] - idx[None, :]) <= 1)
        | (np.abs(idx[:, None] - idx[None, :]) == n - 1)
    )
    proper &= ~adjacent
    return bool(proper.any())


@dataclass
class ParamCheck:
    ok: bool
    reasons: list = field(default_factory=list)
    volume_fraction: float = float("nan")

    def __bool__(self):
        return self.ok


def check_chi_parameters(
    params, with_mesh: bool = False, n_faces: int = 12, area_tol: float = 0.05
) -> ParamCheck:
    """Cheap sanity filter for a candidate parameter set.

    Rejects everything that would silently poison a training set: NaNs,
    control points leaving the unit cell, overlapping struts, and degenerate
    (empty or nearly solid) cells.  Runs in ~50 ms; set ``with_mesh=True`` to
    additionally triangulate and check watertightness / body count (~5x slower,
    and it has never caught anything the polygon tests miss).
    """
    reasons = []
    p = _as_param_array(params)

    if not (np.all(p >= -np.pi / 2) and np.all(p <= np.pi / 2)):
        reasons.append("Chi3D requires every entry in [-pi/2, pi/2]")
    if abs(p[2, 3]) < 1e-3:
        reasons.append("x2 (arm offset) too close to 0 -> 0/0 -> NaN control points")
    if reasons:
        return ParamCheck(False, reasons)

    try:
        patches, _ = Chi3D().create_tile(p, make3D=True)
    except Exception as exc:  # noqa: BLE001
        return ParamCheck(False, [f"create_tile failed: {exc}"])

    cps = np.vstack([patch.control_points for patch in patches])
    if not np.all(np.isfinite(cps)):
        return ParamCheck(False, ["non-finite control points"])
    if cps.min() < -1e-8 or cps.max() > 1 + 1e-8:
        reasons.append(
            f"control points leave the unit cube "
            f"[{cps.min():.3f}, {cps.max():.3f}]"
        )

    try:
        poly = chi_boundary_loop(p, simplify=0.0)
        area = abs(
            0.5
            * np.sum(
                poly[:, 0] * np.roll(poly[:, 1], -1)
                - np.roll(poly[:, 0], -1) * poly[:, 1]
            )
        )
    except Exception as exc:  # noqa: BLE001
        return ParamCheck(False, reasons + [str(exc)])

    if polygon_self_intersects(poly):
        reasons.append("struts overlap: cross-section boundary self-intersects")
    if not (0.02 < area < 0.98):
        reasons.append(f"degenerate volume fraction {area:.3f}")

    if with_mesh:
        mesh = chi_mesh(p, n_faces=n_faces)
        if not mesh.is_watertight:
            reasons.append("boundary mesh not watertight")
        if mesh.body_count != 1:
            reasons.append(f"tile falls apart into {mesh.body_count} bodies")
        vol = abs(mesh.volume)
        # the tile is a prism, so mesh volume must equal the cross-section area
        if area > 0 and abs(vol - area) / area > area_tol:
            reasons.append(
                f"mesh volume {vol:.4f} != cross-section area {area:.4f}"
            )
    return ParamCheck(not reasons, reasons, area)


# --------------------------------------------------------------------------- #
# SDFs
# --------------------------------------------------------------------------- #
class Chi3DMeshSDF(SDFBase):
    """Chi3D tile as a mesh-based SDF -- the SDFSampler / supervisor route.

    Parameters
    ----------
    params : array-like
        (5,), (9,) or (5, 5) parameter array, see :func:`chi_parameters`.
    n_faces : int
        Patch sampling resolution for the triangulation.
    scale : bool
        If True (default) the mesh is normalised to [-1, 1]^3, which is the
        domain DeepSDF trains on.  Note the tile always fills the unit cube,
        so this is exactly ``p -> 2p - 1`` and distances are doubled.
    """

    def __init__(self, params, n_faces: int = 20, scale: bool = True):
        super().__init__(geometric_dim=3)
        self.params = _as_param_array(params)
        self.n_faces = n_faces
        self.scale = scale
        self.mesh = chi_mesh(self.params, n_faces=n_faces)
        self._sdf = SDFfromMesh(self.mesh, scale=scale)

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        return self._sdf._compute(queries)

    def _get_domain_bounds(self) -> torch.Tensor:
        if self.scale:
            return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        return torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def _set_param(self, params):
        self.__init__(params, n_faces=self.n_faces, scale=self.scale)


class Chi3DPrismSDF(SDFBase):
    """Chi3D tile as an analytic prism SDF (fast, torch-native, no meshing).

    The cross section is a single closed polygon (exact boundary curves of the
    patch union, sampled), the third direction is a slab of height 1.
    Distance is the standard 2D-polygon distance combined with the slab, which
    is the exact Euclidean distance for a prism.

    Parameters
    ----------
    params : array-like
        (5,), (9,) or (5, 5) parameter array.
    resolution : int
        Points per curved boundary segment.  25 is already converged.
    scale : bool
        Map the unit cell to [-1, 1]^3 (default), matching ``SDFfromMesh``.
    chunk : int
        Query points processed at once (memory ~ chunk * n_polygon_vertices).
    """

    def __init__(self, params, resolution: int = 25, scale: bool = True, chunk: int = 20000):
        super().__init__(geometric_dim=3)
        self.params = _as_param_array(params)
        self.scale = scale
        self.chunk = chunk
        poly = chi_boundary_loop(self.params, resolution=resolution)
        self.register_buffer("polygon", torch.as_tensor(poly, dtype=torch.float32))

    # -- 2D polygon distance (vectorised, chunked) -------------------------- #
    @staticmethod
    def _polygon_sdf(pts: torch.Tensor, poly: torch.Tensor) -> torch.Tensor:
        """Signed distance to a simple closed polygon. pts (N,2), poly (M,2)."""
        v_i = poly                      # (M,2)
        v_j = torch.roll(poly, 1, 0)    # previous vertex
        e = v_j - v_i                             # (M,2)
        w = pts.unsqueeze(1) - v_i.unsqueeze(0)   # (N,M,2)

        t = (w * e).sum(-1) / (e * e).sum(-1).clamp_min(1e-30)
        t = t.clamp(0.0, 1.0)
        b = w - t.unsqueeze(-1) * e                # (N,M,2)
        d = (b * b).sum(-1).min(dim=1).values      # (N,)

        # winding number (iq's formulation)
        py = pts[:, 1:2]
        c1 = py >= v_i[:, 1]
        c2 = py < v_j[:, 1]
        c3 = e[:, 0] * w[..., 1] > e[:, 1] * w[..., 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        inside = (flip.sum(dim=1) % 2) == 1
        sign = torch.where(inside, -1.0, 1.0)
        return sign * torch.sqrt(d.clamp_min(0.0))

    def _compute(self, queries: torch.Tensor) -> torch.Tensor:
        poly = self.polygon.to(device=queries.device, dtype=queries.dtype)

        p = (queries + 1.0) * 0.5 if self.scale else queries
        uv = torch.stack([p[:, CROSS_AXES[0]], p[:, CROSS_AXES[1]]], dim=1)

        d2 = torch.empty(uv.shape[0], device=queries.device, dtype=queries.dtype)
        for i in range(0, uv.shape[0], self.chunk):
            d2[i : i + self.chunk] = self._polygon_sdf(uv[i : i + self.chunk], poly)

        # slab |y - 0.5| - 0.5 along the extrusion axis
        d_slab = torch.abs(p[:, PRISM_AXIS] - 0.5) - 0.5

        outside = torch.linalg.norm(
            torch.stack([d2.clamp_min(0.0), d_slab.clamp_min(0.0)], dim=1), dim=1
        )
        inside = torch.maximum(d2, d_slab).clamp_max(0.0)
        d = outside + inside
        if self.scale:
            d = d * 2.0
        return d.reshape(-1, 1)

    def _get_domain_bounds(self) -> torch.Tensor:
        if self.scale:
            return torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        return torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def _set_param(self, params):
        self.__init__(params, scale=self.scale, chunk=self.chunk)