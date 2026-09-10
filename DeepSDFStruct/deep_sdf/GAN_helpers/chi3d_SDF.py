import torch
import numpy as np
import splinepy

from DeepSDFStruct.SDF import SDFBase, SDFfromMesh
from DeepSDFStruct.splinepy_unitcells.chi_3D import Chi3D

PARAM_NAMES = ("phi", "t", "x1", "x2", "r")
EVAL_POINTS = ("south", "west", "center", "east", "north")


PRISM_AXIS = 1
CROSS_AXES = (2, 0)  # (u, v) = (queries[:, 2], queries[:, 0])

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
    if isinstance(params, torch.Tensor):
        params = params.detach().cpu().numpy()
    else:
        params = np.asarray(params, dtype=float)

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