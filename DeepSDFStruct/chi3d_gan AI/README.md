# Chi3D → SDF → training samples

Everything needed to turn the splinepy `Chi3D` unit cell into a signed distance
function you can sample, the same way the `CrossMsSDF` radius-only dataset was
made — but with 5 (or 9) shape parameters.

## 1. How the existing samples were made

`Chi3D` is not an SDF and never was one. The sampling route in
[DeepSDFStruct](https://github.com/mkofler96/DeepSDFStruct) (Kofler, TU Wien —
the package behind the paper) is:

```
Chi3D().create_tile(params)      # 13 trivariate Bézier patches (a solid, not a surface)
  → splinepy.Multipatch(...)     # patches + interfaces
  → mp.extract.faces(n_faces)    # OUTER boundary only, interior faces dropped
  → gus.create.faces.to_simplex  # quads → triangles
  → trimesh.Trimesh              # watertight, winding-consistent
  → SDFfromMesh(mesh, scale=True)# libigl signed_distance, mesh normalised to [-1,1]³
  → random_sample_sdf(...)       # 100 000 uniform points in [-1,1]³
  + sample_mesh_surface(...)     # 2 × 50 000 points on the surface, pushed along the
                                 #   face normal by N(0, 0.05) and N(0, 0.025)
  → split by sign → np.savez(name.npz, pos=[x,y,z,d], neg=[x,y,z,d])
```

That is literally what `DeepSDFStruct.sampling.SDFSampler` does — `add_class()`
accepts a `Multipatch` and does the meshing, `process_geometries()` does the
sampling. The README of DeepSDFStruct has the exact snippet, and it uses
`Chi3D` as its example (with `[[phi, t, x1, x2, r]] * 5`). The pretrained model
shipped as `PretrainedModels.ChiAndCross` was trained on exactly this.

So the answer to "how did he make the samples": **he never had an analytic SDF
for the chi cell either — he meshed the spline patches and used a mesh SDF.**

His published dataset is on Hugging Face as `mkofler/lattice_structure_unit_cells`
(`repo_type="dataset"`) if you want to diff your `.npz` files against his.

## 2. Files here

| file | what it does |
|---|---|
| `chi3d_sdf.py` | `Chi3DMeshSDF` and `Chi3DPrismSDF` (both `SDFBase`), tile→mesh helpers, parameter validity check |
| `generate_chi3d_dataset.py` | Sobol/LHS design of experiments + rejection + `SDFSampler` → `.npz` dataset + `parameters.json` |
| `explore_parameter_ranges.py` | prints usable ranges per parameter, draws `chi3d_param_sweep.png` |
| `validate_chi3d_sdf.py` | cross-checks the two SDFs, marching-cubes round trip, draws the slice + gallery figures |

Quick start:

```python
import numpy as np, torch
from chi3d_sdf import Chi3DMeshSDF, Chi3DPrismSDF, check_chi_parameters

params = [-np.pi/8, 0.10, 0.20, 0.10, 0.05]      # phi, t, x1, x2, r
print(check_chi_parameters(params))               # ParamCheck(ok=True, ...)

sdf = Chi3DMeshSDF(params, n_faces=20)            # supervisor's route
d   = sdf(torch.rand(1000, 3) * 2 - 1)            # (1000, 1) signed distances
sdf.plot_slice(normal=(0, 1, 0))                  # the chi cross section
```

Generate a dataset:

```bash
python generate_chi3d_dataset.py --n 500 --out ./training_data \
       --dataset-name chi3d_multiparam --n-samples 100000 --n-faces 20 --workers 8
```

then train unchanged: `train_deep_sdf(exp_dir, "./training_data")`.

## 3. The parameters

`create_tile` wants a **(5, 5)** array: one row per evaluation point, in the
order `south, west, center, east, north`, and 5 numbers per row
`[phi, t, x1, x2, r]` (`Winkel, Dicke, Armposition, Armpositionsverschiebung,
Rundungsradius`).

* Only the **center row** supplies `phi, x1, x2, r` and the nominal thickness `t`.
* Column 1 of the **other four rows** is the thickness of that arm. The tile
  blends linearly from `t` in the middle to the arm thickness at the cell face —
  that is how neighbouring cells in a lattice stay connected.

So your design space is 5-dimensional with uniform arms, or **up to 9-dimensional**
if you let `tS, tW, tE, tN` differ — `chi_parameters(phi, t, x1, x2, r,
arm_thickness=(tS, tW, tE, tN))` builds that array. Nine parameters with four
of them controlling *directional* stiffness is a much more interesting GAN
target than one radius, and it is the same knob the paper's lattice
parametrisation turns.

| parameter | meaning | usable range (1D, around the reference cell) |
|---|---|---|
| `phi` | strut angle [rad], 0 = straight cross, negative = chi tilt | −0.75 … +0.45 |
| `t` | nominal strut thickness | 0.02 … 0.185 |
| `x1` | arm position | 0.136 … 0.307 |
| `x2` | arm offset at the cell face | −0.237 … +0.35, **never 0** |
| `r` | inner fillet radius | 0.005 … 0.133, **never 0** |

`DEFAULT_RANGES` in the generator is slightly wider than these 1D limits on
purpose: the parameters interact, so favourable combinations outside a 1D limit
are still valid, and the rejection test keeps the rest out. Acceptance over the
default box is ~50 %, giving volume fractions from ~0.10 to ~0.58.

## 4. Traps

**`x2 = 0` gives NaN.** The tile code divides by `3*x2/(...)` in a term whose
numerator also goes to zero — mathematically the limit is finite, numerically it
is `0/0`. You get silent `nan` control points, and splinepy then dies inside
`merge_vertices` with "data must be finite". Keep `|x2| ≥ 1e-3`.

**`r = 0` breaks the interface detection** ("Found conflicting interceptions").

**Overlapping struts are the dangerous one.** When `t` gets large relative to
`x1`, or `r` gets large, the patches start to overlap. The extracted boundary
then loops back *into* the material. The mesh is still closed, so
`mesh.is_watertight` is `True`, and the signed volume still equals the
cross-section area — both of the obvious sanity checks pass — but libigl now
measures the distance to a spurious interior sheet, and those samples teach the
network a surface that is not there. Roughly half of a naive box sample is like
this. `check_chi_parameters` catches it by testing the 2D boundary polygon for
proper self-intersections; the red cells in `chi3d_param_sweep.png` are exactly
these cases.

**`n_faces` matters.** The test suite in DeepSDFStruct uses `n_faces=3`, which
is a ~3 % volume error — fine for a unit test, not for training data. Volume
converges by ~20. Use `n_faces=20`.

**`SDFfromMesh(mesh, scale=True)` mutates the mesh in place.** Its `self.mesh`
is your mesh, already normalised to [-1,1]³ — don't be surprised when its
`.volume` is 8× what you computed.

**Normalisation is shape-independent here.** The Chi3D tile always fills the
whole unit cube (that is asserted in the package's own tests), so
`normalize_mesh_to_unit_cube` is always exactly `p → 2p − 1`. Unlike a set of
arbitrary meshes, your parameter sweep is therefore *not* distorted by
per-shape rescaling — the GAN sees the geometry you asked for.

## 5. Conditioning the GAN on the parameters

The `.npz` only holds points and distances; instance identity is the filename.
`generate_chi3d_dataset.py` therefore also writes
`training_data/<dataset>/parameters.json` and `parameters.csv`:

```json
{ "Chi3D_00000": {"phi": -0.6649, "t": 0.0913, "x1": 0.2456, "x2": 0.1599, "r": 0.1070}, ... }
```

Load that alongside the split json and you have `(parameter vector, SDF samples)`
pairs. If your dataloader already reads the `C` field of the npz (the
homogenised elasticity tensor slot), `--store-params-as-C` writes the parameter
vector there too — convenient, but it collides with the homogenisation pipeline,
so the side-car file is the cleaner option.

## 6. The second SDF, and why it exists

`Chi3DPrismSDF` computes the same field without meshing. The chi tile is a
**prism**: `create_tile` builds a 2D cross section and extrudes it, then permutes
the axes so the extrusion runs along **y** and the cross section lives in the
**x–z** plane. So the exact distance is
`combine(2D polygon distance in (z, x), slab |y − ½| − ½)`, where the polygon is
the sampled boundary curve of the patch union.

Verified against the mesh SDF over 100 000 random points:

```
max |mesh − prism|   : 3.4e-04      (domain is [-1,1]^3)
mean |mesh − prism|  : 7.2e-05
sign agreement       : 0.99987      (disagreements are points ~on the surface)
|prism SDF| on the mesh surface : max 3.5e-04
marching cubes on the prism SDF, evaluated with the mesh SDF : max 2.5e-04
```

Two completely different routes agreeing to 3e-4 is the real proof that both are
right. Use it for:

* on-the-fly ground truth during GAN training (no meshing, batched, runs on the
  GPU — the mesh route is a CPU-only libigl call),
* a reference to score your generator against at arbitrary points,
* denser/cheaper resampling than a fixed `.npz` allows.

It is *not* differentiable with respect to the tile parameters (the control
points come out of numpy/splinepy) — only with respect to the query points.
Use the mesh route for the dataset if you want to stay bit-compatible with your
supervisor's data.

## 7. Figures

* `chi3d_param_sweep.png` — cross section vs. each parameter, accepted/rejected.
* `chi3d_arm_thickness.png` — the four arms carrying independent thickness.
* `chi3d_sdf_slices.png` — the SDF itself, mesh vs. analytic, three slices.
* `chi3d_gallery.png` — marching cubes on the SDF for eight parameter sets.
