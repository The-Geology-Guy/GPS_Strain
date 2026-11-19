# Audit of the 2020 `GPS_Strain.py`

The 2020 code is kept, unchanged, at `legacy/GPS_Strain_2020.py`. Findings are grouped by severity. Line numbers refer to that file.

## A. Why it no longer runs

| # | Problem | Where |
|---|---|---|
| A1 | `pip install` fails. `install_requires` lists `math`, `io`, `pylab` and `mpl_toolkits`, which are standard-library modules or parts of matplotlib, not installable packages. | `setup.py` |
| A2 | `DataFrame.append` was removed in pandas 2.0, so building the design matrix crashes. | 353 |
| A3 | `cimgt.Stamen(...)` basemaps. Stamen tiles moved to Stadia Maps in 2023, which needs an API key; its free tier is non-commercial and does not allow printing. | 647, 802, 1147, 1236 |
| A4 | The UNAVCO logo is downloaded from a URL on every plot. UNAVCO is now EarthScope, so the URL is fragile. | 1174, 1258 |
| A5 | Hard-coded fonts (`Georgia`, `Arial`) are missing on Linux, which produces warnings and fallback fonts. | 23, 713 |
| A6 | `V is not 'off'` compares identity with a string literal: a SyntaxWarning now, an error in future Pythons. | 754 |
| A7 | `from pylab import *` pollutes the namespace. Bare names like `array`, `dot` and `svd` depend on it. | 10 |

## B. Scientific errors (results were wrong even when it ran)

| # | Problem | Effect |
|---|---|---|
| B1 | **The velocity was computed as `mean(ΔE) / years`** (lines 61–65). The mean of displacements relative to the first epoch is about half the total displacement, and `relativedelta(...).years` truncates to whole years. | Velocities biased low by roughly 2×, more for short spans. Every strain value scales with this error. The correct velocity is the slope of a fit: published station velocities, MIDAS, or `data.fit_velocity`. |
| B2 | **Uncertainties hard-coded to 0.01 mm/yr** (lines 64, 66). | Formal uncertainties of every output were meaningless; weighting did nothing. |
| B3 | Translation uncertainty reported as `M9[0][0]`, a variance, not its square root. | Translation σ wrong by orders of magnitude. |
| B4 | The ellipse used `theta = ω·180/π` (degrees) inside `np.cos/np.sin` (lines 548–549), scaled `R·S` by 10⁶, then formed `F·Fᵀ` and `sqrtm` again. | The drawn ellipse orientation and shape were not the strain ellipse. |
| B5 | Westernmost UTM zone taken as `floor(mean(zones))`; the antimeridian branch used `abs(lon) − cm`. | Wrong zone for groups spanning more than two zones, or near ±180°. |
| B6 | Eigenvector formula `(1, (λ−exx)/exy)` divides by `exy` (line 434). | NaN for strain aligned with the axes (`exy = 0`), a common case. |
| B7 | Site loop fixed at `range(3)`. | Cannot use more than three stations. |

## C. Limits of the classic method (kept as `frame="utm"`)

* **Meridian convergence.** Velocities are given in each station's own local east/north frame, but positions are on one planar grid. Across a triangle, local north differs by about Δλ·sin φ, which leaks a spurious velocity gradient of order |v|·tan φ / R.
  * In the Cronin example, ENU vs UTM differ by about 1.2 nstrain/yr in e2H and 1.2 nrad/yr in rotation (about 1σ).
  * For a pure rigid plate rotation (≈43 mm/yr, PA–NA-like Euler pole) the classic method reports **≈4.9 nstrain/yr of spurious extension**. The local-ENU frame reports **< 0.04 nstrain/yr** (`tests/test_strain.py::test_rigid_plate_gives_no_strain_in_enu_mode`).
  * The error grows with velocity magnitude, so it matters most with global-frame (IGS) velocities.
* **UTM scale factor:** about 0.1% of the strain. Negligible.
* **Uncertainties:** classically only the six tensor components get σ. The new code adds Monte Carlo 1σ for e1H, e2H, the S1H azimuth, max shear and area strain.

## What the new code reproduces exactly

Cronin & Resor's worked example (TriangleStrainAlgorithm.pdf, 5 Apr 2021: P146, P149, P150). Every printed intermediate and result matches to the printed precision: parameter vector, σ vector, e1H/e2H, azimuths, max shear, area strain, I2, I3, rotation, translation speed and azimuth. See `tests/test_strain.py`.
