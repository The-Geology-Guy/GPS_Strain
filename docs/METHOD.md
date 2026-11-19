# The GPS triangle strain method: math and geophysics

This document explains what the package computes and why. It draws on Cronin & Resor's *Algorithm for triangle strain* (v. 5 Apr 2021), Cronin's *Primer on infinitesimal strain analysis in 1, 2 and 3-D* (13 Jun 2021), the SERC/GETSI module *GPS, Strain, and Earthquakes*, and the geodetic strain literature cited at the end. Detailed research notes with every source URL are in `docs/research/`.

---

## 1. Physical picture

GNSS stations move a few mm/yr to a few cm/yr, mostly as the rigid motion of a tectonic plate. Earthquakes are driven by the *differences* in motion between nearby points: the slow distortion (strain) of crust between locked faults. Three stations forming a triangle tens of km across tell us, on average, how that piece of crust is stretching, shortening, shearing and rotating.

Assumptions:

1. **Uniform (homogeneous) deformation** inside the triangle: velocity varies linearly with position.
2. **Infinitesimal strain:** rates are ~10⁻⁸/yr, so second-order terms are negligible and strains add linearly.
3. **Horizontal plane strain:** vertical motions are ignored.
4. **Tectonic, steady (interseismic) signal:** postseismic transients, magmatic inflation, groundwater and landslides all bias the result.

## 2. The velocity-gradient model

Origin at the triangle centroid; x = east, y = north, in metres:

$$
\mathbf v(\mathbf x) = \mathbf t + \mathbf L\,\mathbf x ,\qquad
\mathbf L = \begin{bmatrix}\dot\varepsilon_{xx} & \dot\varepsilon_{xy}\\ \dot\varepsilon_{xy} & \dot\varepsilon_{yy}\end{bmatrix}
+ \begin{bmatrix}0 & -\omega\\ \omega & 0\end{bmatrix}
$$

- **t**: translation of the centroid. With centred coordinates it equals the average station velocity.
- **ω**: vertical-axis rotation rate. **Positive means counter-clockwise.**
- **ε̇xy is the tensor shear strain rate**: half the engineering shear γ̇xy.

Per station i (Cronin & Resor):

```
vE_i = tx − y_i ω + x_i εxx + y_i εxy
vN_i = ty + x_i ω + x_i εxy + y_i εyy
```

Stacking the stations gives **d = G m**, with rows East `[1 0 −y x y 0]` and North `[0 1 x 0 x y]`, and m = [tx, ty, ω, εxx, εxy, εyy]. Three non-collinear stations determine m exactly; more stations are solved by weighted least squares (§5). Centring (J. C. Savage's suggestion) makes **t** the velocity of the triangle centre and keeps the system well conditioned.

## 3. From the tensor to the quantities on the result sheet

Let a = ½(ε̇xx − ε̇yy), b = ε̇xy and r = √(a² + b²).

| Quantity | Formula | Meaning |
|---|---|---|
| Principal rates | e₁ = ½(ε̇xx+ε̇yy) + r, e₂ = ½(ε̇xx+ε̇yy) − r | e1H max horizontal extension rate; e2H min (most negative = fastest shortening) |
| Direction of e₁ | θ = ½·atan2(2ε̇xy, ε̇xx − ε̇yy) from east (CCW); azimuth = (90° − θ) mod 180° | S1H axis, reported as "57° (or 237°)"; S2H at +90° |
| Max shear strain rate | e₁ − e₂ = 2r | Engineering shear (GETSI definition), on planes at 45° to S1H. Some codes (e.g. Strain_2D) report r. |
| Area (dilatational) strain | e₁ + e₂ = I₁ | Negative means the area is shrinking |
| Second invariant | I₂ = √(ε̇xx² + ε̇yy² + 2ε̇xy²) | Overall magnitude (Kreemer et al. 2014; Cronin's definition) |
| Determinant | I₃ = e₁e₂ | |
| Style | (e₁+e₂)/max(\|e₁\|,\|e₂\|) | −1 shortening, 0 strike-slip-like, +1 extension |

**Units.** ×10⁹ gives **nanostrain/yr** (1 nstrain/yr ≈ 1 mm/yr between points 1000 km apart) and **nanoradians/yr**.

**Geologic reading (GETSI).** Contraction axes (black arrows) are roughly perpendicular to active thrusts and fold hinges. Extension axes (white arrows) are roughly perpendicular to active normal faults. Strike-slip faults lie near the max-shear planes, about 45° from the principal axes.

## 4. The strain ellipse

A unit circle deformed by **F = I + k·E**, with E the strain-rate tensor, becomes an ellipse with semi-axes 1 + k·e₁ (S1H, red) and 1 + k·e₂ (S2H, blue). Rates are ~10⁻⁸/yr, so the drawing is exaggerated by k, which is effectively years of deformation. The package picks k so the largest axis changes by 25%. The 2020 code's ellipse was not reliable (degrees inside sin/cos, double-squared tensor).

## 5. Uncertainty

Each station supplies σE and σN, and optionally the E–N correlation ρ. With data covariance C_d (block-diagonal 2×2 per station) and W = C_d⁻¹:

$$\hat{\mathbf m} = (G^\top W G)^{-1} G^\top W\,\mathbf d,\qquad C_m = (G^\top W G)^{-1}$$

The 1σ of the six parameters is √diag(C_m) (Cronin & Resor Step 16; their worked example confirms the printed values are square roots).

Principal values and azimuths are non-linear in m, and the azimuth is undefined when e₁ ≈ e₂. So the package draws 20,000 samples from N(m̂, C_m) and reports:

- standard deviations of e₁, e₂, max shear and area strain;
- an **axial** circular standard deviation for the S1H azimuth.

With more than three stations, the reduced χ² tests whether one uniform strain fits.

**Where σ comes from matters more than the algebra.** Daily GNSS has coloured (flicker) noise, so white-noise σ's from a simple regression are too small by ~5–10×. MIDAS (Blewitt et al. 2016) and Hector-type analyses (Bos et al. 2013) give realistic values.

## 6. Coordinates: classic UTM vs local tangent plane

**Classic (`frame="utm"`)** reproduces Cronin's spreadsheet:

- WGS84 → UTM with the USGS/Army K1–K5 series and k₀ = 0.9996; it agrees with the exact Krüger series to < 0.5 m here.
- All stations go on the central meridian of the **westernmost** zone.
- Velocities are used as published.

**Default (`frame="enu"`):**

- Geodetic → ECEF → local East/North/Up at the network centre.
- Each station's velocity and covariance are **rotated into the centre's east/north axes**.

**Why the default changed.** Local north at two stations differs by ≈ Δλ·sin φ (meridian convergence). Mixing velocities from different local frames on one planar grid creates a spurious gradient of order |v|·tan φ / R.

- Cronin example: ≈ 1.2 nstrain/yr, about 1σ.
- Pure rigid plate rotation at 43 mm/yr: the classic method shows ~4.9 nstrain/yr of fake strain; ENU shows < 0.04 nstrain/yr. The correct answer is zero.
- The bias is worst with global-frame (IGS20) velocities, another reason to use plate-fixed ones.

UTM scale distortion (≤ 0.1% of the strain) is negligible either way.

## 7. Velocities: what goes in

Inputs must be **station velocities**: slopes of the position time series, in a plate-fixed frame, with honest uncertainties. Supported sources:

1. **NGL MIDAS** (default): free, global, weekly, IGS20 or plate-fixed. Robust median-of-pairs slopes with realistic σ.
2. **Your own table**, e.g. velocities copied from EarthScope (formerly UNAVCO) station pages in NAM14, as in the GETSI exercises.
3. **A fitted time series** (NGL `.tenv3`) over a chosen window with `data.fit_velocity`: offset + rate + annual + semi-annual + optional steps.

The 2020 code divided the *mean* displacement by the duration (≈ half the true rate) and hard-coded σ = 0.01 mm/yr. Both are fixed.

## 8. Choosing a good triangle

- No interior angle below ~30° (Cronin & Resor). Skinny triangles amplify noise. The package warns.
- Size comparable to the structure of interest; tens of km is typical.
- Avoid stations with non-tectonic motion, and time windows spanning big earthquakes unless that is the study.
- Stations within about one locking depth of a fault sample a strongly non-uniform field; the result is then an average.

## 9. References

- Cronin, V. S., & Resor, P. G. (2021). *Algorithm for triangle strain*. https://croninprojects.org/Vince/Geodesy/TriangleStrainAlgorithm.pdf
- Cronin, V. S. (2021). *Primer on infinitesimal strain analysis in 1, 2 and 3-D*. https://croninprojects.org/Vince/Geodesy/GPS_Strain_Primer-20210613.pdf
- *GPS, Strain, and Earthquakes* (GETSI module), SERC/Carleton. https://serc.carleton.edu/getsi/teaching_materials/gps_strain/index.html
- Blewitt, Hammond & Kreemer (2018). *Eos* 99. doi:10.1029/2018EO104623
- Blewitt, Kreemer, Hammond & Gazeaux (2016). MIDAS. *JGR Solid Earth* 121. doi:10.1002/2015JB012552
- Kreemer, Blewitt & Klein (2014). Global Strain Rate Model. *G-cubed* 15. doi:10.1002/2014GC005407
- Savage, Gan & Svarc (2001). *JGR* 106(B10).
- Feigl, King & Jordan (1990). *JGR* 95(B3).
- Shen, Wang, Zeng & Wang (2015). *BSSA* 105(4).
- Cardozo & Allmendinger (2009). SSPX. *Computers & Geosciences* 35(6), 1343–1357.
- Maurer & Materna (2023). Strain_2D. *GJI*.
- Williams (2003). *J. Geodesy* 76.
- Bos, Fernandes, Williams & Bastos (2013). *J. Geodesy* 87.
- Worrell (2016). SLAM applied to the South Napa earthquake. MS thesis, Baylor University.
