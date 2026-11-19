# 02 — Mathematics of horizontal strain-rate tensors from GNSS velocities

Foundations for rebuilding the GETSI "GPS triangle strain" calculator (Vince Cronin, Baylor; Phil Resor, Wesleyan) as a modern Python package.

Compiled 2026-10-14. Items marked **[derived]** are standard results written out here, not quoted from a source. Items marked **[unverified]** could not be checked against the primary document; the reason is given each time.

---

## 0. What the GETSI module actually does (source facts)

- **Module.** GETSI "GPS, Strain, and Earthquakes", Unit 4: *GPS and infinitesimal strain analysis* (Cronin & Resor, published 2016-12-09, files updated 2024-08-15).
  - Unit page: https://serc.carleton.edu/getsi/teaching_materials/gps_strain/unit4.html
  - Module home: https://serc.carleton.edu/getsi/teaching_materials/gps_strain/index.html
- **The problem.** Three Plate Boundary Observatory (now NOTA) stations form a triangle. They give six known velocity components and six unknowns (2 translation, 1 rotation, 3 strain-rate components). The system is "perfectly constrained" (Resor, Cronin, Hammond, Kreemer, Olds, Pratt-Sitaula & West, GSA 2012 abstract: https://gsa.confex.com/gsa/2012AM/webprogram/Paper210930.html).
- **Algorithm document.** Cronin, *GPS Strain Triangle Algorithm* (v3, .docx, 2023), https://cdn.serc.carleton.edu/files/getsi/teaching_materials/gps_strain/gps_strain_triangle_algorithm.v3.docx (landing page https://serc.carleton.edu/details/files/77379.html).
  - **[unverified]** The fetch tool returned only binary data for this .docx, so the exact matrix layout in it was not checked. Download it and read it before freezing the conventions described below.
- **Output explanation.** *Explanation of GPS Strain Calculator output* (v4 PDF): https://cdn.serc.carleton.edu/files/getsi/teaching_materials/gps_strain/explanation_gps_strain_calculator.v4.pdf. It says:
  - Translation is the **average** of the three E and N velocities, in m/yr, with an azimuth measured clockwise from north and a speed.
  - Rotation is reported in deg/yr and nanoradians/yr.
  - Principal horizontal extensions are e1H ≥ e2H, in nanostrain, along the S1H and S2H axes.
  - **Maximum shear strain = e1H − e2H**, i.e. the *engineering* shear, oriented at 45° to S1H.
  - **Area strain = e1H + e2H** (the dilatation).
- **Student worksheet.** The worksheet uses "Rotation direction (+ = counter clockwise, − = clockwise)" and "(+ = extension, − = contraction)" (https://cdn.serc.carleton.edu/files/getsi/teaching_materials/gps_strain/unit_4_gps_infinitesimal.v7.pdf).
- **Triangle-selection rule of thumb.** All interior angles should exceed about 30°, and no side should exceed about 100 km (Unit 6: https://serc.carleton.edu/getsi/teaching_materials/gps_strain/unit6.html).
- **Existing implementations.** There is an Excel calculator (with a simple strain-ellipse visualizer), a roughly 4 kB MATLAB zip by Resor, and a Wolfram CDF ellipse visualizer.
- **Licence.** SERC lists these files under **CC BY-NC-SA** (3.0 for the files, 4.0 for the page text). Copying text or figures into a redistributed package carries the NonCommercial/ShareAlike terms. The mathematics itself is not copyrightable, so a clean-room implementation avoids the issue.

---

## 1. Classic formulation

### 1.1 Velocity gradient, strain rate, rotation rate

Work in a local Cartesian frame with $x$ = East and $y$ = North. Assume the velocity field is linear (homogeneous deformation) over the region:

$$
\mathbf v(\mathbf x) = \mathbf t + \mathbf L\,\mathbf x,\qquad
\mathbf L = \frac{\partial \mathbf v}{\partial \mathbf x} =
\begin{pmatrix} \partial v_x/\partial x & \partial v_x/\partial y\\ \partial v_y/\partial x & \partial v_y/\partial y\end{pmatrix}
=\begin{pmatrix}L_{11}&L_{12}\\L_{21}&L_{22}\end{pmatrix}.
$$

Additive decomposition, valid for infinitesimal strain or for rates:

$$
\mathbf L = \dot{\boldsymbol\varepsilon} + \mathbf W,\qquad
\dot{\boldsymbol\varepsilon}=\tfrac12(\mathbf L+\mathbf L^T),\qquad
\mathbf W=\tfrac12(\mathbf L-\mathbf L^T)=\begin{pmatrix}0&-\omega\\ \omega&0\end{pmatrix}.
$$

The components are

$$
\dot\varepsilon_{xx}=L_{11},\quad \dot\varepsilon_{yy}=L_{22},\quad
\dot\varepsilon_{xy}=\tfrac12(L_{12}+L_{21}),\quad
\omega=\tfrac12(L_{21}-L_{12}) = \tfrac12\left(\frac{\partial v_y}{\partial x}-\frac{\partial v_x}{\partial y}\right).
$$

With $x$ = E and $y$ = N, $\omega>0$ is **counter-clockwise** rotation seen from above. This matches the GETSI worksheet and Strain_2D (`rot = 0.5*(dvdx - dudy)`).

### 1.2 Cronin parameterization $[t_x,t_y,\omega,\dot\varepsilon_{xx},\dot\varepsilon_{xy},\dot\varepsilon_{yy}]$ **[derived]**

Substitute $L_{12}=\dot\varepsilon_{xy}-\omega$ and $L_{21}=\dot\varepsilon_{xy}+\omega$:

$$
\begin{aligned}
v_{E,i} &= t_x + \dot\varepsilon_{xx}x_i + (\dot\varepsilon_{xy}-\omega)\,y_i\\
v_{N,i} &= t_y + (\dot\varepsilon_{xy}+\omega)\,x_i + \dot\varepsilon_{yy}y_i
\end{aligned}
\quad\Longrightarrow\quad
\underbrace{\begin{pmatrix}v_{E,i}\\v_{N,i}\end{pmatrix}}_{\mathbf d_i}=
\underbrace{\begin{pmatrix}1&0&-y_i&x_i&y_i&0\\0&1&x_i&0&x_i&y_i\end{pmatrix}}_{\mathbf G_i}
\mathbf m,\quad
\mathbf m=\begin{pmatrix}t_x\\t_y\\\omega\\\dot\varepsilon_{xx}\\\dot\varepsilon_{xy}\\\dot\varepsilon_{yy}\end{pmatrix}.
$$

Stacking three stations gives a 6×6 system $\mathbf d=\mathbf G\mathbf m$ with $\mathbf m=\mathbf G^{-1}\mathbf d$. $\mathbf G$ is singular only if the three stations are collinear. Its conditioning degrades as the triangle becomes skinny, which is why GETSI asks for interior angles above 30°.

- **Origin choice.** If $(x_i,y_i)$ are measured from the triangle **centroid**, then $\sum x_i=\sum y_i=0$. The first two rows then give $t_x=\bar v_E$ and $t_y=\bar v_N$ exactly, which reproduces the GETSI statement that translation is the average velocity. With raw UTM coordinates, $t$ is the velocity extrapolated to the UTM origin. That number is huge and meaningless, and the system becomes badly conditioned. **Always centre.**
- **Sign-convention trap.** Shen et al. (2015) use the rows $[1,0,\Delta y,\Delta x,\Delta y,0]$ and $[0,1,-\Delta x,0,\Delta x,\Delta y]$. Their $\omega$ is therefore **clockwise-positive**, the opposite of the matrix above. Materna's spherical Delaunay code also carries an in-code note that "there might be a sign issue here compared to other codes." The package should state its convention explicitly and test it on a pure rigid rotation.

### 1.3 Equivalent 2×2 solution (no 6×6 needed) **[derived]**

Let $\Delta\mathbf x_{k}=\mathbf x_k-\mathbf x_1$ and $\Delta\mathbf v_k=\mathbf v_k-\mathbf v_1$ for $k=2,3$. Then

$$
\big[\Delta\mathbf v_2\;\Delta\mathbf v_3\big]=\mathbf L\,\big[\Delta\mathbf x_2\;\Delta\mathbf x_3\big]
\;\Rightarrow\;
\mathbf L=\mathbf V\,\mathbf X^{-1}.
$$

This is the same "linear-triangle" (constant-strain-triangle, CST) gradient used in finite elements. It equals $\nabla$ of the barycentric interpolant. It makes a useful cross-check in unit tests.

### 1.4 Tensor vs engineering shear

- Tensor shear: $\dot\varepsilon_{xy}=\tfrac12(L_{12}+L_{21})$.
- Engineering shear: $\dot\gamma_{xy}=2\dot\varepsilon_{xy}=L_{12}+L_{21}$.

Most geodetic codes (Strain_2D, StrainTool, GSRM) report tensor $\dot\varepsilon_{xy}$. GETSI's "maximum shear = e1 − e2" is the engineering $\dot\gamma_{\max}$. Store tensor components internally and label every derived shear explicitly.

---

## 2. Derived quantities

Let $\bar e=\tfrac12(\dot\varepsilon_{xx}+\dot\varepsilon_{yy})$, $a=\tfrac12(\dot\varepsilon_{xx}-\dot\varepsilon_{yy})$, $b=\dot\varepsilon_{xy}$, and $r=\sqrt{a^2+b^2}$ (the Mohr-circle radius).

| Quantity | Formula | Notes |
|---|---|---|
| Principal rates | $\dot e_{1,2}=\bar e\pm r$ | $\dot e_1\ge \dot e_2$, extension positive (GETSI e1H, e2H). Equivalent to `numpy.linalg.eigh` of the symmetric 2×2. |
| Direction of $\dot e_1$ (math angle from +E, CCW) | $\theta_1=\tfrac12\,\mathrm{atan2}(2\dot\varepsilon_{xy},\ \dot\varepsilon_{xx}-\dot\varepsilon_{yy})$ | Using atan2 removes the quadrant ambiguity of $\tfrac12\tan^{-1}$. |
| Azimuth of $\dot e_1$ (CW from N) | $\alpha_1=(90^\circ-\theta_1)\bmod 180^\circ$ | Axial quantity, so report in $[0,180)$. Azimuth of $\dot e_2$: $\alpha_2=\alpha_1+90^\circ$. |
| Max shear (engineering) | $\dot\gamma_{\max}=\dot e_1-\dot e_2=2r=2\sqrt{a^2+b^2}$ | GETSI definition. Planes of max shear lie at ±45° to $\dot e_1$. |
| Max shear (tensor) | $\dot\varepsilon_{s,\max}=r=\tfrac12(\dot e_1-\dot e_2)$ | Strain_2D: `0.5*sqrt((exx-eyy)**2 + 4*exy**2)`. Same physical quantity at half the value. |
| Dilatation / area strain | $\Delta=\mathrm{tr}\,\dot{\boldsymbol\varepsilon}=\dot e_1+\dot e_2$ | $\dot A/A$. Positive means area increase. |
| "Second invariant" (Kreemer) | $\sqrt{\dot e_1^2+\dot e_2^2}=\sqrt{\dot\varepsilon_{xx}^2+\dot\varepsilon_{yy}^2+2\dot\varepsilon_{xy}^2}$ | GSRM "total strain rate" (Kreemer et al. 2014). Equals pygplates `total_strain_rate` = $\sqrt{\mathrm{tr}(D^2)}$. |
| Algebraic $I_2$ | $\dot\varepsilon_{xx}\dot\varepsilon_{yy}-\dot\varepsilon_{xy}^2=\det\dot{\boldsymbol\varepsilon}$ | Strain_2D's `I2nd`. **Not** the Kreemer quantity. Name both explicitly. |
| Style | $(\dot e_1+\dot e_2)/\max(|\dot e_1|,|\dot e_2|)$ | Kreemer et al. 2014, Fig. 5. Ranges −1 (pure contraction) through 0 (pure shear / strike-slip) to +1 (pure extension). Saturates at ±1 when both eigenvalues share a sign. |
| Rotation units | rad/yr × $(180/\pi)\times10^6$ = °/Myr; ×$10^9$ = nrad/yr | GETSI reports °/yr and nrad/yr. StrainTool reports °/Myr. |
| Strain units | 1 nanostrain/yr = $10^{-9}$ yr$^{-1}$ | Rule of thumb: 1 mm/yr of differential velocity over 100 km ≈ 10 nstrain/yr. |

Edge case: when $r\to 0$ (isotropic strain), the principal directions are undefined. Report NaN or flag them; do not report an arbitrary azimuth.

---

## 3. Uncertainty propagation

### 3.1 Data covariance including E–N correlation

Each station has a per-station 2×2 covariance:

$$
\mathbf C_i=\begin{pmatrix}\sigma_{E}^2&\rho\sigma_E\sigma_N\\\rho\sigma_E\sigma_N&\sigma_N^2\end{pmatrix}_i .
$$

The full data covariance $\mathbf C_d$ is block-diagonal. Use the full matrix instead if a SINEX/velocity-solution covariance with inter-station terms is available. Weight matrix: $\mathbf W=\mathbf C_d^{-1}$.

Sources for the correlation:
- UNAVCO/NOTA velocity files give $\rho_{EN}$.
- NGL MIDAS files give no correlation; set it to 0.

### 3.2 Parameter covariance

- **Weighted least squares** (n ≥ 3 stations):
  $$\hat{\mathbf m}=(\mathbf G^T\mathbf W\mathbf G)^{-1}\mathbf G^T\mathbf W\mathbf d,\qquad \mathbf C_m=(\mathbf G^T\mathbf W\mathbf G)^{-1}.$$
- **Three stations.** The fit is exact and the residuals are zero, so there is no internal check of the linear model. The covariance simplifies to $\mathbf C_m=\mathbf G^{-1}\mathbf C_d\mathbf G^{-T}$.
- **More than three stations.** Report $\chi^2_\nu=\mathbf r^T\mathbf W\mathbf r/(2n-6)$. Optionally scale $\mathbf C_m$ by $\chi^2_\nu$ when it exceeds 1. Large residuals mean the strain field is not uniform.

### 3.3 Propagation to derived quantities (linearized) **[derived]**

For any derived $f(\mathbf m)$, use $\sigma_f^2=\mathbf J\,\mathbf C_m\,\mathbf J^T$, with $\mathbf J=\partial f/\partial(\dot\varepsilon_{xx},\dot\varepsilon_{xy},\dot\varepsilon_{yy})$:

- $\dot e_{1,2}$: $\big(\tfrac12\pm\tfrac{a}{2r},\ \pm\tfrac{b}{r},\ \tfrac12\mp\tfrac{a}{2r}\big)$
- $\theta_1$ (radians): $\big(-\tfrac{b}{4r^2},\ \tfrac{a}{2r^2},\ \tfrac{b}{4r^2}\big)$
- $\dot\gamma_{\max}=2r$: $\big(\tfrac{a}{r},\ \tfrac{2b}{r},\ -\tfrac{a}{r}\big)$
- $\Delta$: $(1,0,1)$
- Second invariant $S=\sqrt{\dot\varepsilon_{xx}^2+\dot\varepsilon_{yy}^2+2\dot\varepsilon_{xy}^2}$: $\big(\dot\varepsilon_{xx},\ 2\dot\varepsilon_{xy},\ \dot\varepsilon_{yy}\big)/S$

These Jacobians blow up as $r\to0$. The quantities $r$ and $S$ are also positively biased by noise, because they are magnitudes.

### 3.4 Monte Carlo (recommended default for teaching)

1. Draw $\mathbf d^{(k)}\sim\mathcal N(\mathbf d,\mathbf C_d)$ with `numpy.random.default_rng().multivariate_normal`, or via a Cholesky factor.
2. Solve for each draw.
3. Report medians and 68%/95% intervals.
4. For azimuths, use **axial circular statistics**: double the angles, take the circular mean and spread, then halve.

Monte Carlo handles the non-linearity and the $r\to0$ singularity gracefully. It is cheap: $10^4$ draws × 6×6 solves takes milliseconds when vectorised.

Materna's spherical Delaunay code also propagates uncertainty this way. It uses $\mathbf C_m=\mathbf M\mathbf C_d\mathbf M^T$ with $\mathbf M=(\mathbf G^T\mathbf W\mathbf G)^{-1}\mathbf G^T\mathbf W$, plus a linearized Jacobian for the Euler-pole terms.

---

## 4. Extensions beyond one triangle

### 4.1 n > 3 stations: overdetermined uniform-strain fit

Stack $\mathbf G_i$ for all stations (coordinates relative to the network centroid) and solve the WLS problem of §3.2. This is the "homogeneous strain of a polygon / sub-network" approach used by:
- Feigl, King & Jordan (1990), JGR 95(B3), 2679–2699, doi:10.1029/JB095iB03p02679;
- Feigl et al. (1993), JGR 98(B12), 21677–21712;
- Savage et al. (2001).

### 4.2 Network: Delaunay triangulation

1. Triangulate the stations with `scipy.spatial.Delaunay`, working in projected coordinates.
2. Compute one uniform $\mathbf L$ per triangle, located at its centroid.
3. Drop slivers using a minimum-angle or maximum-edge filter. Clip large triangles on the hull, for example with an alpha-shape test.

Implementations that follow this pattern:
- **SSPX**: N. Cardozo & R. W. Allmendinger (2009), *SSPX: A program to compute strain from displacement/velocity data*, Computers & Geosciences 35(6), 1343–1357, doi:10.1016/j.cageo.2008.05.008.
  - Offers Delaunay triangles, gridded nearest-neighbour methods, and distance-weighted methods.
  - Builds on Allmendinger, Reilinger & Loveless (2007), *Strain and rotation rate from GPS in Tibet, Anatolia, and the Altiplano*, Tectonics 26, TC3013, doi:10.1029/2006TC002030.
  - SSPX is distributed as free, closed-source desktop binaries for macOS and Windows (https://www.waerfa.com/sspx, https://apps.apple.com/app/id850389108). **[unverified]** its exact licence terms.
- **StrainTool "veis" method** (DSOlab).
- **Strain_2D `delaunay_flat`** and its spherical `delaunay` variant.
- **pygplates `StrainRate`**: per-triangle $\mathbf L$, followed by area-weighted vertex averaging and barycentric or natural-neighbour interpolation (https://www.gplates.org/docs/pygplates/generated/pygplates.StrainRate).

### 4.3 Spherical formulation

In spherical coordinates (colatitude $\theta$, longitude $\phi$, radius $R$), with local South–East velocity components $(v_\theta, v_\phi)$, the surface velocity gradient picks up metric (curvature) terms (pygplates documentation):

$$
L_{\theta\theta}=\frac{1}{R}\frac{\partial v_\theta}{\partial\theta},\quad
L_{\theta\phi}=\frac{1}{R\sin\theta}\frac{\partial v_\theta}{\partial\phi}-\frac{\cot\theta\,v_\phi}{R},\quad
L_{\phi\theta}=\frac{1}{R}\frac{\partial v_\phi}{\partial\theta},\quad
L_{\phi\phi}=\frac{1}{R\sin\theta}\frac{\partial v_\phi}{\partial\phi}+\frac{\cot\theta\,v_\theta}{R}.
$$

The $\cot\theta\,v/R$ terms are what a naive flat-Earth calculation omits. Their size is $|v|\tan(\text{lat})/R$. For example, at 20 mm/yr and 45°N this is about 3 nstrain/yr. That is comparable to intraplate strain rates, but small next to the ~100–1000 nstrain/yr typical of the San Andreas system.

Savage, Gan & Svarc (2001) model velocities on the sphere as a rigid rotation (Euler vector) plus uniform strain about the sub-network centroid. Their derivation is in J. C. Savage, W. Gan & J. L. Svarc (2001), *Strain accumulation and rotation in the Eastern California Shear Zone*, JGR 106(B10), 21995–22007, doi:10.1029/2000JB000127 (https://pubs.usgs.gov/publication/70023061). Strain_2D's spherical Delaunay module cites "Savage et al., JGR October 2001, p.22,005" for its equations. That module also estimates the Euler-vector components and pole location.

### 4.4 Continuous interpolation methods (for context, not needed for the triangle calculator)

- **Shen et al. (1996, 2015) / VISR.**
  - Citations: Shen, Jackson & Ge (1996), JGR 101(B12), 27957–27980, doi:10.1029/96JB02544. Shen, Wang, Zeng & Wang (2015), *Optimal interpolation of spatially discretized geodetic data*, BSSA 105(4), 2117–2127, doi:10.1785/0120140247.
  - At each grid point, fit the same 6-parameter model (with the CW-ω sign noted in §1.2) using station weights $G_i=L_iZ_i$.
  - Distance weight: Gaussian $L_i=e^{-\Delta R_i^2/D^2}$ or quadratic $1/(1+\Delta R_i^2/D^2)$.
  - Spatial weight: Voronoi-area $Z_i=nS_i/\sum S_k$ or azimuthal $n\theta_i/4\pi$.
  - $D$ is chosen per point so that $\sum G_i=W_t$ (e.g. $W_t=24$ for Gaussian with Voronoi weighting).
- **GSRM v2.1** (Kreemer, Blewitt & Klein 2014, *A geodetic plate motion and Global Strain Rate Model*, G³ 15, 3849–3889, doi:10.1002/2014GC005407).
  - Uses the Haines & Holt (1993, JGR 98(B7), 12057–12082) bicubic-spline velocity-gradient method on the sphere, with Bayesian a priori variances.
  - Results page: https://geodesy.unr.edu/cornekreemer/gsrm.htm.
- **gpsgridder.** Sandwell & Wessel (2016), *Interpolation of 2-D vector data using constraints from elasticity*, GRL 43, 10703–10709, doi:10.1002/2016GL070340. Implemented in GMT (`gpsgridder`).
- **Wavelets.** Tape et al. (2009), *Multiscale estimation of GPS velocity fields*, GJI 179, 945–971.
- **Method comparison and uncertainty.** Maurer & Materna (2023), *Quantification of geodetic strain rate uncertainties and implications for seismic hazard estimates*, GJI 234(3), 2128–2142, doi:10.1093/gji/ggad191.

### 4.5 Existing open-source tools (relevant to "rebuild vs reuse")

| Tool | Language / licence | Methods | Relevance |
|---|---|---|---|
| **Strain_2D** (K. Materna, J. Maurer, L. Sandoe) https://github.com/kmaterna/Strain_2D | Python 3, **MIT** | delaunay_flat, delaunay (spherical, after Savage 2001), VISR (Fortran), simple_visr, gpsgridder (GMT), loc_avg_grad, wavelets (MATLAB), geostats/kriging | Research-grade, config-file driven, heavy dependencies (GMT/pygmt, netCDF). Input columns: `lon lat VE VN VU SE SN SU name`, in mm/yr; **no E–N correlation column**. Outputs exx, exy, eyy, rot, I2nd (= det), dilatation, max_shear (tensor, i.e. half the GETSI value), and eigenvectors. Cite Maurer & Materna 2023 and Zenodo doi:10.5281/zenodo.5240908. Good reference implementation for cross-validation; too heavy for a classroom tool. |
| **StrainTool / pystrain** (Dionysos Satellite Observatory, NTUA) https://github.com/DSOlab/StrainTool | Python (numpy/scipy) + GMT bash plotting, **MIT** | Shen 2015 weighted LS (Gaussian/quadratic, $W_t$ search), "veis" Delaunay at barycentres | Outputs vx, vy, ω (°/Myr), exx, exy, eyy, emax, emin, shear, azimuth, dilatation, second invariant, each with σ. Zenodo doi:10.5281/zenodo.1297565. |
| **pygplates `StrainRate` / deforming networks** | Python bindings to GPlates; GPL-family **[unverified]** | Spherical $\mathbf L$ per triangle | Good reference for the spherical metric terms. |
| **GMT `velo` / pygmt `velo`** https://docs.generic-mapping-tools.org/latest/supplements/geodesy/velo.html | LGPL-3 | Plotting only | `-Sx` strain crosses take eps1 (most extensional), eps2 (most compressional), and **azimuth of eps2 in degrees CW from N**. `-Sw` gives rotational wedges; `-Se`/`-Sr` give velocity error ellipses. |
| **Hector / HectorP** (M. Bos) https://pypi.org/project/hectorp | C++ / Python | Time-series trend with power-law + white noise (RMLE) | For velocity estimation (§7). Licence not stated on the PyPI page; check before bundling. |

**Verdict.** The triangle and uniform-strain core is about 150 lines of numpy and is pedagogically transparent. Rebuild it from scratch, with your own tests. Validate against Strain_2D's `delaunay_flat` and StrainTool on the same inputs, and optionally offer an exporter to Strain_2D or GMT formats. Because Strain_2D and StrainTool are MIT, borrowing snippets with attribution is legally fine. Do not copy GETSI documents or text verbatim (CC BY-NC-SA).

---

## 5. Map-projection issues and best practice

### 5.1 What goes wrong with UTM

1. **Scale factor.** UTM uses $k_0=0.9996$ on the central meridian, rising to about 1.0010 at the zone edges near the equator. Positions are scaled by $k$, but velocities are true metres per year. The resulting strain error is a fractional $(k-1)$ of the strain itself, at most about 0.1%. That is negligible.
2. **Meridian (grid) convergence.** This is the real issue.
   - UTM grid north differs from true north by $\gamma\approx\Delta\lambda\,\sin\varphi$. At a zone edge ($\Delta\lambda=3^\circ$) at 45° latitude, $\gamma\approx 2.1^\circ$.
   - GNSS velocities are given in *geographic* E/N. Combining grid coordinates with geographic velocities applies a rotation $R(\gamma)$ to positions only, so the estimate becomes $\mathbf L_{\text{est}}\approx\mathbf L R(\gamma)$.
   - **[derived]** To first order this adds $-\gamma\,\omega$ to the dilatation and mixes rotation into the strain components. It also biases azimuths by γ. With ω ~ 100 nrad/yr and γ ~ 0.04 rad, that is several nstrain/yr of spurious signal.
   - Fix: rotate the velocities into grid orientation using pyproj `Proj.get_factors(...).meridian_convergence`, or avoid UTM entirely.
3. **Cross-zone triangles.** Coordinates from two UTM zones are not in the same Cartesian system, so the result is garbage. The GETSI Excel workflow requires the user to force a single zone.
4. **Differential north directions** (applies even to tangent planes). Local north at each station differs from north at the centroid by $\approx\Delta\lambda\sin\varphi$. This is the flat-Earth form of the $\cot\theta$ terms of §4.3: an error of about $|v|\tan\varphi/R$, roughly 3 nstrain/yr for 20 mm/yr at 45°.
   - Mitigation option 1: express velocities relative to a frame that minimises $|v|$ (e.g. plate-fixed NAM14, as GETSI does).
   - Mitigation option 2: rotate each station's velocity into the centroid frame through ECEF (below).

### 5.2 Recommended approach

Per triangle or sub-network:

1. Compute the centroid $(\varphi_0,\lambda_0)$, either the mean of the ECEF positions normalised back to the ellipsoid, or simply the mean lat/lon for regions under 1000 km.
2. **Positions → local ENU.**
   - Use the pyproj pipeline `+proj=pipeline +step +proj=cart +ellps=GRS80 +step +proj=topocentric +lat_0=φ0 +lon_0=λ0 +h_0=0`. The PROJ `topocentric` operation needs PROJ ≥ 8.0 and corresponds to EPSG:9836/9837 (https://proj.org/en/stable/operations/conversions/topocentric.html).
   - Alternative: an azimuthal-equidistant projection `+proj=aeqd +lat_0=φ0 +lon_0=λ0`. For 100 km triangles, the distance distortion of either choice is of order $(d/R)^2\sim10^{-4}$ and is negligible.
3. **Velocities → centroid ENU.** For each station:
   - Map $(v_E,v_N,v_U)$ to ECEF using the station's own rotation matrix $\mathbf R_i^T$.
   - Map back to ENU at the centroid with $\mathbf R_0$, and keep E and N.
   - Rotate the 2×2 covariance with the same matrix.
   - This removes the convergence and curvature error at first order and costs a few lines of numpy.
4. Solve the WLS system of §3 with coordinates in metres and velocities in m/yr, so strain comes out in yr$^{-1}$.
5. For continental-scale networks, use the spherical formulation (§4.3) or per-triangle local frames as above.

Keep UTM only as an optional "GETSI-compatible" mode, so students can reproduce the original Excel numbers, and emit a warning when stations span zones or $|\gamma|>0.5^\circ$.

---

## 6. Visualisation

### 6.1 Strain ellipse

An infinitesimal rate needs exaggeration. Over a time $\Delta t$ with exaggeration factor $k$:

$$
\mathbf F=\mathbf I+k\,\Delta t\,\mathbf L .
$$

1. Map the unit circle $\mathbf x=(\cos s,\sin s)$ to $\mathbf F\mathbf x$.
2. Choose $k\Delta t$ so that $\max(|\dot e_1|,|\dot e_2|)\,k\Delta t\approx 0.2$–0.3. Below about 0.5 the ellipse is still nearly "infinitesimal".
3. Report the factor on the figure, e.g. "×10⁶ exaggeration" or "1 Myr at rate × 10".
4. Optionally separate pure strain from rotation:
   - Symmetric-only form: $\mathbf F_s=\mathbf I+k\Delta t\,\dot{\boldsymbol\varepsilon}$, with semi-axes $1+k\Delta t\,\dot e_{1,2}$ along $\theta_{1}$ and $\theta_1+90^\circ$.
   - Show rotation separately, either by rotating that ellipse by $k\Delta t\,\omega$ or with the polar decomposition $\mathbf F=\mathbf R\mathbf U$ via `scipy.linalg.polar`.

The GETSI Excel and CDF visualisers work on this principle. A "unit circle → ellipse" with the original circle overlaid, plus the deformed triangle, is the most instructive figure. Draw the deformed triangle as $\mathbf x_i+k\Delta t\,(\mathbf v_i-\bar{\mathbf v})$.

### 6.2 Map symbols

- **Strain-rate crosses** (GMT `velo -Sx`, pygmt `velo(style="x...")`).
  - Two orthogonal bars along the principal axes, with length proportional to $|\dot e|$.
  - Common convention: outward arrows for extension (red) and inward arrows for contraction (blue).
  - GMT input is eps1 (most extensional), eps2 (most compressional), and **azimuth of eps2 CW from N**. Note this is the azimuth of the *compressional* axis.
- **Rotation wedges** (`-Sw`): a wedge or arc, with angle proportional to ω and its sign.
- **Background colour fields**: second invariant (log scale), dilatation (diverging colormap centred at 0), style.
- **Velocity arrows with error ellipses** (`-Se`) at the stations: 95% confidence ellipses from $\mathbf C_i$ scaled by $\sqrt{\chi^2_{2,0.95}}=2.45$.
- **Python stack**: matplotlib (`quiver` and `matplotlib.patches.Ellipse`) is enough for a teaching package. Cartopy or pygmt are optional extras.

---

## 7. Velocity estimation from position time series

### 7.1 Why fit a slope rather than (end − start)/duration

- (last − first)/T uses only two noisy epochs. Its error is set entirely by noise and transients at the endpoints, and the seasonal signal does not cancel unless T is an integer number of years and the endpoints sit at the same seasonal phase.
- Mean displacement divided by duration is also biased, because it mixes intercept and rate.
- Instead, the standard trajectory model is fitted by least squares:

$$
x(t)=a+b\,(t-t_0)+\sum_{k=1}^{2}\big[c_k\sin(2\pi k t)+d_k\cos(2\pi k t)\big]+\sum_j o_j H(t-t_j)\;[+\text{post-seismic log/exp terms}].
$$

Here $b$ is the velocity, $k=1,2$ give the annual and semi-annual terms, $o_j$ are offsets from equipment changes or earthquakes, and $H$ is the Heaviside step.

- Blewitt & Lavallée (2002, JGR 107(B7), 2145, doi:10.1029/2001JB000570) show that unmodelled annual signals bias velocities for short spans. They recommend ≥ 2.5 yr of data.
- For teaching, a linear fit with annual and semi-annual terms in numpy (`lstsq`) is enough.

### 7.2 MIDAS: robust velocities without step detection

Blewitt, Kreemer, Hammond & Gazeaux (2016), JGR Solid Earth 121, 2054–2068, doi:10.1002/2015JB012552.

**Algorithm:**
1. Form slopes between data pairs separated by one year (0.999 < Δt < 1.001 yr). A forward/backward search handles gaps. The one-year separation cancels the seasonal signal, and most pairs do not straddle a step.
2. Take the median slope $\hat v$.
3. Estimate the spread $\sigma=1.4826\,\mathrm{MAD}$.
4. Trim slopes with $|v-\hat v|>2\sigma$, then recompute the median and $\sigma$.

**Uncertainty:**

$$
\sigma_{\hat v}=\sqrt{\pi/2}\;\sigma/\sqrt{N/4}\;(\approx1.2533\,\sigma/\sqrt{N/4}),\qquad\sigma_{\text{MIDAS}}=3\,\sigma_{\hat v}.
$$

The $N/4$ accounts for observations reused across pairs. The factor 3 is an empirical inflation for autocorrelated noise.

NGL publishes weekly MIDAS velocities in IGS14 and plate-fixed frames (NA, PA, EU, …). See https://geodesy.unr.edu/velocities/midas.readme.txt for the 27-column format. There are **no E–N correlations**. This is a practical default data source for the package.

### 7.3 Realistic uncertainties: coloured noise

- GNSS position noise is roughly white plus flicker (power-law, spectral index ≈ −1), sometimes with random walk:
  - Mao, Harrison & Dixon (1999), JGR 104(B2), 2797–2816, doi:10.1029/1998JB900033;
  - Williams et al. (2004), *Error analysis of continuous GPS position time series*, JGR 109, B03412, doi:10.1029/2003JB002741.
- White-noise-only formal velocity errors are optimistic by factors of about 2 to more than 10.
- The rate uncertainty scales as $\sim T^{-3/2}$ for white noise, $\sim T^{-1}$ for flicker noise, and $\sim T^{-1/2}$ for random walk (Williams 2003, *The effect of coloured noise on the uncertainties of rates estimated from geodetic time series*, J. Geod. 76(9–10), 483–494, doi:10.1007/s00190-002-0283-4).
- **Hector** (Bos, Fernandes, Williams & Bastos 2013, *Fast error analysis of continuous GNSS observations with missing data*, J. Geod. 87(4), 351–360, doi:10.1007/s00190-012-0605-0) estimates the trend, seasonal terms and offsets with power-law + white noise by (restricted) maximum likelihood. HectorP is the pure-Python port on PyPI. Older equivalent: CATS (Williams), https://www.ngs.noaa.gov/gps-toolbox/cats.htm.
- **Practical guidance for the package:**
  - Ingest published velocities and σ (MIDAS/NGL, or the UNAVCO/NOTA GAGE solution).
  - Offer an optional educational "fit my own velocity" module (linear + seasonal + offsets) that clearly warns that its white-noise σ is too small. Offer a "×3" MIDAS-style inflation, or call HectorP if it is installed.

---

## 8. Recommended implementation summary

1. **Inputs.** Station id, lat, lon, $v_E$, $v_N$, $\sigma_E$, $\sigma_N$, $\rho_{EN}$, and the reference frame (warn if the frames are mixed).
2. **Geometry.** Build an ECEF-based local ENU about the centroid for both positions and velocity vectors (pyproj `cart` + `topocentric`). Keep UTM only as a "classic GETSI" mode, with convergence and zone checks.
3. **Solve.** Use the 6-parameter $\mathbf m=[t_x,t_y,\omega_{\text{CCW}},\dot\varepsilon_{xx},\dot\varepsilon_{xy},\dot\varepsilon_{yy}]$ with centred coordinates and WLS (exact for n = 3, overdetermined for n > 3). Use `scipy.linalg.lstsq` or `solve` on the whitened system. Check the condition number and the minimum triangle angle.
4. **Derived quantities.** $\dot e_{1,2}$, azimuths (CW from N, axial), $\dot\gamma_{\max}$ in engineering units (GETSI-compatible) and the tensor value, dilatation, Kreemer second invariant, $\det$, style, and rotation in °/Myr and nrad/yr. Use explicit names, e.g. `max_shear_engineering` and `max_shear_tensor`, to avoid the factor-2 confusion.
5. **Uncertainties.** Linear propagation, plus Monte Carlo as the default for principal values and azimuths, using axial circular statistics.
6. **Network mode.** scipy Delaunay with sliver and hull filtering; per-triangle results at the centroids. An optional spherical mode follows Savage et al. (2001) and pygplates.
7. **Visualisation.** Matplotlib unit-circle → ellipse with a stated exaggeration and the deformed triangle; strain crosses with GMT-compatible export (eps1, eps2, azimuth of eps2).
8. **Tests.**
   - Synthetic pure translation should give zero strain and zero rotation.
   - Rigid rotation should give ω only, with the CCW sign.
   - Uniaxial extension along N30E should give the correct azimuth.
   - Simple shear should give $\dot\varepsilon_{xy}=\omega$.
   - Cross-check against the CST formula $\mathbf L=\mathbf V\mathbf X^{-1}$, Strain_2D `delaunay_flat`, and the GETSI Excel outputs.

---

### Retrieval notes

- Not readable by the fetch tool (binary .docx or blocked): the GETSI algorithm .docx and the Explanation .docx. The v4 PDF of the Explanation *was* read.
- Not readable: Scribd copy of "Triangle Strain Algorithm", PMC copy of MIDAS (CAPTCHA), Crossref entries for Allmendinger 2007 and Feigl 1990 (rate-limited). Their citations above are from other pages or standard bibliographic knowledge. Allmendinger 2007 and Feigl 1990 should be spot-checked.
- Not given in the sources read: the precise UTM handling and matrix layout of the GETSI Excel/MATLAB calculators.
