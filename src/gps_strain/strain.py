"""Uniform horizontal infinitesimal strain-rate from GPS station velocities.

Model (Cronin & Resor, "Algorithm for triangle strain"; x = east, y = north,
positions measured from the station centroid)::

    vE = tx - y*w + x*exx + y*exy
    vN = ty + x*w + x*exy + y*eyy

Unknowns ``m = [tx, ty, w, exx, exy, eyy]``:

* ``tx, ty``  translation of the centroid (m/yr) - the mean station velocity
* ``w``       rigid-body rotation rate (rad/yr), **positive = counter-clockwise**
* ``exx, exy, eyy``  infinitesimal strain-rate tensor (1/yr); ``exy`` is the
  *tensor* shear component (half the engineering shear strain)

With three stations the 6x6 system is exactly determined (the GETSI triangle
method). With more stations it is solved by weighted least squares and the
reduced chi-square tests whether a single uniform strain fits.

Uncertainties: ``Cm = (G^T W G)^-1`` with ``W`` the inverse data covariance
(including any E-N correlation). Reported 1-sigma = sqrt(diag(Cm)). Derived
quantities (principal strains, azimuths, max shear, ...) get 1-sigma from a
Monte Carlo draw of the parameter covariance, because linear propagation fails
for nearly isotropic strain (azimuth undefined).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy

PARAM_NAMES = ("tx", "ty", "rotation", "exx", "exy", "eyy")
NANO = 1e9


def design_matrix(x, y):
    """Interleaved design matrix (rows E1, N1, E2, N2, ...)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    G = np.zeros((2 * len(x), 6))
    G[0::2, 0] = 1.0
    G[0::2, 2] = -y
    G[0::2, 3] = x
    G[0::2, 4] = y
    G[1::2, 1] = 1.0
    G[1::2, 2] = x
    G[1::2, 4] = x
    G[1::2, 5] = y
    return G


def data_covariance(sig_e, sig_n, corr_en=None):
    """Block-diagonal covariance of the interleaved velocity vector."""
    sig_e = np.asarray(sig_e, dtype=float)
    sig_n = np.asarray(sig_n, dtype=float)
    if np.any(sig_e <= 0) or np.any(sig_n <= 0):
        raise ValueError("velocity uncertainties must be positive")
    rho = np.zeros_like(sig_e) if corr_en is None else np.asarray(corr_en, dtype=float)
    n = len(sig_e)
    C = np.zeros((2 * n, 2 * n))
    for i in range(n):
        C[2 * i, 2 * i] = sig_e[i] ** 2
        C[2 * i + 1, 2 * i + 1] = sig_n[i] ** 2
        C[2 * i, 2 * i + 1] = C[2 * i + 1, 2 * i] = rho[i] * sig_e[i] * sig_n[i]
    return C


def principal(exx, exy, eyy):
    """Principal strain rates and azimuths.

    Returns ``e1, e2, az1, az2`` where e1 >= e2 (max / min horizontal extension
    rate) and az1/az2 are the azimuths (deg clockwise from north, in [0, 180))
    of the S1H and S2H axes. Works element-wise on arrays.
    """
    exx, exy, eyy = (np.asarray(v, dtype=float) for v in (exx, exy, eyy))
    mean = 0.5 * (exx + eyy)
    r = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy ** 2)
    e1, e2 = mean + r, mean - r
    # angle of e1 axis from +x (east), counter-clockwise
    theta = 0.5 * np.arctan2(2.0 * exy, exx - eyy)
    az1 = (90.0 - np.degrees(theta)) % 180.0
    az2 = (az1 + 90.0) % 180.0
    return e1, e2, az1, az2


def _axial_std(az_deg):
    """Circular standard deviation (deg) of axial (180-periodic) azimuths."""
    a = np.radians(2.0 * np.asarray(az_deg))
    R = np.hypot(np.mean(np.cos(a)), np.mean(np.sin(a)))
    R = min(max(R, 1e-12), 1.0)
    return np.degrees(np.sqrt(-2.0 * np.log(R))) / 2.0


@dataclass
class StrainResult:
    """Everything the GETSI output sheet reports, plus uncertainties."""

    sites: list
    x: np.ndarray            # station coordinates used in the inversion (m, centred)
    y: np.ndarray
    ve: np.ndarray           # velocities used (m/yr, in the inversion frame)
    vn: np.ndarray
    params: np.ndarray       # [tx, ty, w, exx, exy, eyy]
    cov: np.ndarray          # 6x6 parameter covariance
    frame: str               # "enu" or "utm"
    origin: tuple            # (lat0, lon0) or ("UTM zone", zone)
    chi2_reduced: float | None = None
    mc: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    # --- raw parameters -------------------------------------------------
    @property
    def sigma(self):
        return np.sqrt(np.diag(self.cov))

    tx = property(lambda s: s.params[0])
    ty = property(lambda s: s.params[1])
    rotation = property(lambda s: s.params[2])
    exx = property(lambda s: s.params[3])
    exy = property(lambda s: s.params[4])
    eyy = property(lambda s: s.params[5])

    @property
    def strain_tensor(self):
        return np.array([[self.exx, self.exy], [self.exy, self.eyy]])

    @property
    def velocity_gradient(self):
        """L = dv/dx (strain-rate tensor + rotation), 1/yr."""
        w = self.rotation
        return np.array([[self.exx, self.exy - w], [self.exy + w, self.eyy]])

    # --- derived ----------------------------------------------------------
    @property
    def speed(self):
        return float(np.hypot(self.tx, self.ty))

    @property
    def translation_azimuth(self):
        return float(np.degrees(np.arctan2(self.tx, self.ty)) % 360.0)

    @property
    def principal(self):
        return tuple(float(v) for v in principal(self.exx, self.exy, self.eyy))

    e1 = property(lambda s: s.principal[0])
    e2 = property(lambda s: s.principal[1])
    azimuth_s1 = property(lambda s: s.principal[2])
    azimuth_s2 = property(lambda s: s.principal[3])

    @property
    def max_shear(self):
        """Maximum infinitesimal shear strain rate, engineering (= e1 - e2), as GETSI."""
        return self.e1 - self.e2

    @property
    def area_strain(self):
        """Areal (dilatational) strain rate = e1 + e2 = I1."""
        return float(self.exx + self.eyy)

    @property
    def invariants(self):
        """(I1, I2, I3) as defined in Cronin & Resor's algorithm document.

        I1 = trace; I2 = sqrt(exx^2 + eyy^2 + 2 exy^2) (Kreemer et al. 2014
        second invariant); I3 = determinant.
        """
        return (
            self.area_strain,
            float(np.sqrt(self.exx ** 2 + self.eyy ** 2 + 2 * self.exy ** 2)),
            float(np.linalg.det(self.strain_tensor)),
        )

    @property
    def style(self):
        """Kreemer et al. (2014) style: (e1+e2)/max(|e1|,|e2|); -1 contraction ... +1 extension."""
        m = max(abs(self.e1), abs(self.e2))
        return float((self.e1 + self.e2) / m) if m else 0.0

    @property
    def rotation_sense(self):
        return "clockwise" if self.rotation < 0 else "counter-clockwise"

    def summary(self) -> dict:
        """GETSI-style summary in conventional units (m/yr, nanostrain/yr, nrad/yr, deg)."""
        s = self.sigma
        mc = self.mc
        out = {
            "sites": ", ".join(self.sites),
            "frame": self.frame,
            "translation_E_m_per_yr": (self.tx, s[0]),
            "translation_N_m_per_yr": (self.ty, s[1]),
            "translation_azimuth_deg": self.translation_azimuth,
            "translation_speed_m_per_yr": self.speed,
            "rotation_nrad_per_yr": (self.rotation * NANO, s[2] * NANO),
            "rotation_deg_per_yr": (np.degrees(self.rotation), np.degrees(s[2])),
            "rotation_sense": self.rotation_sense,
            "exx_nstrain_per_yr": (self.exx * NANO, s[3] * NANO),
            "exy_nstrain_per_yr": (self.exy * NANO, s[4] * NANO),
            "eyy_nstrain_per_yr": (self.eyy * NANO, s[5] * NANO),
            "e1H_nstrain_per_yr": (self.e1 * NANO, mc.get("e1", np.nan) * NANO),
            "S1H_azimuth_deg": (self.azimuth_s1, mc.get("az1", np.nan)),
            "e2H_nstrain_per_yr": (self.e2 * NANO, mc.get("e2", np.nan) * NANO),
            "S2H_azimuth_deg": (self.azimuth_s2, mc.get("az1", np.nan)),
            "max_shear_nstrain_per_yr": (self.max_shear * NANO, mc.get("max_shear", np.nan) * NANO),
            "area_strain_nstrain_per_yr": (self.area_strain * NANO, mc.get("area", np.nan) * NANO),
            "I1": self.invariants[0],
            "I2": self.invariants[1],
            "I3": self.invariants[2],
            "style": self.style,
        }
        if self.chi2_reduced is not None:
            out["chi2_reduced"] = self.chi2_reduced
        return out

    def report(self) -> str:
        """Human-readable text report (the content of the GETSI output sheet)."""
        s = self.sigma
        mc = self.mc

        def pm(v, e, fmt="{:.4f}"):
            return f"{fmt.format(v)} ± {fmt.format(e)}" if np.isfinite(e) else fmt.format(v)

        def alt(a):
            return f"{a:.1f}° (or {(a + 180) % 360:.1f}°)"

        lines = [
            f"Stations: {', '.join(self.sites)}",
            f"Coordinate frame: {self.frame} {self.origin}",
            "",
            "Translation (velocity of centroid)",
            f"  East   {pm(self.tx * 1000, s[0] * 1000, '{:.3f}')} mm/yr",
            f"  North  {pm(self.ty * 1000, s[1] * 1000, '{:.3f}')} mm/yr",
            f"  Speed  {self.speed * 1000:.3f} mm/yr toward azimuth {self.translation_azimuth:.1f}°",
            "",
            f"Rotation  {pm(self.rotation * NANO, s[2] * NANO, '{:.3f}')} nrad/yr ({self.rotation_sense})",
            "",
            "Strain-rate tensor (nanostrain/yr; exy = tensor shear)",
            f"  exx {pm(self.exx * NANO, s[3] * NANO, '{:.3f}')}",
            f"  exy {pm(self.exy * NANO, s[4] * NANO, '{:.3f}')}",
            f"  eyy {pm(self.eyy * NANO, s[5] * NANO, '{:.3f}')}",
            "",
            "Principal strain rates",
            f"  e1H (max horizontal extension)  {pm(self.e1 * NANO, mc.get('e1', np.nan) * NANO, '{:.3f}')} nstrain/yr",
            f"      S1H azimuth  {alt(self.azimuth_s1)}"
            + (f" ± {mc['az1']:.1f}°" if "az1" in mc else ""),
            f"  e2H (min horizontal extension)  {pm(self.e2 * NANO, mc.get('e2', np.nan) * NANO, '{:.3f}')} nstrain/yr",
            f"      S2H azimuth  {alt(self.azimuth_s2)}",
            f"  Max shear strain rate (e1-e2)   {pm(self.max_shear * NANO, mc.get('max_shear', np.nan) * NANO, '{:.3f}')} nstrain/yr,"
            f" on planes at 45° to S1H",
            f"  Area strain rate (e1+e2)        {pm(self.area_strain * NANO, mc.get('area', np.nan) * NANO, '{:.3f}')} nstrain/yr",
            f"  Invariants I1={self.invariants[0]:.4e}  I2={self.invariants[1]:.4e}  I3={self.invariants[2]:.4e}",
        ]
        if self.chi2_reduced is not None:
            lines.append(f"  Reduced chi-square of uniform-strain fit: {self.chi2_reduced:.2f}")
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def solve_planar(x, y, ve, vn, sig_e, sig_n, corr_en=None, center=True):
    """Solve the uniform-strain model for planar coordinates (metres).

    Returns ``(params, cov, x_c, y_c, chi2_reduced)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        raise ValueError("at least three stations are required")
    if center:
        x = x - x.mean()
        y = y - y.mean()
    G = design_matrix(x, y)
    d = np.empty(2 * len(x))
    d[0::2] = ve
    d[1::2] = vn
    Cd = data_covariance(sig_e, sig_n, corr_en)
    W = np.linalg.inv(Cd)
    N = G.T @ W @ G
    if np.linalg.cond(N) > 1e30:
        raise np.linalg.LinAlgError("stations are (nearly) collinear; strain is undefined")
    cov = np.linalg.inv(N)
    params = cov @ G.T @ W @ d
    dof = len(d) - 6
    chi2 = None
    if dof > 0:
        r = d - G @ params
        chi2 = float(r @ W @ r / dof)
    return params, cov, x, y, chi2


def triangle_quality(x, y):
    """Interior angles (deg) of a triangle; Cronin advises none below ~30 deg."""
    pts = np.column_stack([x, y])
    angles = []
    for i in range(3):
        a, b, c = pts[i], pts[(i + 1) % 3], pts[(i + 2) % 3]
        u, v = b - a, c - a
        cosang = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return angles


def triangle_sides_km(x, y):
    """Side lengths (km) of a triangle; GETSI Unit 6 asks for none over 100 km."""
    pts = np.column_stack([x, y])
    return [float(np.linalg.norm(pts[(i + 1) % 3] - pts[i]) / 1000) for i in range(3)]


def monte_carlo(params, cov, n=20000, seed=0):
    """1-sigma of derived quantities by sampling the parameter covariance."""
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(params, cov, size=n, method="eigh")
    e1, e2, az1, _ = principal(S[:, 3], S[:, 4], S[:, 5])
    return {
        "e1": float(np.std(e1)),
        "e2": float(np.std(e2)),
        "az1": float(_axial_std(az1)),
        "max_shear": float(np.std(e1 - e2)),
        "area": float(np.std(e1 + e2)),
    }


def compute_strain(stations, frame: str = "enu", mc_samples: int = 20000) -> StrainResult:
    """Compute uniform strain rate for a set of stations.

    ``stations`` is a pandas DataFrame (or mapping of columns) with:
    ``site, lat, lon, ve, vn, sig_e, sig_n`` (velocities in **m/yr**) and
    optionally ``corr_en``. Use :func:`gps_strain.data.normalize` to build it.

    ``frame``: ``"enu"`` (local tangent plane, recommended) or ``"utm"``
    (classic Cronin/GETSI spreadsheet method).
    """
    import pandas as pd

    df = pd.DataFrame(stations).reset_index(drop=True)
    sites = [str(s) for s in df["site"]]
    lat, lon = df["lat"].to_numpy(float), df["lon"].to_numpy(float)
    ve, vn = df["ve"].to_numpy(float), df["vn"].to_numpy(float)
    se, sn = df["sig_e"].to_numpy(float), df["sig_n"].to_numpy(float)
    rho = df["corr_en"].to_numpy(float) if "corr_en" in df else np.zeros(len(df))
    warnings = []

    if frame == "enu":
        x, y, origin = geodesy.to_local_enu(lat, lon)
        cov_en = np.zeros((len(df), 2, 2))
        cov_en[:, 0, 0], cov_en[:, 1, 1] = se ** 2, sn ** 2
        cov_en[:, 0, 1] = cov_en[:, 1, 0] = rho * se * sn
        ve, vn, cov_r = geodesy.rotate_velocities_to_frame(lat, lon, ve, vn, *origin, cov_en)
        se, sn = np.sqrt(cov_r[:, 0, 0]), np.sqrt(cov_r[:, 1, 1])
        rho = cov_r[:, 0, 1] / (se * sn)
        origin = (round(origin[0], 6), round(origin[1], 6))
    elif frame == "utm":
        x, y, zone = geodesy.to_utm_classic(lat, lon)
        origin = ("UTM zone", zone)
        if len(np.unique(geodesy.utm_zone(lon))) > 1:
            warnings.append(
                f"stations span several UTM zones; all projected on zone {zone} "
                "(Cronin's pseudo-zone rule). Prefer frame='enu'."
            )
    else:
        raise ValueError("frame must be 'enu' or 'utm'")

    params, cov, xc, yc, chi2 = solve_planar(x, y, ve, vn, se, sn, rho)

    if len(df) == 3:
        angles = triangle_quality(xc, yc)
        if min(angles) < 30:
            warnings.append(
                f"smallest interior angle is {min(angles):.1f}°; Cronin & Resor advise "
                "triangles with no angle below ~30° (results become unstable)."
            )
        sides = triangle_sides_km(xc, yc)
        if max(sides) > 100:
            warnings.append(
                f"longest side is {max(sides):.0f} km; the GETSI Unit 6 project asks for "
                "triangles with no side longer than 100 km."
            )
    span_km = np.hypot(np.ptp(xc), np.ptp(yc)) / 1000
    if span_km > 500:
        warnings.append(f"network spans ~{span_km:.0f} km; uniform-strain assumption is doubtful.")

    res = StrainResult(
        sites=sites, x=xc, y=yc, ve=np.asarray(ve), vn=np.asarray(vn), params=params,
        cov=cov, frame=frame, origin=origin, chi2_reduced=chi2, warnings=warnings,
    )
    if mc_samples:
        res.mc = monte_carlo(params, cov, n=mc_samples)
    return res
