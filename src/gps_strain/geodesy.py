"""Coordinate handling for GPS triangle strain.

Two horizontal coordinate systems are supported:

``"enu"`` (default, recommended)
    A local East-North-Up tangent plane centred on the stations' mean position.
    Station positions go geodetic -> ECEF -> ENU, and each station's velocity is
    rotated from its own local E/N frame into the common frame at the centre.
    This avoids UTM zone boundaries, grid convergence and scale-factor issues.

``"utm"`` (classic Cronin & Resor / GETSI mode)
    UTM coordinates computed with the USGS/Army K1-K5 series (the form in
    Steven Dutch's spreadsheet that Cronin's Excel calculator follows), all
    stations projected on the central meridian of the westernmost zone
    ("compute the coordinates of the site(s) in the eastern zone as if they
    were in the western zone"). Velocities are used as given (no rotation for
    grid convergence), exactly like the spreadsheet.

Only numpy is required.
"""
from __future__ import annotations

import math

import numpy as np

# WGS84 ellipsoid
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)  # 6356752.314245...
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

UTM_K0 = 0.9996


# ---------------------------------------------------------------------------
# Local tangent plane (ENU)
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg, lon_deg, h=0.0):
    """Geodetic (deg, deg, m) -> ECEF X, Y, Z in metres (WGS84)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h, dtype=float)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + h) * np.sin(lat)
    return x, y, z


def enu_basis(lat_deg, lon_deg):
    """Unit vectors (east, north, up) expressed in ECEF, each shape (..., 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    east = np.stack([-so, co, np.zeros_like(so)], axis=-1)
    north = np.stack([-sl * co, -sl * so, cl], axis=-1)
    up = np.stack([cl * co, cl * so, sl], axis=-1)
    return east, north, up


def mean_position(lat_deg, lon_deg):
    """Geographic centre of a set of points (mean of ECEF vectors).

    Robust across the antimeridian. Returns (lat, lon) in degrees.
    """
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg)
    xm, ym, zm = np.mean(x), np.mean(y), np.mean(z)
    lon0 = math.degrees(math.atan2(ym, xm))
    p = math.hypot(xm, ym)
    lat0 = math.atan2(zm, p * (1.0 - WGS84_E2))
    for _ in range(5):
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * math.sin(lat0) ** 2)
        h = p / math.cos(lat0) - n
        lat0 = math.atan2(zm, p * (1.0 - WGS84_E2 * n / (n + h)))
    return math.degrees(lat0), lon0


def to_local_enu(lat_deg, lon_deg, lat0=None, lon0=None, h=0.0):
    """Project stations onto the tangent plane at (lat0, lon0).

    Returns ``(x_east, y_north, (lat0, lon0))`` in metres. If the origin is not
    given, the stations' mean position is used.
    """
    if lat0 is None or lon0 is None:
        lat0, lon0 = mean_position(lat_deg, lon_deg)
    X, Y, Z = geodetic_to_ecef(lat_deg, lon_deg, h)
    X0, Y0, Z0 = geodetic_to_ecef(lat0, lon0, 0.0)
    d = np.stack([X - X0, Y - Y0, Z - Z0], axis=-1)
    e0, n0, _ = enu_basis(lat0, lon0)
    return d @ e0, d @ n0, (lat0, lon0)


def rotate_velocities_to_frame(lat_deg, lon_deg, ve, vn, lat0, lon0, cov_en=None):
    """Rotate each station's (vE, vN) into the E/N axes of the frame at (lat0, lon0).

    Uses the angle between the station's local east axis and the reference
    east axis (a pure 2-D rotation, so horizontal speeds are preserved).
    ``cov_en`` (n, 2, 2) is rotated the same way. Returns ``(ve_r, vn_r, cov_r)``.
    """
    ve = np.asarray(ve, dtype=float)
    vn = np.asarray(vn, dtype=float)
    e_i, _, _ = enu_basis(lat_deg, lon_deg)
    e0, n0, _ = enu_basis(lat0, lon0)
    phi = np.arctan2(e_i @ n0, e_i @ e0)
    c, s = np.cos(phi), np.sin(phi)
    R = np.empty((len(ve), 2, 2))
    R[:, 0, 0], R[:, 0, 1] = c, -s
    R[:, 1, 0], R[:, 1, 1] = s, c
    v = np.einsum("kij,kj->ki", R, np.stack([ve, vn], axis=-1))
    cov_r = None
    if cov_en is not None:
        cov_r = np.einsum("kij,kjl,kml->kim", R, np.asarray(cov_en), R)
    return v[:, 0], v[:, 1], cov_r


# ---------------------------------------------------------------------------
# UTM (classic mode)
# ---------------------------------------------------------------------------

def utm_zone(lon_deg):
    """Standard 6-degree UTM zone number (1-60) for longitude(s) in degrees."""
    lon = (np.asarray(lon_deg, dtype=float) + 180.0) % 360.0 - 180.0
    zone = np.floor((lon + 180.0) / 6.0).astype(int) + 1
    return np.clip(zone, 1, 60)


def zone_central_meridian(zone):
    return -183.0 + 6.0 * np.asarray(zone, dtype=float)


def westernmost_zone(lon_deg):
    """Westernmost UTM zone spanned by a small group of stations.

    Handles groups straddling the antimeridian (zones 60 and 1 -> 60).
    """
    zones = np.unique(utm_zone(lon_deg))
    if zones.max() - zones.min() > 30:  # wraps the antimeridian
        return int(zones[zones > 30].min())
    return int(zones.min())


def utm_dutch_series(lat_deg, lon_deg, lon0_deg):
    """Forward UTM with the USGS/Army K1-K5 series (Cronin's spreadsheet form).

    Returns ``(easting, northing)`` in metres for the given central meridian.
    Northern-hemisphere northing (no 10,000 km false northing), as in the
    spreadsheet; for relative strain coordinates only differences matter.
    """
    a, b = WGS84_A, 6356752.3142  # Cronin uses this rounded b
    e = math.sqrt(1.0 - b ** 2 / a ** 2)
    e2p = (e * a / b) ** 2
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    dlon = (np.asarray(lon_deg, dtype=float) - lon0_deg + 180.0) % 360.0 - 180.0
    p = np.radians(dlon)
    s, c, t = np.sin(lat), np.cos(lat), np.tan(lat)
    nu = a / np.sqrt(1.0 - e ** 2 * s ** 2)
    M = a * (
        (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * lat
        - (3 * e ** 2 / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * np.sin(2 * lat)
        + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * np.sin(4 * lat)
        - (35 * e ** 6 / 3072) * np.sin(6 * lat)
    )
    k0 = UTM_K0
    K1 = M * k0
    K2 = k0 * nu * np.sin(2 * lat) / 4.0
    K3 = (k0 * nu * s * c ** 3 / 24.0) * (5 - t ** 2 + 9 * e2p * c ** 2 + 4 * e2p ** 2 * c ** 4)
    K4 = k0 * nu * c
    K5 = (k0 * nu * c ** 3 / 6.0) * (1 - t ** 2 + e2p * c ** 2)
    northing = K1 + K2 * p ** 2 + K3 * p ** 4
    easting = 500000.0 + K4 * p + K5 * p ** 3
    return easting, northing


# Krüger series (accurate to ~1 mm within several degrees of the CM). Used to
# cross-check the classic series and to invert UTM for test data.
def _kruger_coeffs():
    n = WGS84_F / (2.0 - WGS84_F)
    A = WGS84_A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = [
        n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16,
        13 * n ** 2 / 48 - 3 * n ** 3 / 5,
        61 * n ** 3 / 240,
    ]
    beta = [
        n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96,
        n ** 2 / 48 + n ** 3 / 15,
        17 * n ** 3 / 480,
    ]
    delta = [
        2 * n - 2 * n ** 2 / 3 - 2 * n ** 3,
        7 * n ** 2 / 3 - 8 * n ** 3 / 5,
        56 * n ** 3 / 15,
    ]
    return n, A, alpha, beta, delta


def utm_kruger(lat_deg, lon_deg, lon0_deg):
    n, A, alpha, _, _ = _kruger_coeffs()
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    dl = np.radians((np.asarray(lon_deg, dtype=float) - lon0_deg + 180.0) % 360.0 - 180.0)
    k = 2 * math.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(lat)) - k * np.arctanh(k * np.sin(lat)))
    xi_p = np.arctan2(t, np.cos(dl))
    eta_p = np.arctanh(np.sin(dl) / np.sqrt(1 + t ** 2))
    xi = xi_p + sum(a * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p) for j, a in enumerate(alpha, 1))
    eta = eta_p + sum(a * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p) for j, a in enumerate(alpha, 1))
    return 500000.0 + UTM_K0 * A * eta, UTM_K0 * A * xi


def utm_kruger_inverse(easting, northing, lon0_deg):
    n, A, _, beta, delta = _kruger_coeffs()
    xi = np.asarray(northing, dtype=float) / (UTM_K0 * A)
    eta = (np.asarray(easting, dtype=float) - 500000.0) / (UTM_K0 * A)
    xi_p = xi - sum(b * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, b in enumerate(beta, 1))
    eta_p = eta - sum(b * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, b in enumerate(beta, 1))
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    lat = chi + sum(d * np.sin(2 * j * chi) for j, d in enumerate(delta, 1))
    lon = lon0_deg + np.degrees(np.arctan2(np.sinh(eta_p), np.cos(xi_p)))
    return np.degrees(lat), lon


def to_utm_classic(lat_deg, lon_deg):
    """Cronin-style UTM for a station group: all on the westernmost zone's CM.

    Returns ``(easting, northing, zone)``.
    """
    zone = westernmost_zone(lon_deg)
    e, n = utm_dutch_series(lat_deg, lon_deg, float(zone_central_meridian(zone)))
    return e, n, zone
