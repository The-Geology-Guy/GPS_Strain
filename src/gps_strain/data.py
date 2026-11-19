"""Getting station velocities.

Sources
-------
* **Nevada Geodetic Laboratory (NGL) MIDAS velocities** - default. Free, no
  login, global, updated weekly, IGS20 or plate-fixed frames (e.g. ``NA`` =
  North America fixed). One file holds position, velocity and uncertainty for
  every station. Please cite Blewitt, Hammond & Kreemer (2018, Eos,
  doi:10.1029/2018EO104623) and Blewitt et al. (2016, JGR,
  doi:10.1002/2015JB012552).
* **NGL ``.tenv3`` daily time series** + :func:`fit_velocity` for users who want
  to estimate a velocity themselves over a chosen time window.
* **Your own table** (CSV/Excel), e.g. velocities copied from EarthScope/GAGE
  station pages, as in the GETSI exercises.

Internally velocities are **m/yr**. CSV inputs may be in mm/yr (the default
for :func:`read_table`) as published on station pages.
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

NGL_BASE = "https://geodesy.unr.edu/gps_timeseries/IGS20"
MIDAS_COLUMNS = [
    "site", "version", "t_first", "t_last", "duration", "n_epochs", "n_good", "n_pairs",
    "ve", "vn", "vu", "sig_e", "sig_n", "sig_u", "e0", "n0", "u0",
    "out_e", "out_n", "out_u", "sd_e", "sd_n", "sd_u", "n_steps", "lat", "lon", "height",
]
REQUIRED = ["site", "lat", "lon", "ve", "vn", "sig_e", "sig_n"]
USER_AGENT = "gps-strain/2.0 (+https://github.com/The-Geology-Guy/GPS_Strain)"


def cache_dir() -> Path:
    d = Path(os.environ.get("GPS_STRAIN_CACHE", Path.home() / ".cache" / "gps_strain"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path, max_age_days: float = 7.0) -> Path:
    """Download ``url`` to ``dest`` unless a fresh cached copy exists."""
    if dest.exists() and (time.time() - dest.stat().st_mtime) < max_age_days * 86400:
        return dest
    import requests

    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    r.raise_for_status()
    if r.content[:200].lstrip().lower().startswith((b"<!doctype", b"<html")):
        raise RuntimeError(f"{url} returned an HTML page instead of data")
    dest.write_bytes(r.content)
    return dest


def _is_file(obj) -> bool:
    if isinstance(obj, Path):
        return obj.exists()
    if isinstance(obj, str) and "\n" not in obj and len(obj) < 4096:
        try:
            return Path(obj).exists()
        except OSError:
            return False
    return False


# ---------------------------------------------------------------------------
# Normalising tables
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a station table (m/yr) and wrap longitudes to [-180, 180)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"station table is missing columns: {missing}")
    out = df.copy()
    out["site"] = out["site"].astype(str).str.strip().str.upper()
    out["lon"] = (out["lon"].astype(float) + 180.0) % 360.0 - 180.0
    if "corr_en" not in out:
        out["corr_en"] = 0.0
    return out.reset_index(drop=True)


_ALIASES = {
    "site": ["site", "station", "sta", "name", "id"],
    "lat": ["lat", "latitude"],
    "lon": ["lon", "long", "longitude"],
    "ve": ["ve", "v_e", "east_velocity", "e_velocity", "east", "vel_e"],
    "vn": ["vn", "v_n", "north_velocity", "n_velocity", "north", "vel_n"],
    "sig_e": ["sig_e", "sigma_e", "se", "e_uncertainty", "east_uncertainty", "unc_e"],
    "sig_n": ["sig_n", "sigma_n", "sn", "n_uncertainty", "north_uncertainty", "unc_n"],
    "corr_en": ["corr_en", "rho_en", "rho", "corr"],
}


def read_table(path_or_buffer, units: str = "mm/yr") -> pd.DataFrame:
    """Read a CSV/Excel station table. Column names are matched loosely.

    Units of ``ve, vn, sig_e, sig_n`` are ``"mm/yr"`` (default) or ``"m/yr"``.
    """
    name = str(path_or_buffer).lower()
    df = pd.read_excel(path_or_buffer) if name.endswith((".xls", ".xlsx")) else pd.read_csv(path_or_buffer)

    def key(c):
        return "".join(ch for ch in c.lower().split("(")[0] if ch.isalnum() or ch == "_").strip("_").replace("_", "")

    lookup = {key(c): c for c in df.columns}
    rename = {}
    for std, aliases in _ALIASES.items():
        for a in aliases:
            k = a.replace("_", "")
            if k in lookup:
                rename[lookup[k]] = std
                break
    df = df.rename(columns=rename)
    scale = {"mm/yr": 1e-3, "m/yr": 1.0}[units]
    for c in ("ve", "vn", "sig_e", "sig_n"):
        if c in df:
            df[c] = df[c].astype(float) * scale
    return normalize(df)


def from_records(records, units: str = "mm/yr") -> pd.DataFrame:
    """Build a station table from dicts with keys site, lat, lon, ve, vn, sig_e, sig_n."""
    df = pd.DataFrame(records)
    scale = {"mm/yr": 1e-3, "m/yr": 1.0}[units]
    for c in ("ve", "vn", "sig_e", "sig_n"):
        df[c] = df[c].astype(float) * scale
    return normalize(df)


# ---------------------------------------------------------------------------
# NGL MIDAS
# ---------------------------------------------------------------------------

def parse_midas(text_or_path) -> pd.DataFrame:
    """Parse an NGL MIDAS velocity file (27 whitespace-separated columns).

    Accepts a path or the file's text.
    """
    text = Path(text_or_path).read_text() if _is_file(text_or_path) else str(text_or_path)
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 27 or line.startswith(("-", "#")):
            continue
        rows.append(parts[:27])
    df = pd.DataFrame(rows, columns=MIDAS_COLUMNS)
    num = [c for c in MIDAS_COLUMNS if c not in ("site", "version")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["lat", "lon", "ve", "vn", "sig_e", "sig_n"])
    return normalize(df)


def load_midas(plate: str = "NA", max_age_days: float = 7.0) -> pd.DataFrame:
    """Download (with weekly caching) the NGL MIDAS velocity table.

    ``plate``: ``"IGS"`` for the global IGS20 frame or a plate code for a
    plate-fixed frame (``"NA"`` North America, ``"PA"`` Pacific, ``"EU"`` ...).
    """
    fname = f"midas.{plate}.txt"
    path = _download(f"{NGL_BASE}/midas/{fname}", cache_dir() / f"IGS20_{fname}", max_age_days)
    return parse_midas(path)


def select_sites(table: pd.DataFrame, sites) -> pd.DataFrame:
    """Rows of ``table`` for ``sites`` in the given order (raises if any missing)."""
    sites = [s.strip().upper() for s in sites]
    idx = table.drop_duplicates("site").set_index("site")
    missing = [s for s in sites if s not in idx.index]
    if missing:
        raise KeyError(f"stations not found: {missing}")
    return idx.loc[sites].reset_index()


def stations_in_bbox(table, min_lon, max_lon, min_lat, max_lat, min_duration=None):
    """Stations inside a lon/lat box (handles boxes across the antimeridian)."""
    lon = table["lon"]
    in_lon = (lon >= min_lon) & (lon <= max_lon) if min_lon <= max_lon else (lon >= min_lon) | (lon <= max_lon)
    sel = table[in_lon & table["lat"].between(min_lat, max_lat)]
    if min_duration is not None and "duration" in sel:
        sel = sel[sel["duration"] >= min_duration]
    return sel.reset_index(drop=True)


def stations_near(table, lat, lon, radius_km=50.0):
    """Stations within ``radius_km`` of a point, sorted by distance."""
    R = 6371.0
    p1, p2 = np.radians(lat), np.radians(table["lat"].to_numpy())
    dl = np.radians(table["lon"].to_numpy() - lon)
    d = 2 * R * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
    out = table.assign(distance_km=d)
    return out[out.distance_km <= radius_km].sort_values("distance_km").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time series and velocity estimation
# ---------------------------------------------------------------------------

TENV3_COLUMNS = [
    "site", "date", "decyr", "mjd", "gps_week", "dow", "reflon", "e0", "east", "n0", "north",
    "u0", "up", "ant", "sig_e", "sig_n", "sig_u", "corr_en", "corr_eu", "corr_nu", "lat", "lon", "height",
]


def parse_tenv3(text_or_path) -> pd.DataFrame:
    """Parse an NGL ``.tenv3`` time series; adds ``e``, ``n``, ``u`` in metres."""
    src = text_or_path if _is_file(text_or_path) else io.StringIO(str(text_or_path))
    df = pd.read_csv(src, sep=r"\s+", comment="#", header=None, skiprows=1, names=TENV3_COLUMNS)
    df["e"] = df["e0"] + df["east"]
    df["n"] = df["n0"] + df["north"]
    df["u"] = df["u0"] + df["up"]
    return df


def load_tenv3(site: str, plate: str = "NA", max_age_days: float = 7.0) -> pd.DataFrame:
    site = site.upper()
    if plate == "IGS":
        url, name = f"{NGL_BASE}/tenv3/IGS20/{site}.tenv3", f"{site}.IGS20.tenv3"
    else:
        url, name = f"{NGL_BASE}/tenv3/{plate}/{site}.{plate}.tenv3", f"{site}.{plate}.tenv3"
    return parse_tenv3(_download(url, cache_dir() / name, max_age_days))


def fit_velocity(t, pos, steps=(), seasonal=True, sigma_scale=1.0):
    """Least-squares velocity from a position time series.

    Model: ``pos = a + v*(t - t0) + [annual + semi-annual sin/cos] + sum(H(t - t_k) * o_k)``.

    The rate is the slope of this fit (NOT mean displacement / duration, which
    the 2020 package used and which is biased). The returned sigma is the
    formal white-noise value times ``sigma_scale``. Formal white-noise sigmas
    for daily GNSS are optimistic (often by 5-10x) because the noise is
    coloured (Williams 2003; Bos et al. 2013 / Hector). Prefer MIDAS
    velocities and uncertainties for anything beyond a classroom exercise.

    Returns ``(velocity, sigma)`` in units of pos per unit t.
    """
    t = np.asarray(t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    ok = np.isfinite(t) & np.isfinite(pos)
    t, pos = t[ok], pos[ok]
    t0 = t.mean()
    cols = [np.ones_like(t), t - t0]
    if seasonal:
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
    for ts in steps:
        cols.append((t >= ts).astype(float))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, pos, rcond=None)
    resid = pos - A @ coef
    dof = max(len(t) - A.shape[1], 1)
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(coef[1]), float(np.sqrt(cov[1, 1])) * sigma_scale


def velocities_from_timeseries(sites, plate="NA", start=None, end=None, sigma_scale=5.0):
    """Station table built by fitting NGL ``.tenv3`` series for each site.

    ``sigma_scale`` inflates the formal white-noise uncertainties (default 5x,
    a common rule of thumb for flicker noise); see :func:`fit_velocity`.
    """
    rows = []
    for s in sites:
        ts = load_tenv3(s, plate)
        if start is not None:
            ts = ts[ts.decyr >= start]
        if end is not None:
            ts = ts[ts.decyr <= end]
        if len(ts) < 365:
            raise ValueError(f"{s}: fewer than one year of data in the requested window")
        ve, se = fit_velocity(ts.decyr, ts.e, sigma_scale=sigma_scale)
        vn, sn = fit_velocity(ts.decyr, ts.n, sigma_scale=sigma_scale)
        rows.append(dict(site=s, lat=ts.lat.iloc[-1], lon=ts.lon.iloc[-1], ve=ve, vn=vn, sig_e=se, sig_n=sn))
    return normalize(pd.DataFrame(rows))
