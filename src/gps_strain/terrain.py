"""Grey shaded-relief terrain for printed maps, drawn from a DEM (no map tiles).

This is the GEOSLAM approach (``geoslam.sources.dems`` + ``geoslam.report``):
fetch elevations, compute a hillshade, and draw it in grey under the map
layers. Because nothing comes from a tile service, there are no API keys and
no print or commercial-use limits beyond attribution.

DEM sources
-----------
``"3dep"``
    USGS 3DEP ImageServer (United States and territories, public domain, no
    key). The grid is requested directly in Web Mercator (EPSG:3857) at the
    size of the map image, as raw float32 plus a 1-bit validity mask - the
    same request geoslam.org makes from the browser.
``"copernicus"``
    Copernicus DEM GLO-30 / GLO-90 Cloud-Optimised GeoTIFFs on AWS Open Data
    (global, no key). Whole 1x1 degree tiles are downloaded and cached;
    GLO-90 is used when a map pixel is coarser than about 60 m, which is the
    usual case for GPS triangles, so only a few small files are needed.
``"auto"`` (default)
    3DEP when the map centre is in the US, then Copernicus for any cells
    3DEP left empty (border areas); Copernicus everywhere else.
callable
    ``f(extent_merc, width, height) -> z`` returning elevations in metres
    (rows north to south, NaN = no data). Used for tests and offline previews
    (:func:`synthetic_dem`).

The hillshade is computed on the map's own Web Mercator grid with the true
ground spacing of each row (Mercator cell size x cos(latitude)), so slopes are
not exaggerated toward the poles. Water (no data, or perfectly flat cells, as
3DEP hydro-flattened lakes and Copernicus sea level are) is left out and shows
as a pale fill instead of a flat mid-grey.

Vertical datums differ (3DEP NAVD88, Copernicus EGM2008); this does not
matter for a hillshade.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .basemap import USER_AGENT, merc_to_lonlat

THREE_DEP = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/"
             "ImageServer/exportImage")
COP_URL = "https://copernicus-dem-{res}m.s3.amazonaws.com/{name}/{name}.tif"

CREDIT_3DEP = "Terrain: USGS 3D Elevation Program (public domain)"
CREDIT_COP = ("Terrain produced using Copernicus WorldDEM-{res} © DLR e.V. 2010-2014 and © Airbus "
              "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union "
              "and ESA; all rights reserved")
WATER_RGB = (0.957, 0.957, 0.949)  # GEOSLAM's report background (#f4f4f2)


@dataclass(frozen=True)
class HillshadeStyle:
    """Lighting and grey ramp. Defaults follow GEOSLAM's report figures:
    light from the northwest, vertical exaggeration 2, and the grey ramp
    0.18-0.90 (dark grey shadows, never pure black or white)."""

    azimuth: float = 315.0
    altitude: float = 30.0
    exaggeration: float = 2.0
    dark: float = 0.18
    light: float = 0.90


DEFAULT_STYLE = HillshadeStyle()


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    d = Path(os.environ.get("GPS_STRAIN_CACHE", Path.home() / ".cache" / "gps_strain")) / "dem"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url, params=None, timeout=120, cache=True):
    """HTTP GET returning bytes, cached on disk. Returns None for 403/404
    (Copernicus has no tiles over open ocean; S3 answers 403 or 404)."""
    import requests

    key = hashlib.sha1((url + repr(sorted((params or {}).items()))).encode()).hexdigest()
    f = _cache_dir() / "http" / key
    if cache and f.exists():
        return f.read_bytes()
    last = None
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            if cache:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
            return r.content
        except Exception as exc:  # noqa: BLE001 - retry any network error
            last = exc
            if attempt < 2:
                import time
                time.sleep(2 ** attempt)
    raise last


def in_us(lon, lat) -> bool:
    """Rough 3DEP coverage test (CONUS, Alaska, Hawaii, Puerto Rico), from GEOSLAM."""
    return ((-125.5 <= lon <= -66 and 24 <= lat <= 50) or (-170 <= lon <= -129 and 51 <= lat <= 72)
            or (-161 <= lon <= -154 and 18 <= lat <= 23) or (-68 <= lon <= -65 and 17.5 <= lat <= 18.7))


def grid_lonlat(extent_merc, width, height):
    """Cell-centre longitudes (per column) and latitudes (per row, north first)."""
    x0, x1, y0, y1 = extent_merc
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
    lon, _ = merc_to_lonlat(xs, np.zeros_like(xs))
    _, lat = merc_to_lonlat(np.zeros_like(ys), ys)
    return lon, lat


# ---------------------------------------------------------------------------
# USGS 3DEP
# ---------------------------------------------------------------------------

def parse_bsq_f32(raw: bytes, width: int, height: int) -> np.ndarray:
    """Decode an ImageServer ``format=bsq&pixelType=F32`` response: a raw
    little-endian float32 grid (row 0 = north) followed by a 1-bit validity
    mask, most significant bit first."""
    cells = width * height
    if raw is None or len(raw) < cells * 4:
        n = 0 if raw is None else len(raw)
        snippet = b"" if raw is None else raw[:200]
        raise RuntimeError(f"3DEP returned {n} bytes, expected at least {cells * 4}: {snippet!r}")
    z = np.frombuffer(raw[:cells * 4], dtype="<f4").astype(float).reshape(height, width)
    nmask = (cells + 7) // 8
    if len(raw) >= cells * 4 + nmask:
        bits = np.unpackbits(np.frombuffer(raw[cells * 4:cells * 4 + nmask], np.uint8))[:cells]
        z[bits.reshape(height, width) == 0] = np.nan
    z[(z < -1e4) | (z > 1e5) | ~np.isfinite(z)] = np.nan
    return z


def fetch_3dep(extent_merc, width, height) -> np.ndarray:
    """3DEP elevations on the Web Mercator grid ``extent_merc`` x (width, height)."""
    if width > 8000 or height > 8000:
        raise ValueError("3DEP exportImage is limited to 8000 x 8000 pixels")
    x0, x1, y0, y1 = extent_merc
    params = dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=3857, imageSR=3857, size=f"{width},{height}",
                  format="bsq", pixelType="F32", noDataInterpretation="esriNoDataMatchAny",
                  interpolation="RSP_BilinearInterpolation", f="image")
    return parse_bsq_f32(_get(THREE_DEP, params), width, height)


# ---------------------------------------------------------------------------
# Copernicus GLO-30 / GLO-90
# ---------------------------------------------------------------------------

def copernicus_tile_name(lat0: int, lon0: int, res: int = 30) -> str:
    """Name of the 1x1 degree tile whose south-west corner is (lat0, lon0)."""
    ns = f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}_00"
    ew = f"{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}_00"
    return f"Copernicus_DSM_COG_{10 if res == 30 else 30}_{ns}_{ew}_DEM"


@dataclass
class GeoRaster:
    """A lon/lat raster. ``lon_min``/``lat_max`` are the outer cell edges."""

    z: np.ndarray
    lon_min: float
    lat_max: float
    dlon: float
    dlat: float

    @property
    def lon_max(self):
        return self.lon_min + self.z.shape[1] * self.dlon

    @property
    def lat_min(self):
        return self.lat_max - self.z.shape[0] * self.dlat


def read_geotiff(data: bytes) -> GeoRaster:
    """Read a single-band geographic GeoTIFF with Pillow (libtiff handles the
    DEFLATE/LZW compression and floating-point predictor that tifffile cannot
    decode without the optional ``imagecodecs`` package)."""
    from PIL import Image

    im = Image.open(io.BytesIO(data))
    z = np.asarray(im, dtype=float)
    tags = im.tag_v2
    sx, sy = tags[33550][:2]
    i, j, _, x, y, _ = tags[33922][:6]
    keys = list(tags.get(34735, ()))
    raster_type = 1  # 1 = PixelIsArea, 2 = PixelIsPoint
    for k in range(4, len(keys) - 3, 4):
        if keys[k] == 1025 and keys[k + 1] == 0:
            raster_type = keys[k + 3]
    lon_min, lat_max = x - i * sx, y + j * sy
    if raster_type == 2:  # tie point is a cell centre
        lon_min, lat_max = lon_min - sx / 2, lat_max + sy / 2
    nodata = tags.get(42113)
    if nodata not in (None, ""):
        try:
            z[z == float(str(nodata).strip("\x00 "))] = np.nan
        except ValueError:
            pass
    z[(z < -1e4) | (z > 1e5)] = np.nan
    return GeoRaster(z, float(lon_min), float(lat_max), float(sx), float(sy))


def sample_bilinear(r: GeoRaster, lon, lat) -> np.ndarray:
    """Bilinear sample of ``r`` at points (cell centres at half-cell offsets)."""
    col = (np.asarray(lon) - r.lon_min) / r.dlon - 0.5
    row = (r.lat_max - np.asarray(lat)) / r.dlat - 0.5
    nr, nc = r.z.shape
    col = np.clip(col, 0, nc - 1)
    row = np.clip(row, 0, nr - 1)
    c0 = np.minimum(np.floor(col).astype(int), nc - 2 if nc > 1 else 0)
    r0 = np.minimum(np.floor(row).astype(int), nr - 2 if nr > 1 else 0)
    c1, r1 = np.minimum(c0 + 1, nc - 1), np.minimum(r0 + 1, nr - 1)
    fc, fr = col - c0, row - r0
    z = r.z
    top = z[r0, c0] * (1 - fc) + z[r0, c1] * fc
    bot = z[r1, c0] * (1 - fc) + z[r1, c1] * fc
    return top * (1 - fr) + bot * fr


def copernicus_resolution(extent_merc, width) -> int:
    """30 or 90: GLO-90 when a map pixel is coarser than ~60 m on the ground."""
    x0, x1, y0, y1 = extent_merc
    _, lat = merc_to_lonlat(0.0, (y0 + y1) / 2)
    ground_px = (x1 - x0) / width * math.cos(math.radians(float(lat)))
    return 90 if ground_px > 60 else 30


def fetch_copernicus(extent_merc, width, height, res=None, tile_getter: Callable | None = None,
                     need: np.ndarray | None = None):
    """Copernicus elevations on the Web Mercator grid. Returns ``(z, res)``.

    ``need`` (bool grid) limits the download to tiles containing those cells.
    ``tile_getter(name, res) -> bytes | None`` overrides the download (tests).
    """
    res = res or copernicus_resolution(extent_merc, width)
    lon, lat = grid_lonlat(extent_merc, width, height)
    LON, LAT = np.meshgrid(lon, lat)
    z = np.full((height, width), np.nan)
    if need is None:
        need = np.ones((height, width), bool)
    corners = np.unique(np.stack([np.floor(LAT[need]), np.floor(LON[need])], axis=1), axis=0)
    tiles = [(int(a), int(b)) for a, b in corners]
    getter = tile_getter or (lambda name, rr: _get(COP_URL.format(res=rr, name=name), timeout=300))

    def load(t):
        data = getter(copernicus_tile_name(t[0], t[1], res), res)
        return t, (read_geotiff(data) if data else None)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for (la, lo), r in pool.map(load, tiles):
            if r is None:
                continue  # open ocean: no tile
            inside = (LAT >= la) & (LAT < la + 1) & (LON >= lo) & (LON < lo + 1)
            if inside.any():
                z[inside] = sample_bilinear(r, LON[inside], LAT[inside])
    return z, res


# ---------------------------------------------------------------------------
# offline preview terrain
# ---------------------------------------------------------------------------

def _lattice(i, j, k):
    """Deterministic pseudo-random values in [-1, 1] at integer lattice points."""
    h = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + k * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFF) / 32767.5 - 1.0


def synthetic_dem(extent_merc, width, height, seed=7):
    """Procedural ridged terrain (metres), deterministic in world coordinates.

    For offline previews and tests only: it looks like terrain, but it is not
    the real landscape, and maps drawn with it say so.
    """
    x0, x1, y0, y1 = extent_merc
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y1 - (np.arange(height) + 0.5) * (y1 - y0) / height
    X, Y = np.meshgrid(xs, ys)
    z = np.zeros_like(X)
    amp, spacing = 1.0, 60_000.0
    for k in range(8):
        th = 0.61 * (k + 1)  # rotate each octave so the lattice does not show
        u = (X * math.cos(th) - Y * math.sin(th)) / spacing + 0.37 * k
        v = (X * math.sin(th) + Y * math.cos(th)) / spacing + 0.71 * k
        i, j = np.floor(u), np.floor(v)
        fu, fv = u - i, v - j
        fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
        n = ((_lattice(i, j, k + seed) * (1 - fu) + _lattice(i + 1, j, k + seed) * fu) * (1 - fv)
             + (_lattice(i, j + 1, k + seed) * (1 - fu) + _lattice(i + 1, j + 1, k + seed) * fu) * fv)
        z += amp * (1.0 - np.abs(n)) if k < 4 else amp * n
        amp *= 0.5
        spacing /= 2.0
    z = (z - 0.78) * 1600.0
    z[z < 0] = 0.0  # a few flat "lakes" at sea level, like Copernicus over water
    return z


# ---------------------------------------------------------------------------
# hillshade
# ---------------------------------------------------------------------------

def _window_range(z, size=5):
    """Max minus min over a size x size window (NaN-free input)."""
    from numpy.lib.stride_tricks import sliding_window_view

    p = size // 2
    zp = np.pad(z, p, mode="edge")
    hmax = sliding_window_view(zp, size, axis=1).max(-1)
    hmin = sliding_window_view(zp, size, axis=1).min(-1)
    return sliding_window_view(hmax, size, axis=0).max(-1) - sliding_window_view(hmin, size, axis=0).min(-1)


def water_mask(z: np.ndarray, flat_m: float = 0.01) -> np.ndarray:
    """No-data cells, plus cells in a perfectly flat 5x5 neighbourhood (open
    water: 3DEP lakes are hydro-flattened and Copernicus sea level is flat)."""
    nan = ~np.isfinite(z)
    if nan.all():
        return nan
    zf = np.where(nan, np.nanmean(z), z)
    return nan | (_window_range(zf) < flat_m)


def lambert(z, extent_merc, style: HillshadeStyle = DEFAULT_STYLE) -> np.ndarray:
    """Raw Lambertian illumination in [0, 1] on a Web Mercator grid, using
    each row's true ground spacing (Mercator cell size x cos(latitude))."""
    z = np.asarray(z, float)
    nan = ~np.isfinite(z)
    height, width = z.shape
    x0, x1, y0, y1 = extent_merc
    cx, cy = (x1 - x0) / width, (y1 - y0) / height  # Mercator metres per cell
    _, lat = grid_lonlat(extent_merc, width, height)
    coslat = np.cos(np.radians(lat))[:, None]
    zf = np.where(nan, np.nanmean(z) if (~nan).any() else 0.0, z) * style.exaggeration
    g_row, g_col = np.gradient(zf, cy, cx)
    dzdx, dzdy = g_col / coslat, -g_row / coslat
    az, alt = math.radians(style.azimuth), math.radians(style.altitude)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    return np.clip((-dzdx * lx - dzdy * ly + lz) / np.sqrt(1 + dzdx ** 2 + dzdy ** 2), 0, 1)


def shade(z, extent_merc, style: HillshadeStyle = DEFAULT_STYLE) -> np.ndarray:
    """Hillshade in [0, 1] (NaN = no data): :func:`lambert` with GEOSLAM's
    ``report.shade`` contrast stretch (1st to 99.5th percentile)."""
    z = np.asarray(z, float)
    nan = ~np.isfinite(z)
    if nan.all():
        return np.full(z.shape, np.nan)
    s = lambert(z, extent_merc, style)
    lo, hi = np.nanpercentile(s[~nan], [1, 99.5])
    if hi - lo < 1e-6:  # featureless: no stretch (avoids amplifying round-off)
        s = np.full(z.shape, 0.5)
    else:
        s = np.clip((s - lo) / (hi - lo), 0, 1)
    s[nan] = np.nan
    return s


def hillshade_rgb(z, extent_merc, style: HillshadeStyle = DEFAULT_STYLE) -> np.ndarray:
    """Grey RGB image (rows north to south) with water in a pale fill."""
    water = water_mask(z)
    land = np.where(water, np.nan, z)
    hs = shade(land, extent_merc, style)
    g = style.dark + (style.light - style.dark) * np.nan_to_num(hs, nan=0.5)
    img = np.dstack([g, g, g])
    img[water] = WATER_RGB
    return img


# ---------------------------------------------------------------------------
# the one call plotting needs
# ---------------------------------------------------------------------------

def terrain_image(extent_merc, source="auto", target_px=2400, style: HillshadeStyle = DEFAULT_STYLE):
    """Grey hillshade covering ``extent_merc`` = (x0, x1, y0, y1) in Web Mercator.

    Returns ``(rgb, credit)``; draw with ``imshow(rgb, extent=extent_merc,
    origin="upper")``. ``source`` is ``"auto"``, ``"3dep"``, ``"copernicus"``
    or a callable (see the module docstring).
    """
    x0, x1, y0, y1 = extent_merc
    width = int(target_px)
    height = max(2, int(round(width * (y1 - y0) / (x1 - x0))))
    if height > 8000:
        height, width = 8000, max(2, int(round(8000 * (x1 - x0) / (y1 - y0))))
    if callable(source):
        z = source(extent_merc, width, height)
        credit = ("Synthetic terrain (offline preview, not real relief)" if source is synthetic_dem
                  else "Terrain: user-supplied DEM")
    else:
        clon, clat = merc_to_lonlat((x0 + x1) / 2, (y0 + y1) / 2)
        use_3dep = source == "3dep" or (source == "auto" and in_us(float(clon), float(clat)))
        if source not in ("auto", "3dep", "copernicus"):
            raise ValueError(f"unknown DEM source {source!r}")
        credits = []
        z = None
        if use_3dep:
            z = fetch_3dep(extent_merc, width, height)
            credits.append(CREDIT_3DEP)
        if source == "copernicus" or (source == "auto" and (z is None or np.isnan(z).mean() > 0.02)):
            zc, res = fetch_copernicus(extent_merc, width, height,
                                       need=None if z is None else np.isnan(z))
            if z is None:
                z = zc
            else:
                fill = np.isnan(z) & np.isfinite(zc)
                z[fill] = zc[fill]
            if np.isfinite(zc).any():
                credits.append(CREDIT_COP.format(res=res))
        credit = "\n".join(credits)
    return hillshade_rgb(z, extent_merc, style), credit
