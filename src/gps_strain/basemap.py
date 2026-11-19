"""Web Mercator helpers and optional USGS map tiles, without cartopy.

Map axes use Web Mercator (EPSG:3857) metres as data coordinates; use
:func:`lonlat_to_merc` to place anything on the map. Web Mercator is
conformal, so azimuths (velocity arrows, strain axes) are drawn correctly;
distances must be scaled by ``1/cos(lat)`` (see :func:`ground_scale`).

The default terrain for printed maps is a grey hillshade computed from a DEM
(:mod:`gps_strain.terrain`), as in GEOSLAM. The tile basemaps below are the
optional alternatives, the same USGS The National Map tiles geoslam.org uses:

``usgs-relief``, ``usgs-topo``, ``usgs-imagery``
    US government works: public domain, no key, no print restrictions.
    Coverage is the United States and territories only.

Tiles are cached on disk (``~/.cache/gps_strain/tiles``).
"""
from __future__ import annotations

import io
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

R_MERC = 6378137.0
ORIGIN = math.pi * R_MERC  # half the world width in metres


@dataclass(frozen=True)
class TileProvider:
    name: str
    url: str
    attribution: str
    max_zoom: int = 18
    tile_size: int = 256


PROVIDERS = {
    "usgs-relief": TileProvider(
        "usgs-relief",
        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer/tile/{z}/{y}/{x}",
        "Basemap: USGS The National Map (public domain)",
        max_zoom=13,
    ),
    "usgs-topo": TileProvider(
        "usgs-topo",
        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}",
        "Basemap: USGS The National Map (public domain)",
        max_zoom=16,
    ),
    "usgs-imagery": TileProvider(
        "usgs-imagery",
        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
        "Basemap: USGS The National Map (public domain)",
        max_zoom=16,
    ),
}
DEFAULT_PROVIDER = "usgs-relief"
USER_AGENT = "gps-strain/2.0 (+https://github.com/The-Geology-Guy/GPS_Strain)"


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def lonlat_to_merc(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.clip(np.asarray(lat, dtype=float), -85.05112878, 85.05112878)
    x = R_MERC * np.radians(lon)
    y = R_MERC * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def merc_to_lonlat(x, y):
    lon = np.degrees(np.asarray(x, dtype=float) / R_MERC)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, dtype=float) / R_MERC)) - np.pi / 2)
    return lon, lat


def ground_scale(lat):
    """Web Mercator metres per ground metre at latitude ``lat``."""
    return 1.0 / math.cos(math.radians(lat))


# ---------------------------------------------------------------------------
# Tiles
# ---------------------------------------------------------------------------

def _tile_bounds(z, x, y):
    n = 2 ** z
    size = 2 * ORIGIN / n
    x0 = -ORIGIN + x * size
    y1 = ORIGIN - y * size
    return x0, x0 + size, y1 - size, y1


def choose_zoom(extent_merc, target_px=2000, max_zoom=18, tile_size=256):
    x0, x1, _, _ = extent_merc
    width = abs(x1 - x0)
    z = math.log2(target_px * 2 * ORIGIN / (tile_size * width))
    return int(max(0, min(max_zoom, round(z))))


def _cache_root() -> Path:
    d = Path(os.environ.get("GPS_STRAIN_CACHE", Path.home() / ".cache" / "gps_strain")) / "tiles"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch_tile(provider: TileProvider, z, x, y):
    """Download one tile (cached)."""
    from PIL import Image

    cache = _cache_root() / provider.name / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        return Image.open(cache).convert("RGB")
    import requests

    url = provider.url.format(z=z, x=x, y=y)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache)
    return img


def fetch_basemap(extent_merc, provider=DEFAULT_PROVIDER, zoom=None,
                  tile_source: Callable | None = None, target_px=2000):
    """Stitched basemap image covering ``extent_merc`` = (x0, x1, y0, y1).

    Returns ``(image_array, extent)`` for ``imshow(..., extent=extent)``.
    ``tile_source(z, x, y) -> PIL.Image`` overrides the network (tests/offline).
    """
    from PIL import Image

    prov = PROVIDERS[provider] if isinstance(provider, str) else provider
    z = zoom if zoom is not None else choose_zoom(extent_merc, target_px, prov.max_zoom, 256)
    n = 2 ** z
    x0, x1, y0, y1 = extent_merc
    size = 2 * ORIGIN / n
    tx0 = int(math.floor((min(x0, x1) + ORIGIN) / size))
    tx1 = int(math.floor((max(x0, x1) + ORIGIN) / size))
    ty0 = int(math.floor((ORIGIN - max(y0, y1)) / size))
    ty1 = int(math.floor((ORIGIN - min(y0, y1)) / size))
    ty0, ty1 = max(ty0, 0), min(ty1, n - 1)
    jobs = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]
    getter = tile_source or (lambda zz, xx, yy: _fetch_tile(prov, zz, xx % n, yy))
    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = list(pool.map(lambda j: getter(z, j[0], j[1]), jobs))
    ts = tiles[0].size[0]
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * ts, (ty1 - ty0 + 1) * ts))
    for (tx, ty), im in zip(jobs, tiles):
        if im.size[0] != ts:
            im = im.resize((ts, ts))
        mosaic.paste(im, ((tx - tx0) * ts, (ty - ty0) * ts))
    left, _, _, top = _tile_bounds(z, tx0, ty0)
    _, right, bottom, _ = _tile_bounds(z, tx1, ty1)
    return np.asarray(mosaic), (left, right, bottom, top)


def provider_attribution(provider) -> str:
    prov = PROVIDERS[provider] if isinstance(provider, str) else provider
    return prov.attribution
