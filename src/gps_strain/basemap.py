"""Web-map tile basemaps for matplotlib, without cartopy.

Tiles are fetched in Web Mercator (EPSG:3857), stitched with Pillow and drawn
with ``imshow``. Map axes therefore use Web Mercator metres as data
coordinates; use :func:`lonlat_to_merc` to place anything on the map. Web
Mercator is conformal, so azimuths (velocity arrows, strain axes) are drawn
correctly; distances must be scaled by ``1/cos(lat)`` (see :func:`ground_scale`).

Providers
---------
``stadia-terrain-background``
    The Stamen "terrain-background" style the 2020 maps used, now hosted by
    Stadia Maps. Needs an API key (env ``STADIA_API_KEY`` or ``api_key=``).
    Stadia's free tier is for non-commercial use; printed/published maps may
    need a paid plan - see docs/research/04_costs.md.
``esri-terrain`` (default)
    Esri World Terrain Base: hillshade over a tan/green land palette with blue
    water - the closest match to Stamen terrain-background. For an app you
    distribute, Esri requires an ArcGIS Location Platform account and API key
    (env ``ESRI_API_KEY``; 2 million basemap tiles/month free, then
    $0.15 per 1,000). Keyless use is only for holders of an ArcGIS licence.
``usgs-relief``, ``usgs-topo``, ``usgs-imagery``
    USGS The National Map basemaps (the same tiles the GEOSLAM web app uses).
    US government works: public domain, no key, no print restrictions.
    Coverage is the United States and territories only.
``esri-shaded-relief``, ``esri-physical``, ``esri-imagery``, ``opentopomap``
    Alternatives. All require the attribution printed on the map (done
    automatically).
``synthetic``
    Procedural offline preview tiles (no network), see
    :func:`synthetic_terrain_tile`.

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
    needs_key: bool = False
    tile_size: int = 256


PROVIDERS = {
    "stadia-terrain-background": TileProvider(
        "stadia-terrain-background",
        "https://tiles.stadiamaps.com/tiles/stamen_terrain_background/{z}/{x}/{y}@2x.png?api_key={key}",
        "Map tiles © Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors",
        max_zoom=16, needs_key=True, tile_size=512,
    ),
    "stadia-terrain": TileProvider(
        "stadia-terrain",
        "https://tiles.stadiamaps.com/tiles/stamen_terrain/{z}/{x}/{y}@2x.png?api_key={key}",
        "Map tiles © Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors",
        max_zoom=16, needs_key=True, tile_size=512,
    ),
    "esri-terrain": TileProvider(
        "esri-terrain",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
        "Basemap: Esri World Terrain Base — Sources: Esri, USGS, NOAA",
        max_zoom=13,
    ),
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
    "esri-shaded-relief": TileProvider(
        "esri-shaded-relief",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
        "Basemap: Esri World Shaded Relief — Copyright © 2014 Esri",
        max_zoom=13,
    ),
    "esri-physical": TileProvider(
        "esri-physical",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Physical_Map/MapServer/tile/{z}/{y}/{x}",
        "Basemap: Esri World Physical Map — Source: US National Park Service",
        max_zoom=8,
    ),
    "esri-imagery": TileProvider(
        "esri-imagery",
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        max_zoom=17,
    ),
    "opentopomap": TileProvider(
        "opentopomap",
        "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "Map: © OpenTopoMap (CC-BY-SA) — Data: © OpenStreetMap contributors, SRTM",
        max_zoom=15,
    ),
}
DEFAULT_PROVIDER = "esri-terrain"
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


def _fetch_tile(provider: TileProvider, z, x, y, key=None):
    """Download one tile (cached)."""
    from PIL import Image

    cache = _cache_root() / provider.name / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        return Image.open(cache).convert("RGB")
    import requests

    url = provider.url.format(z=z, x=x, y=y, key=key or "")
    if key and provider.name.startswith("esri"):
        url += f"?token={key}"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    cache.parent.mkdir(parents=True, exist_ok=True)
    img.save(cache)
    return img


def fetch_basemap(extent_merc, provider="esri-terrain", zoom=None, api_key=None,
                  tile_source: Callable | None = None, target_px=2000):
    """Stitched basemap image covering ``extent_merc`` = (x0, x1, y0, y1).

    Returns ``(image_array, extent)`` for ``imshow(..., extent=extent)``.
    ``tile_source(z, x, y) -> PIL.Image`` overrides the network (tests/offline).
    """
    from PIL import Image

    prov = PROVIDERS[provider] if isinstance(provider, str) else provider
    if prov.name.startswith("esri"):
        api_key = api_key or os.environ.get("ESRI_API_KEY")
    if prov.needs_key:
        api_key = api_key or os.environ.get("STADIA_API_KEY")
        if not api_key and tile_source is None:
            raise RuntimeError(
                f"{prov.name} needs an API key: set STADIA_API_KEY or pass api_key= "
                "(keys at https://client.stadiamaps.com/signup/)"
            )
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
    getter = tile_source or (lambda zz, xx, yy: _fetch_tile(prov, zz, xx % n, yy, api_key))
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


def synthetic_terrain_tile(z, x, y, size=256):
    """Offline stand-in tile: procedural shaded relief in a Stamen-like palette.

    Deterministic in world coordinates, so neighbouring tiles join seamlessly.
    Used by tests and when no network is available (``basemap="synthetic"``).
    """
    from PIL import Image

    x0, x1, y0, y1 = _tile_bounds(z, x, y)
    gx = np.linspace(x0, x1, size, endpoint=False)
    gy = np.linspace(y1, y0, size, endpoint=False)
    X, Y = np.meshgrid(gx / 3e4, gy / 3e4)
    h = np.zeros_like(X)
    rng = np.random.default_rng(7)
    for k in range(7):
        f = 0.35 * 1.9 ** k
        th = rng.uniform(0, np.pi, 2)
        ph = rng.uniform(0, 2 * np.pi, 2)
        u = X * np.cos(th[0]) + Y * np.sin(th[0])
        v = X * np.cos(th[1]) + Y * np.sin(th[1])
        h += (0.6 ** k) * np.sin(f * u + ph[0]) * np.sin(f * v + ph[1])
    dy, dx = np.gradient(h)
    shade = np.clip(0.85 + 12.0 * (dx - dy), 0.45, 1.25)
    t = np.clip((h + 1.0) / 2.0, 0, 1)
    low = np.array([226, 220, 186]) / 255.0
    mid = np.array([196, 205, 160]) / 255.0
    high = np.array([245, 247, 250]) / 255.0
    col = np.where(t[..., None] < 0.6, low + (mid - low) * (t[..., None] / 0.6),
                   mid + (high - mid) * ((t[..., None] - 0.6) / 0.4))
    rgb = np.clip(col * shade[..., None], 0, 1)
    water = h < -0.75
    rgb[water] = np.array([160, 190, 220]) / 255.0
    return Image.fromarray((rgb * 255).astype(np.uint8))
