"""Mapped active faults to draw on the maps (ported from GEOSLAM ``sources/faults.py``).

* **USGS Quaternary Fault and Fold Database** (United States; public domain):
  the ArcGIS REST service queried by bounding box, layer 21 (national), the
  same query geoslam.org makes. GETSI Unit 6 asks students to map these.
* **GEM Global Active Faults Database** (global; Styron & Pagani 2020,
  CC BY-SA 4.0), used outside the US. The whole GeoJSON (~20 MB) is
  downloaded once and cached.

Faults are plain dicts ``{"properties": {...}, "lines": [ndarray(N, 2) lon/lat, ...]}``.
"""
from __future__ import annotations

import json

import numpy as np

from .terrain import _cache_dir, _get

QFAULTS = "https://earthquake.usgs.gov/arcgis/rest/services/haz/Qfaults/MapServer/{layer}/query"
QFAULT_FIELDS = "fault_name,section_name,age,slip_sense,slip_rate"
GEM_URL = ("https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/master/"
           "geojson/gem_active_faults_harmonized.geojson")

CREDIT_USGS = "Faults: USGS Quaternary Fault and Fold Database (public domain)"
CREDIT_GEM = "Faults: GEM Global Active Faults Database (Styron & Pagani 2020), CC BY-SA 4.0"


def _lines(geom):
    if geom is None:
        return []
    if geom["type"] == "LineString":
        return [np.asarray(geom["coordinates"], float)[:, :2]]
    if geom["type"] == "MultiLineString":
        return [np.asarray(c, float)[:, :2] for c in geom["coordinates"] if len(c) > 1]
    return []


def parse_geojson(js) -> list[dict]:
    """GeoJSON FeatureCollection (dict or str) -> list of fault dicts."""
    if isinstance(js, (str, bytes)):
        js = json.loads(js)
    out = []
    for f in js.get("features", []):
        ls = [ln for ln in _lines(f.get("geometry")) if len(ln) > 1]
        if ls:
            out.append({"properties": f.get("properties") or {}, "lines": ls})
    return out


def usgs_qfaults(west, south, east, north, layer=21, page=2000) -> list[dict]:
    """Quaternary faults intersecting the box (paged through the service limit)."""
    out, offset = [], 0
    while True:
        raw = _get(QFAULTS.format(layer=layer), dict(
            geometry=f"{west},{south},{east},{north}", geometryType="esriGeometryEnvelope",
            inSR=4326, spatialRel="esriSpatialRelIntersects", outFields=QFAULT_FIELDS, outSR=4326,
            f="geojson", resultOffset=offset, resultRecordCount=page), timeout=90)
        if raw is None:
            raise RuntimeError("USGS Quaternary fault service refused the request")
        js = json.loads(raw)
        if "error" in js:
            raise RuntimeError(f"USGS Quaternary fault service: {js['error']}")
        feats = parse_geojson(js)
        out += feats
        n = len(js.get("features", []))
        if n < page and not js.get("exceededTransferLimit") and not js.get("properties", {}).get("exceededTransferLimit"):
            return out
        offset += n


def _in_box(lines, west, south, east, north):
    return any(ln[:, 0].max() >= west and ln[:, 0].min() <= east and ln[:, 1].max() >= south
               and ln[:, 1].min() <= north for ln in lines)


def gem_faults(west, south, east, north) -> list[dict]:
    path = _cache_dir().parent / "gem_active_faults_harmonized.geojson"
    if not path.exists():
        raw = _get(GEM_URL, timeout=600, cache=False)
        if raw is None:
            raise RuntimeError("GEM fault database not available")
        path.write_bytes(raw)
    return [f for f in parse_geojson(path.read_text()) if _in_box(f["lines"], west, south, east, north)]


def faults_in_box(west, south, east, north, source="auto"):
    """``(faults, credit)`` for a lon/lat box. ``source``: ``"auto"`` (USGS,
    then GEM when USGS has nothing there), ``"usgs"`` or ``"gem"``."""
    if source not in ("auto", "usgs", "gem"):
        raise ValueError(f"unknown fault source {source!r}")
    if source in ("auto", "usgs"):
        try:
            res = usgs_qfaults(west, south, east, north)
            if res or source == "usgs":
                return res, CREDIT_USGS
        except Exception:
            if source == "usgs":
                raise
    return gem_faults(west, south, east, north), CREDIT_GEM
