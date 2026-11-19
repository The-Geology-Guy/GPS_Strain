"""JSON-in / JSON-out bridge for the gps-strain.org web app.

The site runs this package in the browser with Pyodide, inside a Web Worker.
JavaScript sends plain JSON strings and gets JSON strings (or file bytes)
back, so nothing but strings and bytes cross the worker boundary.

Downloads that the Python side makes itself (USGS 3DEP terrain and USGS
Quaternary faults for the printed sheets) go through the browser with a
synchronous ``XMLHttpRequest``, which Web Workers allow; see
:func:`use_browser_http`. Station velocities are not downloaded here: the
page loads a weekly copy of NGL MIDAS and sends the selected stations.

The website covers the United States only (USGS 3DEP and Quaternary-fault
coverage); :func:`analyze` rejects stations elsewhere.

Station records use the units of the station pages and the GETSI
worksheets: velocities and uncertainties in mm/yr.
"""
from __future__ import annotations

import io
import json
import math

import numpy as np

from . import basemap as bm
from . import plotting, terrain
from .data import from_records
from .strain import NANO, compute_strain

US_ONLY_MESSAGE = "gps-strain.org covers stations in the United States only for now"


# ---------------------------------------------------------------------------
# browser transport and fonts (Pyodide only)
# ---------------------------------------------------------------------------

def _xhr_fetch(url, params=None, timeout=120):
    """Synchronous GET from inside a Web Worker. Returns (status, bytes)."""
    from urllib.parse import urlencode

    from js import Uint8Array, XMLHttpRequest  # noqa: PLC0415 - Pyodide only

    full = url + ("?" + urlencode(params) if params else "")
    req = XMLHttpRequest.new()
    req.open("GET", full, False)
    req.responseType = "arraybuffer"
    req.send(None)
    status = int(req.status)
    data = Uint8Array.new(req.response).to_bytes() if req.response else b""
    return status, data


def use_browser_http():
    """Route terrain and fault downloads through the browser (call once in Pyodide)."""
    terrain.HTTP_FETCH = _xhr_fetch
    return "ok"


def add_font(url: str) -> str:
    """Download a TTF/OTF and register it with matplotlib (e.g. Gelasio, the
    free stand-in for Georgia). Returns the family name."""
    import os
    import tempfile

    from matplotlib import font_manager

    status, data = (terrain.HTTP_FETCH or _requests_fetch)(url, None, 60)
    if status != 200:
        raise RuntimeError(f"font download failed (HTTP {status})")
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(url.split("?")[0])[1] or ".ttf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)
    font_manager.fontManager.addfont(path)
    return font_manager.FontProperties(fname=path).get_name()


def _requests_fetch(url, params=None, timeout=60):
    import requests

    r = requests.get(url, params=params, timeout=timeout)
    return r.status_code, r.content


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(v):
    """JSON-safe values (NaN/inf -> None, numpy -> Python)."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    return v


def _tables(req):
    """Station tables (m/yr) for each triangle in a request."""
    tris = req.get("triangles") or []
    if not tris:
        raise ValueError("no triangle given")
    tables = []
    for tri in tris:
        stations = tri["stations"] if isinstance(tri, dict) else tri
        if len(stations) < 3:
            raise ValueError("each triangle needs three stations")
        outside = [s.get("site", "?") for s in stations if not terrain.in_us(float(s["lon"]), float(s["lat"]))]
        if outside:
            raise ValueError(f"{US_ONLY_MESSAGE} ({', '.join(outside)} is outside)")
        tables.append(from_records(stations, units="mm/yr"))
    return tables


def _merc(table):
    return bm.lonlat_to_merc(table["lon"].to_numpy(float), table["lat"].to_numpy(float))


def _ring(x, y):
    lon, lat = bm.merc_to_lonlat(np.asarray(x, float), np.asarray(y, float))
    pts = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def _line(x, y):
    lon, lat = bm.merc_to_lonlat(np.asarray(x, float), np.asarray(y, float))
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]


def _feature(geom_type, coords, **props):
    return {"type": "Feature", "geometry": {"type": geom_type, "coordinates": coords}, "properties": props}


# ---------------------------------------------------------------------------
# station data for the site (built weekly by the deploy workflow)
# ---------------------------------------------------------------------------

MIDAS_CITATION = ("Nevada Geodetic Laboratory MIDAS velocities (IGS20). Blewitt, Hammond & Kreemer 2018, "
                  "Eos 99, doi:10.1029/2018EO104623; Blewitt et al. 2016, JGR 121, doi:10.1002/2015JB012552")


def midas_us_json(table, frame="NA", generated=""):
    """Compact US-only station list for the web page: velocities and
    uncertainties in mm/yr, record length in years."""
    t = table.drop_duplicates("site")
    t = t[[terrain.in_us(float(lo), float(la)) for lo, la in zip(t["lon"], t["lat"])]].sort_values("site")
    fields = ["site", "lat", "lon", "ve", "vn", "sig_e", "sig_n", "years"]
    rows = []
    for r in t.itertuples():
        years = (r.t_last - r.t_first) if hasattr(r, "t_last") else float("nan")
        rows.append([r.site, round(float(r.lat), 6), round(float(r.lon), 6),
                     round(r.ve * 1000, 3), round(r.vn * 1000, 3), round(r.sig_e * 1000, 3),
                     round(r.sig_n * 1000, 3), round(float(years), 1) if math.isfinite(years) else None])
    return {"frame": frame, "generated": generated, "source": "NGL MIDAS", "citation": MIDAS_CITATION,
            "units": "mm/yr", "count": len(rows), "fields": fields, "rows": rows}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _result_json(r, table):
    s = r.summary()
    return _clean({
        "sites": list(r.sites),
        "summary": s,
        "report": r.report(),
        "warnings": list(r.warnings),
        "e1": r.e1 * NANO, "e2": r.e2 * NANO,
        "azimuth_s1": r.azimuth_s1, "azimuth_s2": r.azimuth_s2,
        "max_shear": r.max_shear * NANO, "area_strain": r.area_strain * NANO,
        "rotation": r.rotation * NANO, "rotation_sense": r.rotation_sense,
        "speed_mm": r.speed * 1000, "translation_azimuth": r.translation_azimuth,
        "sigma": {"e1": r.mc.get("e1", np.nan) * NANO, "e2": r.mc.get("e2", np.nan) * NANO,
                  "azimuth": r.mc.get("az1", np.nan), "max_shear": r.mc.get("max_shear", np.nan) * NANO,
                  "area_strain": r.mc.get("area", np.nan) * NANO, "rotation": r.sigma[2] * NANO},
        "centroid": [float(table["lon"].mean()), float(table["lat"].mean())],
    })


def overlays(results, tables, kind="ellipse"):
    """GeoJSON for the web map: triangles, stations, velocity arrows, and the
    strain ellipse (``kind="ellipse"``) or strain symbol (``"symbol"``) for
    each triangle, sized to stay inside its triangle (shared strain scale)."""
    feats = []
    allx = np.concatenate([_merc(t)[0] for t in tables])
    ally = np.concatenate([_merc(t)[1] for t in tables])
    extent = max(np.ptp(allx), np.ptp(ally), 1.0)

    # triangles
    for i, t in enumerate(tables):
        x, y = _merc(t)
        feats.append(_feature("LineString", _line(np.append(x, x[0]), np.append(y, y[0])),
                              layer="triangle", tri=i))

    # stations (once each) and velocity arrows
    seen = {}
    for t in tables:
        for row in t.itertuples():
            seen.setdefault(row.site, row)
    rows = list(seen.values())
    vmax = max(max(math.hypot(r.ve, r.vn) for r in rows), 1e-9)
    L = 0.22 * extent / vmax                  # metres (Mercator) per m/yr
    head = 0.045 * extent
    for i, r in enumerate(rows):
        x0, y0 = bm.lonlat_to_merc(r.lon, r.lat)
        x0, y0 = float(x0), float(y0)
        dx, dy = r.ve * L, r.vn * L
        n = math.hypot(dx, dy)
        feats.append(_feature("Point", _line([x0], [y0])[0], layer="station", site=r.site, order=i,
                              ve=round(r.ve * 1000, 3), vn=round(r.vn * 1000, 3),
                              sig_e=round(r.sig_e * 1000, 3), sig_n=round(r.sig_n * 1000, 3)))
        if n > head * 0.6:
            ux, uy = dx / n, dy / n
            hb = (x0 + dx - ux * head, y0 + dy - uy * head)
            feats.append(_feature("LineString", _line([x0, hb[0]], [y0, hb[1]]), layer="velocity"))
            px, py = -uy * head * 0.45, ux * head * 0.45
            feats.append(_feature("Polygon", [_ring([x0 + dx, hb[0] + px, hb[0] - px],
                                                    [y0 + dy, hb[1] + py, hb[1] - py])], layer="velocity_head"))

    # strain glyphs on one shared scale
    emax = max(max(abs(r.e1), abs(r.e2)) for r in results)
    kexag = 0.25 / max(emax, 1e-30)
    th = np.linspace(0, 2 * np.pi, 121)
    circ = np.vstack([np.cos(th), np.sin(th)])
    for i, (r, t) in enumerate(zip(results, tables)):
        x, y = _merc(t)
        cx, cy = float(x.mean()), float(y.mean())
        if kind == "symbol":
            size = plotting.symbol_max_size(r, x, y, max_strain=max(300.0, emax * NANO)) or 0.1 * extent
            for k, poly in plotting.strain_symbol_polygons(r, (cx, cy), size,
                                                           max_strain=max(300.0, emax * NANO)):
                feats.append(_feature("Polygon", [_ring(poly[:, 0], poly[:, 1])], layer=k, tri=i))
            continue
        R = plotting.ellipse_max_radius(r, x, y, exaggeration=kexag) or 0.1 * extent
        F = np.eye(2) + kexag * r.strain_tensor
        ell = F @ circ
        feats.append(_feature("Polygon", [_ring(cx + R * circ[0], cy + R * circ[1])], layer="circle", tri=i))
        feats.append(_feature("LineString", _line(cx + R * circ[0], cy + R * circ[1]), layer="circle_line", tri=i))
        for az, e, name in ((r.azimuth_s1, r.e1, "s1"), (r.azimuth_s2, r.e2, "s2")):
            a = math.radians(az)
            u = np.array([math.sin(a), math.cos(a)]) * (1 + kexag * e) * R
            feats.append(_feature("LineString", _line([cx - u[0], cx + u[0]], [cy - u[1], cy + u[1]]),
                                  layer=f"axis_{name}", tri=i))
        feats.append(_feature("LineString", _line(cx + R * ell[0], cy + R * ell[1]), layer="ellipse", tri=i))
    lon, lat = bm.merc_to_lonlat(np.array([allx.min(), allx.max()]), np.array([ally.min(), ally.max()]))
    return {"type": "FeatureCollection", "features": feats,
            "bbox": [float(lon[0]), float(lat[0]), float(lon[1]), float(lat[1])]}


def analyze(request: str) -> str:
    """Strain for one or more triangles.

    ``request`` (JSON): ``{"triangles": [[station, station, station], ...],
    "frame": "enu" | "utm", "glyph": "ellipse" | "symbol"}``; a station is
    ``{"site", "lat", "lon", "ve", "vn", "sig_e", "sig_n"}`` (mm/yr).
    Returns JSON ``{"ok": true, "triangles": [...], "geojson": {...}}`` or
    ``{"ok": false, "error": "..."}``.
    """
    try:
        req = json.loads(request)
        tables = _tables(req)
        frame = req.get("frame", "enu")
        results = [compute_strain(t, frame=frame) for t in tables]
        out = {"ok": True, "frame": frame,
               "triangles": [_result_json(r, t) for r, t in zip(results, tables)],
               "geojson": overlays(results, tables, req.get("glyph", "ellipse"))}
        return json.dumps(_clean(out))
    except Exception as exc:  # report to the page instead of throwing across the worker
        return json.dumps({"ok": False, "error": str(exc)})


# ---------------------------------------------------------------------------
# printed sheets
# ---------------------------------------------------------------------------

SHEETS = ("symbol", "strain", "network_symbol", "network_ellipse")


def report(request: str) -> bytes:
    """One map sheet as file bytes (PNG, PDF or SVG).

    ``request`` (JSON) is an :func:`analyze` request plus:
    ``sheet`` (symbol | strain | network_symbol | network_ellipse),
    ``basemap`` (any :data:`plotting.BASEMAPS` key), ``greyscale``,
    ``visibility``, ``faults``, ``light_azimuth``, ``sun_altitude``,
    ``exaggeration``, ``terrain_px``, ``fmt`` (png | pdf | svg), ``dpi``,
    ``title``, ``data_label``, ``velocity_label``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    req = json.loads(request)
    tables = _tables(req)
    frame = req.get("frame", "enu")
    sheet = req.get("sheet", "strain")
    if sheet not in SHEETS:
        raise ValueError(f"unknown sheet {sheet!r}")
    results = [compute_strain(t, frame=frame, mc_samples=0) for t in tables]
    basemap = req.get("basemap", "hillshade")
    if basemap not in plotting.BASEMAPS or basemap in ("hillshade-copernicus", "synthetic"):
        raise ValueError(f"basemap {basemap!r} is not offered on the website")
    style = terrain.HillshadeStyle(azimuth=float(req.get("light_azimuth", 315.0)),
                                   altitude=float(req.get("sun_altitude", 30.0)),
                                   exaggeration=float(req.get("exaggeration", 2.0)))
    terrain.TARGET_PX = int(req.get("terrain_px", 1600))
    kw = dict(basemap=basemap, greyscale=bool(req.get("greyscale")), visibility=bool(req.get("visibility")),
              faults=bool(req.get("faults", True)), fault_source="usgs", terrain_style=style,
              dem_source="3dep")  # US only: USGS 3DEP for every terrain look
    label = req.get("data_label", "NGL MIDAS")
    if sheet == "symbol":
        fig = plotting.symbol_map(results[0], tables[0], None, title=req.get("title") or None,
                                  data_label=label, **kw)
    elif sheet == "strain":
        fig = plotting.strain_map(results[0], tables[0], None, title=req.get("title") or None, data_label=label,
                                  velocity_label=req.get("velocity_label", "Velocity Relative to North America"),
                                  **kw)
    else:
        fig = plotting.network_map(results, tables, None, kind=sheet.split("_")[1],
                                   title=req.get("title") or None, **kw)
    fmt = req.get("fmt", "png")
    if fmt not in ("png", "pdf", "svg"):
        raise ValueError(f"unknown file type {fmt!r}")
    buf = io.BytesIO()
    extra = {} if sheet.startswith("network") else {"edgecolor": "k"}
    fig.savefig(buf, format=fmt, dpi=int(req.get("dpi", 150)), bbox_inches="tight", **extra)
    plt.close(fig)
    return buf.getvalue()


def results_csv(request: str) -> str:
    """Results table (one row per triangle) as CSV text."""
    req = json.loads(request)
    tables = _tables(req)
    frame = req.get("frame", "enu")
    cols = ["triangle", "frame", "e1H_nstrain_yr", "e1H_sigma", "e2H_nstrain_yr", "e2H_sigma",
            "S1H_azimuth_deg", "S1H_azimuth_sigma", "S2H_azimuth_deg", "max_shear_nstrain_yr", "max_shear_sigma",
            "area_strain_nstrain_yr", "area_strain_sigma", "rotation_nrad_yr", "rotation_sigma",
            "translation_speed_mm_yr", "translation_azimuth_deg"]
    lines = [",".join(cols)]
    for t in tables:
        r = compute_strain(t, frame=frame)
        mc = r.mc
        vals = ["-".join(r.sites), frame, r.e1 * NANO, mc.get("e1", np.nan) * NANO, r.e2 * NANO,
                mc.get("e2", np.nan) * NANO, r.azimuth_s1, mc.get("az1", np.nan), r.azimuth_s2,
                r.max_shear * NANO, mc.get("max_shear", np.nan) * NANO, r.area_strain * NANO,
                mc.get("area", np.nan) * NANO, r.rotation * NANO, r.sigma[2] * NANO, r.speed * 1000,
                r.translation_azimuth]
        lines.append(",".join(v if isinstance(v, str) else f"{float(v):.6g}" for v in vals))
    return "\n".join(lines) + "\n"
