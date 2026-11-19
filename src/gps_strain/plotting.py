"""Map products in the style of the original (2020) GPS_Strain figures.

Two layouts, matching the 2020 package:

* :func:`symbol_map` - landscape. Left column: title, infinitesimal strain
  ellipse, contraction / elongation symbol legends, logo. Right: terrain map
  with the station triangle and the strain-rate map symbol at the centroid.
* :func:`strain_map` - portrait. Terrain map with station velocity arrows and
  the strain ellipse at the centroid; below it the title, logo, ellipse legend,
  velocity key, speed table and station table.

The default background is a grey hillshade drawn from a DEM
(:mod:`gps_strain.terrain`), like GEOSLAM's printed maps; USGS tiles are an
option (:mod:`gps_strain.basemap`). No cartopy. Map axes use Web Mercator
metres as data coordinates. Web Mercator is
conformal, so arrow and strain-axis azimuths are drawn correctly.

Symbol conventions (Cronin & Resor / GETSI): black arrows pointing inward =
contraction (S2H when e2H < 0); white double-headed arrow = elongation.
S1H is drawn tomato-red and S2H cornflower-blue on the ellipse.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from . import basemap as bm
from . import faults as qf
from . import terrain
from .strain import NANO, StrainResult

S1_COLOR = "tomato"
S2_COLOR = "cornflowerblue"
SITE_COLORS = ["g", "b", "r", "orange", "m", "c"]
SERIF = ["Georgia", "Gelasio", "DejaVu Serif"]
SANS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
log = logging.getLogger("gps_strain")
HALO = [patheffects.withStroke(linewidth=4, foreground="w")]
HALO_THIN = [patheffects.withStroke(linewidth=3, foreground="w")]
ARROW_CASING = [patheffects.withStroke(linewidth=4, foreground="w"), patheffects.Normal()]


def outline(lw, width=2.5, color="w"):
    """Path effect: a ``color`` edge of ``width`` points around a line of width ``lw``
    (each dash of a dashed line gets its own edge)."""
    return [patheffects.withStroke(linewidth=lw + 2 * width, foreground=color), patheffects.Normal()]
PANEL_ALPHA = 0.85     # white panels behind the scale bar and north arrow


@dataclass(frozen=True)
class MapStyle:
    """How the overlays (triangle, arrows, ellipse, scale bar...) are drawn."""

    triangle_lw: float = 2.0
    edge: bool = False             # white edges under lines/arrows, black rims on dots
    dot_ms: float = 10.0
    arrow_width: float = 0.006
    ellipse_fill_alpha: float = 0.45
    ellipse_lw: float = 2.0
    ellipse_axis_lw: float = 6.0
    scale_fontsize: float = 12.0
    panels: bool = False           # white panels behind the scale bar and north arrow
    legend_alpha: float = 0.75
    credit_fontsize: float = 6.0
    credit_alpha: float = 0.6


CLASSIC = MapStyle()
"""The 2020 look."""
HIGH_VISIBILITY = MapStyle(triangle_lw=3.5, edge=True, dot_ms=12.0, arrow_width=0.0075,
                           ellipse_fill_alpha=0.3, ellipse_lw=3.0, ellipse_axis_lw=7.0,
                           scale_fontsize=14.0, panels=True, legend_alpha=0.92,
                           credit_fontsize=7.0, credit_alpha=PANEL_ALPHA)
"""``visibility=True``: thicker overlays with white edges, and white panels
behind the scale bar and north arrow, so everything reads over dark relief."""


def map_style(visibility=False) -> MapStyle:
    """``visibility`` may be a bool ("Increase visibility") or a MapStyle."""
    if isinstance(visibility, MapStyle):
        return visibility
    return HIGH_VISIBILITY if visibility else CLASSIC

DEFAULT_BASEMAP = "hillshade"
FAULT_COLOR = "#e6d43a"   # GEOSLAM / Cronin report figures: yellow fault traces
FAULT_ZORDER = 10         # above the terrain and grid, below every map symbol
FAULT_LABEL = "Fault"     # legend entry; the credit line names the database
HILLSHADE_SOURCES = {"hillshade": "auto", "hillshade-3dep": "3dep", "hillshade-copernicus": "copernicus"}
BASEMAPS = [*HILLSHADE_SOURCES, *bm.PROVIDERS, "synthetic", "none"]


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _axis_vec(az_deg):
    """Unit vector (east, north) for an azimuth in degrees."""
    a = np.radians(az_deg)
    return np.array([np.sin(a), np.cos(a)])


def _arrow_polygon(length, shaft_w, head_w, head_l, double=False):
    """Arrow outline along +x from 0 to ``length`` (centred on y = 0)."""
    hl = min(head_l, length * (0.45 if double else 0.8))
    s, h = shaft_w / 2, head_w / 2
    if not double:
        return np.array([[0, -s], [length - hl, -s], [length - hl, -h], [length, 0],
                         [length - hl, h], [length - hl, s], [0, s]])
    return np.array([[0, 0], [hl, -h], [hl, -s], [length - hl, -s], [length - hl, -h], [length, 0],
                     [length - hl, h], [length - hl, s], [hl, s], [hl, h]])


def _place(poly, angle_rad, origin):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return poly @ R.T + np.asarray(origin)


def exaggeration_for(result: StrainResult, target: float = 0.25) -> float:
    """Factor k so the largest principal strain rate times k equals ``target``."""
    return target / max(abs(result.e1), abs(result.e2), 1e-30)


def _symbol_fraction(e, min_strain, max_strain, lo=0.3, hi=1.0):
    """Linear size scaling of |e| (nanostrain/yr) between legend extremes."""
    v = abs(e) * NANO
    t = (v - min_strain) / (max_strain - min_strain)
    return lo + (hi - lo) * float(np.clip(t, 0, 1))


# ---------------------------------------------------------------------------
# strain ellipse
# ---------------------------------------------------------------------------

def draw_ellipse(ax, result: StrainResult, center=(0.0, 0.0), radius=1.0, exaggeration=None,
                 fill_alpha=None, lw=None, axis_lw=None, zorder=30, style=CLASSIC):
    """Strain ellipse in the 2020 style: dashed initial circle, black ellipse,
    principal axes (S1H tomato, S2H cornflower), faint initial E-W/N-S axes.

    The deformation is F = I + k*E applied to a circle of ``radius`` (data
    units) at ``center``; k defaults to :func:`exaggeration_for`. Line widths
    and fill default to ``style``; ``style.edge`` adds white edges.
    """
    fill_alpha = style.ellipse_fill_alpha if fill_alpha is None else fill_alpha
    lw = style.ellipse_lw if lw is None else lw
    axis_lw = style.ellipse_axis_lw if axis_lw is None else axis_lw
    k = exaggeration or exaggeration_for(result)
    cx, cy = center
    th = np.linspace(0, 2 * np.pi, 361)
    circ = np.vstack([np.cos(th), np.sin(th)])
    F = np.eye(2) + k * result.strain_tensor
    ell = F @ circ
    edge = (lambda w, e: outline(w, e)) if style.edge else (lambda w, e: None)
    ax.fill(cx + radius * circ[0], cy + radius * circ[1], color="w", alpha=fill_alpha, zorder=zorder, lw=0)
    if style.edge:
        ax.plot(cx + radius * circ[0], cy + radius * circ[1], color="w", lw=lw + 3.5, zorder=zorder + 1)
    circle_lw = lw + 0.5 if style.edge else lw
    ax.plot(cx + radius * circ[0], cy + radius * circ[1], color="slategrey", lw=circle_lw, ls="--",
            zorder=zorder + 1)
    # faint initial axes
    ax.plot([cx - radius, cx + radius], [cy, cy], color=S1_COLOR, alpha=0.25, lw=axis_lw,
            solid_capstyle="butt", zorder=zorder + 1)
    ax.plot([cx, cx], [cy - radius, cy + radius], color=S2_COLOR, alpha=0.25, lw=axis_lw,
            solid_capstyle="butt", zorder=zorder + 1)
    for az, e, col in ((result.azimuth_s1, result.e1, S1_COLOR), (result.azimuth_s2, result.e2, S2_COLOR)):
        u = _axis_vec(az) * (1 + k * e) * radius
        ax.plot([cx - u[0], cx + u[0]], [cy - u[1], cy + u[1]], color=col, lw=axis_lw,
                solid_capstyle="butt", zorder=zorder + 2, path_effects=edge(axis_lw, 2))
    ax.plot(cx + radius * ell[0], cy + radius * ell[1], color="k", lw=lw, zorder=zorder + 3,
            path_effects=edge(lw, 2))
    return k


# ---------------------------------------------------------------------------
# strain-rate map symbol
# ---------------------------------------------------------------------------

def draw_strain_symbol(ax, result: StrainResult, center, size, min_strain=0.1, max_strain=300.0,
                       zorder=40):
    """Map symbol at ``center``: inward black arrows for contraction, a white
    double-headed arrow for elongation, along each principal axis.

    ``size`` is the half-length (data units) of a 300-nanostrain symbol.
    """
    cx, cy = center
    max_strain = max(max_strain, abs(result.e1) * NANO, abs(result.e2) * NANO)
    items = sorted(((result.azimuth_s1, result.e1), (result.azimuth_s2, result.e2)),
                   key=lambda t: t[1] < 0)  # elongation first (drawn underneath)
    for az, e in items:
        if abs(e) * NANO < min_strain:
            continue
        frac = _symbol_fraction(e, min_strain, max_strain)
        half = size * frac
        ang = math.radians(90.0 - az)  # azimuth -> math angle
        if e > 0:  # elongation: white double arrow through the centre
            w = size * 0.16
            poly = _arrow_polygon(2 * half, w, w * 1.9, w * 1.05, double=True) - [half, 0]
            ax.add_patch(Polygon(_place(poly, ang, center), closed=True, fc="w", ec="k", lw=3,
                                 joinstyle="miter", zorder=zorder))
        else:      # contraction: two black arrows pointing at the centre
            w = size * 0.14
            gap = size * 0.06
            for sgn in (1, -1):
                poly = _arrow_polygon(half, w, w * 2.4, w * 1.9)
                start = np.array([cx, cy]) + sgn * _axis_vec(az) * (gap + half)
                a = ang + (math.pi if sgn > 0 else 0.0)
                ax.add_patch(Polygon(_place(poly, a, start), closed=True, fc="k", ec="k", lw=1,
                                     zorder=zorder + 1))


# ---------------------------------------------------------------------------
# map frame: basemap, graticule, scale bar, north arrow, legend
# ---------------------------------------------------------------------------

def _nice_step(span):
    for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10):
        if span / s <= 9:
            return s
    return 20


def _fmt_deg(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def map_extent(table, bounds=0.5):
    """(lon0, lon1, lat0, lat1) padded by ``bounds`` degrees, as in 2020."""
    lon, lat = table["lon"].to_numpy(float), table["lat"].to_numpy(float)
    return lon.min() - bounds, lon.max() + bounds, lat.min() - bounds, lat.max() + bounds


def draw_basemap(ax, ext, basemap=DEFAULT_BASEMAP, tile_source=None, zoom=None, dem_source=None,
                 terrain_style=None):
    """Draw the background of a map whose Web Mercator extent is ``ext``.

    ``basemap``:

    * ``"hillshade"`` (default): grey shaded relief from a DEM, as on GEOSLAM's
      printed maps. USGS 3DEP in the US, Copernicus GLO-30/90 elsewhere; no key.
      ``"hillshade-3dep"`` / ``"hillshade-copernicus"`` force one DEM.
    * ``"usgs-relief"``, ``"usgs-topo"``, ``"usgs-imagery"``: USGS tiles (US only).
    * ``"synthetic"``: the same grey hillshade of made-up terrain (offline preview).
    * ``"none"`` or None: no background.

    ``dem_source(ext, width, height) -> elevations`` overrides the DEM download.
    Returns the attribution text (or None). If the download fails the map gets
    a plain background and a note, and the rest of the figure is still drawn.
    """
    if basemap in (None, "none"):
        return None
    style = terrain_style or terrain.DEFAULT_STYLE
    try:
        if basemap in HILLSHADE_SOURCES or basemap == "synthetic":
            src = terrain.synthetic_dem if basemap == "synthetic" else (dem_source or HILLSHADE_SOURCES[basemap])
            img, credit = terrain.terrain_image(ext, source=src, style=style)
            ax.imshow(img, extent=ext, origin="upper", interpolation="bilinear", zorder=0)
        else:
            prov = bm.PROVIDERS[basemap]
            img, iext = bm.fetch_basemap(ext, prov, zoom=zoom, tile_source=tile_source)
            ax.imshow(img, extent=iext, interpolation="spline36", zorder=0, origin="upper")
            credit = prov.attribution
    except KeyError:
        raise ValueError(f"unknown basemap {basemap!r}; choose from {', '.join(BASEMAPS)}") from None
    except Exception as exc:  # network down etc.: plain background, keep going
        ax.set_facecolor("#e9e9e6")
        credit = f"Basemap unavailable ({type(exc).__name__}: {exc})"[:160]
        log.warning(credit)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    return credit


def draw_faults(ax, ext, faults=True, source="auto", style=CLASSIC):
    """Draw fault traces over the terrain (zorder :data:`FAULT_ZORDER`).

    Returns the credit line, or None. Records the legend label on the axes
    (``ax.gps_strain_fault_label``) so the station legend can show it.
    """
    ax.gps_strain_fault_label = None
    if faults is None or faults is False:
        return None
    credit = None
    if faults is True:
        lon0, lat0 = bm.merc_to_lonlat(ext[0], ext[2])
        lon1, lat1 = bm.merc_to_lonlat(ext[1], ext[3])
        try:
            faults, credit = qf.faults_in_box(float(lon0), float(lat0), float(lon1), float(lat1), source)
        except Exception as exc:  # network down etc.: the map is still drawn
            note = f"Faults unavailable ({type(exc).__name__}: {exc})"[:160]
            log.warning(note)
            return note
    lw = 2.2 if style.edge else 1.5
    edge = outline(lw, 0.9, "k") if style.edge else None
    n = 0
    for f in faults:
        for line in f["lines"]:
            fx, fy = bm.lonlat_to_merc(line[:, 0], line[:, 1])
            ax.plot(fx, fy, color=FAULT_COLOR, lw=lw, alpha=0.95, zorder=FAULT_ZORDER,
                    solid_capstyle="round", path_effects=edge)
            n += 1
    if n:
        ax.gps_strain_fault_label = FAULT_LABEL  # the credit line names the database
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    return credit


def _faults_for(basemap, faults):
    """No fetched faults on the made-up offline terrain."""
    return False if (basemap == "synthetic" and faults is True) else faults


def _fault_handle(ax, style=CLASSIC):
    label = getattr(ax, "gps_strain_fault_label", None)
    if not label:
        return []
    return [Line2D([0], [0], color=FAULT_COLOR, lw=3, label=label,
                   path_effects=outline(3, 0.9, "k") if style.edge else None)]


def setup_map(ax, table, bounds=0.5, basemap=DEFAULT_BASEMAP, tile_source=None, zoom=None,
              attribution=True, dem_source=None, terrain_style=None, max_width=None, style=CLASSIC,
              faults=False, fault_source="auto"):
    """Prepare a Web Mercator map axes with basemap and labelled lon/lat graticule.

    ``max_width`` (Web Mercator metres) zooms the map in to at most that width,
    centred on the stations, but never so far that the stations lose a margin
    of :data:`STATION_PAD` of the map on each side (more at the top, under the
    station legend).

    ``faults``: True fetches mapped faults for the map area (see
    :func:`gps_strain.faults.faults_in_box`, ``fault_source``); a list of
    fault dicts draws those; False draws none. Faults sit above the terrain
    and below everything else.
    """
    lon0, lon1, lat0, lat1 = map_extent(table, bounds)
    x0, y0 = bm.lonlat_to_merc(lon0, lat0)
    x1, y1 = bm.lonlat_to_merc(lon1, lat1)
    # Like the 2020 cartopy maps: pad by `bounds` degrees of longitude, then
    # fit the extent to the axes box (equal aspect), centred on the stations.
    sx0, sy0 = bm.lonlat_to_merc(table["lon"].min(), table["lat"].min())
    sx1, sy1 = bm.lonlat_to_merc(table["lon"].max(), table["lat"].max())
    pad = x1 - sx1  # = bounds degrees of longitude in Mercator metres
    bb = ax.get_position()
    fw, fh = ax.figure.get_size_inches()
    box = (bb.height * fh) / (bb.width * fw)
    w = (sx1 - sx0) + 2 * pad
    h = max(w * box, (sy1 - sy0) + 0.6 * pad)
    w = max(w, h / box)
    if max_width is not None and max_width < w:
        w_min = max((sx1 - sx0) / (1 - 2 * STATION_PAD),
                    (sy1 - sy0) / (1 - 2 * STATION_PAD - LEGEND_PAD) / box)
        w = max(max_width, w_min)
    h = w * box
    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    ext = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal", adjustable="box")
    credit = draw_basemap(ax, ext, basemap, tile_source, zoom, dem_source, terrain_style)
    fault_credit = draw_faults(ax, ext, faults, fault_source, style)
    if fault_credit:
        credit = f"{credit}\n{fault_credit}" if credit else fault_credit

    # graticule labelled on all four sides
    glon0, glat0 = bm.merc_to_lonlat(ext[0], ext[2])
    glon1, glat1 = bm.merc_to_lonlat(ext[1], ext[3])
    lstep, bstep = _nice_step(glon1 - glon0), _nice_step(glat1 - glat0)
    lons = np.arange(math.ceil(glon0 / lstep) * lstep, glon1, lstep)
    lats = np.arange(math.ceil(glat0 / bstep) * bstep, glat1, bstep)
    xt, _ = bm.lonlat_to_merc(lons, np.zeros_like(lons))
    _, yt = bm.lonlat_to_merc(np.zeros_like(lats), lats)
    ax.set_xticks(xt, [_fmt_deg(v) for v in lons])
    ax.set_yticks(yt, [_fmt_deg(v) for v in lats])
    ax.tick_params(axis="both", which="both", length=0, labelsize=8, pad=6,
                   top=True, labeltop=True, right=True, labelright=True)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontfamily(SERIF)
    ax.grid(True, color="gray", lw=0.5, alpha=0.5, zorder=1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)
    if attribution and credit:
        credit = "\n".join(textwrap.fill(line, 120) for line in credit.split("\n"))
        ax.text(0.995, 0.004, credit, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=style.credit_fontsize, color="0.15", zorder=60, fontfamily=SANS,
                bbox=dict(boxstyle="square,pad=0.2", fc="w", ec="none", alpha=style.credit_alpha))
    return ext


STATION_PAD = 0.15     # map fraction kept clear around the stations when zooming in
LEGEND_PAD = 0.05      # extra room at the top for the station legend
ELLIPSE_FRAC = 0.11    # strain-ellipse radius, as a fraction of the map width (2020 layout)
SYMBOL_FRAC = 0.085    # 300-nanostrain symbol half-length, as a fraction of the map width
FIT_MARGIN = 1.12      # keep the glyph this much clear of the triangle sides
ROW_GAP = 0.035        # gap between the three boxes under the portrait map (fraction of its width)
BOX_LW = 1.0           # border width of those boxes (matches the tables)


def _hull(x, y):
    """Convex hull of the stations (counter-clockwise), monotone chain."""
    pts = sorted(set(zip(map(float, x), map(float, y))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _sides_from_centroid(x, y):
    """For each side of the station polygon: (unit normal, distance) from the
    centroid (where the glyph is drawn) to that side, in map units."""
    c = np.array([np.mean(x), np.mean(y)])
    h = _hull(x, y)
    out = []
    for a, b in zip(h, np.roll(h, -1, axis=0)):
        t = b - a
        n = np.array([t[1], -t[0]]) / max(np.hypot(*t), 1e-12)
        d = abs(np.dot(c - a, n))
        out.append((n, d))
    return out


def ellipse_max_radius(result: StrainResult, x, y, exaggeration=None, margin=FIT_MARGIN):
    """Largest ellipse radius (map units) that keeps the drawing inside the
    station polygon (x, y in map units) when centred on the centroid.

    Uses the ellipse's support function: the initial circle and the deformed
    ellipse F = I + kE reach ``R max(1, |F^T n|)`` toward a side with
    outward normal n.
    """
    k = exaggeration or exaggeration_for(result)
    F = np.eye(2) + k * result.strain_tensor
    radii = [d / (max(1.0, float(np.linalg.norm(F.T @ n))) * margin) for n, d in _sides_from_centroid(x, y)]
    return min(radii) if radii else None


def symbol_max_size(result: StrainResult, x, y, min_strain=0.1, max_strain=300.0, margin=FIT_MARGIN):
    """Largest symbol ``size`` (see :func:`draw_strain_symbol`) that keeps the
    strain-rate symbol inside the station polygon (treated as a circle of its
    longest arm, plus the arrow-head half width)."""
    max_strain = max(max_strain, abs(result.e1) * NANO, abs(result.e2) * NANO)
    reach = 0.0
    for e in (result.e1, result.e2):
        if abs(e) * NANO < min_strain:
            continue
        f = _symbol_fraction(e, min_strain, max_strain)
        reach = max(reach, f if e > 0 else f + 0.06)  # contraction arrows start 0.06 out
    if reach == 0:
        return None
    reach = math.hypot(reach, 0.17)
    return min(d / (reach * margin) for _, d in _sides_from_centroid(x, y))


def _merc_xy(table):
    return bm.lonlat_to_merc(table["lon"].to_numpy(float), table["lat"].to_numpy(float))


def ellipse_fit_width(result: StrainResult, table, frac=ELLIPSE_FRAC, margin=FIT_MARGIN):
    """Largest map width (Web Mercator metres) at which the strain ellipse,
    drawn with radius ``frac`` x map width at the centroid, stays inside the
    station triangle."""
    r = ellipse_max_radius(result, *_merc_xy(table), margin=margin)
    return None if r is None else r / frac


def symbol_fit_width(result: StrainResult, table, min_strain=0.1, max_strain=300.0,
                     frac=SYMBOL_FRAC, margin=FIT_MARGIN):
    """Largest map width at which the strain-rate symbol (half-length ``frac``
    x map width for 300 nanostrain) stays inside the triangle."""
    s = symbol_max_size(result, *_merc_xy(table), min_strain, max_strain, margin)
    return None if s is None else s / frac


def draw_triangle_and_sites(ax, table, colors=SITE_COLORS, legend_loc="upper center", style=CLASSIC):
    x, y = bm.lonlat_to_merc(table["lon"].to_numpy(float), table["lat"].to_numpy(float))
    xs, ys = np.append(x, x[0]), np.append(y, y[0])
    if style.edge:  # solid white line under the dashes: a blue-and-white dashed line
        ax.plot(xs, ys, color="w", lw=style.triangle_lw + 4, solid_joinstyle="round", zorder=19)
        ax.plot(xs, ys, color="blue", ls=(0, (4, 2.5)), lw=style.triangle_lw, zorder=20)
    else:
        ax.plot(xs, ys, color="blue", ls="--", lw=style.triangle_lw, zorder=20)
    handles = []
    for i, site in enumerate(table["site"]):
        c = colors[i % len(colors)]
        ax.plot(x[i], y[i], "o", ms=style.dot_ms, color=c, mec="k" if style.edge else c,
                mew=1.5 if style.edge else 0, zorder=50)
        handles.append(Line2D([0], [0], marker="o", color="b", ls="--", markerfacecolor=c,
                              markeredgewidth=0, markersize=15, label=site))
    handles += _fault_handle(ax, style)
    leg = ax.legend(handles=handles, ncol=len(handles), loc=legend_loc,
                    prop={"family": SERIF, "size": 15})
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(style.legend_alpha)
    leg.set_zorder(70)
    return x, y


def draw_scale_bar(ax, length_km=None, loc=(0.5, 0.05), segments=5, lw=15, style=CLASSIC):
    """Alternating black/white bar with '0 km' and 'N km' labels (on a white
    panel when ``style.panels``)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xc, yc = x0 + loc[0] * (x1 - x0), y0 + loc[1] * (y1 - y0)
    _, latc = bm.merc_to_lonlat(xc, yc)
    k = bm.ground_scale(latc)
    if length_km is None:
        ground_w = (x1 - x0) / k / 1000
        length_km = min((1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500), key=lambda v: abs(v - ground_w * 0.28))
    L = length_km * 1000 * k
    start = xc - L / 2
    seg = L / segments
    if style.panels:
        ax.plot([start, start + L], [yc, yc], color="k", lw=lw + 3, solid_capstyle="butt", zorder=55)
    for i in range(segments):
        ax.plot([start + i * seg, start + (i + 1) * seg], [yc, yc], color="k" if i % 2 == 0 else "w",
                lw=lw, solid_capstyle="butt", zorder=55)
    dy = (0.014 if style.panels else 0.012) * (y1 - y0)
    halo = HALO_THIN if style.panels else [patheffects.withStroke(linewidth=1.5, foreground="w")]
    fs = style.scale_fontsize
    labels = [ax.text(start, yc + dy, "0 km", ha="right", va="bottom", fontsize=fs, fontfamily=SANS,
                      path_effects=halo, zorder=56),
              ax.text(start + L, yc + dy, f"{length_km:g} km", ha="left", va="bottom", fontsize=fs,
                      fontfamily=SANS, path_effects=halo, zorder=56)]
    if style.panels:
        _panel_behind(ax, [(start, yc), (start + L, yc)], labels, pad_px=10)
    return length_km


def _panel_behind(ax, points, texts, pad_px=8):
    """White rounded panel behind map furniture (data points + text artists)."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in texts]
    px = ax.transData.transform(np.asarray(points, float))
    xs = [b.x0 for b in boxes] + [b.x1 for b in boxes] + list(px[:, 0])
    ys = [b.y0 for b in boxes] + [b.y1 for b in boxes] + list(px[:, 1] - 12) + list(px[:, 1] + 12)
    inv = ax.transAxes.inverted()
    (ax0, ay0), (ax1, ay1) = inv.transform([(min(xs) - pad_px, min(ys) - pad_px),
                                            (max(xs) + pad_px, max(ys) + pad_px)])
    ax.add_patch(FancyBboxPatch((ax0, ay0), ax1 - ax0, ay1 - ay0, boxstyle="round,pad=0,rounding_size=0.01",
                                transform=ax.transAxes, fc="w", ec="k", lw=0.8, alpha=PANEL_ALPHA,
                                zorder=54))


def draw_north_arrow(ax, loc=(0.11, 0.05), fontsize=35, style=CLASSIC):
    if style.panels:
        ax.text(loc[0], loc[1], "▲\nN", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=fontsize, fontfamily=SANS, linespacing=0.9, zorder=60,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc="w", ec="k", lw=0.8,
                          alpha=PANEL_ALPHA))
    else:
        ax.text(loc[0], loc[1], "▲\nN", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=fontsize, fontfamily=SANS, linespacing=0.9, path_effects=HALO, zorder=60)


def _load_logo(logo):
    if logo is None:
        return None
    if isinstance(logo, np.ndarray):
        return logo
    s = str(logo)
    if s.startswith(("http://", "https://")):
        import requests
        from PIL import Image
        import io

        r = requests.get(s, timeout=30)
        r.raise_for_status()
        return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    return plt.imread(Path(s))


WORDMARK = "GPS STRAIN"
BLOCK = ["Arial Black", "Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"]


def draw_logo(ax, logo=None, align="center"):
    """Logo slot: an image file/URL, or by default the block-letter "GPS STRAIN" wordmark.

    ``align="right"`` puts the logo's right edge on the slot's right edge.
    Pass ``logo=False`` to leave the slot empty.
    """
    ax.axis("off")
    if logo is False:
        return
    if logo is None or (isinstance(logo, str) and not logo.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".webp")) and "://" not in logo):
        x, ha = (1.0, "right") if align == "right" else (0.5, "center")
        ax.text(x, 0.5, WORDMARK if logo is None else logo, transform=ax.transAxes, ha=ha,
                va="center", fontsize=40, fontweight="black", fontfamily=BLOCK, color="#9e2a2b")
        return
    ax.imshow(_load_logo(logo), aspect="equal")
    if align == "right":
        ax.set_anchor("E")


# ---------------------------------------------------------------------------
# legend panels
# ---------------------------------------------------------------------------

def _ellipse_title_legend(ax, title_y=1.12):
    ax.set_title("Infinitesimal Strain Ellipse", y=title_y, fontsize=16, fontfamily=SERIF)
    h = [Line2D([0], [0], color=S1_COLOR, lw=1.5, label="$S_{1H}$"),
         Line2D([0], [0], color=S2_COLOR, lw=1.5, label="$S_{2H}$")]
    ax.legend(handles=h, ncol=2, loc="upper center", bbox_to_anchor=(0.5, title_y + 0.02),
              fontsize="x-large", frameon=False)


def contraction_legend(ax, min_strain=0.1, max_strain=300.0):
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    big_h, small_h = 0.62, 0.62 * 0.3
    bx, sx, base = 0.35, -0.35, 0.08
    wb, ws = 0.1, 0.1 * 0.45
    pb = _place(_arrow_polygon(big_h, wb, wb * 2.4, wb * 1.9), math.pi / 2, (bx, base))
    ps = _place(_arrow_polygon(small_h, ws, ws * 2.4, ws * 1.9), math.pi / 2, (sx, base + 0.25))
    xs = np.array([sx, bx])
    top = np.array([base + 0.25 + small_h, base + big_h])
    bot = np.array([base + 0.25, base])
    ax.plot(xs, top, color="slategrey", lw=1, ls="--")
    ax.plot(xs, bot, color="slategrey", lw=1, ls="--")
    ax.fill_between(xs, bot, top, color="slategrey", alpha=0.15, lw=0)
    ax.add_patch(Polygon(pb, closed=True, fc="k", ec="k", zorder=5))
    ax.add_patch(Polygon(ps, closed=True, fc="k", ec="k", zorder=5))
    ax.text(0, 0.96, "Infinitesimal Strain (Contraction)", ha="center", va="top", fontsize=16,
            fontfamily=SERIF)
    ax.text(-0.75, 0.42, f"{min_strain:g}\nnano-strain", ha="center", va="center", fontsize=12,
            fontfamily=SERIF)
    ax.text(0.75, 0.42, f"{max_strain:g}\nnano-strain", ha="center", va="center", fontsize=12,
            fontfamily=SERIF)


def elongation_legend(ax, min_strain=0.1, max_strain=300.0):
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    big_l, small_l = 1.3, 1.3 * 0.47
    wb, ws = 0.11, 0.11 * 0.6
    cb, cs = (0.2, 0.2), (0.1, 0.62)
    pb = _arrow_polygon(big_l, wb, wb * 1.9, wb * 1.05, double=True) - [big_l / 2, 0] + cb
    ps = _arrow_polygon(small_l, ws, ws * 1.9, ws * 1.05, double=True) - [small_l / 2, 0] + cs
    shade = np.array([[cs[0] - small_l / 2, cs[1]], [cs[0] + small_l / 2, cs[1]],
                      [cb[0] + big_l / 2, cb[1]], [cb[0] - big_l / 2, cb[1]]])
    ax.add_patch(Polygon(shade, closed=True, fc="slategrey", alpha=0.15, lw=0))
    ax.plot(shade[[0, 3], 0], shade[[0, 3], 1], color="slategrey", lw=1, ls="--")
    ax.plot(shade[[1, 2], 0], shade[[1, 2], 1], color="slategrey", lw=1, ls="--")
    ax.add_patch(Polygon(pb, closed=True, fc="w", ec="k", lw=3, joinstyle="miter", zorder=5))
    ax.add_patch(Polygon(ps, closed=True, fc="w", ec="k", lw=3, joinstyle="miter", zorder=5))
    ax.text(0, 0.96, "Infinitesimal Strain (Elongation)", ha="center", va="top", fontsize=16,
            fontfamily=SERIF)
    ax.text(-0.68, 0.74, f"{min_strain:g}\nnano-strain", ha="center", va="center", fontsize=12,
            fontfamily=SERIF)
    ax.text(-0.68, 0.0, f"{max_strain:g}\nnano-strain", ha="center", va="center", fontsize=12,
            fontfamily=SERIF)


def ellipse_legend_box(fig, rect):
    """Strain-ellipse legend centred in a bordered box at ``rect`` = (x, y, w, h)
    in figure coordinates."""
    x, y, w, h = rect
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, fc="w", ec="k", lw=BOX_LW))
    h_ = [Line2D([0], [0], color=S1_COLOR, lw=1.5, label="$S_{1H}$"),
          Line2D([0], [0], marker="$◌$", color="w", markeredgecolor="slategrey", markeredgewidth=0.5,
                 markerfacecolor="slategrey", markersize=20, label="Initial State"),
          Line2D([0], [0], color=S2_COLOR, lw=1.5, label="$S_{2H}$"),
          Line2D([0], [0], marker="o", color="w", markeredgecolor="k", markeredgewidth=1.1,
                 markerfacecolor="w", markersize=18, label="Strain Ellipse")]
    leg = fig.legend(handles=h_, ncol=2, loc="center", bbox_to_anchor=(x + w / 2, y + h / 2),
                     bbox_transform=fig.transFigure, frameon=False, title="Strain Ellipse Legend",
                     prop={"family": SERIF, "size": 14})
    plt.setp(leg.get_title(), fontsize=14, fontfamily=SERIF)
    return leg


def _table(ax, cells, header, fontsize=11.25, scale=1.75):
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, scale)
    for cell in t.get_celld().values():
        cell.get_text().set_fontfamily(SERIF)
    return t


def _default_title(data_label, one_line):
    sep = " " if one_line else "\n"
    return f"GPS Triangle-Strain Map{sep}Using {data_label} Data"


# ---------------------------------------------------------------------------
# the two map products
# ---------------------------------------------------------------------------

def symbol_map(result: StrainResult, table, path=None, title=None, data_label="NGL MIDAS",
               logo=None, basemap=DEFAULT_BASEMAP, tile_source=None, dem_source=None, bounds=0.5,
               scale_km=None, min_strain=0.1, max_strain=300.0, zoom=None, terrain_style=None,
               fit_symbol=True, visibility=False, faults=True, fault_source="auto"):
    """Landscape strain-rate symbol map (the 2020 ``symbol_map_full`` layout).

    ``visibility=True`` ("Increase visibility") draws the overlays thicker
    with white edges; see :data:`HIGH_VISIBILITY`.

    ``faults=True`` draws mapped Quaternary faults (USGS; GEM outside the US)
    between the terrain and the map symbols; ``faults=False`` leaves them off.
    With ``basemap="synthetic"`` they are off unless a list is given (real
    faults over made-up terrain would mislead).

    With ``fit_symbol`` (default) the map zooms in, if needed, so the symbol
    at the centroid stays inside the triangle; the symbol keeps its size on
    the page. If the stations would not fit at that zoom, the symbol is drawn
    smaller instead.
    """
    style = map_style(visibility)
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(30, 40, wspace=0.0, hspace=0.0, left=0.02, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[:, 11:])
        fit = symbol_fit_width(result, table, min_strain, max_strain) if fit_symbol else None
        setup_map(ax, table, bounds, basemap, tile_source, zoom, dem_source=dem_source,
                  terrain_style=terrain_style, max_width=fit, style=style,
                  faults=_faults_for(basemap, faults), fault_source=fault_source)
        x, y = draw_triangle_and_sites(ax, table, style=style)
        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        size = SYMBOL_FRAC * (min(width, fit) if fit else width)
        draw_strain_symbol(ax, result, (x.mean(), y.mean()), size, min_strain, max_strain)
        draw_scale_bar(ax, scale_km, style=style)
        draw_north_arrow(ax, style=style)

        ax_t = fig.add_subplot(gs[:3, :10])
        ax_t.axis("off")
        ax_t.text(0.5, 0.925, title or _default_title(data_label, False), ha="center", va="top",
                  fontsize=24, transform=ax_t.transAxes)

        ax_e = fig.add_subplot(gs[4:12, 1:9])
        draw_ellipse(ax_e, result)
        ax_e.set_aspect("equal", adjustable="box")
        ax_e.set_xlim(-1.3, 1.3)
        ax_e.set_ylim(-1.3, 1.3)
        ax_e.axis("off")
        _ellipse_title_legend(ax_e)

        contraction_legend(fig.add_subplot(gs[13:18, :10]), min_strain, max_strain)
        elongation_legend(fig.add_subplot(gs[20:25, :10]), min_strain, max_strain)
        draw_logo(fig.add_subplot(gs[27:30, 1:9]), logo)
        if path:
            fig.savefig(path, dpi=150, edgecolor="k", bbox_inches="tight")
    return fig


def strain_map(result: StrainResult, table, path=None, title=None, data_label="NGL MIDAS",
               velocity_label="Velocity Relative to North America", logo=None, basemap=DEFAULT_BASEMAP,
               tile_source=None, dem_source=None, bounds=0.5, scale_km=None, ref_velocity=None,
               zoom=None, terrain_style=None, fit_ellipse=True, visibility=False, faults=True,
               fault_source="auto"):
    """Portrait strain-ellipse map with velocities (the 2020 ``strain_map_full`` layout).

    ``visibility=True`` ("Increase visibility") draws the overlays thicker
    with white edges; see :data:`HIGH_VISIBILITY`.

    ``faults=True`` draws mapped Quaternary faults (USGS; GEM outside the US)
    between the terrain and the map symbols; ``faults=False`` leaves them off.
    With ``basemap="synthetic"`` they are off unless a list is given (real
    faults over made-up terrain would mislead).

    With ``fit_ellipse`` (default) the map zooms in, if needed, so the strain
    ellipse at the centroid stays inside the triangle, as on the 2020 maps;
    the ellipse keeps its size on the page (radius 11% of the map width). If
    the stations would not fit at that zoom, the ellipse is drawn smaller.
    """
    style = map_style(visibility)
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(40, 30, left=0.03, right=0.97, top=0.98, bottom=0.02, wspace=0.4, hspace=0.35)
        ax = fig.add_subplot(gs[:30, :])
        fit = ellipse_fit_width(result, table) if fit_ellipse else None
        setup_map(ax, table, bounds, basemap, tile_source, zoom, dem_source=dem_source,
                  terrain_style=terrain_style, max_width=fit, style=style,
                  faults=_faults_for(basemap, faults), fault_source=fault_source)
        x, y = draw_triangle_and_sites(ax, table, style=style)
        mx0, mx1 = ax.get_xlim()
        width = mx1 - mx0

        ve = table["ve"].to_numpy(float) * 1000  # mm/yr, published frame
        vn = table["vn"].to_numpy(float) * 1000
        vmax = max(np.hypot(ve, vn).max(), 1e-6)
        ref = ref_velocity or min((1, 2, 5, 10, 20, 50), key=lambda v: abs(v - vmax * 0.9))
        # map metres per (mm/yr): the largest arrow is ~11% of the map width
        k = 0.11 * width / vmax
        ax.quiver(x, y, ve, vn, angles="xy", scale_units="xy", scale=1 / k, color="k",
                  width=style.arrow_width, headwidth=3.5, headlength=4.5, headaxislength=4, zorder=45,
                  path_effects=ARROW_CASING if style.edge else None)
        radius = ELLIPSE_FRAC * (min(width, fit) if fit else width)
        draw_ellipse(ax, result, center=(x.mean(), y.mean()), radius=radius, zorder=30, style=style)
        draw_scale_bar(ax, scale_km, style=style)
        draw_north_arrow(ax, loc=(0.085, 0.07), style=style)

        ax_t = fig.add_subplot(gs[30:33, :22])
        ax_t.axis("off")
        title_txt = ax_t.text(0.0, 0.45, title or _default_title(data_label, True), ha="left", va="center",
                              fontsize=24, transform=ax_t.transAxes)
        # right edge of the logo on the map's right axis line (not its tick labels)
        ax.apply_aspect()
        mpos = ax.get_position()
        ax_logo = fig.add_subplot(gs[30:33, 23:])
        lpos = ax_logo.get_position()
        ax_logo.set_position([lpos.x0, lpos.y0, mpos.x1 - lpos.x0, lpos.height])
        draw_logo(ax_logo, logo, align="right")

        speed = np.hypot(ve, vn)
        speed_tab = _table(fig.add_subplot(gs[33:36, 21:]),
                           [[s, f"{v:.6f}"] for s, v in zip(table["site"], speed)], ["sites", "Speed (mm/yr)"])
        cells = [[r.site, f"{r.lon:.6f}", f"{r.lat:.6f}", f"{r.ve * 1000:.6f}", f"{r.sig_e * 1000:.6f}",
                  f"{r.vn * 1000:.6f}", f"{r.sig_n * 1000:.6f}"] for r in table.itertuples()]
        station_tab = _table(fig.add_subplot(gs[37:, :]), cells,
                             ["site", "longitude", "latitude", "E velocity (mm/yr)", "E uncertainty (mm/yr)",
                              "N velocity (mm/yr)", "N uncertainty (mm/yr)"])

        # Row of three boxes under the title, laid out from the station table's
        # edges: legend flush left, speed table flush right, velocity key centred
        # between them with equal gaps; all three share the speed table's height,
        # and the row sits midway between the title and the station table.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        (L, _), (R, _) = to_fig.transform(station_tab.get_window_extent(renderer).get_points())
        (s0, sb), (s1, top) = to_fig.transform(speed_tab.get_window_extent(renderer).get_points())
        w_side, h = s1 - s0, top - sb
        # centre the row vertically between the title and the station table
        title_bottom = to_fig.transform((0, title_txt.get_window_extent(renderer).y0))[1]
        table_top = to_fig.transform((0, station_tab.get_window_extent(renderer).y1))[1]
        new_top = (title_bottom + table_top) / 2 + h / 2
        p = speed_tab.axes.get_position()
        speed_tab.axes.set_position([p.x0 + R - s1, p.y0 + new_top - top, p.width, p.height])
        top = new_top
        gap = ROW_GAP * (R - L)
        w_mid = (R - L) - 2 * w_side - 2 * gap
        legend_rect = (L, top - h, w_side, h)
        velocity_rect = (L + w_side + gap, top - h, w_mid, h)

        ellipse_legend_box(fig, legend_rect)
        ax_v = fig.add_axes(velocity_rect)
        ax_v.axis("off")
        ax_v.add_patch(Rectangle((0, 0), 1, 1, transform=ax_v.transAxes, fc="w", ec="k", lw=BOX_LW,
                                 clip_on=False))
        ax_v.text(0.5, 0.72, velocity_label, ha="center", va="center", fontsize=14,
                  transform=ax_v.transAxes)
        # reference arrow with exactly the on-map length of `ref` mm/yr
        p0 = ax.transData.transform((mx0, 0.0))
        p1 = ax.transData.transform((mx0 + ref * k, 0.0))
        La = (p1[0] - p0[0]) / ax_v.get_window_extent(renderer).width  # in ax_v axes fraction
        x0 = 0.5 - (La + 0.04 + 0.16) / 2  # centre arrow + label in the box
        ax_v.annotate("", xy=(x0 + La, 0.32), xytext=(x0, 0.32), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=1.2", lw=7,
                                      color="k", shrinkA=0, shrinkB=0))
        ax_v.text(x0 + La + 0.04, 0.32, f"{ref:g} mm/yr", transform=ax_v.transAxes, va="center",
                  fontsize=12.5)
        if path:
            fig.savefig(path, dpi=150, edgecolor="k", bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# networks: several triangles on one map (the 2021 Raton, NM maps)
# ---------------------------------------------------------------------------

TRI_COLORS = ["m", "k", "r", "c", "b", "y", "g", "tab:orange", "tab:brown", "tab:pink"]


def _stations_union(tables):
    import pandas as pd

    allt = pd.concat(tables, ignore_index=True)
    return allt.drop_duplicates("site").reset_index(drop=True)


def network_map(results, tables, path=None, kind="symbol", basemap=DEFAULT_BASEMAP,
                tile_source=None, dem_source=None, bounds=0.5, scale_km=None, tri_colors=None, lines=None,
                velocities=True, min_strain=0.1, max_strain=None, zoom=None, title=None,
                figsize=(15, 15), terrain_style=None, fit=True, visibility=False, faults=True,
                fault_source="auto"):
    """Several triangles on one map, like the 2021 Raton, NM figures.

    ``kind="symbol"`` draws each triangle's strain-rate symbol at its centroid;
    ``kind="ellipse"`` draws each triangle's strain ellipse (outlined in the
    triangle's colour) with the dashed triangle. Symbols and ellipses share one
    strain scale, so their shapes are comparable between triangles; with
    ``fit=True`` (default) each one is drawn smaller where needed so it stays
    inside its own triangle (the map cannot zoom in for every triangle at
    once). Station velocity arrows are drawn once per station.
    """
    if len(results) != len(tables):
        raise ValueError("results and tables must have the same length")
    colors = tri_colors or [TRI_COLORS[i % len(TRI_COLORS)] for i in range(len(results))]
    lines = (kind == "ellipse") if lines is None else lines
    stations = _stations_union(tables)
    style = map_style(visibility)
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0.04, 0.78, 0.9])
        setup_map(ax, stations, bounds, basemap, tile_source, zoom, dem_source=dem_source,
                  terrain_style=terrain_style, style=style, faults=_faults_for(basemap, faults),
                  fault_source=fault_source)
        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        sx, sy = bm.lonlat_to_merc(stations["lon"].to_numpy(float), stations["lat"].to_numpy(float))
        if velocities:
            ve = stations["ve"].to_numpy(float) * 1000
            vn = stations["vn"].to_numpy(float) * 1000
            k = 0.08 * width / max(np.hypot(ve, vn).max(), 1e-6)
            ax.quiver(sx, sy, ve, vn, angles="xy", scale_units="xy", scale=1 / k, color="k",
                      width=style.arrow_width, headwidth=3.5, headlength=4.5, headaxislength=4,
                      zorder=45, path_effects=ARROW_CASING if style.edge else None)
        ax.scatter(sx, sy, s=90, c=[SITE_COLORS[i % len(SITE_COLORS)] for i in range(len(sx))],
                   zorder=50, edgecolors="k" if style.edge else "none", linewidths=1.5 if style.edge else 0)
        for x0, y0, s in zip(sx, sy, stations["site"]):
            ax.annotate(s, (x0, y0), xytext=(6, 6), textcoords="offset points", fontsize=10,
                        zorder=51, path_effects=[patheffects.withStroke(linewidth=3, foreground="w")])

        emax = max(max(abs(r.e1), abs(r.e2)) for r in results)
        kexag = 0.25 / max(emax, 1e-30)
        mmax = max_strain or max(emax * NANO, min_strain * 10)
        handles = []
        for r, t, c in zip(results, tables, colors):
            x, y = bm.lonlat_to_merc(t["lon"].to_numpy(float), t["lat"].to_numpy(float))
            cen = (x.mean(), y.mean())
            if lines:
                lx, ly = np.append(x, x[0]), np.append(y, y[0])
                if style.edge:
                    ax.plot(lx, ly, color="w", lw=style.triangle_lw + 3, alpha=0.8, zorder=19)
                    ax.plot(lx, ly, color=c, ls=(0, (4, 2.5)), lw=style.triangle_lw - 0.5, zorder=20)
                else:
                    ax.plot(lx, ly, color=c, ls="--", lw=2, alpha=0.6, zorder=20)
            if kind == "ellipse":
                rad = 0.06 * width
                if fit:
                    rad = min(rad, ellipse_max_radius(r, x, y, exaggeration=kexag) or rad)
                draw_ellipse(ax, r, center=cen, radius=rad, exaggeration=kexag, axis_lw=3 + style.edge,
                             zorder=30, style=style)
                th = np.linspace(0, 2 * np.pi, 361)
                E = (np.eye(2) + kexag * r.strain_tensor) @ np.vstack([np.cos(th), np.sin(th)])
                ax.plot(cen[0] + rad * E[0], cen[1] + rad * E[1], color=c, lw=2, zorder=34)
            else:
                size = 0.12 * width
                if fit:
                    size = min(size, symbol_max_size(r, x, y, min_strain, mmax) or size)
                draw_strain_symbol(ax, r, cen, size, min_strain, mmax)
            handles.append(Line2D([0], [0], color=c, ls="--", lw=2, label="–".join(r.sites)))
        handles += _fault_handle(ax, style)
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        prop={"family": SERIF, "size": 10}, title="Triangles", framealpha=0.9)
        leg.set_zorder(70)
        draw_scale_bar(ax, scale_km, style=style)
        draw_north_arrow(ax, style=style)
        if title:
            ax.set_title(title, fontsize=20, pad=28)
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
