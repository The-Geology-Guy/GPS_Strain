"""Map products in the style of the original (2020) GPS_Strain figures.

Two layouts, matching the 2020 package:

* :func:`symbol_map` - landscape. Left column: title, infinitesimal strain
  ellipse, contraction / elongation symbol legends, logo. Right: terrain map
  with the station triangle and the strain-rate map symbol at the centroid.
* :func:`strain_map` - portrait. Terrain map with station velocity arrows and
  the strain ellipse at the centroid; below it the title, logo, ellipse legend,
  velocity key, speed table and station table.

Basemaps come from :mod:`gps_strain.basemap` (Web Mercator tiles, no cartopy).
Map axes use Web Mercator metres as data coordinates. Web Mercator is
conformal, so arrow and strain-axis azimuths are drawn correctly.

Symbol conventions (Cronin & Resor / GETSI): black arrows pointing inward =
contraction (S2H when e2H < 0); white double-headed arrow = elongation.
S1H is drawn tomato-red and S2H cornflower-blue on the ellipse.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon

from . import basemap as bm
from .strain import NANO, StrainResult

S1_COLOR = "tomato"
S2_COLOR = "cornflowerblue"
SITE_COLORS = ["g", "b", "r", "orange", "m", "c"]
SERIF = ["Georgia", "Gelasio", "DejaVu Serif"]
SANS = ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"]
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
HALO = [patheffects.withStroke(linewidth=4, foreground="w")]


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
                 fill_alpha=0.45, lw=2.0, axis_lw=6.0, zorder=30):
    """Strain ellipse in the 2020 style: dashed initial circle, black ellipse,
    principal axes (S1H tomato, S2H cornflower), faint initial E-W/N-S axes.

    The deformation is F = I + k*E applied to a circle of ``radius`` (data
    units) at ``center``; k defaults to :func:`exaggeration_for`.
    """
    k = exaggeration or exaggeration_for(result)
    cx, cy = center
    th = np.linspace(0, 2 * np.pi, 361)
    circ = np.vstack([np.cos(th), np.sin(th)])
    F = np.eye(2) + k * result.strain_tensor
    ell = F @ circ
    ax.fill(cx + radius * circ[0], cy + radius * circ[1], color="w", alpha=fill_alpha, zorder=zorder, lw=0)
    ax.plot(cx + radius * circ[0], cy + radius * circ[1], color="slategrey", lw=lw, ls="--", zorder=zorder + 1)
    # faint initial axes
    ax.plot([cx - radius, cx + radius], [cy, cy], color=S1_COLOR, alpha=0.25, lw=axis_lw,
            solid_capstyle="butt", zorder=zorder + 1)
    ax.plot([cx, cx], [cy - radius, cy + radius], color=S2_COLOR, alpha=0.25, lw=axis_lw,
            solid_capstyle="butt", zorder=zorder + 1)
    for az, e, col in ((result.azimuth_s1, result.e1, S1_COLOR), (result.azimuth_s2, result.e2, S2_COLOR)):
        u = _axis_vec(az) * (1 + k * e) * radius
        ax.plot([cx - u[0], cx + u[0]], [cy - u[1], cy + u[1]], color=col, lw=axis_lw,
                solid_capstyle="butt", zorder=zorder + 2)
    ax.plot(cx + radius * ell[0], cy + radius * ell[1], color="k", lw=lw, zorder=zorder + 3)
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


def setup_map(ax, table, bounds=0.5, basemap="esri-terrain", api_key=None, tile_source=None,
              zoom=None, attribution=True):
    """Prepare a Web Mercator map axes with basemap and labelled lon/lat graticule."""
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
    h = w * box
    cx, cy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    ext = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal", adjustable="box")
    credit = None
    if basemap not in (None, "none"):
        src = tile_source
        if basemap == "synthetic":
            src, prov = bm.synthetic_terrain_tile, bm.PROVIDERS["esri-terrain"]
            credit = "Synthetic offline basemap (preview only)"
        else:
            prov = bm.PROVIDERS[basemap]
            credit = prov.attribution
        try:
            img, iext = bm.fetch_basemap(ext, prov, zoom=zoom, api_key=api_key, tile_source=src)
            ax.imshow(img, extent=iext, interpolation="spline36", zorder=0, origin="upper")
        except Exception as exc:  # network down etc.: plain background, keep going
            ax.set_facecolor("#e8e4d2")
            credit = f"Basemap unavailable ({type(exc).__name__}: {exc})"[:120]
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

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
        ax.text(0.995, 0.004, credit, transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                color="0.25", zorder=60, fontfamily=SANS,
                bbox=dict(boxstyle="square,pad=0.15", fc="w", ec="none", alpha=0.6))
    return ext


def draw_triangle_and_sites(ax, table, colors=SITE_COLORS, legend_loc="upper center", ms=10):
    x, y = bm.lonlat_to_merc(table["lon"].to_numpy(float), table["lat"].to_numpy(float))
    xs, ys = np.append(x, x[0]), np.append(y, y[0])
    ax.plot(xs, ys, color="blue", ls="--", lw=2, zorder=20)
    handles = []
    for i, site in enumerate(table["site"]):
        c = colors[i % len(colors)]
        ax.plot(x[i], y[i], "o", ms=ms, color=c, zorder=50)
        handles.append(Line2D([0], [0], marker="o", color="b", ls="--", markerfacecolor=c,
                              markeredgewidth=0, markersize=15, label=site))
    leg = ax.legend(handles=handles, ncol=len(handles), loc=legend_loc,
                    prop={"family": SERIF, "size": 15})
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.75)
    leg.set_zorder(70)
    return x, y


def draw_scale_bar(ax, length_km=None, loc=(0.5, 0.05), segments=5, lw=15):
    """Alternating black/white bar with '0 km' and 'N km' labels."""
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
    for i in range(segments):
        ax.plot([start + i * seg, start + (i + 1) * seg], [yc, yc], color="k" if i % 2 == 0 else "w",
                lw=lw, solid_capstyle="butt", zorder=55)
    halo = [patheffects.withStroke(linewidth=1.5, foreground="w")]
    dy = 0.012 * (y1 - y0)
    ax.text(start, yc + dy, "0 km", ha="right", va="bottom", fontsize=12, fontfamily=SANS,
            path_effects=halo, zorder=56)
    ax.text(start + L, yc + dy, f"{length_km:g} km", ha="left", va="bottom", fontsize=12,
            fontfamily=SANS, path_effects=halo, zorder=56)
    return length_km


def draw_north_arrow(ax, loc=(0.11, 0.05), fontsize=35):
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


def draw_logo(ax, logo=None):
    """Logo slot: an image file/URL, or by default the block-letter "GPS STRAIN" wordmark.

    Pass ``logo=False`` to leave the slot empty.
    """
    ax.axis("off")
    if logo is False:
        return
    if logo is None or (isinstance(logo, str) and not logo.lower().endswith(
            (".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff", ".webp")) and "://" not in logo):
        ax.text(0.5, 0.5, WORDMARK if logo is None else logo, transform=ax.transAxes, ha="center",
                va="center", fontsize=40, fontweight="black", fontfamily=BLOCK, color="#9e2a2b")
        return
    ax.imshow(_load_logo(logo), aspect="equal")


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


def ellipse_legend_box(ax):
    ax.axis("off")
    h = [Line2D([0], [0], color=S1_COLOR, lw=1.5, label="$S_{1H}$"),
         Line2D([0], [0], marker="$◌$", color="w", markeredgecolor="slategrey", markeredgewidth=0.5,
                markerfacecolor="slategrey", markersize=20, label="Initial State"),
         Line2D([0], [0], color=S2_COLOR, lw=1.5, label="$S_{2H}$"),
         Line2D([0], [0], marker="o", color="w", markeredgecolor="k", markeredgewidth=1.1,
                markerfacecolor="w", markersize=18, label="Strain Ellipse")]
    leg = ax.legend(handles=h, ncol=2, loc="center", frameon=True, title="Strain Ellipse Legend",
                    prop={"family": SERIF, "size": 14})
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_linewidth(0.5)
    leg.get_frame().set_alpha(0.5)
    plt.setp(leg.get_title(), fontsize=14, fontfamily=SERIF)


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
               logo=None, basemap="esri-terrain", api_key=None, tile_source=None, bounds=0.5,
               scale_km=None, min_strain=0.1, max_strain=300.0, zoom=None):
    """Landscape strain-rate symbol map (the 2020 ``symbol_map_full`` layout)."""
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(30, 40, wspace=0.0, hspace=0.0, left=0.02, right=0.97, top=0.97, bottom=0.03)
        ax = fig.add_subplot(gs[:, 11:])
        setup_map(ax, table, bounds, basemap, api_key, tile_source, zoom)
        x, y = draw_triangle_and_sites(ax, table)
        size = 0.085 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        draw_strain_symbol(ax, result, (x.mean(), y.mean()), size, min_strain, max_strain)
        draw_scale_bar(ax, scale_km)
        draw_north_arrow(ax)

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
               velocity_label="Velocity Relative to North America", logo=None, basemap="esri-terrain",
               api_key=None, tile_source=None, bounds=0.5, scale_km=None, ref_velocity=None,
               zoom=None):
    """Portrait strain-ellipse map with velocities (the 2020 ``strain_map_full`` layout)."""
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=(15, 20))
        gs = fig.add_gridspec(40, 30, left=0.03, right=0.97, top=0.98, bottom=0.02, wspace=0.4, hspace=0.35)
        ax = fig.add_subplot(gs[:30, :])
        setup_map(ax, table, bounds, basemap, api_key, tile_source, zoom)
        x, y = draw_triangle_and_sites(ax, table)
        mx0, mx1 = ax.get_xlim()
        width = mx1 - mx0

        ve = table["ve"].to_numpy(float) * 1000  # mm/yr, published frame
        vn = table["vn"].to_numpy(float) * 1000
        vmax = max(np.hypot(ve, vn).max(), 1e-6)
        ref = ref_velocity or min((1, 2, 5, 10, 20, 50), key=lambda v: abs(v - vmax * 0.9))
        # map metres per (mm/yr): the largest arrow is ~11% of the map width
        k = 0.11 * width / vmax
        ax.quiver(x, y, ve, vn, angles="xy", scale_units="xy", scale=1 / k, color="k",
                  width=0.006, headwidth=3.5, headlength=4.5, headaxislength=4, zorder=45)
        draw_ellipse(ax, result, center=(x.mean(), y.mean()), radius=0.11 * width, zorder=30)
        draw_scale_bar(ax, scale_km)
        draw_north_arrow(ax, loc=(0.085, 0.07))

        ax_t = fig.add_subplot(gs[30:33, :22])
        ax_t.axis("off")
        ax_t.text(0.0, 0.45, title or _default_title(data_label, True), ha="left", va="center",
                  fontsize=24, transform=ax_t.transAxes)
        draw_logo(fig.add_subplot(gs[30:33, 23:]), logo)

        ellipse_legend_box(fig.add_subplot(gs[33:37, :10]))

        ax_v = fig.add_subplot(gs[33:37, 10:21])
        ax_v.axis("off")
        ax_v.add_patch(FancyBboxPatch((0.03, 0.08), 0.94, 0.84, boxstyle="square,pad=0.0",
                                      transform=ax_v.transAxes, fc="w", ec="0.6", lw=1))
        ax_v.text(0.5, 0.72, velocity_label, ha="center", va="center", fontsize=14,
                  transform=ax_v.transAxes)
        fig.canvas.draw()
        # reference arrow with exactly the on-map length of `ref` mm/yr
        p0 = ax.transData.transform((mx0, 0.0))
        p1 = ax.transData.transform((mx0 + ref * k, 0.0))
        L = (p1[0] - p0[0]) / fig.dpi / fig.get_size_inches()[0]  # figure fraction
        pos = ax_v.get_position()
        La = L / pos.width  # in ax_v axes fraction
        ax_v.annotate("", xy=(0.2 + La, 0.36), xytext=(0.2, 0.36), xycoords="axes fraction",
                      arrowprops=dict(arrowstyle="-|>,head_width=0.6,head_length=1.2", lw=7,
                                      color="k", shrinkA=0, shrinkB=0))
        ax_v.text(0.2 + La + 0.04, 0.36, f"{ref:g} mm/yr", transform=ax_v.transAxes, va="center",
                  fontsize=12.5)

        speed = np.hypot(ve, vn)
        _table(fig.add_subplot(gs[33:36, 21:]), [[s, f"{v:.6f}"] for s, v in zip(table["site"], speed)],
               ["sites", "Speed (mm/yr)"])

        cells = [[r.site, f"{r.lon:.6f}", f"{r.lat:.6f}", f"{r.ve * 1000:.6f}", f"{r.sig_e * 1000:.6f}",
                  f"{r.vn * 1000:.6f}", f"{r.sig_n * 1000:.6f}"] for r in table.itertuples()]
        _table(fig.add_subplot(gs[37:, :]), cells,
               ["site", "longitude", "latitude", "E velocity (mm/yr)", "E uncertainty (mm/yr)",
                "N velocity (mm/yr)", "N uncertainty (mm/yr)"])
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


def network_map(results, tables, path=None, kind="symbol", basemap="esri-terrain", api_key=None,
                tile_source=None, bounds=0.5, scale_km=None, tri_colors=None, lines=None,
                velocities=True, min_strain=0.1, max_strain=None, zoom=None, title=None,
                figsize=(15, 15)):
    """Several triangles on one map, like the 2021 Raton, NM figures.

    ``kind="symbol"`` draws each triangle's strain-rate symbol at its centroid;
    ``kind="ellipse"`` draws each triangle's strain ellipse (outlined in the
    triangle's colour) with the dashed triangle. All symbols/ellipses share one
    scale, so sizes are comparable between triangles. Station velocity arrows
    are drawn once per station.
    """
    if len(results) != len(tables):
        raise ValueError("results and tables must have the same length")
    colors = tri_colors or [TRI_COLORS[i % len(TRI_COLORS)] for i in range(len(results))]
    lines = (kind == "ellipse") if lines is None else lines
    stations = _stations_union(tables)
    with plt.rc_context({"font.family": SERIF}):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.05, 0.04, 0.78, 0.9])
        setup_map(ax, stations, bounds, basemap, api_key, tile_source, zoom)
        width = ax.get_xlim()[1] - ax.get_xlim()[0]
        sx, sy = bm.lonlat_to_merc(stations["lon"].to_numpy(float), stations["lat"].to_numpy(float))
        if velocities:
            ve = stations["ve"].to_numpy(float) * 1000
            vn = stations["vn"].to_numpy(float) * 1000
            k = 0.08 * width / max(np.hypot(ve, vn).max(), 1e-6)
            ax.quiver(sx, sy, ve, vn, angles="xy", scale_units="xy", scale=1 / k, color="k",
                      width=0.006, headwidth=3.5, headlength=4.5, headaxislength=4, zorder=45)
        ax.scatter(sx, sy, s=90, c=[SITE_COLORS[i % len(SITE_COLORS)] for i in range(len(sx))],
                   zorder=50, edgecolors="none")
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
                ax.plot(np.append(x, x[0]), np.append(y, y[0]), color=c, ls="--", lw=2, alpha=0.6, zorder=20)
            if kind == "ellipse":
                draw_ellipse(ax, r, center=cen, radius=0.06 * width, exaggeration=kexag, axis_lw=3, zorder=30)
                th = np.linspace(0, 2 * np.pi, 361)
                E = (np.eye(2) + kexag * r.strain_tensor) @ np.vstack([np.cos(th), np.sin(th)])
                ax.plot(cen[0] + 0.06 * width * E[0], cen[1] + 0.06 * width * E[1], color=c, lw=2, zorder=34)
            else:
                draw_strain_symbol(ax, r, cen, 0.12 * width, min_strain, mmax)
            handles.append(Line2D([0], [0], color=c, ls="--", lw=2, label="–".join(r.sites)))
        leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        prop={"family": SERIF, "size": 10}, title="Triangles", framealpha=0.9)
        leg.set_zorder(70)
        draw_scale_bar(ax, scale_km)
        draw_north_arrow(ax)
        if title:
            ax.set_title(title, fontsize=20, pad=28)
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
