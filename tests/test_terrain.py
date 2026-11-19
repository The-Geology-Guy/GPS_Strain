"""Terrain (DEM hillshade) tests. All offline: network calls are replaced by
in-memory responses built to the services' formats."""
import io
import math

import numpy as np
import pytest

from gps_strain import basemap as bm
from gps_strain import terrain


def _merc_ext(lon0, lon1, lat0, lat1):
    x0, y0 = bm.lonlat_to_merc(lon0, lat0)
    x1, y1 = bm.lonlat_to_merc(lon1, lat1)
    return float(x0), float(x1), float(y0), float(y1)


def _geotiff_bytes(z, lon_min, lat_max, dlon, dlat, point=False, nodata=None):
    """A float32 GeoTIFF the way Copernicus ships them (DEFLATE + floating-point
    predictor), written with Pillow/libtiff."""
    from PIL import Image

    tie_lon, tie_lat = (lon_min + dlon / 2, lat_max - dlat / 2) if point else (lon_min, lat_max)
    geokeys = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 2 if point else 1, 2048, 0, 1, 4326)
    info = {33550: (float(dlon), float(dlat), 0.0),
            33922: (0.0, 0.0, 0.0, float(tie_lon), float(tie_lat), 0.0), 34735: geokeys, 317: 3}
    if nodata is not None:
        info[42113] = str(nodata)
    buf = io.BytesIO()
    Image.fromarray(np.asarray(z, np.float32), mode="F").save(buf, format="TIFF",
                                                              compression="tiff_adobe_deflate", tiffinfo=info)
    return buf.getvalue()


# --- USGS 3DEP response format -----------------------------------------------
def test_parse_bsq_with_validity_mask():
    w, h = 5, 3
    z = np.arange(w * h, dtype="<f4").reshape(h, w) * 10
    valid = np.ones(w * h, np.uint8)
    valid[[0, 7, 14]] = 0
    raw = z.tobytes() + np.packbits(valid).tobytes()
    out = terrain.parse_bsq_f32(raw, w, h)
    assert np.isnan(out.ravel()[[0, 7, 14]]).all()
    assert out[1, 1] == pytest.approx(60.0)
    assert np.isfinite(out).sum() == w * h - 3


def test_parse_bsq_rejects_error_page():
    with pytest.raises(RuntimeError, match="3DEP returned"):
        terrain.parse_bsq_f32(b'{"error": {"code": 400}}', 100, 100)


# --- Copernicus -----------------------------------------------------------------
def test_copernicus_tile_names():
    assert terrain.copernicus_tile_name(39, -121) == "Copernicus_DSM_COG_10_N39_00_W121_00_DEM"
    assert terrain.copernicus_tile_name(-34, 18, res=90) == "Copernicus_DSM_COG_30_S34_00_E018_00_DEM"


@pytest.mark.parametrize("point", [False, True])
def test_read_geotiff_georeferencing(point):
    z = np.arange(12, dtype=float).reshape(3, 4)
    data = _geotiff_bytes(z, -121.0, 40.0, 0.25, 1 / 3, point=point, nodata=-32767)
    r = terrain.read_geotiff(data)
    assert r.lon_min == pytest.approx(-121.0)
    assert r.lat_max == pytest.approx(40.0)
    assert r.lon_max == pytest.approx(-120.0)
    assert r.lat_min == pytest.approx(39.0)
    np.testing.assert_allclose(r.z, z)


def test_fetch_copernicus_mosaic_is_exact_for_a_linear_field():
    """Two 1-degree tiles (different longitude spacing, as above 50 N) plus a
    missing ocean tile. Bilinear sampling reproduces a linear field exactly."""
    def field(lon, lat):
        return 100 + 50 * lon + 30 * lat

    def tile(la, lo, nlon, nlat):
        dlon, dlat = 1 / nlon, 1 / nlat
        lon = lo + (np.arange(nlon) + 0.5) * dlon
        lat = la + 1 - (np.arange(nlat) + 0.5) * dlat
        return _geotiff_bytes(field(*np.meshgrid(lon, lat)), lo, la + 1, dlon, dlat)

    tiles = {terrain.copernicus_tile_name(51, 10, 90): tile(51, 10, 800, 1200),
             terrain.copernicus_tile_name(51, 11, 90): tile(51, 11, 600, 1200)}
    ext = _merc_ext(10.2, 11.8, 51.2, 52.5)  # the row above 52 N has no tiles (ocean)
    z, res = terrain.fetch_copernicus(ext, 300, 200, res=90, tile_getter=lambda name, r: tiles.get(name))
    assert res == 90
    lon, lat = terrain.grid_lonlat(ext, 300, 200)
    LON, LAT = np.meshgrid(lon, lat)
    land = LAT < 52
    assert np.isnan(z[~land]).all()
    np.testing.assert_allclose(z[land], field(LON[land], LAT[land]), atol=1e-3)


def test_copernicus_resolution_choice():
    assert terrain.copernicus_resolution(_merc_ext(-121, -119, 39, 40.5), 2400) == 90   # ~70 m pixels
    assert terrain.copernicus_resolution(_merc_ext(-121, -120.8, 39, 39.2), 2400) == 30  # ~7 m pixels


# --- hillshade --------------------------------------------------------------------
def _plane(ext, w, h, slope_e, slope_n):
    """Locally planar terrain rising slope_e (m/m) to the east and slope_n to
    the north on the ground (sphere of radius R; good to ~1% over a map)."""
    lon, lat = terrain.grid_lonlat(ext, w, h)
    LON, LAT = np.meshgrid(np.radians(lon - lon.mean()), np.radians(lat))
    R = bm.R_MERC
    return 500 + slope_e * R * np.cos(LAT) * LON + slope_n * R * (LAT - LAT.mean())


def test_lambert_uses_true_ground_slope():
    """At 63 N a Mercator cell is 2.2x its ground size; the hillshade must see
    the true ground slope, equal in every row."""
    ext = _merc_ext(-146.5, -145.0, 62.8, 63.6)
    style = terrain.HillshadeStyle(azimuth=315.0, altitude=30.0, exaggeration=1.0)
    se, sn = 0.3, -0.2
    s = terrain.lambert(_plane(ext, 80, 60, se, sn), ext, style)
    az, alt = math.radians(315), math.radians(30)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    expected = (-se * lx - sn * ly + lz) / math.sqrt(1 + se ** 2 + sn ** 2)
    np.testing.assert_allclose(s, expected, atol=0.01)
    # (Mercator spacing without the cos(latitude) correction would give 0.44 here)


def test_shade_lights_slopes_facing_the_light():
    ext = _merc_ext(-121.0, -120.0, 39.0, 39.5)
    w = 60
    lit = _plane(ext, w, 40, 0.3, -0.3)     # rises to the southeast: faces the NW light
    dark = _plane(ext, w, 40, -0.3, 0.3)
    ext2 = (ext[0], ext[1] + (ext[1] - ext[0]), ext[2], ext[3])
    s = terrain.shade(np.hstack([lit, dark + 5000]), ext2)
    assert np.nanmean(s[:, 3:w - 3]) > np.nanmean(s[:, w + 3:-3]) + 0.3
    flat = terrain.shade(np.full((10, 10), 250.0), ext)
    assert np.allclose(flat, 0.5)


def test_water_is_masked_and_drawn_pale():
    ext = _merc_ext(-123.5, -122.5, 48.5, 49.5)
    z = _plane(ext, 120, 120, 0.05, 0.02)
    z[:, :40] = 0.0          # sea level, perfectly flat (Copernicus over the sea)
    z[100:, 60:] = np.nan    # no data (3DEP outside the US)
    water = terrain.water_mask(z)
    assert water[:, :37].all() and water[100:, 60:].all()
    assert not water[10:90, 50:110].any()
    img = terrain.hillshade_rgb(z, ext)
    np.testing.assert_allclose(img[50, 10], terrain.WATER_RGB)
    g = img[10:90, 50:110, 0]
    assert g.min() >= terrain.DEFAULT_STYLE.dark - 1e-9 and g.max() <= terrain.DEFAULT_STYLE.light + 1e-9


def test_water_mask_closes_dem_seams_across_a_lake():
    """3DEP tile seams leave a small step between two flattened halves of a
    lake (seen on Lake Tahoe); the seam must not show as land."""
    ext = _merc_ext(-120.2, -119.9, 38.9, 39.3)
    z = _plane(ext, 200, 200, 0.08, 0.05) + 3000
    z[60:140, 40:100] = 1897.0
    z[60:140, 100:160] = 1897.6        # other side of the seam
    z[60:140, 99:101] = 1897.3         # resampled step
    water = terrain.water_mask(z)
    assert water[62:138, 42:158].all()
    assert not water[:50].any() and not water[150:].any()


def test_terrain_image_with_callable_source():
    ext = _merc_ext(-120.6, -119.9, 39.1, 39.7)
    img, credit = terrain.terrain_image(ext, source=terrain.synthetic_dem, target_px=400)
    assert img.shape[1] == 400 and img.shape[2] == 3
    assert img.shape[0] == round(400 * (ext[3] - ext[2]) / (ext[1] - ext[0]))
    assert "Synthetic" in credit
    img2, credit2 = terrain.terrain_image(ext, source=lambda e, w, h: _plane(e, w, h, 0.1, 0.1), target_px=200)
    assert credit2 == "Terrain: user-supplied DEM"
    with pytest.raises(ValueError):
        terrain.terrain_image(ext, source="stamen")


def test_synthetic_dem_is_seamless_in_world_coordinates():
    ext = _merc_ext(-120.6, -119.9, 39.1, 39.7)
    a = terrain.synthetic_dem(ext, 200, 160)
    x0, x1, y0, y1 = ext
    half = (x0, (x0 + x1) / 2, y0, y1)
    b = terrain.synthetic_dem(half, 100, 160)
    np.testing.assert_allclose(a[:, :100], b, atol=1e-6)


# --- maps -------------------------------------------------------------------------
def test_map_survives_terrain_failure(tmp_path):
    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    t = from_records([dict(site="P146", lat=39.337464, lon=-120.5373, ve=-10.31, vn=6.25, sig_e=0.3, sig_n=0.3),
                      dict(site="P149", lat=39.60213, lon=-120.104975, ve=-9.42, vn=5.20, sig_e=0.3, sig_n=0.3),
                      dict(site="P150", lat=39.292381, lon=-120.033854, ve=-10.86, vn=5.92, sig_e=0.3, sig_n=0.3)])
    r = compute_strain(t, mc_samples=0)

    def offline(ext, w, h):
        raise ConnectionError("no network")

    fig = plotting.symbol_map(r, t, tmp_path / "s.png", dem_source=offline, faults=False)
    texts = [x.get_text() for x in fig.axes[0].texts]
    assert any("Basemap unavailable" in s for s in texts)
    with pytest.raises(ValueError, match="unknown basemap"):
        plotting.symbol_map(r, t, None, basemap="esri-terrain")


# --- glyphs stay inside the triangle (zoom in, don't shrink) ---------------------
TAHOE = [dict(site="P146", lat=39.337463, lon=-120.537302, ve=-9.927, vn=8.814, sig_e=0.175, sig_n=0.156),
         dict(site="P149", lat=39.602129, lon=-120.104977, ve=-8.553, vn=7.557, sig_e=0.217, sig_n=0.161),
         dict(site="P150", lat=39.292380, lon=-120.033855, ve=-9.186, vn=8.139, sig_e=0.174, sig_n=0.160)]
ALASKA = [dict(site="AC63", lat=63.502425, lon=-145.847246, ve=2.563, vn=1.006, sig_e=0.204, sig_n=0.195),
          dict(site="AC62", lat=63.083606, lon=-146.312698, ve=-13.145, vn=8.035, sig_e=0.3, sig_n=0.152),
          dict(site="AB37", lat=62.967322, lon=-145.451881, ve=-13.75, vn=10.503, sig_e=0.297, sig_n=0.21)]


def _inside(px, py, tx, ty):
    """All points strictly inside the triangle (same side of every edge)."""
    sgn = []
    for i in range(3):
        ax_, ay_, bx, by = tx[i], ty[i], tx[(i + 1) % 3], ty[(i + 1) % 3]
        sgn.append(np.sign((bx - ax_) * (py - ay_) - (by - ay_) * (px - ax_)))
    s = np.array(sgn)
    return bool(np.all(s == s[0, 0]))


@pytest.mark.parametrize("stations", [TAHOE, ALASKA], ids=["tahoe", "alaska"])
def test_strain_ellipse_fits_inside_triangle(stations):
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    t = from_records(stations)
    r = compute_strain(t, mc_samples=0)
    fig = plotting.strain_map(r, t, None, basemap="none", faults=False)
    ax = fig.axes[0]
    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    fit = plotting.ellipse_fit_width(r, t)
    default = plotting.strain_map(r, t, None, basemap="none", fit_ellipse=False, faults=False).axes[0]
    assert width < default.get_xlim()[1] - default.get_xlim()[0]   # zoomed in ...
    assert fit * (1 - 1e-9) <= width <= fit * 1.1                  # ... to the fit (or the station margin)
    R = plotting.ELLIPSE_FRAC * min(width, fit)                    # 11% of the map when it can be
    tx, ty = bm.lonlat_to_merc(t["lon"].to_numpy(float), t["lat"].to_numpy(float))
    th = np.linspace(0, 2 * np.pi, 721)
    circ = np.vstack([np.cos(th), np.sin(th)])
    F = np.eye(2) + plotting.exaggeration_for(r) * r.strain_tensor
    for pts in (circ, F @ circ):
        assert _inside(tx.mean() + R * pts[0], ty.mean() + R * pts[1], tx, ty)
    # every station keeps a margin inside the map
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert ((tx - x0) / (x1 - x0)).min() >= plotting.STATION_PAD - 1e-9
    assert ((x1 - tx) / (x1 - x0)).min() >= plotting.STATION_PAD - 1e-9
    plt.close("all")


def test_skinny_triangle_shrinks_glyph_instead_of_hiding_stations():
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    skinny = [dict(TAHOE[0]), dict(TAHOE[1]), dict(TAHOE[2])]
    skinny[2].update(lat=39.47, lon=-120.30)   # nearly on the P146-P149 line
    t = from_records(skinny)
    r = compute_strain(t, mc_samples=0)
    fit = plotting.ellipse_fit_width(r, t)
    ax = plotting.strain_map(r, t, None, basemap="none", faults=False).axes[0]
    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    assert width > fit                       # stations' margin wins over the zoom
    tx, _ = bm.lonlat_to_merc(t["lon"].to_numpy(float), t["lat"].to_numpy(float))
    assert ((tx - ax.get_xlim()[0]) / width).min() >= plotting.STATION_PAD - 1e-9
    plt.close("all")


def test_wordmark_right_edge_on_map_axis():
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    t = from_records(TAHOE)
    r = compute_strain(t, mc_samples=0)
    fig = plotting.strain_map(r, t, None, basemap="none", faults=False)
    fig.canvas.draw()
    ax = fig.axes[0]
    mark = next(x for a in fig.axes for x in a.texts if x.get_text() == plotting.WORDMARK)
    right_px = mark.get_window_extent().x1
    axis_px = ax.get_window_extent().x1
    assert abs(right_px - axis_px) < 2.0
    plt.close("all")


def test_network_ellipses_shrink_to_fit_their_own_triangles():
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    big = from_records(TAHOE)
    small = from_records([TAHOE[0], TAHOE[1],                     # a smaller triangle inside the map
                          dict(TAHOE[2], site="P999", lat=39.40, lon=-120.25)])
    rs = [compute_strain(t, mc_samples=0) for t in (big, small)]
    fig = plotting.network_map(rs, [big, small], None, kind="ellipse", basemap="none", faults=False)
    ax = fig.axes[0]
    outlines = [ln for ln in ax.lines if ln.get_zorder() == 34]
    assert len(outlines) == 2
    width = ax.get_xlim()[1] - ax.get_xlim()[0]
    radii = []
    for ln, t in zip(outlines, (big, small)):
        tx, ty = bm.lonlat_to_merc(t["lon"].to_numpy(float), t["lat"].to_numpy(float))
        px, py = ln.get_data()
        assert _inside(np.asarray(px), np.asarray(py), tx, ty)
        radii.append(np.max(np.hypot(px - tx.mean(), py - ty.mean())))
    assert radii[1] < radii[0] <= 0.06 * width * 1.3   # only the small triangle's ellipse shrank
    plt.close("all")


def test_portrait_box_row_layout():
    """Legend flush with the station table's left edge, speed table flush with
    its right edge, velocity key centred with equal gaps; same top and height."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.table import Table

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    t = from_records(ALASKA)
    r = compute_strain(t, mc_samples=0)
    fig = plotting.strain_map(r, t, None, basemap="none", faults=False)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    tables = [c for a in fig.axes for c in a.get_children() if isinstance(c, Table)]
    speed = next(tb for tb in tables if len(tb.get_celld()) == 2 * (len(t) + 1))
    station = next(tb for tb in tables if tb is not speed)
    legend_box = next(a for a in fig.artists if isinstance(a, Rectangle))
    velocity_box = next(p for a in fig.axes for p in a.patches if isinstance(p, Rectangle))
    st, sp = station.get_window_extent(rend), speed.get_window_extent(rend)
    lg, vb = legend_box.get_window_extent(rend), velocity_box.get_window_extent(rend)
    assert abs(lg.x0 - st.x0) < 2 and abs(sp.x1 - st.x1) < 2
    assert abs((vb.x0 - lg.x1) - (sp.x0 - vb.x1)) < 2 and vb.x0 - lg.x1 > 10
    for b in (lg, vb):
        assert abs(b.y1 - sp.y1) < 2 and abs(b.height - sp.height) < 2
    title = next(x for a in fig.axes for x in a.texts if x.get_text().startswith("GPS Triangle-Strain"))
    above = title.get_window_extent(rend).y0 - sp.y1
    below = sp.y0 - st.y1
    assert abs(above - below) < 2
    plt.close("all")


@pytest.mark.parametrize("vis", [False, True])
def test_visibility_toggle(vis, tmp_path):
    """visibility=True draws thicker overlays with white edges and panels;
    False keeps the 2020 look. Both render every map product."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    from gps_strain import compute_strain, plotting
    from gps_strain.data import from_records

    t = from_records(TAHOE)
    r = compute_strain(t, mc_samples=0)
    fig = plotting.strain_map(r, t, tmp_path / "m.png", basemap="none", visibility=vis, faults=False)
    ax = fig.axes[0]
    blue = [ln for ln in ax.lines if ln.get_color() == "blue"]
    white_under = [ln for ln in ax.lines if ln.get_color() == "w" and ln.get_zorder() == 19]
    panels = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    style = plotting.HIGH_VISIBILITY if vis else plotting.CLASSIC
    assert blue[0].get_linewidth() == style.triangle_lw
    assert bool(white_under) == vis and bool(panels) == vis
    assert ax.collections[0].get_path_effects() if vis else not ax.collections[0].get_path_effects()
    plotting.symbol_map(r, t, tmp_path / "s.png", basemap="none", visibility=vis, faults=False)
    plotting.network_map([r], [t], tmp_path / "n.png", kind="ellipse", basemap="none", visibility=vis, faults=False)
    plt.close("all")


# --- Quaternary faults ---------------------------------------------------------------
QFAULT_PAGE = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"fault_name": "Mohawk Valley fault zone", "age": "<15,000"},
     "geometry": {"type": "LineString", "coordinates": [[-120.45, 39.30], [-120.30, 39.45], [-120.20, 39.55]]}},
    {"type": "Feature", "properties": {"fault_name": "North Tahoe fault", "age": "<130,000"},
     "geometry": {"type": "MultiLineString", "coordinates": [[[-120.15, 39.25], [-120.05, 39.35]],
                                                             [[-120.05, 39.35], [-120.00, 39.40]]]}},
    {"type": "Feature", "properties": {}, "geometry": None},
]}


def test_parse_qfaults_geojson():
    from gps_strain import faults

    fs = faults.parse_geojson(QFAULT_PAGE)
    assert [f["properties"]["fault_name"] for f in fs] == ["Mohawk Valley fault zone", "North Tahoe fault"]
    assert [len(f["lines"]) for f in fs] == [1, 2]
    assert fs[0]["lines"][0].shape == (3, 2)


def test_usgs_qfaults_pages_through_results(monkeypatch):
    import json

    from gps_strain import faults

    calls = []

    def fake_get(url, params=None, timeout=0, cache=True):
        calls.append(params["resultOffset"])
        if params["resultOffset"] == 0:
            return json.dumps(dict(QFAULT_PAGE, exceededTransferLimit=True)).encode()
        return json.dumps({"type": "FeatureCollection", "features": [QFAULT_PAGE["features"][0]]}).encode()

    monkeypatch.setattr(faults, "_get", fake_get)
    fs = faults.usgs_qfaults(-121, 39, -119.5, 40, page=3)
    assert calls == [0, 3] and len(fs) == 3
    out, credit = faults.faults_in_box(-121, 39, -119.5, 40)
    assert credit == faults.CREDIT_USGS and len(out) == 3


def test_faults_sit_between_terrain_and_symbols():
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, faults, plotting
    from gps_strain.data import from_records

    fs = faults.parse_geojson(QFAULT_PAGE)
    t = from_records(TAHOE)
    r = compute_strain(t, mc_samples=0)
    for vis in (False, True):
        fig = plotting.strain_map(r, t, None, dem_source=lambda e, w, h: _plane(e, w, h, 0.05, 0.02),
                                  faults=fs, visibility=vis)
        ax = fig.axes[0]
        fault_lines = [ln for ln in ax.lines if ln.get_color() == plotting.FAULT_COLOR]
        assert len(fault_lines) == 3
        terrain_z = max(im.get_zorder() for im in ax.images)
        symbols_z = min(a.get_zorder() for a in ax.lines + ax.collections + ax.patches
                        if a not in fault_lines)
        assert terrain_z < plotting.FAULT_ZORDER < symbols_z
        labels = [tx.get_text() for tx in ax.get_legend().get_texts()]
        assert "Quaternary fault" in labels
    # fetched faults are not drawn on the made-up offline terrain
    fig = plotting.strain_map(r, t, None, basemap="synthetic")
    assert not [ln for ln in fig.axes[0].lines if ln.get_color() == plotting.FAULT_COLOR]
    plt.close("all")


def test_map_survives_fault_service_failure(monkeypatch):
    import matplotlib.pyplot as plt

    from gps_strain import compute_strain, faults, plotting
    from gps_strain.data import from_records

    def down(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(faults, "faults_in_box", down)
    t = from_records(TAHOE)
    r = compute_strain(t, mc_samples=0)
    fig = plotting.strain_map(r, t, None, basemap="none")
    assert any("Faults unavailable" in x.get_text() for x in fig.axes[0].texts)
    plt.close("all")
