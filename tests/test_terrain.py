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

    fig = plotting.symbol_map(r, t, tmp_path / "s.png", dem_source=offline)
    texts = [x.get_text() for x in fig.axes[0].texts]
    assert any("Basemap unavailable" in s for s in texts)
    with pytest.raises(ValueError, match="unknown basemap"):
        plotting.symbol_map(r, t, None, basemap="esri-terrain")
