"""Tests for the web-app bridge (gps_strain.web), with the browser's network
replaced by in-memory responses in the services' formats."""
import json

import numpy as np
import pytest
from urllib.parse import parse_qs

from gps_strain import terrain, web

TAHOE = [dict(site="P146", lat=39.337463, lon=-120.537302, ve=-9.927, vn=8.814, sig_e=0.175, sig_n=0.156),
         dict(site="P149", lat=39.602129, lon=-120.104977, ve=-8.553, vn=7.557, sig_e=0.217, sig_n=0.161),
         dict(site="P150", lat=39.292380, lon=-120.033855, ve=-9.186, vn=8.139, sig_e=0.174, sig_n=0.160)]
P151 = dict(site="P151", lat=39.75, lon=-120.45, ve=-9.5, vn=8.0, sig_e=0.2, sig_n=0.2)
QFAULTS = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"fault_name": "Mohawk Valley fault zone"},
     "geometry": {"type": "LineString", "coordinates": [[-120.45, 39.30], [-120.30, 39.45]]}}]}


@pytest.fixture
def fake_browser(monkeypatch, tmp_path):
    """Stand-in for the browser's XMLHttpRequest: 3DEP raw grids and Qfaults pages."""
    monkeypatch.setenv("GPS_STRAIN_CACHE", str(tmp_path))
    calls = []

    def fetch(url, params=None, timeout=0):
        calls.append(url)
        if url == terrain.THREE_DEP:
            w, h = (int(v) for v in params["size"].split(","))
            yy, xx = np.mgrid[0:h, 0:w]
            z = (1500 + 300 * np.sin(xx / 37.0) * np.cos(yy / 23.0)).astype("<f4")
            return 200, z.tobytes() + np.packbits(np.ones(w * h, np.uint8)).tobytes()
        if "Qfaults" in url:
            return 200, json.dumps(QFAULTS).encode()
        return 404, b""

    monkeypatch.setattr(terrain, "HTTP_FETCH", fetch)
    return calls


def test_analyze_one_triangle():
    out = json.loads(web.analyze(json.dumps({"triangles": [TAHOE]})))
    assert out["ok"]
    t = out["triangles"][0]
    assert t["sites"] == ["P146", "P149", "P150"]
    assert t["e1"] > 0 > t["e2"] and 0 <= t["azimuth_s1"] < 180
    assert t["sigma"]["e1"] > 0 and "e1H" in t["report"]
    layers = {f["properties"]["layer"] for f in out["geojson"]["features"]}
    assert {"triangle", "station", "velocity", "velocity_head", "circle", "ellipse", "axis_s1", "axis_s2"} <= layers
    json.dumps(out)  # strict JSON (no NaN)


def test_analyze_symbol_and_network():
    second = [TAHOE[0], TAHOE[1], P151]
    out = json.loads(web.analyze(json.dumps({"triangles": [TAHOE, second], "glyph": "symbol"})))
    assert out["ok"] and len(out["triangles"]) == 2
    feats = out["geojson"]["features"]
    assert {f["properties"]["tri"] for f in feats if f["properties"]["layer"] in ("elongation", "contraction")} == {0, 1}
    assert sum(f["properties"]["layer"] == "station" for f in feats) == 4   # P146, P149 shared


def test_map_glyphs_stay_inside_their_triangle():
    out = json.loads(web.analyze(json.dumps({"triangles": [TAHOE]})))
    from matplotlib.path import Path

    tri = Path([[s["lon"], s["lat"]] for s in TAHOE])
    for f in out["geojson"]["features"]:
        if f["properties"]["layer"] in ("ellipse", "circle_line"):
            assert tri.contains_points(np.array(f["geometry"]["coordinates"])).all()


def test_analyze_rejects_stations_outside_the_us():
    bad = [dict(TAHOE[0]), dict(TAHOE[1]), dict(TAHOE[2], site="TOKY", lat=35.7, lon=139.7)]
    out = json.loads(web.analyze(json.dumps({"triangles": [bad]})))
    assert not out["ok"] and "United States" in out["error"] and "TOKY" in out["error"]
    out = json.loads(web.analyze(json.dumps({"triangles": [TAHOE[:2]]})))
    assert not out["ok"] and "three stations" in out["error"]


@pytest.mark.parametrize("sheet,fmt,magic", [("strain", "png", b"\x89PNG"), ("symbol", "pdf", b"%PDF"),
                                             ("network_ellipse", "svg", b"<?xml")])
def test_report_sheets(fake_browser, sheet, fmt, magic):
    req = {"triangles": [TAHOE], "sheet": sheet, "fmt": fmt, "dpi": 40, "terrain_px": 300,
           "basemap": "hillshade", "visibility": True}
    data = web.report(json.dumps(req))
    assert data.startswith(magic) and len(data) > 5000
    assert any(u == terrain.THREE_DEP for u in fake_browser)     # terrain from 3DEP only
    assert any("Qfaults" in u for u in fake_browser)
    assert not any("copernicus" in u for u in fake_browser)


def test_report_terrain_choices(fake_browser):
    for basemap in ("hillshade-color", "elevation", "slope"):
        assert web.report(json.dumps({"triangles": [TAHOE], "basemap": basemap, "dpi": 30,
                                      "terrain_px": 200, "faults": False})).startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="not offered"):
        web.report(json.dumps({"triangles": [TAHOE], "basemap": "hillshade-copernicus"}))


def test_results_csv():
    text = web.results_csv(json.dumps({"triangles": [TAHOE]}))
    head, row = text.strip().split("\n")
    assert head.startswith("triangle,frame,e1H") and row.startswith("P146-P149-P150,enu,")


def test_midas_us_mirror():
    import pandas as pd

    table = pd.DataFrame([
        dict(site="P146", lat=39.337463, lon=-120.537302, ve=-0.009927, vn=0.008814, sig_e=0.000175,
             sig_n=0.000156, t_first=2005.1, t_last=2026.8),
        dict(site="TOKY", lat=35.7, lon=139.7, ve=0.0, vn=0.0, sig_e=0.001, sig_n=0.001, t_first=2010, t_last=2026),
        dict(site="AC63", lat=63.502425, lon=-145.847246, ve=0.002563, vn=0.001006, sig_e=0.000204,
             sig_n=0.000195, t_first=2006.5, t_last=2026.8),
    ])
    d = web.midas_us_json(table, frame="NA", generated="2026-11-19")
    assert d["count"] == 2 and [r[0] for r in d["rows"]] == ["AC63", "P146"]
    row = dict(zip(d["fields"], d["rows"][1]))
    assert row["ve"] == -9.927 and row["sig_n"] == 0.156 and row["years"] == 21.7
    assert "Blewitt" in d["citation"]
