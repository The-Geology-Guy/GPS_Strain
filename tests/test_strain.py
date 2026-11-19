"""Validation tests.

The primary test reproduces the worked example in Cronin & Resor,
"Algorithm for triangle strain" (TriangleStrainAlgorithm.pdf, 5 Apr 2021),
stations P146, P149, P150.
"""
import numpy as np
import pytest

from gps_strain import compute_strain, geodesy, strain
from gps_strain.data import fit_velocity, from_records, parse_midas

# --- Cronin & Resor worked example --------------------------------------
UTM_E = np.array([712245.807, 748566.739, 755806.255])
UTM_N = np.array([4357118.796, 4387604.015, 4353418.458])
VE = np.array([-0.01031, -0.00942, -0.01086])
VN = np.array([0.00625, 0.00520, 0.00592])
SIG = np.array([0.00001, 0.00003, 0.00003])


def cronin_result():
    params, cov, x, y, chi2 = strain.solve_planar(UTM_E, UTM_N, VE, VN, SIG, SIG)
    return strain.StrainResult(
        sites=["P146", "P149", "P150"], x=x, y=y, ve=VE, vn=VN, params=params, cov=cov,
        frame="utm", origin=("UTM zone", 10),
    )


def test_cronin_parameters():
    r = cronin_result()
    expected = [-0.0101967, 0.00579, -2.48541e-8, -9.21367e-9, 1.53178e-8, -2.30811e-8]
    np.testing.assert_allclose(r.params, expected, rtol=2e-5)


def test_cronin_centered_coordinates():
    r = cronin_result()
    # the document's Step 3 uses slightly different eastings (a typo), hence 0.1 m
    np.testing.assert_allclose(r.x, [-26627.1, 9693.8, 16933.4], atol=0.1)
    np.testing.assert_allclose(r.y, [-8928.3, 21556.9, -12628.6], atol=0.1)


def test_cronin_uncertainties():
    r = cronin_result()
    expected = [1.453e-5, 1.453e-5, 6.7227e-10, 6.7197e-10, 6.7227e-10, 1.1646e-9]
    np.testing.assert_allclose(r.sigma, expected, rtol=2e-4)


def test_cronin_derived():
    r = cronin_result()
    assert r.e1 * 1e9 == pytest.approx(0.66663, rel=1e-4)
    assert r.e2 * 1e9 == pytest.approx(-32.9614, rel=1e-5)
    assert r.max_shear * 1e9 == pytest.approx(33.628, rel=1e-4)
    assert r.area_strain * 1e9 == pytest.approx(-32.2948, rel=1e-5)
    I1, I2, I3 = r.invariants
    assert I2 == pytest.approx(3.2968e-8, rel=1e-4)
    assert I3 == pytest.approx(-2.1973e-17, rel=1e-4)
    # S1H "237° (or 57°)", S2H "327° (or 147°)"; we report axial [0, 180)
    assert r.azimuth_s1 == pytest.approx(57.18, abs=0.02)
    assert r.azimuth_s2 == pytest.approx(147.18, abs=0.02)
    assert r.speed == pytest.approx(0.0117259, rel=1e-5)
    assert r.translation_azimuth == pytest.approx(360 - 60.4107, abs=1e-3)
    assert r.rotation * 1e9 == pytest.approx(-24.8541, rel=1e-5)
    assert r.rotation_sense == "clockwise"


# --- conventions on synthetic fields -------------------------------------
def _synthetic(L, w=0.0):
    x = np.array([-10e3, 12e3, 3e3])
    y = np.array([-6e3, -4e3, 11e3])
    Lw = np.array(L, float) + np.array([[0, -w], [w, 0]])
    v = (Lw @ np.vstack([x, y])).T + [0.002, -0.001]
    params, cov, *_ = strain.solve_planar(x, y, v[:, 0], v[:, 1], [1e-4] * 3, [1e-4] * 3)
    return params


def test_rigid_ccw_rotation_positive():
    p = _synthetic(np.zeros((2, 2)), w=1e-7)
    assert p[2] == pytest.approx(1e-7, rel=1e-9)
    np.testing.assert_allclose(p[3:], 0, atol=1e-18)


def test_ew_extension_azimuth():
    p = _synthetic([[50e-9, 0], [0, -20e-9]])
    e1, e2, az1, az2 = strain.principal(*p[3:])
    assert e1 == pytest.approx(50e-9)
    assert az1 == pytest.approx(90.0)
    assert az2 == pytest.approx(0.0, abs=1e-9)


def test_ne_extension_azimuth():
    # pure shear with extension toward N45E: exy > 0, exx = eyy = 0
    p = _synthetic([[0, 30e-9], [30e-9, 0]])
    e1, e2, az1, _ = strain.principal(*p[3:])
    assert az1 == pytest.approx(45.0)
    assert e1 == pytest.approx(30e-9) and e2 == pytest.approx(-30e-9)


def test_overdetermined_chi2():
    x = np.array([-10e3, 12e3, 3e3, 20e3, -15e3])
    y = np.array([-6e3, -4e3, 11e3, 9e3, 14e3])
    L = np.array([[40e-9, 10e-9], [10e-9, -25e-9]])
    v = (L @ np.vstack([x, y])).T
    rng = np.random.default_rng(1)
    sig = np.full(5, 2e-4)
    ve = v[:, 0] + rng.normal(0, 2e-4, 5)
    vn = v[:, 1] + rng.normal(0, 2e-4, 5)
    params, cov, *_, chi2 = strain.solve_planar(x, y, ve, vn, sig, sig)
    assert chi2 is not None and 0 < chi2 < 10


# --- geodesy ---------------------------------------------------------------
def test_dutch_series_matches_kruger():
    lats = np.array([39.3375, 39.6021, 39.2924, 45.0, 60.0])
    lons = np.array([-120.5373, -120.1050, -120.0339, -125.9, -121.0])
    e1, n1 = geodesy.utm_dutch_series(lats, lons, -123.0)
    e2, n2 = geodesy.utm_kruger(lats, lons, -123.0)
    assert np.max(np.abs(e1 - e2)) < 0.5
    assert np.max(np.abs(n1 - n2)) < 0.5


def test_kruger_reference_point():
    # On the central meridian at 45N, northing = k0 * meridian arc (4984944.378 m)
    e, n = geodesy.utm_kruger(45.0, -123.0, -123.0)
    assert e == pytest.approx(500000.0)
    assert n == pytest.approx(0.9996 * 4984944.378, abs=0.01)


def test_kruger_roundtrip():
    lat, lon = geodesy.utm_kruger_inverse(UTM_E, UTM_N, -123.0)
    e, n = geodesy.utm_kruger(lat, lon, -123.0)
    np.testing.assert_allclose(e, UTM_E, atol=1e-6)
    np.testing.assert_allclose(n, UTM_N, atol=1e-6)


def test_westernmost_zone():
    assert geodesy.westernmost_zone([-120.5, -119.0, -117.5]) == 10
    assert geodesy.westernmost_zone([-119.5, -117.5]) == 11
    assert geodesy.westernmost_zone([-121.0, -119.0]) == 10
    assert geodesy.westernmost_zone([179.5, -179.5]) == 60


def test_velocity_rotation_preserves_length():
    lat = np.array([40.0, 41.0, 39.5])
    lon = np.array([-120.0, -118.0, -121.0])
    ve = np.array([0.01, -0.005, 0.0])
    vn = np.array([0.0, 0.007, 0.012])
    ver, vnr, _ = geodesy.rotate_velocities_to_frame(lat, lon, ve, vn, 40.0, -119.5)
    np.testing.assert_allclose(np.hypot(ver, vnr), np.hypot(ve, vn), rtol=1e-6)


# --- end-to-end -------------------------------------------------------------
def _cronin_latlon_table():
    lat, lon = geodesy.utm_kruger_inverse(UTM_E, UTM_N, -123.0)
    return from_records(
        [dict(site=s, lat=a, lon=o, ve=e * 1000, vn=n * 1000, sig_e=g * 1000, sig_n=g * 1000)
         for s, a, o, e, n, g in zip(["P146", "P149", "P150"], lat, lon, VE, VN, SIG)]
    )


def test_compute_strain_utm_mode_matches_spreadsheet():
    r = compute_strain(_cronin_latlon_table(), frame="utm", mc_samples=0)
    assert r.origin == ("UTM zone", 10)
    assert r.e2 * 1e9 == pytest.approx(-32.9614, rel=2e-4)
    assert r.rotation * 1e9 == pytest.approx(-24.8541, rel=2e-4)


def test_enu_mode_close_to_utm_mode():
    t = _cronin_latlon_table()
    a = compute_strain(t, frame="utm", mc_samples=0)
    b = compute_strain(t, frame="enu", mc_samples=0)
    # The classic method mixes velocities given in each station's own local
    # E/N frame with planar coordinates. Meridian convergence then leaks a
    # spurious gradient of order |v| tan(lat) / R (~1.5 nstrain/yr here).
    bound = 2 * np.hypot(a.tx, a.ty) * np.tan(np.radians(39.4)) / 6.371e6
    np.testing.assert_array_less(np.abs(b.params[2:] - a.params[2:]), bound)


def _euler_pole_velocities(lat, lon, pole_lat, pole_lon, deg_per_myr):
    """Rigid-plate velocities (m/yr, local E/N) from an Euler pole."""
    w = np.radians(deg_per_myr) / 1e6
    P = np.array(geodesy.geodetic_to_ecef(pole_lat, pole_lon))
    Om = w * P / np.linalg.norm(P)
    X = np.stack(geodesy.geodetic_to_ecef(lat, lon), axis=-1)
    v = np.cross(Om, X)
    e, n, _ = geodesy.enu_basis(lat, lon)
    return np.sum(v * e, axis=1), np.sum(v * n, axis=1)


def test_rigid_plate_gives_no_strain_in_enu_mode():
    """A rigid rotation on the sphere must produce (essentially) zero strain."""
    lat = np.array([39.34, 39.60, 39.29])
    lon = np.array([-120.54, -120.10, -120.03])
    ve, vn = _euler_pole_velocities(lat, lon, -48.7, 101.8, 0.75)  # PA-NA-like pole
    tbl = from_records([dict(site=f"S{i}", lat=a, lon=o, ve=e * 1e3, vn=n * 1e3, sig_e=0.1, sig_n=0.1)
                        for i, (a, o, e, n) in enumerate(zip(lat, lon, ve, vn))])
    enu = compute_strain(tbl, frame="enu", mc_samples=0)
    utm = compute_strain(tbl, frame="utm", mc_samples=0)
    assert np.hypot(enu.tx, enu.ty) > 0.03          # ~4-5 cm/yr of plate motion
    assert max(abs(enu.e1), abs(enu.e2)) < 0.05e-9   # < 0.05 nstrain/yr
    # the classic UTM method reports spurious strain for the same rigid motion
    assert max(abs(utm.e1), abs(utm.e2)) > 1e-9


def test_monte_carlo_uncertainties_reasonable():
    r = compute_strain(_cronin_latlon_table(), frame="enu", mc_samples=20000)
    # e2 is well constrained (~1 nstrain/yr), azimuth to a few degrees
    assert 0.3e-9 < r.mc["e2"] < 3e-9
    assert 0 < r.mc["az1"] < 5


def test_triangle_rule_warnings():
    good = compute_strain(_cronin_latlon_table(), mc_samples=0)
    assert not any("100 km" in w for w in good.warnings)  # Tahoe sides are 39-48 km
    big = from_records([dict(site=s, lat=a, lon=o, ve=1, vn=1, sig_e=0.1, sig_n=0.1)
                        for s, a, o in [("A", 39.0, -121.0), ("B", 40.5, -120.0), ("C", 39.0, -119.0)]])
    assert any("100 km" in w for w in compute_strain(big, mc_samples=0).warnings)


def test_report_runs():
    r = compute_strain(_cronin_latlon_table())
    text = r.report()
    assert "S1H azimuth" in text and "clockwise" in text


# --- data ------------------------------------------------------------------
MIDAS_SAMPLE = """\
------------------------------------------------------------------------------------
P146 MIDAS5 2005.1000 2026.7000 21.6000 7800 7600 9000 -0.010310 0.006250 0.000100 0.000250 0.000210 0.000800 0.0 0.0 0.0 0.01 0.01 0.02 0.001 0.001 0.003 2 39.3375 239.4627 1500.0
P149 MIDAS5 2005.2000 2026.7000 21.5000 7700 7500 8900 -0.009420 0.005200 0.000200 0.000260 0.000220 0.000810 0.0 0.0 0.0 0.01 0.01 0.02 0.001 0.001 0.003 1 39.6021 -120.1050 1800.0
"""


def test_parse_midas():
    df = parse_midas(MIDAS_SAMPLE)
    assert list(df.site) == ["P146", "P149"]
    assert df.lon.iloc[0] == pytest.approx(-120.5373)
    assert df.ve.iloc[0] == pytest.approx(-0.01031)
    assert df.sig_n.iloc[1] == pytest.approx(0.00022)


def test_fit_velocity_recovers_slope():
    t = np.arange(2010, 2020, 1 / 365.25)
    rng = np.random.default_rng(3)
    pos = 0.1 + 0.0123 * (t - 2010) + 0.002 * np.sin(2 * np.pi * t) + rng.normal(0, 0.002, t.size)
    v, s = fit_velocity(t, pos)
    assert v == pytest.approx(0.0123, abs=3 * s + 1e-5)
    # the 2020 method (mean displacement / duration) is badly biased:
    old = np.mean(pos - pos[0]) / 10
    assert abs(old - 0.0123) > 0.005


# --- plotting (offline) ---------------------------------------------------
def test_maps_render_offline(tmp_path):
    from gps_strain import plotting

    t = _cronin_latlon_table()
    r = compute_strain(t, mc_samples=0)
    plotting.symbol_map(r, t, tmp_path / "s.png", basemap="synthetic")
    plotting.strain_map(r, t, tmp_path / "m.png", basemap="synthetic")
    assert (tmp_path / "s.png").stat().st_size > 10000
    assert (tmp_path / "m.png").stat().st_size > 10000
