"""Recreate the 2021 Raton, NM network maps (seven triangles on one map).

The 2021 study (Luke Pajer with Eric Lindsey, UNM) used these station
triangles. Velocities below are the values saved in the 2021 notebook, which
were computed with the flawed 2020 velocity method (mean displacement / years
over the previous 365 days), so treat them as illustrative only. Several of
these triangles are larger than GETSI Unit 6's 100 km side limit; the package
warns about that.

For current values:
    midas = data.load_midas("NA"); tables = [data.select_sites(midas, t) for t in TRIANGLES]

Run with ``--offline`` for the synthetic preview basemap.
"""
import sys

from gps_strain import compute_strain, plotting
from gps_strain.data import from_records

STATIONS = {  # site: (lat, lon, vE mm/yr, vN mm/yr)
    "RG24": (37.963525, -104.966767, -0.285099, 0.170120),
    "P036": (36.420274, -105.293654, -0.319222, 0.128859),
    "P039": (36.448113, -103.153954, -0.102473, 0.339794),
    "P070": (36.044780, -104.698000, -0.264772, 0.278004),
    "P040": (38.071499, -102.686980, -0.160552, -0.059304),
    "RG23": (37.743873, -105.498515, -0.072888, 0.080673),
    "P123": (36.635171, -105.910847, -0.196786, 0.123793),
    "RG13": (36.491259, -104.211489, -0.345939, -0.331060),
    "P038": (34.147255, -103.407339, -0.166641, 0.296628),
    "RG21": (37.767847, -107.109195, -0.207118, 0.087253),
}
TRIANGLES = [
    ["RG24", "P036", "P039"], ["P070", "P039", "RG24"], ["P070", "P040", "RG23"],
    ["P123", "P039", "RG24"], ["P123", "RG13", "RG24"], ["P123", "P070", "P040"],
    ["P038", "P040", "RG21"],
]


def table(sites, sigma=0.05):
    return from_records([dict(site=s, lat=STATIONS[s][0], lon=STATIONS[s][1], ve=STATIONS[s][2],
                              vn=STATIONS[s][3], sig_e=sigma, sig_n=sigma) for s in sites])


if __name__ == "__main__":
    basemap = "synthetic" if "--offline" in sys.argv else "esri-terrain"
    tables = [table(t) for t in TRIANGLES]
    results = [compute_strain(t) for t in tables]
    for r in results:
        print("-".join(r.sites), f"e1H {r.e1 * 1e9:8.2f}  e2H {r.e2 * 1e9:8.2f}  S1H {r.azimuth_s1:5.1f}°",
              "|", "; ".join(r.warnings) or "ok")
    plotting.network_map(results, tables, "raton_network_symbol.png", kind="symbol", basemap=basemap)
    plotting.network_map(results, tables, "raton_network_ellipse.png", kind="ellipse", basemap=basemap)
