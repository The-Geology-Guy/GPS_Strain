"""Reproduce Cronin & Resor's worked example (P146, P149, P150) and draw the maps.

The algorithm document gives UTM zone 10 coordinates; lat/lon here are the
inverse projection of those values. Run with ``--offline`` for the synthetic
preview basemap.
"""
import sys

from gps_strain import compute_strain, geodesy, plotting
from gps_strain.data import from_records

E = [712245.807, 748566.739, 755806.255]
N = [4357118.796, 4387604.015, 4353418.458]
lat, lon = geodesy.utm_kruger_inverse(E, N, -123.0)
table = from_records([
    dict(site="P146", lat=lat[0], lon=lon[0], ve=-10.31, vn=6.25, sig_e=0.01, sig_n=0.01),
    dict(site="P149", lat=lat[1], lon=lon[1], ve=-9.42, vn=5.20, sig_e=0.03, sig_n=0.03),
    dict(site="P150", lat=lat[2], lon=lon[2], ve=-10.86, vn=5.92, sig_e=0.03, sig_n=0.03),
])  # mm/yr

for frame in ("utm", "enu"):
    r = compute_strain(table, frame=frame)
    print(f"===== frame = {frame} =====")
    print(r.report(), "\n")

basemap = "synthetic" if "--offline" in sys.argv else "esri-terrain"
plotting.symbol_map(r, table, "cronin_symbol.png", basemap=basemap)
plotting.strain_map(r, table, "cronin_strain.png", basemap=basemap)
