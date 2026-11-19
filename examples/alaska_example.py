"""Recreate the 2020 AC63-AC62-AB37 (Alaska) maps.

Velocities below are the values printed on the 2020 map (which were computed
with the flawed 2020 velocity method, so treat them as illustrative only).
Use ``data.select_sites(data.load_midas("NA"), ["AC63", "AC62", "AB37"])``
for current values.

Run with ``--offline`` to use the synthetic preview basemap, and
``--logo=path/to/logo.png`` to add a logo.
"""
import sys

from gps_strain import compute_strain, plotting
from gps_strain.data import from_records

table = from_records([
    dict(site="AC63", lat=63.502426, lon=-145.847246, ve=2.626487, vn=-1.008585, sig_e=0.3, sig_n=0.3),
    dict(site="AC62", lat=63.083606, lon=-146.312695, ve=-8.66512, vn=3.464374, sig_e=0.3, sig_n=0.3),
    dict(site="AB37", lat=62.967322, lon=-145.451877, ve=-9.815231, vn=5.922223, sig_e=0.3, sig_n=0.3),
])
basemap = "synthetic" if "--offline" in sys.argv else "esri-terrain"
logo = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--logo=")), None)

r = compute_strain(table)
print(r.report())
plotting.symbol_map(r, table, "AC63_symbol.png", basemap=basemap, logo=logo)
plotting.strain_map(r, table, "AC63_strain.png", basemap=basemap, logo=logo)
