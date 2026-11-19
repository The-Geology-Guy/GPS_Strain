"""Command line interface.

Examples::

    gps-strain P146 P149 P150                       # report only (NGL MIDAS, North-America fixed)
    gps-strain P146 P149 P150 -o tahoe              # writes tahoe_symbol.pdf and tahoe_strain.pdf
    gps-strain P146 P149 P150 -o tahoe --format png --basemap stadia-terrain-background
    gps-strain --table my_stations.csv -o out --logo my_logo.png
    gps-strain --near 38.2 -122.3 --radius 30       # list stations near a point
    gps-strain --triangles RG24,P036,P039 P070,P039,RG24 -o raton   # several triangles on one map
"""
from __future__ import annotations

import argparse
import sys

from . import data, plotting
from .strain import compute_strain

MIDAS_CREDIT = ("NGL MIDAS velocities (IGS20, {plate}-fixed); Blewitt, Hammond & Kreemer 2018, "
                "Eos 99, doi:10.1029/2018EO104623; Blewitt et al. 2016, JGR, doi:10.1002/2015JB012552")


def main(argv=None):
    p = argparse.ArgumentParser(prog="gps-strain", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sites", nargs="*", help="three or more station IDs")
    p.add_argument("--table", help="CSV/Excel with site, lat, lon, ve, vn, sig_e, sig_n")
    p.add_argument("--units", default="mm/yr", choices=["mm/yr", "m/yr"], help="units in --table")
    p.add_argument("--plate", default="NA", help="MIDAS frame: IGS or plate code (NA, PA, EU, ...)")
    p.add_argument("--frame", default="enu", choices=["enu", "utm"],
                   help="enu = local tangent plane (default); utm = classic GETSI spreadsheet")
    p.add_argument("--near", nargs=2, type=float, metavar=("LAT", "LON"), help="list stations near a point")
    p.add_argument("--radius", type=float, default=50.0, help="search radius in km for --near")
    p.add_argument("-o", "--output", help="output name prefix for the maps")
    p.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--map", default="both", choices=["both", "symbol", "strain"])
    p.add_argument("--basemap", default="esri-terrain",
                   help="esri-terrain, stadia-terrain-background, usgs-relief, usgs-topo, usgs-imagery, "
                        "esri-shaded-relief, esri-physical, "
                        "opentopomap, synthetic (offline preview) or none")
    p.add_argument("--api-key", help="basemap API key (or set STADIA_API_KEY / ESRI_API_KEY)")
    p.add_argument("--logo", help='logo image file or URL, or text (default: block letters "GPS STRAIN")')
    p.add_argument("--title", help="map title")
    p.add_argument("--triangles", nargs="+", metavar="A,B,C",
                   help="several triangles (comma-separated station IDs) for network maps")
    a = p.parse_args(argv)

    if a.triangles:
        src = data.read_table(a.table, units=a.units) if a.table else data.load_midas(a.plate)
        tables = [data.select_sites(src, t.split(",")) for t in a.triangles]
        results = [compute_strain(t, frame=a.frame) for t in tables]
        for r in results:
            print(r.report(), "\n")
        if a.output:
            for kind in ("symbol", "ellipse"):
                out = f"{a.output}_network_{kind}.{a.format}"
                plotting.network_map(results, tables, out, kind=kind, basemap=a.basemap,
                                     api_key=a.api_key, title=a.title)
                print(f"Wrote {out}")
        return 0

    if a.near:
        tbl = data.stations_near(data.load_midas(a.plate), *a.near, radius_km=a.radius)
        cols = ["site", "lat", "lon", "distance_km", "duration", "ve", "vn", "sig_e", "sig_n"]
        print(tbl[cols].to_string(index=False))
        return 0

    if a.table:
        table = data.read_table(a.table, units=a.units)
        if a.sites:
            table = data.select_sites(table, a.sites)
        label = "Station"
    else:
        if len(a.sites) < 3:
            p.error("give at least three station IDs, or --table")
        table = data.select_sites(data.load_midas(a.plate), a.sites)
        label = "NGL MIDAS"
        print(MIDAS_CREDIT.format(plate=a.plate))

    result = compute_strain(table, frame=a.frame)
    print(result.report())
    if a.output:
        kw = dict(title=a.title, data_label=label, logo=a.logo, basemap=a.basemap, api_key=a.api_key)
        if a.map in ("both", "symbol"):
            out = f"{a.output}_symbol.{a.format}"
            plotting.symbol_map(result, table, out, **kw)
            print(f"Wrote {out}")
        if a.map in ("both", "strain"):
            out = f"{a.output}_strain.{a.format}"
            frame = "North America" if a.plate == "NA" else a.plate
            plotting.strain_map(result, table, out, velocity_label=f"Velocity Relative to {frame}", **kw)
            print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
