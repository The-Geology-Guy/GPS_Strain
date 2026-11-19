"""GPS triangle strain: horizontal infinitesimal strain rate from GNSS velocities.

Python implementation of the method of Vincent Cronin (Baylor) and Phil Resor
(Wesleyan) used in the SERC/GETSI module "GPS, Strain, and Earthquakes".

Quick start::

    from gps_strain import data, compute_strain, plotting
    table = data.load_midas("NA")                     # NGL MIDAS, North America fixed
    sta = data.select_sites(table, ["P146", "P149", "P150"])
    result = compute_strain(sta)                      # local ENU frame
    print(result.report())
    plotting.symbol_map(result, sta, "symbol.pdf")   # landscape strain-symbol map
    plotting.strain_map(result, sta, "strain.pdf")   # portrait velocity/ellipse map
"""
from .strain import StrainResult, compute_strain, principal, solve_planar  # noqa: F401

__version__ = "2.0.0.dev0"
