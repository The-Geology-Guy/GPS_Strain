# GPS_Strain

**Horizontal infinitesimal strain rate from GPS/GNSS station velocities**: a Python implementation of the triangle strain method of [Vincent Cronin](https://croninprojects.org/Vince/) (Baylor) and Phil Resor (Wesleyan), taught in the SERC/GETSI module [*GPS, Strain, and Earthquakes*](https://serc.carleton.edu/getsi/teaching_materials/gps_strain/index.html).

> **Version 2 (in development).** The 2020 code no longer installed or ran, and it computed velocities incorrectly (see [docs/AUDIT_2020.md](docs/AUDIT_2020.md)). The core has been rebuilt and is validated against Cronin & Resor's published worked example. The original file is kept in `legacy/`.

Give it three or more stations and it reports the following, then draws the two classic map sheets (a strain-symbol map and a strain-ellipse/velocity map):

- the translation;
- the rotation;
- the strain-rate tensor;
- the principal strain rates **e1H / e2H** and their azimuths **S1H / S2H**;
- the maximum shear and area strain;
- 1σ uncertainties for all of the above.

<p align="center"><img src="examples/AC63_symbol.png" width="800"/></p>
<p align="center"><img src="examples/AC63_strain.png" width="600"/></p>

<sub>Example images were rendered offline with the synthetic preview basemap. With network access the default basemap is Esri World Terrain Base.</sub>

## Install

```bash
pip install git+https://github.com/The-Geology-Guy/GPS_Strain@v2-rebuild
```

Requires Python ≥ 3.11 with numpy, pandas, matplotlib, requests and Pillow. No cartopy or GDAL is needed.

## Use

Command line:

```bash
# velocities from the Nevada Geodetic Laboratory (MIDAS, North America fixed)
# writes tahoe_symbol.pdf and tahoe_strain.pdf
gps-strain P146 P149 P150 -o tahoe

# the original Stamen terrain-background look (needs a Stadia Maps key)
gps-strain P146 P149 P150 -o tahoe --basemap stadia-terrain-background --api-key YOUR_KEY

# add a logo (file or URL) to the sheets
gps-strain P146 P149 P150 -o tahoe --logo earthscope_logo.png

# which stations are near a point?
gps-strain --near 38.22 -122.31 --radius 30

# your own velocities (mm/yr): columns site, lat, lon, ve, vn, sig_e, sig_n
gps-strain --table examples/cronin_example.csv -o example

# reproduce the classic GETSI spreadsheet (UTM) instead of the local ENU frame
gps-strain --table examples/cronin_example.csv --frame utm
```

Python:

```python
from gps_strain import data, compute_strain, plotting

midas = data.load_midas("NA")            # cached for a week
sta = data.select_sites(midas, ["P146", "P149", "P150"])
r = compute_strain(sta)                  # frame="enu" (default) or "utm"
print(r.report())
r.e1, r.e2, r.azimuth_s1, r.max_shear, r.rotation, r.sigma, r.mc
plotting.symbol_map(r, sta, "symbol.pdf", logo="earthscope_logo.png")
plotting.strain_map(r, sta, "strain.pdf", basemap="stadia-terrain-background")
```

## Basemaps

| `--basemap` | Look | Key / cost |
|---|---|---|
| `esri-terrain` (default) | Hillshade, tan/green land, blue water: closest to Stamen terrain-background | A distributed app needs an ArcGIS Location Platform key (`ESRI_API_KEY`); 2M tiles/month free |
| `stadia-terrain-background` | The exact Stamen style the 2020 maps used | Stadia key (`STADIA_API_KEY`). The free tier is non-commercial; paid plans start at $20/mo. Print rights are limited (see `docs/research/04_costs.md`). |
| `esri-shaded-relief`, `esri-physical`, `esri-imagery`, `opentopomap` | Alternatives | Attribution printed automatically |
| `synthetic` | Offline procedural preview | None |

Tiles are cached in `~/.cache/gps_strain/tiles`.

## Tests

```bash
pip install -e .[test]
pytest
```

## Documentation

- [docs/METHOD.md](docs/METHOD.md): the math and geophysics, step by step, with references.
- [docs/AUDIT_2020.md](docs/AUDIT_2020.md): what was wrong with the 2020 version.
- [docs/ROADMAP.md](docs/ROADMAP.md): next steps, including the Mac/PC desktop app and gps-strain.org.
- [docs/research/](docs/research/): full research notes with source URLs, including the cost analysis.

## Data and credits

- **Method:** Cronin, V. S., & Resor, P. G. (2021), *Algorithm for triangle strain*; Cronin (2021), *Primer on infinitesimal strain analysis*.
- **Default velocities:** Nevada Geodetic Laboratory MIDAS. Please cite Blewitt, Hammond & Kreemer (2018), *Eos* 99, doi:10.1029/2018EO104623, and Blewitt et al. (2016), *JGR* 121, doi:10.1002/2015JB012552.
- **Python package:** Luke Pajer ([ORCID 0000-0002-5218-7650](https://orcid.org/0000-0002-5218-7650)).

MIT License. Issues and contributions are welcome; see the [code of conduct](CODE_OF_CONDUCT.md).
