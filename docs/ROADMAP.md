# Roadmap: from Python package to Mac/PC app and gps-strain.org

## Where things stand (v2.0.0.dev0, on `main`)

- **Done:** strain solver for 3+ stations: weighted least squares, full covariance, Monte Carlo uncertainties on derived quantities.
- **Done:** classic UTM mode (matches Cronin's spreadsheet) and a local-ENU default with velocity rotation.
- **Done:** data sources: NGL MIDAS (free, global, weekly) with caching; CSV/Excel import; NGL time series with velocity fitting.
- **Done:** the two 2020 map layouts recreated without cartopy:
  - `plotting.symbol_map`: landscape strain-symbol map;
  - `plotting.strain_map`: portrait velocity and strain-ellipse map with tables.

  The default background is GEOSLAM's grey hillshade, computed from a DEM (`terrain.py`): USGS 3DEP in the US, Copernicus GLO-30/90 elsewhere, no key and no tile licence. USGS The National Map tiles are optional (`basemap.py`). Esri and Stadia were dropped (unclear print and commercial terms, and no longer needed). The logo slot shows a block-letter "GPS STRAIN" wordmark by default (or any image file/URL).
- **Done:** `plotting.network_map`: several triangles on one map (symbol or ellipse), like the 2021 Raton, NM figures, plus `gps-strain --triangles`.
- **Done:** command-line tool `gps-strain`.
- **Done:** the 2020 AC63 results table is reproduced to every printed digit in UTM mode. The 2020 strain algebra was right; its velocities were not.
- **Done:** 36 tests, including exact reproduction of Cronin & Resor's worked example, a rigid-plate null test, offline map renders and the terrain pipeline (3DEP and Copernicus response formats, hillshade geometry, water mask).

### Live tests

Run on a normal connection on 18 Nov 2026 (`gps-strain P146 P149 P150 -o tahoe`, `gps-strain AC63 AC62 AB37 -o alaska`, and Tahoe again with `--basemap hillshade-copernicus`):

- **Working:** the MIDAS download and parse; USGS 3DEP terrain (CONUS and Alaska); Copernicus terrain.
- **Fixed after that run:** 3DEP tile seams left a thin grey line across Lake Tahoe and Pyramid Lake; the water mask now closes such gaps.
- **Not yet tested live:** the `.tenv3` loader and the USGS tile basemaps.

## Next steps (package)

1. Done (18 Nov 2026): live test of MIDAS, 3DEP and Copernicus. Still to do: `.tenv3` and USGS tiles; add a real `midas.NA.txt` excerpt as a test fixture.
2. Logo: decided. No EarthScope logo; the sheets carry a block-letter "GPS STRAIN" wordmark (custom logos still possible with `--logo`).
3. Done: grey DEM hillshade as the default printed-map terrain (decided 18 Nov 2026). An optional colour tint only if wanted later.
4. Basemap licences: settled. The hillshade needs none; Esri and Stadia were removed. `docs/research/04_costs.md` is kept for the record.
5. **GETSI Unit 6 (final project) support.** Unit 6 asks students to choose three adjacent NOTA stations in an actively straining area near people or infrastructure:
   - all interior angles above ~30° and no side over 100 km (both now checked and warned about);
   - map active faults from the USGS Quaternary Fault and Fold Database;
   - compute strain, interpret it, and present tectonics, hazard, risk and mitigation in a 5–7 minute talk.

   To add:
   - Done (18 Nov 2026): Quaternary fault overlay on all maps (`faults.py`, ported from GEOSLAM; USGS, GEM outside the US), drawn between the terrain and the map symbols;
   - a "suggest triangles" helper that searches MIDAS stations near a point for valid triangles;
   - a slide-ready export (PNG at 16:9).

   The strain ellipse figure replaces the Wolfram CDF ellipse visualizer the unit relies on.
6. Second validation case from the GETSI Excel calculator and Tohoku results, at `https://cdn.serc.carleton.edu/files/getsi/teaching_materials/gps_strain/` (`gps_strain_calculator_excel.v3.xls`, `gps_strain_calculator_results.xlsx`). Cross-check against Strain_2D (`delaunay_flat`).
7. Networks: automatic Delaunay triangulation over many stations with a skinny/oversized-triangle filter, feeding `network_map` (which already draws hand-picked triangles).
8. Optional EarthScope/GAGE NAM14 source (free login and token).
9. GitHub Actions CI on Linux, Mac and Windows; publish to PyPI.

## gps-strain.org (web app)

Built to match **geoslam.org**: the same Windows-98 toolkit, desktop and mobile layouts, and Cronin reviewing the site before it is indexed. The solver is small enough to port to JavaScript or to run with Pyodide.

- **Data:** mirror `midas.NA.txt` to JSON on the same domain with a scheduled job, since NGL may not allow direct browser requests.
- **Basemap:** the same as geoslam.org: USGS The National Map tiles (topo default, shaded relief, imagery) plus OpenStreetMap, in MapLibre.
- **Printing:** like geoslam.org, run this package in the browser (Pyodide, matplotlib loaded on first use) and draw the two sheets there, with the grey hillshade from 3DEP fetched by the browser.

## Desktop app

**Recommended stack:** PySide6 (Qt) GUI + this package + matplotlib, frozen with PyInstaller and built on GitHub Actions for macOS (arm64 and x86_64) and Windows.

App flow:

1. **Pick stations:** search, map click, "near a point", or CSV import.
2. **Choose** the source, frame and basemap.
3. **Preview** both sheets live.
4. **Export** PDF/PNG/SVG/CSV, or print.

Distribution costs:

- **macOS:** Apple Developer ID $99/yr plus notarization. Ship a notarized DMG first; the Mac App Store sandbox is hard for bundled Python.
- **Windows:** the Microsoft Store is free for individuals. Direct downloads can be signed with Azure Artifact Signing (~$120/yr).

## Credits and licensing to keep

- **Method:** Vincent Cronin (Baylor) and Phil Resor (Wesleyan). GETSI materials are CC BY-NC-SA: link to them, don't copy their text.
- **Data:** cite NGL/MIDAS (Blewitt et al. 2016, 2018) on every output that uses it.
- **Basemap attribution:** printed on every map automatically.
