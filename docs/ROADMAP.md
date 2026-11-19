# Roadmap: from Python package to Mac/PC app and gps-strain.org

## Where things stand (v2.0.0.dev0, branch `v2-rebuild`)

- **Done:** strain solver for 3+ stations: weighted least squares, full covariance, Monte Carlo uncertainties on derived quantities.
- **Done:** classic UTM mode (matches Cronin's spreadsheet) and a local-ENU default with velocity rotation.
- **Done:** data sources: NGL MIDAS (free, global, weekly) with caching; CSV/Excel import; NGL time series with velocity fitting.
- **Done:** the two 2020 map layouts recreated without cartopy:
  - `plotting.symbol_map`: landscape strain-symbol map;
  - `plotting.strain_map`: portrait velocity and strain-ellipse map with tables.

  Basemaps come from web-map tiles (`basemap.py`): Esri World Terrain (default), Stadia/Stamen terrain-background (needs a key), and others, plus an offline synthetic preview. A logo slot takes any image file or URL.
- **Done:** `plotting.network_map`: several triangles on one map (symbol or ellipse), like the 2021 Raton, NM figures, plus `gps-strain --triangles`.
- **Done:** command-line tool `gps-strain`.
- **Done:** the 2020 AC63 results table is reproduced to every printed digit in UTM mode. The 2020 strain algebra was right; its velocities were not.
- **Done:** 23 tests, including exact reproduction of Cronin & Resor's worked example, a rigid-plate null test and an offline map-render test.

### Not yet verified live

The build sandbox had no network access to NGL, Esri or Stadia. Untested against real servers:

- the MIDAS download (`data.load_midas`);
- the `.tenv3` loader;
- tile fetching, including the Esri `?token=` key format.

Run `gps-strain P146 P149 P150 -o tahoe --format png` once on a normal connection, then regenerate `examples/AC63_*.png` with real tiles.

## Next steps (package)

1. Live smoke test of MIDAS, `.tenv3` and tiles. Add a real `midas.NA.txt` excerpt as a test fixture.
2. Logo: get the official EarthScope logo (check EarthScope's usage terms) and pass it with `--logo`, or bundle it. The default data is NGL MIDAS, so match the logo to the data source.
3. **Licence-free terrain basemap (recommended):** do what GEOSLAM does for its printed maps and draw the terrain from a DEM hillshade:
   - USGS 3DEP for the US, Copernicus GLO-30 on AWS globally, neither needing a key;
   - add a hypsometric tint and water to get the Stamen terrain-background look with no tile licence at all.

   `usgs-relief` / `usgs-topo` tiles (public domain, US only) are already available as basemaps.
4. Basemap licence decision (see `docs/research/04_costs.md`): Esri Location Platform key (free at this scale; terms silent on printing), or a Stadia paid plan for the exact Stamen look (print rights only clearly cover its static-map product, so email Stadia first).
5. **GETSI Unit 6 (final project) support.** Unit 6 asks students to choose three adjacent NOTA stations in an actively straining area near people or infrastructure:
   - all interior angles above ~30° and no side over 100 km (both now checked and warned about);
   - map active faults from the USGS Quaternary Fault and Fold Database;
   - compute strain, interpret it, and present tectonics, hazard, risk and mitigation in a 5–7 minute talk.

   To add:
   - a Qfaults overlay on both maps (reuse GEOSLAM's `sources/faults.py`, the USGS Qfaults REST service, no key);
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
- **Basemap:** use whatever geoslam.org uses, subject to the licensing above.
- **Printing:** render the two sheets server-side with this package, or draw them in the browser.

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
