# 03 — Data Sources, Basemaps, Packaging, and Library Versions (state as of 2026-10-14)

Research context: modernizing the 2020 `GPS_Strain` package, which (a) pulled station metadata and position time series from `web-services.unavco.org`, and (b) used `cartopy.io.img_tiles.Stamen('terrain-background')` basemaps.

Verification notes: everything below was checked with WebSearch/WebFetch on 2026-10-14. The shell sandbox could not reach the data hosts (proxy 403) or PyPI, so **no file was downloaded and parsed locally**. Items marked **[unverified]** come from prior knowledge and should be confirmed on first real download.

---

## 1. UNAVCO → EarthScope Consortium (GAGE / NOTA)

### 1.1 Organisational status
- UNAVCO and IRIS merged into **EarthScope Consortium** (2023). The GNSS facility is now branded **NSF GAGE** / "NSF National Geophysical Facility (NGF)", operated by EarthScope. The legacy `www.unavco.org` pages are still live, and still updated: the web-services hub footer reads "Last modified: 2026-08-16". A new portal also exists at `ngf.earthscope.org`.
  - Web services hub: https://www.unavco.org/data/web-services/web-services.html
  - Release notes: https://www.unavco.org/data/web-services/release-notes.html
  - Data tools index: https://ngf.earthscope.org/data/tools

### 1.2 The legacy REST web services still exist, with no announced replacement
- `web-services.unavco.org` is still the documented service host. The **release notes have an entry dated 2026-04-02**: endpoints that support several formats, including `/gps/metadata/sites/v1`, now take a `?format=json|csv|xml` query parameter, which overrides the `Accept` header. So the old endpoints are **maintained, not retired**.
- Relevant historical change (2019-12-02, `/gps/data/position/{station}/v3`): NAM14 and IGS14 were added, the default analysis center changed from PBO to **CWU**, and the default frame changed from IGS08 to **NAM14**. The original GPS_Strain query parameters (`analysisCenter=cwu&referenceFrame=nam14&report=long&dataPostProcessing=Cleaned&refCoordOption=first_epoch`) therefore match the current defaults.
- Deprecated items: position `v2` (2016) and `/beta` (2014). v3 is current.
- **No IGS20/NAM20 option** is documented for the web services. Release notes and parameter docs list only IGS08/NAM08/IGS14/NAM14 (plus discontinued IGS05/SNF01, and NA12 for UNR).
- Documentation pages:
  - Position: https://www.unavco.org/data/web-services/documentation/gps-position-documentation.html (analysisCenter: PBO, NMT, CWU, UNR; report long/short; refCoordOption)
  - Velocity: https://www.unavco.org/data/web-services/documentation/gps-velocity-documentation.html (station, analysisCenter PBO/NMT/CWU, referenceFrame, report short/long, solutionType `snaps` (monthly) or `final` (annual))
  - Swagger URL builder: https://www.unavco.org/data/web-services/documentation/documentation.html#!/gps/getPositionByStationId and `#!/gps/getVelocityByStationId`
- **Authentication for web services: not stated anywhere.** The EarthScope identity-management page says IdM's "first release ... is for its HTTPS data file server" and that other systems will be announced before they adopt it. No announcement for web services was found. The `ngf.earthscope.org/data/tools` index labels all tools "Open".
- **Could not live-test**: WebFetch is blocked by robots.txt on `web-services.unavco.org`, and the shell proxy refuses the host. Treat "still works without auth" as **likely but unverified**. Code should handle HTTP 401/403/redirects cleanly and fall back to NGL (section 2).

Example URLs (the same patterns as 2020; patterns are documented, but live responses are unverified):
```
https://web-services.unavco.org/gps/metadata/sites/v1?minlatitude=32&maxlatitude=36&minlongitude=-120&maxlongitude=-115&summary=true&format=csv
https://web-services.unavco.org/gps/data/position/P123/v3?analysisCenter=cwu&referenceFrame=nam14&starttime=2015-01-01&endtime=2026-01-01&report=long&dataPostProcessing=Cleaned&refCoordOption=first_epoch
https://web-services.unavco.org/gps/data/velocity/P123/beta?analysisCenter=cwu&referenceFrame=nam14&report=long&solutionType=final   # [unverified path/version]
```

### 1.3 File server (data.earthscope.org): login required
- "As of March 30, 2023, user registration and login is now required" for the file server (https://www.unavco.org/data/gps-gnss/data-access-methods/data-access-methods.html). The file-server examples page says: "We now require all data users to register and login to access the NGF Data". **Derived products are included**, since the examples page uses `/archive/gnss/products/position/` with a token.
- Auth mechanism: the request header `Authorization: Bearer <token>`. Tokens expire after **24 h**. Get a token with the EarthScope CLI (`es login` once, then `es user get-access-token`), or copy a short-lived key from https://www.earthscope.org/user/info. Sign-up: https://data.earthscope.org (Google / CILogon supported).
  ```bash
  curl -L -O -f https://data.earthscope.org/archive/gnss/products/velocity/cwu.snaps_nam14.vel \
       --header "Authorization: Bearer $(es user get-access-token)"
  ```
  Docs: https://docs.earthscope.org/cli/http-examples , https://unavco.org/data/gps-gnss/file-server/file-server-access-examples.html
- Unauthenticated fetches of these URLs return the generic "EarthScope Data Server" navigation page, not the file. Verified for both the directory and the `.vel` URL.

GAGE derived-product URLs (from https://unavco.org/data/gps-gnss/derived-products/derived-products.html):

| Product | URL |
|---|---|
| Velocity dir | https://data.earthscope.org/archive/gnss/products/velocity/ |
| CWU monthly "snaps" NAM14 | https://data.earthscope.org/archive/gnss/products/velocity/cwu.snaps_nam14.vel |
| CWU snaps IGS14 | …/velocity/cwu.snaps_igs14.vel |
| Antarctic ANT14 | …/velocity/cwu.fanet_ant14.vel |
| Legacy IGS08 / NAM08 | …/velocity/cwu.snaps_igs08.vel , cwu.snaps_nam08.vel |
| Annual "final" releases | in release folders, e.g. `…/velocity/rel_201612/pbo.final_nam08.20161203.vel`; the latest release notes (20241228) name `cwu.final_nam14.vel.20250122220535`, `cwu.final_igs14.vel.…`, folder `rel_241228` |
| Station coordinate lists | …/products/position/gage_gps.nam14.txt , gage_gps.igs14.txt |
| Per-station time series | `…/products/position/<STA>/<STA>.cwu.nam14.csv` / `.pos` [path pattern unverified] |
| SINEX | https://data.earthscope.org/archive/gnss/products/sinex/ |

- Frames: **IGS14 / NAM14 / ANT14 are still current for GAGE**. The 2024-12-28 velocity release notes (https://www.unavco.org/data/gps-gnss/derived-products/docs/GAGE_Velocity_Field_Release_Notes_20241228.pdf) still use the ITRF2014-based NAM14. **No IGS20/NAM20 GAGE velocity product was found.**
- Note: `pbo.final_*` (the combined product) appears only in older releases (up to 2016). Current releases are CWU-only (`cwu.final_*`).
- Citation: each release has a DOI (e.g. https://doi.org/10.7283/P25H2H for the 2016 final). See https://ngf.earthscope.org/data/how-to-cite.

GAGE `.vel` column layout **[unverified, from prior knowledge of PBO velocity files]**: a header block, then one line per station:
`Dot# Name Ref_epoch Ref_jday Ref_X Ref_Y Ref_Z Ref_Nlat Ref_Elong Ref_Up dX/dt dY/dt dZ/dt SXd SYd SZd Rxy Rxz Ryz dN/dt dE/dt dU/dt SNd SEd SUd Rne Rnu Reu first_epoch last_epoch` (velocities in m/yr). Parse by header names, not by position.

`.pos` time-series format (verified, https://www.unavco.org/data/gps-gnss/derived-products/docs/knowledgetree-docs-old/gps_timeseries_format.pdf):
`YYYYMMDD HHMMSS JJJJJJ X Y Z xx yy zz xy xz yz N E U Ndel Edel Udel nn ee uu ne nu eu <quality>`. N/E are lat/lon (deg), U is height. `Ndel/Edel/Udel` are offsets (m) from the reference position. `quality` is final or rapid.

### 1.4 `earthscope-sdk` (Python)
- PyPI: https://pypi.org/project/earthscope-sdk , **v1.9.3 (2026-10-01)**, Python ≥3.10. Docs: https://docs.earthscope.org/sdk . Repo: https://gitlab.com/earthscope/public/earthscope-sdk . Companion CLI: `earthscope-cli` (`es`).
- Auth: you need a free EarthScope account. Credentials come from `es login` on the same machine, or from a refresh token in settings or the environment. Access tokens refresh automatically.
- Data features (changelog https://docs.earthscope.org/sdk/changelog): datasource discovery and data access (v1.2.0, 2025-12), `gnss_ephemeris_positions` (v1.4), "GNSS instantaneous positions" (v1.6.0, 2026-07), `es.data.gnss_observations(...)` returning PyArrow tables (raw phase/range/SNR, https://docs.earthscope.org/sdk/gnss-obs-tutorial), and boto3/S3 direct access helpers.
- **It does not (yet) expose daily station position time series, velocity fields, or a site-metadata bounding-box search.** It is the wrong tool for a strain-from-velocities app and adds a login requirement. Not recommended as the primary path.
- API root: https://api.earthscope.org (e.g. `/beta/user/profile`).

---

## 2. Free, no-auth velocity and time-series sources

### 2.1 Nevada Geodetic Laboratory (NGL, UNR): recommended primary source
Global, ~20k+ stations, updated weekly, **no login**, plain HTTPS. Products are now in **IGS20**. The old `/velocities/` folder is IGS14 and labelled "superseded" in https://geodesy.unr.edu/velocities/README.txt.

Directory: https://geodesy.unr.edu/gps_timeseries/IGS20/ (subdirs `kenv llh midas qa qa_rapid rapids rapids_5min tenv tenv3 trop tsplots txyz models`)

| Item | URL (verified listing, files dated 2026-10-12/13) |
|---|---|
| MIDAS velocities, IGS20 global | https://geodesy.unr.edu/gps_timeseries/IGS20/midas/midas.IGS.txt (also `.vel`, ~5.2 MB) |
| MIDAS, North-America-fixed | https://geodesy.unr.edu/gps_timeseries/IGS20/midas/midas.NA.txt (~2.3 MB) |
| Other plate-fixed frames | `midas.{AF,AN,AR,AU,BG,BU,CA,CO,EU,IN,MA,NB,NZ,OK,ON,PA,PM,PS,SA,SB,SC,SL,SO,SU,WL}.txt` |
| Step database / errors | `midas.IGS.step`, `midas.IGS.err` |
| Station lat/lon/height list | https://geodesy.unr.edu/gps_timeseries/IGS20/llh/llh.out (~1.0 MB; also `llh.out_sorted_by_add_date`) |
| Daily time series, IGS20 | `https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{STA}.tenv3` (e.g. `0ABI.tenv3`) |
| Daily time series, NA-fixed | `https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/NA/{STA}.NA.tenv3` (e.g. `ALBH.NA.tenv3`, `1LNT.NA.tenv3`) |
| Data holdings | https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt |
| Legacy IGS14 MIDAS | https://geodesy.unr.edu/velocities/midas.IGS14.txt , midas.NA.txt (frozen ~2024–2025) |

MIDAS line format (whitespace-separated, **27 columns**, file begins with a `---` line). Spec at https://geodesy.unr.edu/velocities/midas.readme.txt, and a real line from `midas.NA.txt` confirmed the column count:
```
1 sta  2 label(MIDAS5)  3 t_first(decyr)  4 t_last  5 duration(yr)  6 n_epochs  7 n_good  8 n_vel_pairs
9-11  vE vN vU (m/yr)          12-14 sigE sigN sigU (m/yr)
15-17 E N U offset at first epoch (m)   18-20 outlier fraction E N U
21-23 std dev of velocity pairs E N U   24 n_steps
25 lat(deg) 26 lon(deg) 27 height(m)
```
Example: `1LSU MIDAS5 2003.3073 2026.7543 23.4470 8079 7108 10379 -0.000713 0.001425 -0.002732 0.000260 0.000217 0.000684 ...`
Lat/lon are included, so **one file gives everything a strain calculation needs**: bounding-box filtering, vE/vN, and sigmas. Longitudes may be 0–360 or negative; normalise them. Check the uncertainties: MIDAS sigmas are realistic but conservative.

`.tenv3` format (https://geodesy.unr.edu/gps_timeseries/readmes/README_tenv3.txt), header line then 23 columns:
`site YYMMMDD decyr MJD GPSweek dow reflon e0(int) east(frac) n0 north u0 up ant_h sig_e sig_n sig_u corr_en corr_eu corr_nu lat lon h`. Positions are integer + fractional metres. Easting is from the reference meridian, using Blewitt (2024) graticule coordinates.

Terms and citation:
- Cite **Blewitt, G., W. C. Hammond, and C. Kreemer (2018), Harnessing the GPS data explosion for interdisciplinary science, Eos, 99, https://doi.org/10.1029/2018EO104623**. For MIDAS velocities, also cite **Blewitt et al. (2016), JGR 121, doi:10.1002/2015JB012552**. NGL also asks users to cite the original station data DOIs where available (https://geodesy.unr.edu/).
- Disclaimer (https://geodesy.unr.edu/Disclaimer.php): NBMG data policy. Data is supplied "as-is", with no warranties. No registration or explicit redistribution restriction was found. Be polite: cache, and don't hammer per-station downloads.

### 2.2 GSRM (Global Strain Rate Model, Kreemer et al. 2014)
- Files: https://geodesy.unr.edu/GSRM/ (e.g. `GPS_*.gmt` site velocity files per region, `vel_0.1deg*.gmt` model velocity grids, `GSRM_strain.txt.Z`, `GSRM_average_strain_v2.1.txt.Z`; `v2.2/` subfolder dated 2021). Results page: https://geodesy.unr.edu/cornekreemer/gsrm.htm . Also at https://www.globalquakemodel.org/product/gsrm .
- Static (2014–2021). Useful as a **validation reference** for computed strain rates, not as a live data source. Cite Kreemer, Blewitt & Klein (2014), G-cubed, doi:10.1002/2014GC005407.

### 2.3 JPL GNSS time series
- https://sideshow.jpl.nasa.gov/pub/JPL_GNSS_Timeseries/ , with `repro2018a/` and a new **`repro2026a/`** (dirs `raw/ post/ rep/ repeatability/ dops/`, updated 2026-10-14). The old `JPL_GPS_Timeseries/` has been frozen since 2019. No auth. Global, but fewer stations (~2–3k) than NGL. Format: per-station `.series` files **[column layout unverified]**. Second-tier option.

### 2.4 Coverage summary
NGL is global and dense (all of NOTA/PBO plus most public networks worldwide), no-auth, and IGS20 or plate-fixed. GAGE is North America/Caribbean/Antarctica, NAM14, and needs login for files. Recommendation: **NGL MIDAS as the default** (works anywhere). GAGE through the web services or an authenticated file server is an optional "NOTA/CWU" source.

---

## 3. Basemaps

### 3.1 cartopy status
- **cartopy 0.26.0 (2026-09-17), Python ≥3.11** (https://pypi.org/project/Cartopy/). It ships binary wheels and depends on shapely, pyproj, and pyshp. **No GDAL is required.**
- `cimgt.Stamen` **still exists but is effectively dead**: its constructor warns "The Stamen styles are no longer served by Stamen and are now served by Stadia Maps". Since 0.23 (2024-04), use `StadiaMapsTiles` (https://cartopy.readthedocs.io/stable/_sources/whatsnew/v0.23.rst.txt).
- Classes in `cartopy.io.img_tiles` (source: https://github.com/SciTools/cartopy/blob/main/lib/cartopy/io/img_tiles.py): `GoogleWTS, GoogleTiles, OSM, StadiaMapsTiles, Stamen, ThunderforestTiles, MapboxTiles, MapboxStyleTiles, QuadtreeTiles, OrdnanceSurvey, AzureMapsTiles, LINZMapsTiles, MapQuestOSM, MapQuestOpenAerial`. You can subclass `GoogleWTS` with `_image_url()` to hit any XYZ server (Esri, OpenTopoMap).
- `StadiaMapsTiles(apikey, style="alidade_smooth", resolution="", cache=False)`. **`apikey` is required** (no default). Styles include `stamen_terrain`, `stamen_terrain_background`, `stamen_toner`, `outdoors`, `osm_bright`. Attribution: "© Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors".

### 3.2 Options compared

| Source | Key? | Licence / terms | Fit for printed PDF |
|---|---|---|---|
| **Natural Earth via `cartopy.feature`** (LAND, OCEAN, COASTLINE, BORDERS, STATES, LAKES, RIVERS) + `ax.stock_img()` | No | **Public domain** | **Best default.** Vector, crisp at any print size, no attribution. Data downloads on first use from naturalearth S3, so **pre-bundle it** (set `cartopy.config['pre_existing_data_dir']`) for offline desktop use. |
| Natural Earth raster shaded relief (NE1/NE2/SR_HR/GRAY_HR, 1:10m) | No | Public domain | Good "terrain-background" substitute. Bundle one GeoTIFF/PNG (~50–300 MB at 10m; the 50m versions are small) and `imshow` with `transform=PlateCarree()`. |
| Esri World Shaded Relief `https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}` | No key in practice (xyzservices lists none) | "Copyright:(c) 2014 Esri", Esri terms (https://www.arcgis.com/home/item.html?id=9c5370d0b54f4de1b48a3792d7377ff2). Service is "In mature support; no longer updated". Max zoom 13. | Closest visual match to Stamen terrain-background. Fine for research or non-commercial use **with attribution**. Commercial redistribution falls under Esri's terms, so not "reliably free". |
| OpenTopoMap `https://{a-c}.tile.opentopomap.org/{z}/{x}/{y}.png` | No | **CC-BY-SA 3.0**, attribution "© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors". Contact the maintainers for heavy use. | Prints are OK with attribution. Share-alike applies to derived *map imagery*. Volunteer-run, so no SLA. |
| OSM standard `cimgt.OSM()` | No | ODbL data. **OSMF tile policy** (https://operations.osmfoundation.org/policies/tiles/): requires a unique User-Agent (cartopy's default `CartoPy/x.y` is generic; set your own), prohibits bulk/offline prefetch, requires visible attribution. | Usable for occasional interactive figures. Not for an app that many users will run. |
| Stadia (`StadiaMapsTiles`) | **Yes** for desktop/server (keyless only for browser domain auth or localhost dev, rate-limited) | Free tier is **non-commercial only**: individuals/non-profits. A for-profit org is commercial even without revenue. **Printing (≤5000 copies) needs a paid plan**, and so does offline caching (https://stadiamaps.com/faqs/, https://docs.stadiamaps.com/authentication/). | Avoid for a distributed app, unless each user brings their own key. |
| `cimgt.GoogleTiles(style='terrain')` | No key (undocumented endpoint) | Violates Google ToS | Don't use. |
| contextily 1.7.1 (2026-07-10) + xyzservices | Depends on provider | Same as the underlying provider | Convenient catalogue (`xyz.Esri.WorldShadedRelief`, `xyz.OpenTopoMap`). Mainly for Web-Mercator axes. With cartopy, a `GoogleWTS` subclass is simpler. |

Recommendation: **Natural Earth vectors plus an optional bundled Natural Earth shaded-relief raster** as the default, so the map works offline, needs no key, is public domain, and prints cleanly. Offer Esri Shaded Relief or OpenTopoMap as optional online backgrounds, with an automatic attribution string stamped on the figure.

---

## 4. Desktop packaging (Mac + Windows, printable PDF maps)

### 4.1 Current tool versions
- PySide6 **6.12.0** (2026-10-08), Python 3.10–3.15, LGPL-3.0 / GPL
- PyInstaller **6.22.3** (2026-09-12), Python 3.8–3.15
- Briefcase **0.4.6** (2026-10-13), Python ≥3.11 (BeeWare; builds .app/.dmg on macOS and .msi on Windows; has `--identity` signing and notarization support **[signing details unverified on PyPI page]**)
- pywebview **6.2.1** (2026-04-15)
- Tauri **v2.11.5** (core), tauri-cli 2.11.4 (https://github.com/tauri-apps/tauri/releases)

### 4.2 Options

| Option | Pros | Cons |
|---|---|---|
| **A. PySide6 + matplotlib/cartopy, frozen with PyInstaller (or Briefcase)** | Reuses existing Python science code. matplotlib's `savefig('x.pdf')` gives true vector PDFs. Cartopy ≥0.22 wheels contain no GDAL; pyproj wheels bundle PROJ and proj.db; shapely wheels bundle GEOS. So packaging pain is much lower than in 2020. | Bundle size ~150–300 MB. You must ship Natural Earth data and the PROJ data dir (PyInstaller hooks for pyproj and cartopy exist). One build per OS/arch (macOS arm64 + x86_64 or universal2; Windows x64). |
| B. pywebview (Python backend + HTML/Leaflet UI) | Lighter UI work, uses the system webview | Still freezes Python, so the same packaging cost as A, plus two UI stacks |
| C. Tauri/Electron front-end + PyInstaller "sidecar" | Modern UI, small Tauri shell | Two toolchains (Rust/Node + Python), and both binaries need signing. Most complex. |
| D. **Pure JS/TS (static web app, optionally wrapped in Tauri)** | Strain math (least-squares on velocity triangles/grids: Delaunay, 2×2 tensor, eigen) is trivial in JS (`d3-delaunay`, small linear algebra). Projections via `proj4js`. Maps via Leaflet/MapLibre/OpenLayers. Can be deployed as a website with **zero install and zero signing**. | **NGL/UNAVCO send no CORS headers [unverified]**, so a pure browser app needs a proxy or pre-built data. Tauri's Rust HTTP plugin avoids CORS. Print-quality vector PDF maps are harder: tiles are raster; a vector PDF needs a custom SVG/D3-geo renderer + jsPDF/svg2pdf, or headless print. Rewrites the existing Python. |

### 4.3 Code signing and notarization (2026)
- **macOS**: distributing outside the App Store needs an Apple Developer Program membership ($99/yr), a **Developer ID Application** certificate, hardened runtime, and **notarization** (`notarytool`) plus stapling. Since **macOS 15 Sequoia the Control-click → Open bypass is gone**: users must go to System Settings → Privacy & Security → "Open Anyway" for un-notarized apps (https://cybersecuritynews.com/macos-gatekeeper/). Every bundled `.dylib`/`.so` (numpy, PROJ, GEOS, Qt) must be signed. PyInstaller's `--codesign-identity` and Briefcase handle this, but expect iteration.
- **Windows**: unsigned or new binaries trigger **SmartScreen** "Windows protected your PC". **EV certificates stopped bypassing SmartScreen in 2024**; all certs now build reputation over time (https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options). The cheapest legitimate route is **Azure Artifact Signing** (formerly Trusted Signing; Basic tier ~$9.99/mo **[price from memory; page showed placeholders]**). It is open to individuals in the **USA and Canada** and orgs in US/CA/EU/UK. Alternative: MSIX via the **Microsoft Store**, where signing is free and done by Microsoft.

### 4.4 Recommendation
**Option A: PySide6 + matplotlib/cartopy + PyInstaller (one-dir mode, then .dmg/.msi or Inno Setup), built per-OS in GitHub Actions** (macos-14 arm64, macos-13 x86_64 or universal2, windows-latest).
Reasoning:
1. The core deliverable is a **printable vector PDF map**. matplotlib/cartopy does this natively and well; the JS stack would need a custom vector renderer.
2. The 2020 "cartopy/GDAL/PROJ packaging pain" has largely gone: all of cartopy's geo dependencies ship as self-contained wheels, and GDAL isn't needed.
3. It reuses the existing numerical Python code. Section 5 lists the required upgrades.
4. With an offline Natural Earth basemap and NGL text files, the app needs **no API keys and no accounts**.
5. Budget for $99/yr Apple + ~$120/yr Azure Artifact Signing if a warning-free install matters. Otherwise ship unsigned builds with "Open Anyway" instructions.

Secondary track: if broad zero-install reach matters more than PDF fidelity, build a static web version (Option D). Use a tiny CORS proxy (or a nightly GitHub Action that mirrors `midas.NA.txt`/`midas.IGS.txt` to a JSON file), MapLibre for the interactive map, and SVG → PDF export for print.

---

## 5. Library versions (PyPI, checked 2026-10-14)

| Package | Latest | Released | Python | Notes |
|---|---|---|---|---|
| numpy | **2.5.4** | 2026-10-10 | ≥3.12 | NumPy 2.x removed aliases like `np.float`/`np.int` (gone since 1.24) and changed some scalar promotion rules (NEP 50) |
| pandas | **3.0.6** | 2026-09-17 | ≥3.11 | `DataFrame.append` was **removed in 2.0**: use `pd.concat`. pandas 3.0 makes **Copy-on-Write the default**, so chained assignment `df[c][i] = x` no longer works (use `.loc`), and the default string dtype is now `str` (pyarrow-backed) |
| matplotlib | **3.11.2** | 2026-09-11 | ≥3.11 | |
| cartopy | **0.26.0** | 2026-09-17 | ≥3.11 | `Stamen` is deprecated, use `StadiaMapsTiles` (needs a key) |
| pyproj | **3.8.0** | 2026-09-05 | ≥3.12 | Wheels bundle PROJ. Use `Transformer.from_crs(...)`; the old `pyproj.transform()` and `Proj(init=...)` were removed |
| contextily | 1.7.1 | 2026-07-10 | ≥3.10 | |
| earthscope-sdk | 1.9.3 | 2026-10-01 | ≥3.10 | |

The minimum Python for the full stack is **3.12** (numpy, pyproj). Target Python 3.12 or 3.13 for the frozen app.

---

## Sources
- UNAVCO web services hub / release notes / docs: https://www.unavco.org/data/web-services/web-services.html , https://www.unavco.org/data/web-services/release-notes.html , https://www.unavco.org/data/web-services/documentation/gps-position-documentation.html , https://www.unavco.org/data/web-services/documentation/gps-velocity-documentation.html
- GAGE derived products: https://unavco.org/data/gps-gnss/derived-products/derived-products.html ; release notes 20241228: https://www.unavco.org/data/gps-gnss/derived-products/docs/GAGE_Velocity_Field_Release_Notes_20241228.pdf ; DOI example https://www.unavco.org/data/doi/10.7283/P25H2H
- EarthScope auth: https://www.unavco.org/data/gps-gnss/data-access-methods/data-access-methods.html , https://unavco.org/data/gps-gnss/file-server/file-server-access-examples.html , https://www.unavco.org/data/gps-gnss/file-server/user-profile.html , https://docs.earthscope.org/cli/http-examples
- EarthScope SDK: https://pypi.org/project/earthscope-sdk , https://docs.earthscope.org/sdk/changelog , https://docs.earthscope.org/sdk/gnss-obs-tutorial
- NGL: https://geodesy.unr.edu/ , https://geodesy.unr.edu/gps_timeseries/IGS20/ , https://geodesy.unr.edu/velocities/midas.readme.txt , https://geodesy.unr.edu/gps_timeseries/readmes/README_tenv3.txt , https://geodesy.unr.edu/Disclaimer.php
- GSRM: https://geodesy.unr.edu/GSRM/ , https://geodesy.unr.edu/cornekreemer/gsrm.htm
- JPL: https://sideshow.jpl.nasa.gov/pub/JPL_GNSS_Timeseries/
- cartopy: https://pypi.org/project/Cartopy/ , https://github.com/SciTools/cartopy/blob/main/lib/cartopy/io/img_tiles.py , https://cartopy.readthedocs.io/stable/_sources/whatsnew/v0.23.rst.txt
- Tiles: https://docs.stadiamaps.com/authentication/ , https://stadiamaps.com/faqs/ , https://operations.osmfoundation.org/policies/tiles/ , https://wiki.openstreetmap.org/wiki/OpenTopoMap , https://elevation3d.arcgis.com/arcgis/rest/services/World_Shaded_Relief/MapServer?f=pjson , https://xyzservices.readthedocs.io/en/stable/introduction.html
- Signing: https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/code-signing-options , https://azure.microsoft.com/pricing/details/trusted-signing/ , https://cybersecuritynews.com/macos-gatekeeper/
- Versions: https://pypi.org/project/numpy/ , https://pypi.org/project/pandas/ , https://pypi.org/project/matplotlib/ , https://pypi.org/project/pyproj/ , https://pypi.org/project/contextily/ , https://pypi.org/project/PySide6/ , https://pypi.org/project/pyinstaller/ , https://pypi.org/project/briefcase/ , https://pypi.org/project/pywebview/ , https://github.com/tauri-apps/tauri/releases
