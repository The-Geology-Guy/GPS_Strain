# Cronin & Resor GPS Triangle-Strain Method: Research Notes

Compiled 2026-10-14 for rebuilding the `The-Geology-Guy/GPS_Strain` Python package.
Tools: WebSearch and WebFetch only. Shell network was blocked (proxy 403), so binary files (xls, docx, pptx, zip, nb) could not be downloaded or parsed. Their URLs are recorded below. The PDFs were readable through WebFetch.

**Primary source:** Cronin & Resor, *Algorithm for triangle strain* ("TriangleStrainAlgorithm.pdf", version of 5 April 2021). It contains a complete worked example (P146, P149, P150). I re-ran that example independently in numpy and reproduced every printed number (see section 4).

---

## 1. URLs found

### 1a. Cronin's site (croninprojects.org)

| URL | Description |
|---|---|
| https://croninprojects.org/Vince/Geodesy/TriangleStrainAlgorithm.pdf | **KEY.** "Algorithm for triangle strain", v. 5 Apr 2021. 17 steps plus the P146/P149/P150 worked example. Contacts: Vince_Cronin@baylor.edu, presor@wesleyan.edu |
| http://CroninProjects.org/Vince/Geodesy/TriangleStrainAlgorithm.doc | Older .doc version of the algorithm |
| http://CroninProjects.org/Vince/Geodesy/gps_strain_triangle_algorithm.docx | Docx version, flagged "needs updating" (binary, not readable) |
| https://croninprojects.org/Vince/Geodesy/GPS_Strain_Primer-20210613.pdf | **KEY.** "Primer on infinitesimal strain analysis in 1, 2 and 3-D", 13 Jun 2021. Theory, G-matrix, covariance |
| http://CroninProjects.org/Vince/Geodesy/GPS_Strain_Primer.doc | Older .doc primer |
| https://croninprojects.org/Vince/Geodesy/GPS-Ed.html | "GPS Strain in Cronin's Structural Geology Course": index of all course materials |
| http://CroninProjects.org/Vince/Geodesy/GPS_Triangle_Strain.xls | Original Excel triangle-strain calculator (binary) |
| http://CroninProjects.org/Vince/GPS-strain/gps_strain_calculator_excel.xls | Older copy of the Excel calculator |
| http://croninprojects.org/Vince/Geodesy/Calculators/gps-strain-calculator-mathematica.nb | Mathematica calculator. WebFetch errored on it; worth retrying manually |
| http://CroninProjects.org/Vince/GPS-strain/explanation_gps_strain_calcula.docx | Explanation of the calculator output |
| http://CroninProjects.org/Vince/GPS-strain/strain_ellipse_visualization_t.zip | Wolfram CDF strain-ellipse visualizer |
| http://CroninProjects.org/Vince/GPS-strain/PBOWorksheet15.doc ; http://CroninProjects.org/Vince/Geodesy/PBOWorksheet14.doc | Blank worksheets for a triplet of PBO sites |
| http://CroninProjects.org/Vince/GPS-strain/gps_infinitesimal_strain_analy.pdf | Cascadia/Wasatch/San Andreas datasheet. Blank; frame given as SNARF |
| http://CroninProjects.org/Vince/GPS-strain/gps_infinitesimal_strain_analy_1443091344.v2.docx | Student exercise |
| http://CroninProjects.org/Vince/GPS-strain/oregon_cascadia_post-exercise_.pptx | Cascadia post-exercise slides |
| http://CroninProjects.org/Vince/GPS-strain/GPS_Triangle_Strain.ppt ; http://CroninProjects.org/Vince/Geodesy/GPS_Triangle_Strain.ppt | "Basics of triangle strain analysis" slides |
| http://CroninProjects.org/Vince/Geodesy/GPS_ProjDescr_2014.docx | Project handout |
| http://CroninProjects.org/Vince/Geodesy/FindingGPSdata.doc | How to find PBO GPS data |
| http://CroninProjects.org/Vince/Geodesy/TriangleStrainPics.doc | Library of triangle-strain images |
| http://CroninProjects.org/Vince/Geodesy/Rubric13.doc | Grading rubric |
| http://CroninProjects.org/Vince/Geodesy/GPSstrainSGTFshortcourse.docx | 2014 SGTF short-course documents |
| http://CroninProjects.org/Vince/Geodesy/CascadiaSlides.pptx, CascadiaVelocityMap.doc, TahoeVelocityMap.doc, GPS_Intro.ppt | Supporting slides and maps |
| http://CroninProjects.org/Vince/Geodesy/{VectorSummary30Aug2012.docx, VectorWorksheet.docx, Matrices.doc, DotProd&MatrixWorksheet.docx, OrthogonalCoordTrans.doc, CoordTransHW.doc, ExtensionWorksheet.doc, 1D-ext-slides.ppt} | Math background |
| https://croninprojects.org/Vince/Course/IntroStructGeol/index.htm | Structural geology course page, linking everything above |
| http://croninprojects.org/Vince/Course/IntroStructGeol/{VectorSummary30Aug2012.pdf, Matrices.pdf, OrthogonalCoordTrans.pdf, StdDevNotes.pdf} | Math PDFs |
| http://CroninProjects.org/Vince/SLAM/DVFZ-example/{SLAM-code-2017.nb, Input_EQ_datafile.xls, thinDVFZCrop.dat} | SLAM Mathematica code and example data |
| https://croninprojects.org/Vince/Geodesy/index.htm | Geodesy index. Mostly "under construction" |

### 1b. GETSI / SERC "GPS, Strain, and Earthquakes"

Base URL for files: `https://cdn.serc.carleton.edu/files/getsi/teaching_materials/gps_strain/`

| File | Description |
|---|---|
| gps_strain_triangle_algorithm.v3.docx | "GPS Strain Triangle Algorithm", Cronin. Landing page: https://serc.carleton.edu/details/files/77379.html. Probably the same content as TriangleStrainAlgorithm.pdf |
| gps_strain_calculator_excel.v3.xls | **The Excel GPS Strain Calculator.** UTM conversion is in rows 41–88 (after Steven Dutch); eigenvector code is in rows 155–162. Binary, not read |
| gps_strain_calculator_matlab.zip | Phil Resor's MATLAB calculator, 4 kB. Binary, not read |
| explanation_gps_strain_calculator.v4.pdf / explanation_gps_strain_calcula.v5.docx | Explanation of calculator output, v. 4 Dec 2017, Cronin rev. Pratt-Sitaula |
| gps_strain_calculator_results.xlsx | Tohoku example results. Binary, not read; **high-value validation target** |
| example_interseismic_data_from.v5.pptx | Japan interseismic example data |
| gps_infinitesimal_strain_analy.v3.pptx | Unit 4 presentation |
| unit_4_gps_infinitesimal.v7.pdf | Unit 4 student exercise. Blank fields; gives plate rates |
| gps_infinitesimal_strain_analy_1443091344.v9.docx | Unit 4 exercise (docx) |
| gps_infinitesimal_strain_analy.v8.pdf / unit_4_gps_infinitesimal_1479532150627822284.v3.pdf | Vector map worksheets. Frame: NAM08 |
| unit_4_velocity_field.v5.pptx | Editable vector maps |
| gps-triangle-vector-map_gps_triangl.zip | Illustrator template for the vector map |
| strain_ellipse_visualization_t.zip | CDF strain-ellipse visualizer |
| gps_triangle_strain_ellipse.docx | Unit 2 triangle and strain-ellipse diagrams |
| infinitesimal_strain_analysis_.v2.docx | Unit 2 infinitesimal-strain reading (the primer) |
| finding_gps_data_oregon.v9.docx / unit_3_finding_gps.v10.pdf | Unit 3: finding GPS data (P395/P396/P404, NAM14) |
| unit_5_2014_south_14795312721393990867.v4.pdf | **Unit 5 Napa GPS data tables.** Real velocities; see section 4 |
| unit_5_2014_south_1479530399629605027.v8.pdf | Unit 5 Napa exercise (P198, P200, SVIN) |
| unit_5_2014_south.v7.pdf, 2014_south_napa_earthquake*.docx/pptx | Unit 5 reading, instructor notes, slides |
| unit_6_quotgps_strain.v3.pdf | Unit 6 final project |
| velocity-symbol.png.webp (under /images/...) | Blank velocity graph |

GETSI pages:
- Unit pages: https://serc.carleton.edu/getsi/teaching_materials/gps_strain/{index,overview,unit1..unit6,assessment}.html
- Student materials: https://serc.carleton.edu/getsi/teaching_materials/gps_strain/student_materials/index.html
- Offline ZIP of the whole module: https://serc.carleton.edu/admin/ajax/ajax_handler.php?action=generate_offline_content&type=commoncartridge1p0&module_id=6233&is_dev=
- Instructor-only (gated): notes file_id=77358; answer keys file_id=106541, which uses November 2016 data and includes the Tohoku solutions.

### 1c. Other sources

- Worrell (2016), Baylor MS thesis (SLAM plus GPS strain, South Napa): https://baylor-ir.tdl.org/bitstream/handle/2104/9796/WORRELL-THESIS-2016.pdf?sequence=1&isAllowed=y
- Steven Dutch, UTM formulas (the basis of the Excel UTM block): https://stevedutch.net/usefuldata/utmformulasold.htm (USGS/Army K1–K5 form) and https://stevedutch.net/usefuldata/utmformulas.htm (Karney)
- NGS NCAT converter, named in the algorithm doc: https://www.ngs.noaa.gov/NCAT/
- UNAVCO Velocity Viewer and frames: https://www.unavco.org/software/visualization/GPS-Velocity-Viewer/GPS-Velocity-Viewer.html and ...-frames.html
- NOTA position time series pattern: `https://data.unavco.org/archive/gnss/products/position/XXXX/XXXX.cwu.nam14.csv`
- Goodell (2013) southern California MATLAB activity, based on Resor's calculator: https://serc.carleton.edu/integrate/workshops/engineering2013/activities/goodell.html. MATLAB zip (218 MB): https://cdn.serc.carleton.edu/files_large/integrate/workshops/engineering2013/activities/matlab_code_crustal_strain.zip
- Original Python repo: https://github.com/The-Geology-Guy/GPS_Strain. Code is in `GPS_Strain/GPS_Strain.py`; raw access works via raw.githubusercontent.com.
- Scribd copy of an older "Triangle Strain Algorithm" (blocked by JavaScript): https://www.scribd.com/document/148902857/Triangles-Train-Algorithm

---

## 2. Computational procedure

Source: TriangleStrainAlgorithm.pdf (5 Apr 2021), with the primer (Jun 2021) and the original Python code. Step numbers follow the algorithm doc.

### Step 1: Inputs
- Three GPS stations forming a triangle. They must be non-collinear, and **no interior angle may be less than ~30°**.
- Horizontal velocities ve and vn in **m/yr** with 1σ uncertainties, "relative to a stable-North-America reference frame such as NAM08" (Step 9 says "SNARF").
- Convert locations (WGS84 geographic) to UTM coordinates **in the same UTM zone**. Tools: NGS NCAT, or rows 41–88 of the Excel calculator, "patterned after a conversion spreadsheet by Steven Dutch".
- **Zone-boundary rule (quote):** "compute the coordinates of the site(s) in the eastern zone as if they were in the western zone." In other words, use the westernmost zone's central meridian for all three stations. This is the "pseudo zone".
- Keep all values unrounded until the end.

### UTM forward conversion (Dutch / USGS-Army form, as used in the Excel calculator and the original Python)
Constants (WGS84): a = 6378137, b = 6356752.3142, k0 = 0.9996.
- e = sqrt(1 − b²/a²)
- e'² = (e·a/b)² = e²/(1−e²)
- n = (a−b)/(a+b)

Per station (lat in radians):
- ρ = a(1−e²)/(1−e² sin²lat)^(3/2). Computed but unused in the forward formulas.
- ν = a/(1−e² sin²lat)^(1/2)
- p = (lon − lon0) **in radians**. Dutch: "Calculate p in radians and omit the sin1" terms and the large power of ten multipliers."
- Meridional arc:
  M = a[(1 − e²/4 − 3e⁴/64 − 5e⁶/256)·lat − (3e²/8 + 3e⁴/32 + 45e⁶/1024)·sin 2lat + (15e⁴/256 + 45e⁶/1024)·sin 4lat − (35e⁶/3072)·sin 6lat]
- K1 = M·k0
- K2 = k0·ν·sin(2lat)/4
- K3 = [k0·ν·sin(lat)·cos³(lat)/24]·[5 − tan²lat + 9e'²cos²lat + 4e'⁴cos⁴lat]
- K4 = k0·ν·cos(lat)
- K5 = [k0·ν·cos³(lat)/6]·[1 − tan²lat + e'²cos²lat]
- Northing = K1 + K2·p² + K3·p⁴. Northern hemisphere only; no false northing is handled.
- Easting = 500000 + K4·p + K5·p³

Zone logic in the original Python (`GPS_Strain.py`, `strain_data.output_data`):
- zone = ceil((lon+180)/6), via a custom helper.
- Central meridian lon0 = −183 + 6·zone.
- Pseudo (western) CM = lon0 − 6, except −177 maps to 177.
- Westernmost zone selection: `if std(zones) > 5: 60 else floor(mean(zones))`.
- Each station uses its own zone's p if its zone equals the westernmost zone, otherwise the pseudo-zone p.
- Caveats:
  - floor(mean) is not min(zones) in general. It works only when the stations span at most two zones.
  - The pseudo-zone formula computes `abs(x) − z` in the antimeridian case, which looks dubious.
  - A rebuild should use min(zone) (with antimeridian handling) and the CM of that zone for every station. That matches Cronin's rule.

Cronin also points to NCAT. Any exact transverse Mercator projection (pyproj) on the westernmost CM agrees with this series to well under 1 m, and the strain is insensitive at that level.

### Steps 2–3: Centroid and centering
- cx = (x1+x2+x3)/3, cy = (y1+y2+y3)/3.
- Centered coordinates: xi = Ei − cx, yi = Ni − cy.
- The doc credits J.C. Savage with the suggestion to center. Centering makes the translation refer to the triangle center.

### Steps 4–5: Design matrix m2 (6×6)
Parameter vector m = [tx, ty, ω, exx, exy, eyy]ᵀ. Data vector m4 = [ve1, vn1, ve2, vn2, ve3, vn3]ᵀ, interleaved by station.

Rows for station i:
```
East : [1, 0, -yi, xi, yi, 0 ]
North: [0, 1,  xi, 0,  xi, yi]
```
These come from the primer's equations:
- vx = tx − yo·ω + xo·exx + yo·exy
- vy = ty + xo·ω + xo·exy + yo·eyy

Interpretation: the velocity gradient is L = [[exx, exy−ω], [exy+ω, eyy]]. The symmetric part is the strain rate ε; the antisymmetric part Ω has Ω21 = ω.
- **ω > 0 is counter-clockwise** (anticlockwise looking down). The doc's wording: "negative W means clockwise".
- **exy is TENSOR shear** (εxy = ½γxy). The primer says tensor shear is half the engineering shear, and the strain tensor is [[exx, exy], [exy, eyy]].

Worked-example m2 as printed (row order E1, N1, E2, N2, E3, N3):
```
1 0   8928.3 -26627.1  -8928.3        0
0 1 -26627.1        0 -26627.1  -8928.3
1 0 -21556.9   9693.8  21556.9        0
0 1   9693.8        0   9693.8  21556.9
1 0  12628.6  16933.4 -12628.6        0
0 1  16933.4        0  16933.4 -12628.6
```

### Steps 6–8: Solve
- m3 = m2⁻¹; m5 = m3·m4. The system is exactly determined: 6 equations, 6 unknowns.
- The doc suggests a Moore–Penrose pseudoinverse if the system is ill-conditioned.

### Step 9: Translation
- Translation (tx, ty) = m5[1], m5[2], "the coordinates of the mean translational velocity vector of the sites", in m/yr.
  - With centered coordinates, tx and ty equal the plain average of the three velocities exactly. Verified: tx = mean(ve) = −0.0101967.
  - The Explanation doc defines translation as that average.
- Speed = sqrt(tx² + ty²).
- Unit vector t̂ = (tx, ty)/speed.
- q1 = arccos(n̂·t̂) in degrees, with n̂ = (0, 1).
- Azimuth = 360 − q1 if tx < 0, otherwise q1. Azimuth is measured clockwise from north.

### Step 10: Rotation
- ω = m5[3], in rad/yr. Report it ×10⁹ as nano-rad/yr.
- The Excel and Python versions also report degrees/yr = ω·180/π.
- Negative means clockwise.

### Step 11: Strain tensor
- m6 = [[exx, exy], [exy, eyy]], the "Lagrangian infinitesimal strain (rate) tensor".

### Step 12: Principal strains (eigen decomposition)
- e1,2 = [(exx+eyy) ± sqrt(4·exy·eyx + (exx−eyy)²)]/2, with e1 ≥ e2. These are e1H and e2H, the max and min horizontal extension.
- Eigenvector proportional to (λ − e22, e12), or equivalently (e12, λ − e11). The primer writes it as (λ−ε11, ε12) normalized; for a symmetric tensor, use any correct form.
- Azimuth of S1H: q2 = arccos(n̂·ŝ1) in degrees. Azimuth = 360 − q2 if the eigenvector's x (east) component is negative, otherwise q2.
- The axis is bidirectional, so the doc reports both values: "~237° (or 57°)".
- S2H = S1H + 90°, reported as "327° (or 147°)".
- **Recommendation:** reduce axial azimuths to [0, 180) and report the alternative as +180.

### Step 13: Maximum shear
- Max infinitesimal shear strain = e1 − e2, in engineering-shear sense (γmax).
- It acts on planes at 45° to S1H. The Napa exercise gives the "azimuth (S1H + 45°)".
- The original Python computes 2·sqrt(((exx−eyy)/2)² + exy²), which is identical.

### Step 14: Area strain
- Areal strain = e1 + e2 = exx + eyy. Negative means area loss ("constriction"). Equivalently S1·S2 − 1, with S = 1 + e.

### Step 15: Invariants
- I1 = exx + eyy
- "I2" as defined in the algorithm doc = sqrt(exx² + eyy² + 2·exy²). This is the Kreemer et al. 2014 second invariant (Frobenius norm), **not** the classic principal invariant.
  - The primer's 3-D classic I2 = e1e2 + … reduces in 2-D to exx·eyy − exy² = det.
- I3 = det(m6) = e1·e2.
- Note: the Step 17 summary mislabels I2 as −2.1973e−8. Step 15 gives I2 = 3.2968e−8 and I3 = −2.1973e−17.

### Step 16: Uncertainties
- m7 = diag(1/σi²), the weight matrix, with order matching m4.
- m8 = m2ᵀ.
- m9 = (m8·m7·m2)⁻¹ = (GᵀWG)⁻¹. This equals G⁻¹ cov_d G⁻ᵀ because G is square.
- Reported uncertainties are **sqrt(diag(m9))**, i.e. 1σ. The primer says standard deviations are the square roots of the diagonal. The algorithm doc text calls the diagonal "squares of the uncertainties" but prints the square roots; my check confirms the square roots.
- Order: tx, ty, ω, exx, exy, eyy.
- No propagation to e1, e2 or the azimuths is given in Cronin's doc. Only tensor-component σ is reported.
- **Doc inconsistency:** the input table lists σ = 1e−5 and 3e−5, but the m7 printout shows 1/0.000012 and 1/0.000032. The printed results match **σ = 1e−5 and 3e−5** (weights 1/σ²); see section 4. The original Python hardcodes σ = 0.01 mm/yr, which is wrong and should be replaced with real uncertainties.

### Step 17: Summary output fields
- Translation E and N ± σ (m/yr)
- Translation azimuth and speed
- Rotation ± σ (nano-rad/yr, and deg/yr)
- Strain tensor components ± σ
- e1H (n-strain/yr) and azimuth of S1H
- e2H and azimuth of S2H
- Max shear at 45° to S1H
- Areal strain
- I1, I2, I3

Units: velocities m/yr; strain rates are dimensionless /yr, reported ×10⁹ as nanostrain/yr.

### Strain ellipse and map symbols
- **Cronin's docs:** a unit circle deforms to an ellipse with semi-axes S1 = 1 + e1 and S2 = 1 + e2. S1H axes are drawn in red and S2H in blue.
  - The infinitesimal change "would be so small" that it must be exaggerated. No factor is given.
  - The Excel calculator has a simple ellipse visualizer. The CDF tool provides an adjustable exaggeration.
- **Explanation doc (exact):** "Map symbols typically show black-filled arrows indicating contraction and white-filled arrows indicating stretching." No sizing rule is stated in Cronin or GETSI.
- **Original Python implementation (to replicate or improve):**
  - Ellipse: F_ = R·S·10⁶ + I, with S the principal strains and R a rotation. Then F = F_·F_ᵀ and an SVD of F gives the axes. The plot uses 180 points.
  - **Bug:** `theta = ω·180/π` (degrees) is passed into np.cos/np.sin.
  - Exaggeration: 10⁶.
  - Arrow symbols: size = linear rescale of |e|·10⁹ from [0.1, 300] nanostrain to [40, 80] (matplotlib units). Style: "simple, head_length=0.35, head_width=0.5, tail_width=0.2", black fill. "darrow" boxes are used for extension.
  - Colors: S1H 'tomato', S2H 'cornflowerblue'. Contraction and elongation fills are slategrey with alpha 0.15. Sites are g/b/r.
- Interpretation figure (Explanation doc):
  - Strike-slip faults lie at ≤45° to the minor axis.
  - Normal faults strike roughly parallel to the minor axis. (As worded; physically, normal faults strike perpendicular to S1H, i.e. parallel to S2H.)
  - Reverse faults and fold hinges are roughly perpendicular to the minor axis.

---

## 3. Velocity data sources and reference frames

- **Cronin's algorithm and primer:** velocities relative to stable North America, "such as NAM08", and the "Stable North American Reference Frame (SNARF)". Datum WGS84. The primer says velocities are the "mean velocities computed from time series" of PBO daily solutions.
- **Unit 4 worksheet (current):** NOTA site pages, decimal lat/long, WGS84, **NAM08**. N, E and Up velocities in mm/yr, each ± uncertainty.
- **Unit 3 (2024 update):** **NAM14**. Values come from the station overview page:
  - Coordinates: the "GPS Monument Coordinates" box, lat/lon/elev.
  - Velocities and uncertainties: the "Station Position" box, using the cleaned/detrended time-series view, which gives a numerical mean velocity ± uncertainty per component.
  - Example: **P395**, 45.02228, −123.85753, elev 53.050 m. N 9.10 ± 0.04, E 7.44 ± 0.06, U 0.02 ± 0.20 mm/yr. Horizontal 11.75 mm/yr. Accessed 18 Jul 2024.
- **Unit 5:** NOTA position CSV `XXXX.cwu.nam14.csv` (CWU analysis center, NAM14).
- **Velocity product:** students use the **station velocity estimates and their formal uncertainties** (the linear-fit velocity from the time series), **not averaged displacements.**
  - The original Python instead used `mean(delta E)/years`, which is wrong. It is biased by roughly a factor of two because it averages displacement relative to the first epoch.
  - A rebuild should use the official velocity files: the GAGE/NOTA `.vel` products in NAM14/IGS14 (CWU/NMT/combined), or a proper linear regression with uncertainties.
- Unit 4 plate-motion references:
  - JdF–NA: azimuth 46°, 0.019 m/yr at 47.055154, −126.258545 (ITRF2020, NA fixed).
  - PA–NA: azimuth 325°, 0.049 m/yr at 37.696318, −123.009224 (GSRM v1.2).
- GETSI Unit 4 recommended stations:
  - Cascadia: NEAH, P401, P403. P401 is co-located with P815 and P816.
  - Wasatch: P088, P116, COON.
  - San Andreas: P538, P539, P541.
  - The answer keys use November 2016 data.

---

## 4. Worked examples

### 4a. Cronin's algorithm example (TriangleStrainAlgorithm.pdf): primary validation case

Inputs (UTM, presumably zone 10. Inverse projection with zone 10 gives Lake Tahoe area coordinates: P146 ≈ 39.3375 N, −120.5373; P149 ≈ 39.6021, −120.1050; P150 ≈ 39.2924, −120.0339. Those lat/longs are my inference, not printed in the doc.)

| Site | UTM E (m) | UTM N (m) | vE (m/yr) | σE | vN (m/yr) | σN |
|---|---|---|---|---|---|---|
| P146 | 712245.807 | 4357118.796 | −0.01031 | 0.00001 | 0.00625 | 0.00001 |
| P149 | 748566.739 | 4387604.015 | −0.00942 | 0.00003 | 0.00520 | 0.00003 |
| P150 | 755806.255 | 4353418.458 | −0.01086 | 0.00003 | 0.00592 | 0.00003 |

Step 3 of the doc uses eastings 748566.261 and 755806.266 for P149 and P150, which is a typo-level inconsistency. Either set gives the same results to about 4–5 significant figures.

Printed intermediates and results:
- Center: (738872.9, 4366047.1).
- Centered coordinates: P146 (−26627.1, −8928.3); P149 (9693.8, 21556.9); P150 (16933.4, −12628.6).
- m5:
  - tx = −0.0101967
  - ty = 0.00579
  - ω = −2.48541e−8 rad/yr
  - exx = −9.21367e−9
  - exy = 1.53178e−8
  - eyy = −2.30811e−8
- Unit translation vector (−0.869587, 0.49378). q1 ≈ 60.41°. Azimuth ≈ 300°. Speed 0.0117259 m/yr.
- Rotation: −24.8541 ± 0.6723 nrad/yr (clockwise).
- Eigenvalues: e1 = 6.6663e−10 (0.66663 nstrain/yr); e2 = −3.2961e−8 (−32.9614 nstrain/yr).
- S1 unit eigenvector (−0.8404, −0.5420). q2 ≈ 123°. S1H azimuth ≈ 237° (or 57°). S2H ≈ 327° (or 147°).
- Max shear: 33.628 nstrain/yr. Areal strain = I1 = −32.2948 nstrain/yr. I2 = 3.2968e−8. I3 = −2.1973e−17.
- σ (sqrt diag m9):
  - tx, ty: 1.453e−5 each
  - ω: 6.7227e−10
  - exx: 6.7197e−10
  - exy: 6.7227e−10
  - eyy: 1.1646e−9
  - The summary prints exy as "1.5318 ± 0.06723×10⁻⁸" and eyy as "−2.3081 ± 0.1165×10⁻⁸".

**My independent reproduction (numpy, table eastings, σ = 1e−5/3e−5, W = diag(1/σ²)):**
- Centroid: 738872.93367, 4366047.08967.
- m: tx = −1.01966667e−2, ty = 5.79e−3, ω = −2.48541089e−8, exx = −9.21364518e−9, exy = 1.53177587e−8, eyy = −2.30810503e−8.
- Eigenvalues: −3.29613289e−8 and 6.66633410e−10.
- e1 eigenvector: (−0.84035, −0.54204), i.e. (E, N).
- sqrt diag: 1.45297e−5, 1.45297e−5, 6.72270e−10, 6.71967e−10, 6.72270e−10, 1.16458e−9. This **matches exactly**. With σ = 1.2e−5/3.2e−5 the values would be 1.5606e−5 etc., which do not match.
- I2 = 3.29681e−8. det = −2.19731e−17. Speed 0.01172587. q1 = 60.4107°. q2 = 122.823°, giving 237.18°.
- Rotation: −1.42404e−6 deg/yr.

This confirms:
- the matrix layout;
- ω positive = CCW;
- exy is tensor shear;
- uncertainties are sqrt(diag((GᵀWG)⁻¹)) with W = 1/σ².

### 4b. GETSI Unit 5 Napa data tables (real PBO/USGS velocities, doc dated 12 Nov 2014; mm/yr, frame stable NA)

| Sta | Lat | Lon | vN | σN | vE | σE |
|---|---|---|---|---|---|---|
| P193 | 38.1229 | −122.9081 | 32.506 | 0.65 | −22.583 | 0.495 |
| P194 | 38.1857 | −122.8163 | 28.22 | 0.634 | −19.368 | 0.496 |
| P196 | 38.2981 | −122.7426 | 25.248 | 0.652 | −17.129 | 0.499 |
| P197 | 38.4286 | −122.7674 | 25.523 | 0.654 | −15.895 | 0.51 |
| P198 | 38.2599 | −122.6074 | 23.147 | 0.64 | −16.213 | 0.494 |
| P199 | 38.2637 | −122.5034 | 19.802 | 0.641 | −14.274 | 0.498 |
| P200 | 38.2398 | −122.4517 | 19.059 | 0.674 | −14.555 | 0.514 |
| P201 | 38.5598 | −122.6584 | 18.219 | 0.666 | −12.424 | 0.5 |
| P202 | 38.4236 | −122.496 | 17.663 | 0.643 | −13.492 | 0.493 |
| P261 | 38.153 | −122.2175 | 14.35 | 0.657 | −11.962 | 0.548 |
| P263 | 38.5777 | −122.4292 | 14.93 | 0.644 | −11.494 | 0.512 |
| P264 | 38.4442 | −122.1953 | 10.379 | 0.744 | −10.724 | 0.579 |
| P265 | 38.5302 | −121.9542 | 7.754 | 0.666 | −9.88 | 0.516 |
| SVIN | 38.0332 | −122.5263 | 23.596 | 0.657 | −19.189 | 0.524 |

Coseismic displacements (mm; dN, σ, dE, σ):

| Sta | dN | σN | dE | σE |
|---|---|---|---|---|
| P193 | 0.8 | 1.0 | −1.7 | 0.8 |
| P194 | 0.8 | 0.9 | −3.3 | 0.8 |
| P196 | 2.0 | 1.1 | −5.7 | 1.1 |
| P197 | 3.5 | 0.8 | −6.2 | 1.2 |
| P198 | 4.5 | 0.9 | −9.5 | 1.4 |
| P199 | 9.9 | 1.1 | −21.7 | 2.5 |
| P200 | 17.3 | 1.7 | −19.1 | 2.1 |
| P201 | 1.4 | 0.9 | −3.6 | 1.1 |
| P202 | 8.7 | 1.4 | −16.1 | 1.8 |
| P261 | −14.6 | 1.8 | 25.7 | 3.0 |
| P263 | −1.3 | 0.8 | −2.1 | 0.8 |
| P264 | −16.8 | 2.2 | −10.8 | 1.4 |
| P265 | −2.0 | 1.1 | −3.1 | 1.0 |
| SVIN | 6.4 | 1.3 | 3.1 | 1.1 |

- Exercise triangle: **P198–P200–SVIN**.
- Recurrence (yr) = coseismic max shear (nstrain) / interseismic max shear rate (nstrain/yr).
- West Napa fault trend: 175°.
- No answer values are public; the answer key is gated.

### 4c. Worrell (2016) thesis, Napa

- Station inputs (Table C.1, pre-EQ, mm/yr, NA-fixed): same values as 4b, with more decimals in lat/lon. For example, P194 38.18571777, −122.8162587; P200 38.239832642, −122.451703296.
- Also P266: 38.183968807, −121.843527331, vN 8.21 ± 1.05, vE −10.209 ± 0.791.
- And P267: 38.380336146, −121.823235255, vN 8.031 ± 0.875, vE −9.317 ± 0.699.
- Outputs per group (pre-EQ, nstrain/yr; speed m/yr; azimuth; rotation sense; e1H; S1H azimuth; e2H):
  - Across: 0.0177 ± 0.0005, 319°, CCW, 349.1998, 7.7926°, −129.1814
  - E1: 0.0135 ± 0.0006, 310.5°, CCW, 16.0653, 38.6898°, −29.9524
  - E2: 0.0175 ± 0.0005, 319.2°, CW, 134.4977, 118.5902°, −179.3208
  - W1: 0.248 [sic, ≈0.0248] ± 0.0005, 323.5°, CW, 180.9922, 111.1966°, −178.2544
  - W2: 0.031 ± 0.0005, 325.5°, CW, 129.8219, 101.5256°, −134.6649
- **The station-to-group membership is only in Figure 10**, which was not extractable. The groups can be identified by brute-force testing of station triplets against these outputs: a useful validation exercise.
- Coseismic (Table 5) and post-EQ (Table 6) results are also printed in the thesis.

### 4d. Not yet obtained
- The Tohoku results xlsx (gps_strain_calculator_results.xlsx).
- The Excel calculator's embedded default example.
- The MATLAB zip and the Mathematica notebook.

All four are binary and need a manual download. They would give extra validation cases and the exact UTM implementation.

---

## 5. Caveats stated by Cronin / Resor

- The three sites must be non-collinear, with no interior angle < ~30°. Avoid skinny triangles, which make the system ill-conditioned.
- No triangle size limit is stated.
- The strain is a spatial average and is assumed **homogeneous** across the triangle. The method is "a first look". Use all local GPS sites and geologic/geophysical data for real characterization.
- Horizontal plane strain only. Vertical velocities are ignored; this is poor where uplift or subsidence is significant. One principal axis is assumed vertical.
- Rotation is a rigid-body spin and is meaningful only under plane strain.
- Infinitesimal: strains below about 1%.
- Three sites give an exact fit with no redundancy, but the data uncertainties still propagate. Station errors are assumed independent (diagonal covariance).
- Non-tectonic signals bias results: groundwater and surface-water loading, magmatism, earthquake-cycle transients and postseismic deformation, and landslides.
- All sites must be projected in one UTM zone, using the western zone for straddling triangles.
- The Napa recurrence model assumes small postseismic and inelastic strain and linear elastic superposition. Nearby faults add interseismic strain.

---

## 6. SLAM (Seismo-Lineament Analysis Method) relevance

- SLAM (Cronin and Baylor students) projects earthquake focal-mechanism nodal planes, with uncertainty, to the ground surface as "seismo-lineaments" to locate seismogenic faults. Code: http://CroninProjects.org/Vince/SLAM/DVFZ-example/SLAM-code-2017.nb. There is also a PyPI package, `geoslam` / `GEOSLAM`, likely related to the same original author.
- GPS triangle strain is a **complementary** input. Worrell (2016) computed GPS strain before, during and after the South Napa earthquake to characterize the regional strain and stress field. The kinematics (S1H/S2H orientations) are compared with the seismo-lineament trends, for example to check consistency with right-lateral slip on NNW-striking faults.
- The original Python README describes its outputs as "maps with seismo-lineament bounds defined" (AC63_strain.jpg / AC63_symbol.jpg). The strain-ellipse and symbol maps are drawn over a SLAM study area.
- There is no mathematical coupling between SLAM and the strain calculation.

---

## 7. Implementation notes for the rebuild (derived)

1. Use real velocities and uncertainties: NOTA/GAGE velocity products, NAM14 recommended, or regression.
2. Project to UTM on the westernmost zone's CM. Use exact TM (pyproj) or the Dutch series above.
3. Center on the centroid.
4. Build G with rows [1,0,−y,x,y,0] and [0,1,x,0,x,y].
5. Solve m = G⁻¹d.
6. Compute Cov = (GᵀWG)⁻¹ with W = diag(1/σ²) and report σ = sqrt(diag).
7. Compute e1,2 with eigh, and S1H azimuth = atan2(vE, vN) mod 180.
8. Compute γmax = e1 − e2, area strain = e1 + e2, and the invariants.
9. Report ω in nrad/yr and deg/yr; positive is CCW.
10. Optionally propagate the covariance to e1, e2 and the azimuths, e.g. by Monte Carlo. Cronin does not do this.
11. Validate against section 4a, which must reproduce to at least 5 significant figures.
