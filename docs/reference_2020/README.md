# Reference figures from the 2020–2021 package

These are the original outputs of Luke Pajer's 2020 GPS_Strain notebook ("Adding a Pythonic resource to the GPS station strain calculator", June 2020) and the 2021 Raton, NM study with Eric Lindsey (UNM). The v2 maps are designed to match them.

| File | What it is |
|---|---|
| `AC63_strain.jpg`, `AC63_symbol.jpg` | AC63–AC62–AB37 (Alaska): portrait strain-ellipse map and landscape symbol map |
| `P146_strain.jpg`, `P146_symbol.jpg` | P146–P142–P090 (Lake Tahoe area) |
| `Symbol_types.png` | All possible map symbols for combinations of e1H and e2H |
| `raton_symbol_all.jpg`, `raton_strain_all.jpg` | Seven triangles around Raton, NM on one map (`network_map` in v2) |

The basemaps are Stamen terrain-background tiles (© Stamen Design, CC BY 3.0; data © OpenStreetMap contributors). That service is now hosted by Stadia Maps.

**Numbers on these figures used the 2020 velocity method** (mean displacement ÷ whole years over the previous 365 days, σ fixed at 0.01 mm/yr), so the strain values are not reliable. Given the same velocities, though, v2's classic UTM mode reproduces the 2020 strain outputs exactly (`tests/test_strain.py::test_matches_2020_ac63_results_table`).
